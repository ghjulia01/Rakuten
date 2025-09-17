# models/cnn_features.py
# =======================================================
# Extraire un embedding CNN (ResNet) compatible scikit-learn
# → lire imageid/productid, batcher, normaliser, renvoyer csr_matrix
# =======================================================
from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from concurrent.futures import ThreadPoolExecutor, as_completed

from main.profiling_tools import profile_func

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

ARCH_REGISTRY = {
    "resnet18":  (resnet18,  ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50":  (resnet50,  ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2, 2048),
}


class CNNFeaturizer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn qui :
      - lire les fichiers images à partir de imageid/productid
      - extraire un embedding CNN pré-entraîné (par défaut ResNet50 → 2048d)
      - renvoyer un csr_matrix (bien compatible avec TF-IDF sparse)
    """
    @profile_func
    def __init__(
        self,
        image_dir: str,
        arch: str = "resnet50",
        batch_size: int = 32,
        device: str = "auto",           # "auto" | "cpu" | "cuda"
        use_imagenet_norm: bool = True, # normaliser comme ImageNet
        fallback_zero: bool = True,     # image manquante → vecteur 0
        dtype: str = "float32",         # "float32" conseillé (mémoire)
        num_workers: int = 0,
        # --- nouveaux paramètres ---
        trainable_last_n: int = 0,     # nb de paramètres finaux à défiger (0 = tout figé, features “classiques”)
        finetune_epochs: int = 0,      # nb d’époques de fine-tuning (0 = pas de FT)
        finetune_lr: float = 3e-4,
        finetune_weight_decay: float = 0.01,
        finetune_max_n: int = 8000,    # échantillon max utilisé pour FT (pour rester rapide)
        trainable_last_layers: int = 1, # nb de couches (transformer blocks) à défiger si HF
        hf_model_name: Optional[str] = None,  # ex: "google/vit-base-patch16-224"
        hf_revision: Optional[str] = None,    # ex: "main"
        hf_feature_dim: Optional[int] = None, # si les dim embeddings HF sont connus, sinon déduit 
    ):
        self.image_dir = image_dir
        self.arch = arch
        self.batch_size = int(batch_size)
        self.device = device
        self.use_imagenet_norm = use_imagenet_norm
        self.fallback_zero = fallback_zero
        self.dtype = dtype
        self.num_workers = int(num_workers)

        # internes
        self._model = None
        self._preprocess = None
        self._feat_dim = None
        self._device_resolved = None

        # stats diagnostics
        self.n_total = 0
        self.n_loaded = 0
        self.n_missing = 0
        self.n_failed = 0

        # --- nouveaux paramètres (unfreeze / FT / HF) ---
        self.trainable_last_n     = int(trainable_last_n)
        self.finetune_epochs      = int(finetune_epochs)
        self.finetune_lr          = float(finetune_lr)
        self.finetune_weight_decay= float(finetune_weight_decay)
        self.finetune_max_n       = int(finetune_max_n)
        self.trainable_last_layers = int(trainable_last_layers)

        self.hf_model_name  = hf_model_name
        self.hf_revision    = hf_revision
        self.hf_feature_dim = int(hf_feature_dim) if hf_feature_dim is not None else None

    # -------- Utilitaires -------------------------------------------------------
    @profile_func
    def set_image_dir(self, new_dir: str):
        """Mettre à jour le dossier images (utile pour passer TRAIN → TEST)."""
        self.image_dir = new_dir

    @profile_func
    def _resolve_device(self):
        if self._device_resolved is not None:
            return self._device_resolved
        if self.device == "cuda":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cpu":
            dev = "cpu"
        else:  # auto
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_resolved = torch.device(dev)
        return self._device_resolved

    @profile_func
    def _build_model(self):
        # HF branch (ViT & co)
        if self.hf_model_name:
            from transformers import AutoImageProcessor, AutoModel
            device = self._resolve_device()
            processor = AutoImageProcessor.from_pretrained(self.hf_model_name, revision=self.hf_revision)
            base = AutoModel.from_pretrained(self.hf_model_name, revision=self.hf_revision).to(device)
            base.eval()

            class HFBackbone(torch.nn.Module):
                def __init__(self, base, processor, device):
                    super().__init__()
                    self.base = base
                    self.processor = processor
                    self.device = device
                def forward(self, x):   # x: (B,3,H,W) in [0,1]
                    imgs = [transforms.ToPILImage()(xi.cpu()) for xi in x]
                    inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
                    out = self.base(**inputs)
                    if hasattr(out, "pooler_output") and out.pooler_output is not None:
                        z = out.pooler_output
                    elif hasattr(out, "last_hidden_state"):
                        z = out.last_hidden_state[:, 0]   # [CLS]
                    else:
                        z0 = out[0] if isinstance(out, (tuple, list)) else out
                        z = z0.mean(dim=1) if z0.ndim == 3 else z0
                    return z

            model = HFBackbone(base, processor, device).to(device)
            self._feat_dim = int(self.hf_feature_dim or 768)  # ViT-base = 768
            # Préprocess standard (le processor fait le reste)
            preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
            # On emballe pour garder la même API que torchvision
            return model, preprocess

        # Torchvision branch (la logique actuelle, résumée)
        arch_key = str(self.arch).lower()
        if arch_key not in ARCH_REGISTRY:
            raise ValueError(f"Architecture inconnue: {self.arch} (supportées: {list(ARCH_REGISTRY)})")
        ctor, weights_enum, feat_dim = ARCH_REGISTRY[arch_key]
        weights = weights_enum
        model = ctor(weights=weights)
        model.fc = nn.Identity()
        model.eval()
        model.to(self._resolve_device())
        preprocess = weights.transforms() if self.use_imagenet_norm else transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
        ])
        self._feat_dim = feat_dim
        return model, preprocess

    def _set_trainable_tail(self, n_params: int):
        """Défige la queue du backbone :
           - HF/ViT : N derniers blocs Transformer (trainable_last_layers)
           - ResNet : layer4 complet
           - Fallback : derniers `n_params` paramètres
        """
        # Cas HF (ViT) : dernier(s) bloc(s) du Transformer
        if hasattr(self._model, "base") and hasattr(self._model.base, "encoder") and hasattr(self._model.base.encoder, "layer"):
            for p in self._model.base.parameters():
                p.requires_grad = False
            n_layers = max(1, int(getattr(self, "trainable_last_layers", 1)))
            for blk in list(self._model.base.encoder.layer)[-n_layers:]:
                for p in blk.parameters():
                    p.requires_grad = True
            print(f"[INFO] HF unfreeze: last {n_layers} transformer block(s).")
            return

        # Cas ResNet (torchvision) : défige layer4
        if hasattr(self._model, "layer4"):
            for p in self._model.parameters():
                p.requires_grad = False
            for p in self._model.layer4.parameters():
                p.requires_grad = True
            print("[INFO] ResNet unfreeze: layer4")
            return

        # Fallback générique : défige les n derniers paramètres
        for p in self._model.parameters():
            p.requires_grad = False
        if n_params > 0:
            tail = list(self._model.parameters())[-n_params:]
            for p in tail:
                p.requires_grad = True
            print(f"[INFO] Unfreeze last {n_params} parameters (generic)")

    @profile_func
    def _lazy_load(self):
        if self._model is None:
            self._model, self._preprocess = self._build_model()
    
    @profile_func
    def _path_from_row(self, row) -> str:
        # Nommage Rakuten : image_{imageid}_product_{productid}.jpg
        fname = f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"
        return os.path.join(self.image_dir, fname)

    # -------- API sklearn -------------------------------------------------------
    @profile_func
    def fit(self, X, y=None):
        self._lazy_load()
        self.n_total = self.n_loaded = self.n_missing = self.n_failed = 0


        # --- Fine-tuning optionnel (supervisé) ---
        if self.finetune_epochs and y is not None and (self.trainable_last_n > 0 or getattr(self, "trainable_last_layers", 0) > 0):
            from sklearn.preprocessing import LabelEncoder
            from torch.utils.data import TensorDataset, DataLoader
            import math

            # 1) Encode labels
            le = LabelEncoder()
            y_enc = le.fit_transform(np.asarray(y))

            # 2) Sous-échantillon rapide
            n = min(len(X), int(self.finetune_max_n))
            idx = np.random.RandomState(42).permutation(len(X))[:n]
            X_ft = X.iloc[idx].reset_index(drop=True)
            y_ft = y_enc[idx]

            # 3) Préparer training minimal
            device = self._resolve_device()
            self._set_trainable_tail(self.trainable_last_n)
            self._model.train()

            head = nn.Linear(self._feat_dim, int(len(le.classes_))).to(device)
            opt = torch.optim.AdamW(
                [{"params": [p for p in self._model.parameters() if p.requires_grad], "lr": self.finetune_lr},
                {"params": head.parameters(), "lr": self.finetune_lr}],
                weight_decay=self.finetune_weight_decay
            )
            criterion = nn.CrossEntropyLoss()

            # petit loader CPU -> tensor
            bs = int(self.batch_size)
            steps_per_epoch = math.ceil(len(X_ft)/bs)

            for epoch in range(int(self.finetune_epochs)):
                i = 0
                losses = []
                while i < len(X_ft):
                    j = min(i + bs, len(X_ft))
                    paths_slice = [self._path_from_row(X_ft.iloc[k]) for k in range(i, j)]
                    imgs = []
                    ys = []
                    for k, p in enumerate(paths_slice, start=i):
                        if os.path.exists(p):
                            try:
                                img = Image.open(p).convert("RGB")
                                imgs.append(self._preprocess(img))
                                ys.append(y_ft[k])
                            except Exception:
                                pass
                    if imgs:
                        batch = torch.stack(imgs, dim=0).to(device)
                        yb = torch.tensor(ys, dtype=torch.long, device=device)
                        # forward
                        feats = self._model(batch)      # [B, feat_dim]
                        # (option) L2 normalisation pendant FT — à tester/couper si besoin
                        norms = torch.norm(feats, dim=1, keepdim=True) + 1e-12
                        feats = feats / norms
                        logits = head(feats)
                        loss = criterion(logits, yb)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        losses.append(loss.item())
                    i = j
                # (log rapide)
                # print(f"[FT] epoch {epoch+1}/{self.finetune_epochs} loss={np.mean(losses):.4f}")

            # remettre en eval, retirer la tête
            self._model.eval()

        return self

    @profile_func
    def _load_one(self, path: str):
        """Charge + prétraite 1 image (ou None si manquante/erreur)."""
        if not os.path.exists(path):
            return None
        try:
            img = Image.open(path).convert("RGB")
            return self._preprocess(img)
        except Exception:
            return None
        
    @profile_func
    def transform(self, X):
        """
        X : DataFrame avec colonnes 'imageid' et 'productid'
        Retour : csr_matrix (n_samples, feat_dim)
        """
        self._lazy_load()
        device = self._resolve_device()

        n  = len(X)
        bs = int(self.batch_size)
        d  = int(self._feat_dim)
        out = np.zeros((n, d), dtype=self.dtype)

        # Chemins des fichiers
        paths = [self._path_from_row(X.iloc[i]) for i in range(n)]

        # Stats
        self.n_total = n
        self.n_loaded = 0
        self.n_missing = 0
        self.n_failed = 0

        with torch.no_grad():
            i = 0
            while i < n:
                j = min(i + bs, n)
                paths_slice = paths[i:j]

                imgs, idxs = [], []

                if self.num_workers > 0:
                    # Chargement multi-threads des images (I/O bound)
                    with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                        futs = {ex.submit(self._load_one, p): k for k, p in enumerate(paths_slice, start=i)}
                        for fut in as_completed(futs):
                            k = futs[fut]
                            t = fut.result()
                            if t is None:
                                # manquante / échec
                                if not os.path.exists(paths[k]):
                                    self.n_missing += 1
                                else:
                                    self.n_failed += 1
                            else:
                                imgs.append(t)
                                idxs.append(k)
                else:
                    # Chargement séquentiel
                    for k, p in enumerate(paths_slice, start=i):
                        if os.path.exists(p):
                            try:
                                img = Image.open(p).convert("RGB")
                                imgs.append(self._preprocess(img))
                                idxs.append(k)
                            except Exception:
                                self.n_failed += 1
                        else:
                            self.n_missing += 1

                if imgs:
                    batch = torch.stack(imgs, dim=0).to(device)
                    feats = self._model(batch).detach().cpu().numpy().astype(self.dtype, copy=False)
                    # L2-normalisation
                    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
                    feats = feats / norms
                    for t, k in enumerate(idxs):
                        out[k, :] = feats[t]
                    self.n_loaded += len(idxs)

                i = j

        return sparse.csr_matrix(out)

    # -------- Diagnostics -------------------------------------------------------
    @profile_func
    def get_diagnostics(self) -> Dict[str, object]:
        """Retourner un petit résumé du run (device, arch, tailles, taux manquants)."""
        input_size = None
        try:
            tf = getattr(self, "_preprocess", None)
            # torchvision >=0.13: souvent crop_size ou size disponible
            input_size = getattr(tf, "crop_size", None) or getattr(tf, "size", None)
            if isinstance(input_size, (tuple, list)):
                input_size = input_size[0]
        except Exception:
            pass
        return {
            "arch": self.arch,
            "device": str(self._resolve_device()),
            "feat_dim": int(self._feat_dim or 0),
            "batch_size": int(self.batch_size),
            "use_imagenet_norm": getattr(self, "use_imagenet_norm", True),
            "n_total": int(self.n_total),
            "n_loaded": int(self.n_loaded),
            "n_missing": int(self.n_missing),
            "n_failed": int(self.n_failed),
            "loaded_ratio": float(self.n_loaded / max(1, self.n_total)),
            "num_workers" : int(self.num_workers),
            "input_size": int(input_size) if input_size else None,
            "trainable_last_n": int(getattr(self, "trainable_last_n", 0)),
            "finetune_epochs": int(getattr(self, "finetune_epochs", 0)),
            "hf_model_name": getattr(self, "hf_model_name", None),
        }
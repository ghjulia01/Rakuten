# models/cnn_features.py
# =======================================================
# Extraire un embedding CNN (ResNet) compatible scikit-learn
# → lit imageid/productid, batch, normalise, renvoie csr_matrix
# =======================================================
from __future__ import annotations
import os
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from concurrent.futures import ThreadPoolExecutor, as_completed

from main.profiling_tools import profile_func
from transformers import AutoImageProcessor, AutoModel

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)
import logging
log = logging.getLogger("models.cnn_features")

ARCH_REGISTRY = {
    "resnet18":  (resnet18,  ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50":  (resnet50,  ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2, 2048),
}

class HFBackbone(torch.nn.Module):
    def __init__(self, base, processor, device):
        super().__init__()
        self.base = base
        self.processor = processor
        self.device = device
        self.log = logging.getLogger("models.cnn_features")
    def forward(self, x):   # x: (B,3,H,W) in [0,1]
        imgs = [transforms.ToPILImage()(xi.cpu()) for xi in x]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        out = self.base(**inputs)
        # utiliser [CLS], pas le pooler
        if hasattr(out, "last_hidden_state"):
            z = out.last_hidden_state[:, 0]
        else:
            z0 = out[0] if isinstance(out, (tuple, list)) else out
            z = z0.mean(dim=1) if z0.ndim == 3 else z0
        return z

class CNNFeaturizer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn qui :
      - lit les fichiers images à partir de imageid/productid
      - extrait un embedding CNN pré-entraîné (par défaut ResNet50 → 2048d)
      - renvoie un csr_matrix (bien compatible avec TF-IDF sparse)
    """
    @profile_func
    def __init__(
        self,
        image_dir: str,
        arch: str = "resnet50",
        batch_size: int = 32,
        device: str = "auto",           # "auto" | "cpu" | "cuda" | "dml"
        use_imagenet_norm: bool = True, # normaliser comme ImageNet
        fallback_zero: bool = True,     # image manquante → vecteur 0
        dtype: str = "float32",         # "float32" conseillé (mémoire)
        num_workers: int = 0,
        # --- paramètres unfreeze / FT / HF ---
        trainable_last_n: int = 0,      # nb de paramètres finaux à défiger (0 = tout figé)
        finetune_epochs: int = 0,       # nb d’époques de fine-tuning (0 = pas de FT)
        finetune_lr: float = 3e-4,
        finetune_weight_decay: float = 0.01,
        finetune_max_n: int = 8000,     # échantillon max utilisé pour FT
        trainable_last_layers: int = 1, # nb de blocs Transformer à défiger (HF)
        hf_model_name: Optional[str] = None,  # ex: "google/vit-base-patch16-224"
        hf_revision: Optional[str] = None,    # ex: "main"
        hf_feature_dim: Optional[int] = None, # si connu, sinon déduit
        save_head_path: Optional[str] = None,  # si renseigné, sauvegarde la tête FT + classes
        save_head_normalize: bool = True,      # normalise l'embedding avant la tête (comme en FT)
        foreach: bool = True,                  # peut être forçé à False sur DML
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

        # unfreeze / FT / HF
        self.trainable_last_n      = int(trainable_last_n)
        self.finetune_epochs       = int(finetune_epochs)
        self.finetune_lr           = float(finetune_lr)
        self.finetune_weight_decay = float(finetune_weight_decay)
        self.finetune_max_n        = int(finetune_max_n)
        self.trainable_last_layers = int(trainable_last_layers)

        self.hf_model_name  = hf_model_name
        self.hf_revision    = hf_revision
        self.hf_feature_dim = int(hf_feature_dim) if hf_feature_dim is not None else None
        self.save_head_path = save_head_path
        self.save_head_normalize = bool(save_head_normalize)
        self._trained_head: Optional[nn.Module] = None
        self.label_classes_: Optional[np.ndarray] = None
        self.foreach = bool(foreach)
        self.log = logging.getLogger("models.cnn_features")

    # -------- Utilitaires -------------------------------------------------------

    def _load_one(self, path: str):
        if not os.path.exists(path):
            return None
        try:
            with Image.open(path).convert("RGB") as im:
                return self._preprocess(im)
        except UnicodeDecodeError as e:
            self.log.warning("UnicodeDecodeError on path=%s -> %r", path, e)
            return None
        except Exception as e:
            self.log.warning("PIL failed for path=%s -> %s: %r", path, type(e).__name__, e)
            return None

        
        
    @profile_func
    def set_image_dir(self, new_dir: str):
        """Mettre à jour le dossier images (utile pour passer TRAIN → TEST)."""
        self.image_dir = new_dir

    @profile_func
    def _resolve_device(self):
        if self._device_resolved is not None:
            return self._device_resolved

        if self.device == "cuda":
            self._device_resolved = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        elif self.device == "cpu":
            self._device_resolved = torch.device("cpu")
        elif self.device == "dml":
            import torch_directml
            self._device_resolved = torch_directml.device()
        else:  # "auto"
            if torch.cuda.is_available():
                self._device_resolved = torch.device("cuda")
            else:
                try:
                    import torch_directml
                    self._device_resolved = torch_directml.device()
                except Exception:
                    self._device_resolved = torch.device("cpu")

        return self._device_resolved

    @profile_func
    def _build_model(self):
        # HF branch (ViT & co)
        if self.hf_model_name:
            device = self._resolve_device()
            processor = AutoImageProcessor.from_pretrained(
                self.hf_model_name, revision=self.hf_revision, use_fast=True
            )
            base = AutoModel.from_pretrained(
                self.hf_model_name, revision=self.hf_revision
            ).to(device)
            base.eval()

            model = HFBackbone(base, processor, device).to(device)
            self._feat_dim = int(self.hf_feature_dim or 768)  # ViT-base = 768
            # Le processor gère resize/crop/normalisation → juste ToTensor ici
            preprocess = transforms.ToTensor()
            return model, preprocess

        # Torchvision branch (ResNet)
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
            self.log.info(f"[INFO] HF unfreeze: last {n_layers} Transformer blocks.")
            return

        # Cas ResNet (torchvision) : défige layer4
        if hasattr(self._model, "layer4"):
            for p in self._model.parameters():
                p.requires_grad = False
            for p in self._model.layer4.parameters():
                p.requires_grad = True
            self.log.info("[INFO] ResNet unfreeze: layer4.")
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

        # --- Fine-tuning optionnel (supervisé), SANS LR-FINDER ---
        if self.finetune_epochs and y is not None and (self.trainable_last_n > 0 or getattr(self, "trainable_last_layers", 0) > 0):
            from sklearn.preprocessing import LabelEncoder
            import math

            # 1) Encode labels
            le = LabelEncoder()
            y_enc = le.fit_transform(np.asarray(y))

            # 2) Sous-échantillon rapide
            n = min(len(X), int(self.finetune_max_n))
            rng = np.random.RandomState(42)
            idx = rng.permutation(len(X))[:n]
            X_ft = X.iloc[idx].reset_index(drop=True)
            y_ft = y_enc[idx]

            # 3) Préparer training minimal
            device = self._resolve_device()
            self._set_trainable_tail(self.trainable_last_n)
            self._model.train()

            head = nn.Linear(self._feat_dim, int(len(le.classes_))).to(device)

            # Optim principal (utilise self.finetune_lr)
            # Sécurisation DirectML optionnelle : foreach=False
            is_dml = (str(self.device).lower() == "dml") or ("directml" in str(type(device)).lower())

            param_groups = [
                {"params": [p for p in self._model.parameters() if p.requires_grad], "lr": self.finetune_lr},
                {"params": head.parameters(), "lr": self.finetune_lr},
            ]

            adamw_kwargs = dict(
                weight_decay=self.finetune_weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

            # Certains PyTorch n’acceptent pas foreach/fused → on tente puis on retombe sans
            try:
                opt = torch.optim.AdamW(
                param_groups,
                **adamw_kwargs,
                foreach=False if is_dml else bool(self.foreach),  # DML: False impératif
                fused=False,  # éviter les implémentations fusionnées non supportées
                )
            except TypeError:
                # Ancienne version de PyTorch: pas de foreach/fused
                opt = torch.optim.AdamW(param_groups, **adamw_kwargs)

            criterion = nn.CrossEntropyLoss()

            # 4) Entraînement court
            bs = int(self.batch_size)
            steps_per_epoch = math.ceil(len(X_ft) / bs)

            i = 0
            for _ in range(int(self.finetune_epochs)):
                i = 0
                while i < len(X_ft):
                    j = min(i + bs, len(X_ft))
                    paths_slice = [self._path_from_row(X_ft.iloc[k]) for k in range(i, j)]
                    imgs, ys = [], []
                    for k, p in enumerate(paths_slice, start=i):
                        if os.path.exists(p):
                            try:
                                with Image.open(p).convert("RGB") as im:
                                    imgs.append(self._preprocess(im))
                                ys.append(y_ft[k])
                            except Exception:
                                pass
                    if imgs:
                        batch = torch.stack(imgs, dim=0).to(device)
                        yb = torch.tensor(ys, dtype=torch.long, device=device)
                        feats = self._model(batch)
                        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
                        logits = head(feats)
                        loss = criterion(logits, yb)

                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(self._model.parameters()) + list(head.parameters()), max_norm=1.0)
                        opt.step()
                    i = j

            # 5) Eval + stockage tête
            self._model.eval()
            self._trained_head = head.to(device).eval()
            self.label_classes_ = le.classes_
            if self.save_head_path:
                to_save = {
                    "state_dict": self._trained_head.state_dict(),
                    "feat_dim": int(self._feat_dim),
                    "classes": self.label_classes_.tolist(),
                    "normalize_feat": bool(self.save_head_normalize),
                }
                torch.save(to_save, self.save_head_path)

        return self

    @profile_func
    def attach_head(self, head: nn.Module, classes: List[str] | np.ndarray, normalize_feat: Optional[bool] = None):
        """Attache une tête Linear entraînée + liste de classes."""
        self._lazy_load()
        self._trained_head = head.to(self._resolve_device()).eval()
        self.label_classes_ = np.asarray(classes)
        if normalize_feat is not None:
            self.save_head_normalize = bool(normalize_feat)

    @profile_func
    def load_head(self, path: str):
        """Charge une tête Linear + classes depuis torch.save(...)"""
        self._lazy_load()
        chk = torch.load(path, map_location=self._resolve_device())
        feat_dim = int(chk.get("feat_dim", int(self._feat_dim or 0)))
        classes = chk["classes"]
        head = nn.Linear(feat_dim, len(classes))
        head.load_state_dict(chk["state_dict"])
        self._trained_head = head.to(self._resolve_device()).eval()
        self.label_classes_ = np.asarray(classes)
        self.save_head_normalize = bool(chk.get("normalize_feat", True))

    @torch.no_grad()
    def _embed_batch(self, batch: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Embeddings [B, feat_dim] avec option de L2-normalisation (comme en FT)."""
        self._lazy_load()
        feats = self._model(batch)  # [B, feat_dim]
        if normalize:
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
        return feats

    @torch.no_grad()
    def predict_logits_from_paths(self, paths: List[str]) -> np.ndarray:
        """
        Calcule les logits par classe pour une liste de chemins d'images.
        Requiert self._trained_head et self.label_classes_.
        """
        assert self._trained_head is not None, "Aucune tête entraînée attachée/chargée. Utilise attach_head(...) ou load_head(...)."
        device = self._resolve_device()
        bs = int(self.batch_size)
        logits_all = []

        i = 0
        while i < len(paths):
            j = min(i + bs, len(paths))
            imgs = []
            for p in paths[i:j]:
                if os.path.exists(p):
                    try:
                        with Image.open(p).convert("RGB") as im:
                            imgs.append(self._preprocess(im))
                    except Exception:
                        imgs.append(None)
                else:
                    imgs.append(None)

            if any(t is not None for t in imgs):
                batch = torch.stack([t for t in imgs if t is not None], dim=0).to(device)
                feats = self._embed_batch(batch, normalize=self.save_head_normalize)
                logits = self._trained_head(feats).detach().cpu().numpy()
                # Remettre dans l’ordre avec des lignes vides pour les manquantes
                it = iter(logits)
                for t in imgs:
                    if t is None:
                        logits_all.append(np.full((1, len(self.label_classes_)), np.nan))
                    else:
                        logits_all.append(next(it)[None, :])
            else:
                # tout manquant dans ce batch
                logits_all.extend([np.full((1, len(self.label_classes_)), np.nan) for _ in imgs])

            i = j

        return np.vstack(logits_all)

    def idx_to_label(self, class_idx: int) -> str:
        """Map index de classe -> libellé d’origine (si disponible)."""
        if self.label_classes_ is None:
            return str(class_idx)
        return str(self.label_classes_[class_idx])

    @torch.no_grad()
    def predict_proba_from_paths(self, paths: List[str]) -> np.ndarray:
        """Probabilités softmax (même gabarit que predict_logits_from_paths)."""
        logits = self.predict_logits_from_paths(paths)
        # Gestion NaN: on laisse NaN si ligne entière NaN
        mask = ~np.isnan(logits).any(axis=1)
        proba = np.full_like(logits, np.nan, dtype=np.float64)
        if mask.any():
            e = np.exp(logits[mask] - np.max(logits[mask], axis=1, keepdims=True))
            proba[mask] = e / (e.sum(axis=1, keepdims=True) + 1e-12)
        return proba

    @torch.no_grad()
    def topk_from_paths(self, paths: List[str], k: int = 5):
        """Retourne pour chaque image: [(idx,label,logit,proba), ...] triés par logit desc."""
        logits = self.predict_logits_from_paths(paths)
        proba  = self.predict_proba_from_paths(paths)
        out = []
        for i in range(logits.shape[0]):
            if np.isnan(logits[i]).all():
                out.append([])
                continue
            idxs = np.argsort(-logits[i])[:k]
            items = []
            for j in idxs:
                items.append((int(j), self.idx_to_label(int(j)), float(logits[i, j]), float(proba[i, j])))
            out.append(items)
        return out

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
                                with Image.open(p).convert("RGB") as im:
                                    imgs.append(self._preprocess(im))
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
                if i % 100 == 0:
                    self.log.info("CNN progress: sample %d/%d", i, n)

                i = j

        return sparse.csr_matrix(out)

    # -------- Diagnostics -------------------------------------------------------
    @profile_func
    def get_diagnostics(self) -> Dict[str, object]:
        """Résumé du run (device, arch, tailles, taux manquants)."""
        input_size = None
        try:
            tf = getattr(self, "_preprocess", None)
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
            "foreach": bool(getattr(self, "foreach", False)),
        }

    @profile_func
    def save_model(self, path):
        """Sauvegarder le modèle et ses paramètres."""
        state = {
            'state_dict': self._model.state_dict(),
            'arch': self.arch,
            'feat_dim': self._feat_dim,
            'use_imagenet_norm': self.use_imagenet_norm,
        }
        torch.save(state, path)

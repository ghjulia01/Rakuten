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
        arch_key = str(self.arch).lower()
        if arch_key not in ARCH_REGISTRY:
            raise ValueError(f"Architecture inconnue: {self.arch} (supportées: {list(ARCH_REGISTRY)})")
        ctor, weights_enum, feat_dim = ARCH_REGISTRY[arch_key]
        weights = weights_enum
        model = ctor(weights=weights)
        # retirer la dernière couche de classification → embedding
        model.fc = nn.Identity()
        model.eval()
        model.to(self._resolve_device())

        # preprocessing officiel des weights (Resize 224, CenterCrop, ToTensor, Norm)
        preprocess = weights.transforms() if self.use_imagenet_norm else transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        self._feat_dim = feat_dim
        return model, preprocess

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
        """Charger paresseusement le modèle ; réinitialiser les compteurs."""
        self._lazy_load()
        self.n_total = self.n_loaded = self.n_missing = self.n_failed = 0
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
        }
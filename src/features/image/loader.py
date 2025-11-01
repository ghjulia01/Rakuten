# features/image_loader.py
from __future__ import annotations

import os
from typing import Tuple, Optional, List, Any

import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.profiling import profile_func, list_debug_add

class ImageLoader(BaseEstimator, TransformerMixin):
    """
    Charger les pixels d'images à partir de `imageid` et `productid` en construisant
    des chemins de type: image_{imageid}_product_{productid}<ext>

    - Laisser les paramètres *inchangés* dans __init__ (compatibilité sklearn.clone).
    - Redimensionner à `image_size` et normaliser dans [0,1].
    - Renvoyer un tenseur (n_samples, H, W, 3) en float32.
    - En cas d'erreur/fichier manquant, retourner un vecteur zéro (fallback).
    """

    @profile_func
    def __init__(
        self,
        image_dir: str,
        image_size: Any = (128, 128),   # ne pas forcer le type ici
        imgid_col: str = "imageid",
        pid_col: str = "productid",
        ext: str = ".jpg",
    ):
        # Ne pas modifier les valeurs reçues (ex: pas de tuple(), pas de int())
        self.image_dir = image_dir
        self.image_size = image_size
        self.imgid_col = imgid_col
        self.pid_col = pid_col
        self.ext = ext

    # --- API sklearn ------------------------------------------------------------

    @profile_func
    def fit(self, X=None, y=None):
        # ne rien apprendre
        return self

    @profile_func
    def _resolve_size(self) -> Tuple[int, int]:
        """
        Sécuriser/convertir la taille *au moment de l'usage* (pas dans __init__).
        Accepter tuple, liste, numpy array…
        """
        sz = self.image_size
        try:
            H = int(sz[0])
            W = int(sz[1])
        except Exception:
            # défaut robuste
            H, W = 128, 128
        return H, W

    @profile_func
    def _build_path(self, imgid: Any, pid: Any) -> str:
        """Construire le chemin d'une image à partir des ids."""
        # convertir prudemment en int -> str
        try:
            iid = str(int(imgid))
        except Exception:
            iid = str(imgid)
        try:
            pid_str = str(int(pid))
        except Exception:
            pid_str = str(pid)
        fname = f"image_{iid}_product_{pid_str}{self.ext}"
        return os.path.join(self.image_dir, fname)

    @profile_func
    def transform(self, X):
        """
        X: DataFrame avec colonnes `imageid` et `productid`.
        Retour: np.ndarray (n, H, W, 3) float32 dans [0,1].
        """
        list_debug_add("ImageLoader.transform : " + str(X.shape[0]))
        H, W = self._resolve_size()

        # extraire colonnes (laisser pandas gérer les types)
        imgids = X[self.imgid_col].tolist()
        pids   = X[self.pid_col].tolist()

        paths: List[str] = [self._build_path(iid, pid) for iid, pid in zip(imgids, pids)]
        out = np.zeros((len(paths), H, W, 3), dtype=np.float32)

        for i, p in enumerate(paths):
            try:
                with Image.open(p) as img:
                    img = img.convert("RGB").resize((W, H))  # PIL: (width, height)
                    arr = np.asarray(img, dtype=np.float32)
                # normaliser en [0,1] seulement si besoin (éviter /0)
                if arr.max() > 1.0:
                    arr /= 255.0
                if arr.shape == (H, W, 3):
                    out[i] = arr
                # sinon: garder les zéros (fallback)
            except Exception:
                # fichier manquant/corrompu -> garder le vecteur zéro
                pass

        return out

# features/image_loader.py
# ------------------------------------------------------------
# Rôle : charger et prétraiter les images depuis imageid/productid
# Nom de fichier attendu : image_{imageid}_product_{productid}.jpg
# Exemple : image_216810030_product_6342052.jpg
# ------------------------------------------------------------
from __future__ import annotations

import os
from typing import Tuple, List

import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin


class ImageLoader(BaseEstimator, TransformerMixin):
    """
    Charger les pixels d'images à partir de `imageid` et `productid` en
    construisant le chemin :
        image_{imageid}_product_{productid}{ext}

    - Tolérer les fichiers manquants/corrompus (remplir avec des zéros)
    - Redimensionner à `image_size` et normaliser dans [0, 1]
    - Retourner un tableau (n_samples, H, W, 3) float32

    Args:
        image_dir: dossier des images (train ou test)
        image_size: (H, W) en pixels
        imgid_col: nom de la colonne image id
        pid_col: nom de la colonne product id
        ext: extension de fichier (".jpg" par défaut)
    """

    def __init__(
        self,
        image_dir: str,
        image_size: Tuple[int, int] = (128, 128),
        imgid_col: str = "imageid",
        pid_col: str = "productid",
        ext: str = ".jpg",
    ) -> None:
        self.image_dir = str(image_dir)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.imgid_col = str(imgid_col)
        self.pid_col = str(pid_col)
        self.ext = str(ext)

    # sklearn API -----------------------------------------------------

    def fit(self, X=None, y=None) -> "ImageLoader":
        # ne rien apprendre (transformeur statique)
        return self

    def transform(self, X):
        """
        X : DataFrame/Series (avec colonnes imgid/pid) ou itérable de dicts
        Retour : np.ndarray (N, H, W, 3) en float32
        """
        H, W = self.image_size

        # extraire listes de ids (garantir chaînes sans espaces)
        if hasattr(X, "loc"):  # DataFrame/Series
            imgids = X[self.imgid_col].astype("int64").astype(str).tolist()
            pids   = X[self.pid_col].astype("int64").astype(str).tolist()
        else:
            # itérable de dicts/objets : chercher les attributs
            imgids, pids = [], []
            for row in X:
                imgids.append(str(int(row[self.imgid_col])))
                pids.append(str(int(row[self.pid_col])))

        # construire chemins
        paths: List[str] = [
            os.path.join(self.image_dir, f"image_{iid}_product_{pid}{self.ext}")
            for iid, pid in zip(imgids, pids)
        ]

        # allouer sortie
        out = np.zeros((len(paths), H, W, 3), dtype=np.float32)

        # charger + prétraiter
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as img:
                    # convertir en RGB, redimensionner (Pillow attend (W, H))
                    img = img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                if arr.shape == (H, W, 3):
                    out[i] = arr  # sinon: garder zéros
            except Exception:
                # fichier manquant/corrompu → garder l'image noire
                pass

        return out
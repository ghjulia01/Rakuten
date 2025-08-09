# features/image_stats.py
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ImageStatsFeaturizer(BaseEstimator, TransformerMixin):
    """
    Extrait 3 features par image AVANT tout resize :
      - width_obj : largeur de l'objet détecté (px)
      - height_obj: hauteur de l'objet détecté (px)
      - occupancy : surface objet / surface image ∈ [0,1]
    L'objet est tout pixel dont le niveau de gris est ∈ 
    (black_threshold, white_threshold).
    """
    def __init__(
        self,
        image_dir: str,
        imgid_col: str = "imageid",
        pid_col: str = "productid",
        white_threshold: int = 230,
        black_threshold: int = 25,
        min_area: int = 16,
        out_prefix: str = "img_w230_b25_",
    ):
        self.image_dir = Path(image_dir)
        self.imgid_col = imgid_col
        self.pid_col   = pid_col
        self.white_threshold = int(white_threshold)
        self.black_threshold = int(black_threshold)
        self.min_area = int(min_area)
        self.out_prefix = out_prefix

    def set_image_dir(self, new_dir: str):
        """Permet de re-pointer le dossier (utile pour X_test)."""
        self.image_dir = Path(new_dir)

    def fit(self, X, y=None):
        return self

    def _measure_one(self, img_path: Path):
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.uint8)
            H, W = arr.shape[:2]
            gray = arr.mean(axis=2)
            mask = (gray < self.white_threshold) & (gray > self.black_threshold)
            area = int(mask.sum())
            if area < self.min_area:
                return 0, 0, 0.0
            ys, xs = np.nonzero(mask)
            h = int(ys.max() - ys.min() + 1)
            w = int(xs.max() - xs.min() + 1)
            occ = float(area) / float(H * W)
            return w, h, occ
        except Exception:
            return 0, 0, 0.0

    def transform(self, X: pd.DataFrame):
        # Construit le nom Rakuten : image_{imageid}_product_{productid}.jpg
        if self.imgid_col not in X.columns or self.pid_col not in X.columns:
            raise ValueError(f"Colonnes requises absentes: '{self.imgid_col}', '{self.pid_col}'")

        rows = []
        for idx, row in X[[self.imgid_col, self.pid_col]].iterrows():
            try:
                imgid = int(row[self.imgid_col])
                pid   = int(row[self.pid_col])
            except Exception:
                rows.append((idx, 0, 0, 0.0))
                continue
            name = f"image_{imgid}_product_{pid}.jpg"
            path = self.image_dir / name
            w, h, occ = self._measure_one(path)
            rows.append((idx, w, h, occ))

        out = pd.DataFrame({
            f"{self.out_prefix}width":  [r[1] for r in rows],
            f"{self.out_prefix}height": [r[2] for r in rows],
            f"{self.out_prefix}occ":    [r[3] for r in rows],
        }, index=[r[0] for r in rows])

        # dtype compacts
        out = out.astype({
            f"{self.out_prefix}width":  "float32",
            f"{self.out_prefix}height": "float32",
            f"{self.out_prefix}occ":    "float32",
        })
        return out
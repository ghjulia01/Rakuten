# features/image_stats.py
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ImageStatsFeaturizer(BaseEstimator, TransformerMixin):
    """
    Extrait 5 features par image AVANT tout resize :
      - width  : largeur de l'objet détecté (px)
      - height : hauteur de l'objet détecté (px)
      - occupancy : surface objet / surface image ∈ [0,1]
      - white_ratio : part de pixels >= white_threshold
      - black_ratio : part de pixels <= black_threshold
    L'objet est tout pixel dont le niveau de gris est ∈ (black_threshold, white_threshold).
    """

    def __init__(
        self,
        image_dir: str = None,
        imgid_col: str = "imageid",
        pid_col: str = "productid",
        white_threshold: int = 230,
        black_threshold: int = 25,
        min_area: int = 16,
        out_prefix: str = "auto",  # "auto" nomme les colonnes selon les seuils
        use_cache: bool = True,
        cache_filename: str | None = None,
    ):
        # Ne rien faire d'autre que stocker les paramètres tels quels (sklearn clone-friendly)
        self.image_dir = image_dir
        self.imgid_col = imgid_col
        self.pid_col = pid_col
        self.white_threshold = int(white_threshold)
        self.black_threshold = int(black_threshold)
        self.min_area = int(min_area)
        self.out_prefix = out_prefix
        self.use_cache = use_cache
        self.cache_filename = cache_filename

    def set_image_dir(self, new_dir: str):
        """Permet de re-pointer le dossier (utile pour X_test)."""
        self.image_dir = new_dir
        if hasattr(self, "image_dir_"):
            self.image_dir_ = Path(new_dir)
        return self

    def fit(self, X, y=None):
        # Toute dérivation/validation ici (pas dans __init__)
        self.image_dir_ = Path(self.image_dir) if self.image_dir is not None else None
        # Gérer le préfixe des noms de colonnes
        if self.out_prefix in (None, "auto"):
            self.out_prefix_ = f"img_w{self.white_threshold}_b{self.black_threshold}_"
        else:
            self.out_prefix_ = str(self.out_prefix)

        self.columns_ = np.array([
            f"{self.out_prefix_}width",
            f"{self.out_prefix_}height",
            f"{self.out_prefix_}occupancy",
            f"{self.out_prefix_}white_ratio",
            f"{self.out_prefix_}black_ratio",
        ])
        return self

    def _measure_one(self, img_path: Path):
        try:
            with Image.open(img_path) as im:
                im = im.convert("L")  # grayscale
                arr = np.asarray(im, dtype=np.uint8)

            H, W = arr.shape[:2]
            if H == 0 or W == 0:
                return 0, 0, 0.0, 0.0, 0.0

            white_mask = (arr >= self.white_threshold)
            black_mask = (arr <= self.black_threshold)
            mid_mask = (~white_mask) & (~black_mask)

            area = int(mid_mask.sum())
            # ratios globaux (rapport de pixels)
            white_ratio = float(white_mask.mean())
            black_ratio = float(black_mask.mean())

            if area < self.min_area:
                # Trop petit signal -> 0 pour width/height/occupancy,
                # mais on garde les ratios globaux utiles
                return 0, 0, 0.0, white_ratio, black_ratio

            ys, xs = np.nonzero(mid_mask)
            h = int(ys.max() - ys.min() + 1)
            w = int(xs.max() - xs.min() + 1)
            occ = float(area) / float(H * W)

            return w, h, occ, white_ratio, black_ratio

        except Exception:
            # Image manquante ou corrompue
            return 0, 0, 0.0, 0.0, 0.0

    def transform(self, X: pd.DataFrame):
        if self.imgid_col not in X.columns or self.pid_col not in X.columns:
            raise ValueError(f"Colonnes requises absentes: '{self.imgid_col}', '{self.pid_col}'")
        if not hasattr(self, "image_dir_") or self.image_dir_ is None:
            raise RuntimeError("image_dir_ non défini : as-tu appelé fit() ?")

        n = len(X)
        data = np.zeros((n, 5), dtype=np.float32)
        idxs = []

        for i, (idx, row) in enumerate(X[[self.imgid_col, self.pid_col]].iterrows()):
            try:
                imgid = int(row[self.imgid_col])
                pid = int(row[self.pid_col])
                name = f"image_{imgid}_product_{pid}.jpg"
                path = self.image_dir_ / name
                w, h, occ, wr, br = self._measure_one(path)
                data[i, :] = (w, h, occ, wr, br)
            except Exception:
                # ligne déjà à zéro
                pass
            idxs.append(idx)

        return pd.DataFrame(data, index=idxs, columns=self.columns_)

    def get_feature_names_out(self, input_features=None):
        return self.columns_
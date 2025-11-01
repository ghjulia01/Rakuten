# features/image_stats.py
from math import hypot
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed

from src.utils.profiling import profile_func

class ImageStatsCombinedFeaturizer(BaseEstimator, TransformerMixin):
    """
    Combine BASIC et PRO stats en un seul passage.
    Améliorations:
      - drop_size_features: supprime img_width/img_height (parasites)
      - keep_bpp: conserve ou non pro_file_bpp
      - fast + fast_size: downscale systématique pour vitesse/robustesse
      - n_jobs: parallélisation du transform
      - adaptive_edge_thr: seuil d'arêtes relatif à la dynamique
      - offset_power/clip: stabilité du bbox_center_offset
    """
    @profile_func
    def __init__(
        self,
        image_dir=None,
        imgid_col="imageid",
        pid_col="productid",
        white_threshold=230,
        black_threshold=25,
        min_area=16,
        prefix_basic="img_",
        prefix_pro="pro_",
        fast=True,
        fast_size=128,
        entropy_bins=128,
        drop_size_features=True,
        keep_bpp=False,
        adaptive_edge_thr=True,
        edge_thr_min=8.0,           # seuil minimal en niveau de gris
        edge_thr_rel=0.12,          # % de la plage dynamique
        offset_power=2.0,           # accentue les gros décentrages
        offset_clip=1.0,
        n_jobs=1
    ):
        self.image_dir = image_dir
        self.imgid_col = imgid_col
        self.pid_col = pid_col
        self.white_threshold = int(white_threshold)
        self.black_threshold = int(black_threshold)
        self.min_area = int(min_area)
        self.prefix_basic = str(prefix_basic)
        self.prefix_pro = str(prefix_pro)
        self.fast = bool(fast)
        self.fast_size = int(fast_size)
        self.entropy_bins = int(entropy_bins)

        self.drop_size_features = bool(drop_size_features)
        self.keep_bpp = bool(keep_bpp)
        self.adaptive_edge_thr = bool(adaptive_edge_thr)
        self.edge_thr_min = float(edge_thr_min)
        self.edge_thr_rel = float(edge_thr_rel)
        self.offset_power = float(offset_power)
        self.offset_clip = float(offset_clip)
        self.n_jobs = int(n_jobs)

    @profile_func
    def fit(self, X, y=None):
        self.image_dir_ = Path(self.image_dir) if self.image_dir is not None else None

        # BASIC
        self.basic_cols_ = []
        if not self.drop_size_features:
            self.basic_cols_.extend([
                f"{self.prefix_basic}width",
                f"{self.prefix_basic}height",
            ])
        self.basic_cols_.extend([
            f"{self.prefix_basic}occupancy",
            f"{self.prefix_basic}white_ratio",
            f"{self.prefix_basic}black_ratio",
        ])

        # PRO
        self.pro_cols_ = [
            f"{self.prefix_pro}gray_mean", f"{self.prefix_pro}gray_std",
            f"{self.prefix_pro}p10", f"{self.prefix_pro}p90", f"{self.prefix_pro}dyn_range",
            f"{self.prefix_pro}entropy", f"{self.prefix_pro}lap_var", f"{self.prefix_pro}edge_density",
            f"{self.prefix_pro}aspect_ratio", f"{self.prefix_pro}bbox_center_offset",
            f"{self.prefix_pro}sat_mean", f"{self.prefix_pro}colorfulness",
            f"{self.prefix_pro}border_white_ratio",
        ]
        if self.keep_bpp:
            self.pro_cols_.append(f"{self.prefix_pro}file_bpp")

        self.columns_ = np.array(self.basic_cols_ + self.pro_cols_)
        return self

    def get_feature_names_out(self, input_features=None):
        return self.columns_

    @profile_func
    def set_image_dir(self, image_dir):
        self.image_dir = image_dir
        if hasattr(self, "image_dir_"):
            self.image_dir_ = Path(image_dir)

    # ---------- helpers ----------
    @staticmethod
    def _entropy(gray, bins=256):
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        if bins != 256:
            factor = 256 // bins
            hist = hist.reshape(bins, factor).sum(axis=1)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    @staticmethod
    def _lap_var_fast(gray):
        g = gray.astype(np.float32, copy=False)
        gx = np.zeros_like(g, dtype=np.float32); gy = np.zeros_like(g, dtype=np.float32)
        gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
        gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
        lap = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
        return float(np.var(lap))

    @staticmethod
    def _edge_density(gray, thr):
        g = gray.astype(np.float32, copy=False)
        gx = np.zeros_like(g, dtype=np.float32); gy = np.zeros_like(g, dtype=np.float32)
        gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
        gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
        mag = np.hypot(gx, gy)
        return float((mag > thr).mean())

    @staticmethod
    def _colorfulness(rgb):
        r = rgb[..., 0].astype(np.float32, copy=False)
        g = rgb[..., 1].astype(np.float32, copy=False)
        b = rgb[..., 2].astype(np.float32, copy=False)
        rg = r - g
        yb = 0.5 * (r + g) - b
        std_rg, mean_rg = np.std(rg), np.mean(rg)
        std_yb, mean_yb = np.std(yb), np.mean(yb)
        return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

    @staticmethod
    def _sat_mean(rgb_pil):
        hsv = rgb_pil.convert("HSV")
        s = np.asarray(hsv)[..., 1].astype(np.float32, copy=False)
        return float(s.mean() / 255.0)

    def _load_rgb(self, p: Path):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = im.size
            if self.fast:
                max_side = max(W, H)
                if max_side > self.fast_size:
                    scale = self.fast_size / max_side
                    new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
                    im = im.resize(new_size, Image.BILINEAR)
            return im, (W, H)   # W0,H0 = taille d’origine

    def _featurize_one(self, p: Path):
        if p is None:
            return None
        try:
            img, (W0, H0) = self._load_rgb(p)
            rgb = np.asarray(img)
            H, W = rgb.shape[:2]
            gray = np.asarray(img.convert("L"))

            white_mask = gray >= self.white_threshold
            black_mask = gray <= self.black_threshold
            obj_mask = ~(white_mask | black_mask)

            # Occupancy & bbox
            if obj_mask.any():
                ys, xs = np.where(obj_mask)
                h_obj = ys.max() - ys.min() + 1
                w_obj = xs.max() - xs.min() + 1
                if h_obj * w_obj < self.min_area:
                    obj_mask[:] = False
            occupancy = float(obj_mask.mean())
            white_ratio = float(white_mask.mean())
            black_ratio = float(black_mask.mean())

            # Offset centre (normalisé par la diagonale)
            if obj_mask.any():
                ys, xs = np.where(obj_mask)
                cy = ys.mean(); cx = xs.mean()
                offset = hypot(cx - (W - 1) / 2.0, cy - (H - 1) / 2.0) / hypot(W, H)
            else:
                offset = 0.0
            offset = min(self.offset_clip, max(0.0, offset))
            if self.offset_power != 1.0:
                offset = float(offset ** self.offset_power)

            # Stats gris
            gray_f = gray.astype(np.float32, copy=False)
            gmean = float(gray_f.mean())
            gstd  = float(gray_f.std())
            p10   = float(np.percentile(gray, 10))
            p90   = float(np.percentile(gray, 90))
            dyn   = float(max(1.0, p90 - p10))

            ent   = self._entropy(gray, bins=self.entropy_bins)
            lapv  = self._lap_var_fast(gray)

            # Seuil d’arêtes adaptatif
            thr = self.edge_thr_min
            if self.adaptive_edge_thr:
                thr = max(self.edge_thr_min, self.edge_thr_rel * dyn)
            edged = self._edge_density(gray, thr=thr)

            # Bord blanc
            bw = max(1, min(H, W) // 20)
            border = np.zeros_like(gray, dtype=bool)
            border[:bw, :] = True; border[-bw:, :] = True
            border[:, :bw] = True; border[:, -bw:] = True
            border_white_ratio = float((gray[border] >= self.white_threshold).mean())

            sat_mean = self._sat_mean(img)
            colorf   = self._colorfulness(rgb)

            vals = {}

            # BASIC
            if not self.drop_size_features:
                vals[f"{self.prefix_basic}width"]  = float(W0)
                vals[f"{self.prefix_basic}height"] = float(H0)
            vals[f"{self.prefix_basic}occupancy"]   = occupancy
            vals[f"{self.prefix_basic}white_ratio"] = white_ratio
            vals[f"{self.prefix_basic}black_ratio"] = black_ratio

            # PRO
            vals[f"{self.prefix_pro}gray_mean"]  = gmean
            vals[f"{self.prefix_pro}gray_std"]   = gstd
            vals[f"{self.prefix_pro}p10"]        = p10
            vals[f"{self.prefix_pro}p90"]        = p90
            vals[f"{self.prefix_pro}dyn_range"]  = dyn
            vals[f"{self.prefix_pro}entropy"]    = ent
            vals[f"{self.prefix_pro}lap_var"]    = lapv
            vals[f"{self.prefix_pro}edge_density"] = edged
            vals[f"{self.prefix_pro}aspect_ratio"]  = float(W0 / max(1.0, H0))
            vals[f"{self.prefix_pro}bbox_center_offset"] = offset
            vals[f"{self.prefix_pro}sat_mean"]   = sat_mean
            vals[f"{self.prefix_pro}colorfulness"] = colorf
            vals[f"{self.prefix_pro}border_white_ratio"] = border_white_ratio

            if self.keep_bpp:
                try:
                    fsz = p.stat().st_size
                    bpp = float(fsz / max(1, W0 * H0))
                except Exception:
                    bpp = 0.0
                vals[f"{self.prefix_pro}file_bpp"] = bpp

            # Restituer dans l’ordre des colonnes
            return [vals.get(k, 0.0) for k in self.columns_]

        except Exception:
            return None

    @profile_func
    def transform(self, X):
        if not hasattr(self, "columns_"):
            self.fit(X)

        paths = []
        if self.image_dir_ is None:
            paths = [None] * len(X)
        else:
            for imgid, pid in zip(X[self.imgid_col].values, X[self.pid_col].values):
                fname = f"image_{int(imgid)}_product_{int(pid)}.jpg"
                p = self.image_dir_ / fname
                paths.append(p if p.exists() else None)

        # Parallélisation
        if self.n_jobs != 1:
            rows = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._featurize_one)(p) for p in paths
            )
        else:
            rows = [self._featurize_one(p) for p in paths]

        out = np.zeros((len(paths), len(self.columns_)), dtype=np.float32)
        for i, r in enumerate(rows):
            if r is not None:
                out[i, :] = np.asarray(r, dtype=np.float32)

        # on retourne un ndarray pour compat sklearn ; noms via get_feature_names_out()
        return out
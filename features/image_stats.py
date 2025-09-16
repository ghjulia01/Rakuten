from math import hypot
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from main.profiling_tools import profile_func

class ImageStatsCombinedFeaturizer(BaseEstimator, TransformerMixin):
    """
    Combine BASIC (width,height,occupancy,white_ratio,black_ratio)
    et PRO (gray_mean/std, p10/p90, dyn_range, entropy, lap_var, edge_density,
    aspect_ratio, bbox_center_offset, sat_mean, colorfulness, border_white_ratio, file_bpp)
    en un seul passage par image.

    Accélérateurs:
      - fast: si True, calcule sur une version downscalée (fast_size) en préservant les ratios.
      - fast_size: côté max pour la version downscalée.
      - entropy_bins: nb de bins histogramme (plus petit = plus rapide).
    """
    @profile_func
    def __init__(self, image_dir=None, imgid_col="imageid", pid_col="productid",
                 white_threshold=230, black_threshold=25, min_area=16,
                 prefix_basic="img_", prefix_pro="pro_",
                 fast=False, fast_size=96, entropy_bins=256):
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

    @profile_func
    def fit(self, X, y=None):
        self.image_dir_ = Path(self.image_dir) if self.image_dir is not None else None
        self.basic_cols_ = [
            f"{self.prefix_basic}width",
            f"{self.prefix_basic}height",
            f"{self.prefix_basic}occupancy",
            f"{self.prefix_basic}white_ratio",
            f"{self.prefix_basic}black_ratio",
        ]
        self.pro_cols_ = [
            f"{self.prefix_pro}gray_mean", f"{self.prefix_pro}gray_std",
            f"{self.prefix_pro}p10", f"{self.prefix_pro}p90", f"{self.prefix_pro}dyn_range",
            f"{self.prefix_pro}entropy", f"{self.prefix_pro}lap_var", f"{self.prefix_pro}edge_density",
            f"{self.prefix_pro}aspect_ratio", f"{self.prefix_pro}bbox_center_offset",
            f"{self.prefix_pro}sat_mean", f"{self.prefix_pro}colorfulness",
            f"{self.prefix_pro}border_white_ratio", f"{self.prefix_pro}file_bpp",
        ]
        self.columns_ = np.array(self.basic_cols_ + self.pro_cols_)
        return self

    @profile_func
    def set_image_dir(self, image_dir):
        self.image_dir = image_dir
        if hasattr(self, "image_dir_"):
            self.image_dir_ = Path(image_dir)

    # ---------- helpers rapides ----------
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
        g = gray.astype(np.float32)
        gx = np.gradient(g, axis=1)
        gy = np.gradient(g, axis=0)
        lap = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
        return float(np.var(lap))

    @staticmethod
    def _edge_density(gray, thr=20.0):
        g = gray.astype(np.float32)
        gx = np.zeros_like(g, dtype=np.float32)
        gy = np.zeros_like(g, dtype=np.float32)
        gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
        gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
        mag = np.hypot(gx, gy)
        return float((mag > thr).mean())

    @staticmethod
    def _colorfulness(rgb):
        r = rgb[..., 0].astype(np.float32)
        g = rgb[..., 1].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)
        rg = r - g
        yb = 0.5 * (r + g) - b
        std_rg, mean_rg = np.std(rg), np.mean(rg)
        std_yb, mean_yb = np.std(yb), np.mean(yb)
        return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

    @staticmethod
    def _sat_mean(rgb_pil):
        hsv = rgb_pil.convert("HSV")
        s = np.asarray(hsv)[..., 1].astype(np.float32)
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
            return im, (W, H)

    @profile_func
    def transform(self, X):
        if not hasattr(self, "columns_"):
            self.fit(X)

        paths = []
        for imgid, pid in zip(X[self.imgid_col].values, X[self.pid_col].values):
            if self.image_dir_ is None:
                paths.append(None); continue
            fname = f"image_{int(imgid)}_product_{int(pid)}.jpg"
            p = self.image_dir_ / fname
            paths.append(p if p.exists() else None)

        out = np.zeros((len(paths), len(self.columns_)), dtype=np.float32)

        for i, p in enumerate(paths):
            if p is None:
                continue
            try:
                img, (W0, H0) = self._load_rgb(p)
                rgb = np.asarray(img)
                H, W = rgb.shape[:2]
                gray = np.asarray(img.convert("L"))

                white_mask = gray >= self.white_threshold
                black_mask = gray <= self.black_threshold
                obj_mask = ~(white_mask | black_mask)

                occupancy = float(obj_mask.mean())
                white_ratio = float(white_mask.mean())
                black_ratio = float(black_mask.mean())

                if obj_mask.any():
                    ys, xs = np.where(obj_mask)
                    h_obj = ys.max() - ys.min() + 1
                    w_obj = xs.max() - xs.min() + 1
                    cy = ys.mean(); cx = xs.mean()
                    offset = hypot(cx - (W - 1) / 2.0, cy - (H - 1) / 2.0) / hypot(W, H)
                else:
                    h_obj = w_obj = 0.0
                    offset = 0.0

                gray_f = gray.astype(np.float32)
                gmean = float(gray_f.mean())
                gstd  = float(gray_f.std())
                p10   = float(np.percentile(gray, 10))
                p90   = float(np.percentile(gray, 90))
                dyn   = float(p90 - p10)
                ent   = self._entropy(gray, bins=(128 if self.fast else self.entropy_bins))
                lapv  = self._lap_var_fast(gray)
                edged = self._edge_density(gray, thr=20.0)

                bw = max(1, min(H, W) // 20)
                border = np.zeros_like(gray, dtype=bool)
                border[:bw, :] = True; border[-bw:, :] = True
                border[:, :bw] = True; border[:, -bw:] = True
                border_white_ratio = float((gray[border] >= self.white_threshold).mean())

                sat_mean = self._sat_mean(img)
                colorf   = self._colorfulness(rgb)

                try:
                    fsz = p.stat().st_size
                    bpp = float(fsz / (max(1, W0 * H0)))
                except Exception:
                    bpp = 0.0

                vals = [
                    float(W0), float(H0), occupancy, white_ratio, black_ratio,
                    gmean, gstd, p10, p90, dyn, ent, lapv, edged,
                    (float(W0) / max(1.0, H0)), offset, sat_mean, colorf, border_white_ratio, bpp
                ]
                out[i, :] = np.array(vals, dtype=np.float32)
            except Exception:
                continue

        df = pd.DataFrame(out, columns=list(self.columns_), index=X.index)
        return df.values
# features/image_stats.py
from math import hypot
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ImageStatsCombinedFeaturizer(BaseEstimator, TransformerMixin):
    """
    Combine les features BASIC (width,height,occupancy,white_ratio,black_ratio)
    et PRO (gray_mean/std, p10/p90, dyn_range, entropy, lap_var, edge_density,
    aspect_ratio, bbox_center_offset, sat_mean, colorfulness, border_white_ratio, file_bpp)
    en une seule lecture par image.
    """
    def __init__(self, image_dir=None, imgid_col="imageid", pid_col="productid",
                 white_threshold=230, black_threshold=25, min_area=16,
                 prefix_basic="img_", prefix_pro="pro_"):
        self.image_dir = image_dir
        self.imgid_col = imgid_col
        self.pid_col = pid_col
        self.white_threshold = int(white_threshold)
        self.black_threshold = int(black_threshold)
        self.min_area = int(min_area)
        self.prefix_basic = str(prefix_basic)
        self.prefix_pro = str(prefix_pro)

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
    
    def set_image_dir(self, image_dir):
        self.image_dir = image_dir
        if hasattr(self, "image_dir_"):
            self.image_dir_ = Path(image_dir)
    
    @staticmethod
    def _entropy(gray):
        hist, _ = np.histogram(gray, bins=256, range=(0,255), density=True)
        hist = hist[hist>0]
        return float(-(hist * np.log2(hist)).sum())

    @staticmethod
    def _lap_var(gray):
        k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
        from scipy.signal import convolve2d
        g = convolve2d(gray.astype(np.float32), k, mode="same", boundary="symm")
        return float(g.var())

    @staticmethod
    def _edge_density(gray, thr=20.0):
        gx = np.zeros_like(gray, dtype=np.float32)
        gy = np.zeros_like(gray, dtype=np.float32)
        gx[:,1:-1] = (gray[:,2:].astype(np.float32) - gray[:,:-2].astype(np.float32)) * 0.5
        gy[1:-1,:] = (gray[2:,:].astype(np.float32) - gray[:-2,:].astype(np.float32)) * 0.5
        mag = np.hypot(gx, gy)
        return float((mag > thr).mean())

    @staticmethod
    def _colorfulness(rgb):
        R,G,B = rgb[...,0].astype(np.float32), rgb[...,1].astype(np.float32), rgb[...,2].astype(np.float32)
        rg, yb = np.abs(R-G), np.abs(0.5*(R+G)-B)
        return float(np.sqrt(rg.var()+yb.var()) + 0.3*np.sqrt(rg.mean()**2 + yb.mean()**2))

    def transform(self, X: pd.DataFrame):
        if self.imgid_col not in X.columns or self.pid_col not in X.columns:
            raise ValueError(f"Colonnes requises absentes: '{self.imgid_col}', '{self.pid_col}'")
        if not hasattr(self, "image_dir_") or self.image_dir_ is None:
            raise RuntimeError("image_dir_ non défini : as-tu appelé fit() ?")

        n = len(X)
        out = np.zeros((n, len(self.columns_)), dtype=np.float32)
        idxs = []

        for i, (idx, row) in enumerate(X[[self.imgid_col, self.pid_col]].iterrows()):
            idxs.append(idx)
            try:
                imgid = int(row[self.imgid_col]); pid = int(row[self.pid_col])
                path = self.image_dir_ / f"image_{imgid}_product_{pid}.jpg"

                with Image.open(path) as im:
                    im_rgb = im.convert("RGB")
                    rgb = np.asarray(im_rgb)
                    gray = np.asarray(im_rgb.convert("L"))

                H, W = gray.shape
                if H == 0 or W == 0:
                    continue

                # masques
                white_mask = (gray >= self.white_threshold)
                black_mask = (gray <= self.black_threshold)
                mid = (~white_mask) & (~black_mask)

                # BASIC
                area = int(mid.sum())
                white_ratio = float(white_mask.mean())
                black_ratio = float(black_mask.mean())
                if area >= self.min_area:
                    ys, xs = np.nonzero(mid)
                    h = int(ys.max() - ys.min() + 1)
                    w = int(xs.max() - xs.min() + 1)
                    occ = float(area) / float(H * W)
                else:
                    h = w = 0
                    occ = 0.0

                # PRO
                p10, p90 = np.percentile(gray, [10, 90])
                gray_mean, gray_std = float(gray.mean()), float(gray.std())
                dyn = float(p90 - p10)
                ent = self._entropy(gray)
                lapv = self._lap_var(gray)
                edged = self._edge_density(gray)

                if area >= self.min_area and h > 0:
                    aspect = (w / max(1, h))
                    ys, xs = np.nonzero(mid)
                    cy, cx = float(ys.mean()), float(xs.mean())
                    off = hypot(cx - (W-1)/2.0, cy - (H-1)/2.0) / hypot(W/2.0, H/2.0)
                else:
                    aspect, off = 0.0, 1.0

                # saturation moyenne (sous-échantillonnée si gros)
                try:
                    import colorsys
                    flat = rgb.reshape(-1,3)/255.0
                    step = max(1, flat.shape[0]//5000)
                    s = 0.0
                    for r,g,b in flat[::step]:
                        s += colorsys.rgb_to_hsv(r,g,b)[1]
                    sat_mean = float(s / (flat[::step].shape[0]))
                except Exception:
                    sat_mean = 0.0

                cf = self._colorfulness(rgb)

                # bordures blanches (5% de bande)
                bw = int(max(1, 0.05*min(H,W)))
                top = (gray[:bw,:] >= self.white_threshold).mean()
                bottom = (gray[-bw:,:] >= self.white_threshold).mean()
                left = (gray[:,:bw] >= self.white_threshold).mean()
                right = (gray[:,-bw:] >= self.white_threshold).mean()
                border_white = float(np.mean([top,bottom,left,right]))

                try:
                    fsize = path.stat().st_size
                    bpp = float(fsize) / float(max(1, H*W*3))
                except Exception:
                    bpp = 0.0

                # write row
                out[i, :len(self.basic_cols_)] = [w, h, occ, white_ratio, black_ratio]
                out[i, len(self.basic_cols_):] = [
                    gray_mean, gray_std, float(p10), float(p90), dyn,
                    ent, lapv, edged,
                    aspect, off,
                    sat_mean, cf,
                    border_white, bpp
                ]
            except Exception:
                # ligne à zéro si problème d'I/O
                pass

        return pd.DataFrame(out, index=idxs, columns=self.columns_)

    def get_feature_names_out(self, input_features=None):
        return self.columns_
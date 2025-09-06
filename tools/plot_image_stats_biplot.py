# tools/plot_image_stats_biplot.py
# -*- coding: utf-8 -*-
"""
Biplot des features images interprétables + résumé numérique OK/Erreur.
"""
from __future__ import annotations
import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
try: os.chdir(str(REPO_ROOT))
except Exception: pass

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from features.image_stats import ImageStatsFeaturizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _load_ok_mask(preds_csv: str, index_like=None):
    if not preds_csv or not os.path.exists(preds_csv): return None
    df = pd.read_csv(preds_csv, index_col=0)
    if not {"y_true","y_pred"}.issubset(df.columns): return None
    ok = (df["y_true"].astype(str) == df["y_pred"].astype(str))
    if index_like is not None: ok = ok.reindex(index_like, fill_value=False)
    return ok.values

def load_cfg(path):
    with open(path, "rb") as f: return tomllib.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="features/config.toml")
    ap.add_argument("--out", default="results/figures/biplot_image_stats.png")
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--preds", default="", help="CSV OOF (y_true,y_pred) pour OK/Erreur")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg.get("paths", {})
    X_train = pd.read_csv(paths.get("x_train_csv", "data/X_train_update.csv"), index_col=0)

    img_dir = cfg.get("images", {}).get("train_dir", "data/images/train")
    stats_cfg = cfg.get("images", {}).get("stats", {})
    white_th = int(stats_cfg.get("white_threshold", 230))
    black_th = int(stats_cfg.get("black_threshold", 25))
    min_area = int(stats_cfg.get("min_area", 16))

    with_ids = X_train.dropna(subset=["imageid","productid"])
    if with_ids.empty:
        raise RuntimeError("Aucune image référencée.")

    n = len(with_ids)
    take = min(args.sample, n)
    sample_df = with_ids.sample(n=take, random_state=42)

    stats = ImageStatsFeaturizer(
        image_dir=img_dir, white_threshold=white_th, black_threshold=black_th,
        min_area=min_area, out_prefix="auto", use_cache=False
    ).fit(sample_df)
    Z = stats.transform(sample_df)
    cols = getattr(stats, "columns_", ["width","height","occupancy","white_ratio","black_ratio"])
    df = pd.DataFrame(Z, index=sample_df.index, columns=cols)

    # Résumé numérique OK vs Erreur (si preds fournis)
    ok_mask = _load_ok_mask(args.preds, index_like=sample_df.index)
    if ok_mask is not None:
        df2 = df.copy()
        df2["ok"] = ok_mask
        summary = df2.groupby("ok")[cols].mean().rename(index={True:"OK", False:"Erreur"})
        out_csv = Path(args.out).with_suffix("").as_posix() + "_summary.csv"
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv)
        print(f"[OK] résumé OK/Erreur → {out_csv}")

    # Standardisation + PCA
    Zs = StandardScaler(with_mean=True, with_std=True).fit_transform(df.values)
    XY = PCA(n_components=2, random_state=42).fit_transform(Zs)

    # Loadings 2D (flèches)
    loadings = np.linalg.svd(Zs, full_matrices=False)[2][:2].T  # équivalent à PCA.components_.T
    span = XY.max(axis=0) - XY.min(axis=0); span[span==0]=1.0
    scale = 0.3 * span
    scaled_vecs = loadings * (scale / (np.max(np.abs(loadings), axis=0) + 1e-12))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9,6))
    if ok_mask is None:
        plt.scatter(XY[:,0], XY[:,1], s=10, alpha=0.25, label="Images")
    else:
        plt.scatter(XY[ok_mask,0], XY[ok_mask,1], s=10, alpha=0.20, label="OK")
        plt.scatter(XY[~ok_mask,0], XY[~ok_mask,1], s=10, alpha=0.35, label="Erreur")
    plt.title("Biplot — image stats (PCA 2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    origin = np.zeros(2)
    for (dx, dy), lab in zip(scaled_vecs, cols):
        plt.arrow(origin[0], origin[1], dx, dy, head_width=0.02*span.max(), length_includes_head=True)
        plt.text(dx*1.07, dy*1.07, lab, fontsize=9)

    plt.legend(); plt.tight_layout(); plt.savefig(args.out, dpi=180); plt.close()
    print(f"[OK] {args.out}")

if __name__ == "__main__":
    main()
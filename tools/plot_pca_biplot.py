# tools/plot_pca_biplot.py
# -*- coding: utf-8 -*-
"""
Biplot 2D (docs + flèches de mots) pour B2, avec coloration par thématique possible.
"""
from __future__ import annotations
import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
try: os.chdir(str(REPO_ROOT))
except Exception: pass

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from features.text_cleaner import TextCleaner
from features.text_vectorizer import TextTfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def load_cfg(path): 
    with open(path, "rb") as f: return tomllib.load(f)

def _coerce_df_param(x, default_int=2, default_float=0.95):
    try:
        v = float(x)
        if v >= 1: return int(round(v))
        if 0.0 <= v <= 1.0: return v
    except Exception:
        pass
    return default_int if default_int is not None else default_float

def build_text_pipeline_from_cfg(cfg):
    tc = TextCleaner(
        remove_html=True,
        translate_map_path=cfg.get("text", {}).get("translate_map_path", None),
        use_stem=bool(cfg.get("text", {}).get("use_stem", True)),
        clean_special=bool(cfg.get("text", {}).get("clean_special", True)),
        handle_emojis=bool(cfg.get("text", {}).get("handle_emojis", True)),
        remove_numbers=bool(cfg.get("text", {}).get("remove_numbers", False)),
    )
    tfidf = TextTfidfVectorizer(
        analyzer="word",
        max_features=int(cfg.get("text", {}).get("max_features", 100_000)),
        ngram_range=(int(cfg.get("text", {}).get("ngram_min", 1)),
                     int(cfg.get("text", {}).get("ngram_max", 2))),
        min_df=_coerce_df_param(cfg.get("text", {}).get("min_df", 2)),
        max_df=_coerce_df_param(cfg.get("text", {}).get("max_df", 0.95), default_float=0.95),
        sublinear_tf=bool(cfg.get("text", {}).get("sublinear_tf", True)),
        norm=str(cfg.get("text", {}).get("norm", "l2")),
        strip_accents=cfg.get("text", {}).get("strip_accents", "unicode"),
        lowercase=False,
        dtype="float64",
    )
    return tc, tfidf

def _load_theme_map(path="features/theme_map.json"):
    p = Path(path)
    if not p.exists(): return None
    import json
    with open(p, "r", encoding="utf-8") as f: raw = json.load(f)
    return {str(k): v for k, v in raw.items()}

def make_biplot(XY_docs, loadings_2d, feat_names, y_true=None, theme_map=None,
                top_terms=15, sample_docs=12000, out_path="results/figures/biplot_b2.png",
                title="Biplot (texte — TF-IDF + SVD 2D)"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = XY_docs.shape[0]
    take = min(sample_docs, n)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=take, replace=False)) if take < n else np.arange(n)
    Xd = XY_docs[idx]
    colors = None; labels = None

    if (y_true is not None) and (theme_map is not None):
        themes = pd.Series(y_true).astype(str).map(theme_map).fillna("Autres").values
        themes = themes[idx]
        # palette limitée (10)
        uniq = pd.Series(themes).value_counts().index[:10].tolist()
        cmap = plt.cm.get_cmap("tab10", len(uniq))
        plt.figure(figsize=(9,6))
        for i, t in enumerate(uniq):
            m = (themes == t)
            plt.scatter(Xd[m,0], Xd[m,1], s=8, alpha=0.20, label=t, c=[cmap(i)])
        plt.legend(loc="best", fontsize=8)
    else:
        plt.figure(figsize=(9,6))
        plt.scatter(Xd[:,0], Xd[:,1], s=8, alpha=0.15)

    # top termes (limités par défaut à 15 pour lisibilité)
    norms = np.linalg.norm(loadings_2d, axis=1)
    top_idx = np.argsort(norms)[-top_terms:]
    terms = feat_names[top_idx]; vecs = loadings_2d[top_idx]
    span = (Xd.max(axis=0) - Xd.min(axis=0)); span[span==0]=1.0
    scale = 0.20 * span
    scaled_vecs = vecs * (scale / (np.max(np.abs(vecs), axis=0) + 1e-12))

    origin = np.zeros(2)
    for (dx, dy), lab in zip(scaled_vecs, terms):
        plt.arrow(origin[0], origin[1], dx, dy, head_width=0.02*span.max(), length_includes_head=True)
        plt.text(dx*1.05, dy*1.05, lab, fontsize=8)

    plt.title(title)
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()
    print(f"[OK] biplot → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="features/config.toml")
    ap.add_argument("--out", default="results/figures/biplot_b2.png")
    ap.add_argument("--top-terms", type=int, default=15)  # flèches plus lisibles
    ap.add_argument("--sample", type=int, default=12000)
    ap.add_argument("--color-theme", action="store_true", help="Colorer par thématique si Y_train + theme_map dispo")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg.get("paths", {})
    X_train = pd.read_csv(paths.get("x_train_csv", "data/X_train_update.csv"), index_col=0)
    y_train = pd.read_csv(paths.get("y_train_csv", "data/Y_train_update.csv"), index_col=0).squeeze()

    tc, tfidf = build_text_pipeline_from_cfg(cfg)
    corpus = (X_train["designation"].fillna("") + " " + X_train["description"].fillna("")).map(tc.clean_text)
    Xtf = tfidf.fit_transform(corpus)
    feat_names = np.array(tfidf.get_feature_names_out())

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    XY_docs = svd.fit_transform(Xtf)
    loadings_2d = svd.components_.T

    theme_map = _load_theme_map() if args.color-theme_map else None
    make_biplot(
        XY_docs=XY_docs, loadings_2d=loadings_2d, feat_names=feat_names,
        y_true=y_train.astype(str).values, theme_map=theme_map,
        top_terms=args.top_terms, sample_docs=args.sample, out_path=args.out,
        title="Biplot (texte — TF-IDF + SVD 2D)"
    )

if __name__ == "__main__":
    main()
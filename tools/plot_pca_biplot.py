# tools/plot_pca_biplot.py
# -*- coding: utf-8 -*-
"""
Biplot 2D (documents + flèches de mots) pour la branche texte (B2).

Usage :
  python -m tools.plot_pca_biplot --config features/config.toml --out results/figures/biplot_b2.png --top-terms 30 --sample 12000

Notes :
- On reconstruit le pipeline texte (TextCleaner + TF-IDF) à partir du TOML.
- On applique TruncatedSVD(n_components=2) pour projeter documents ET récupérer les "loadings" des mots.
- On trace un nuage de points (docs, échantillonné) + les flèches/labels des top mots.
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Rendre le paquet projet importable même lancé depuis tools/ ---
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    os.chdir(str(REPO_ROOT))
except Exception:
    pass

# TOML loader
try:
    import tomllib  # py311+
except ModuleNotFoundError:
    import tomli as tomllib

# Import projet
from features.text_cleaner import TextCleaner
from features.text_vectorizer import TextTfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def load_cfg(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)

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
        ngram_range=(
            int(cfg.get("text", {}).get("ngram_min", 1)),
            int(cfg.get("text", {}).get("ngram_max", 2)),
        ),
        # coercion robuste min_df / max_df
        min_df=_coerce_df_param(cfg.get("text", {}).get("min_df", 2), default_int=2),
        max_df=_coerce_df_param(cfg.get("text", {}).get("max_df", 0.95), default_float=0.95),
        sublinear_tf=bool(cfg.get("text", {}).get("sublinear_tf", True)),
        norm=str(cfg.get("text", {}).get("norm", "l2")),
        strip_accents=cfg.get("text", {}).get("strip_accents", "unicode"),
        lowercase=False,
        dtype="float64",
    )
    return tc, tfidf

def _coerce_df_param(x, default_int=2, default_float=0.95):
    try:
        v = float(x)
        if v >= 1:
            return int(round(v))
        if 0.0 <= v <= 1.0:
            return v
    except Exception:
        pass
    return (default_int if default_int is not None else default_float)

def make_biplot(
    XY_docs: np.ndarray,
    loadings_2d: np.ndarray,
    feat_names: np.ndarray,
    top_terms: int = 30,
    sample_docs: int = 10000,
    out_path: str = "results/figures/biplot_b2.png",
    title: str = "Biplot (texte — TF-IDF + SVD 2D)",
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Échantillonnage documents pour tracer plus léger
    n = XY_docs.shape[0]
    take = min(sample_docs, n)
    if take < n:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=take, replace=False))
        Xd = XY_docs[idx]
    else:
        Xd = XY_docs

    # Sélection des top termes par norme du vecteur (PC1, PC2)
    norms = np.linalg.norm(loadings_2d, axis=1)
    top_idx = np.argsort(norms)[-top_terms:]
    terms = feat_names[top_idx]
    vecs = loadings_2d[top_idx]

    # Mise à l'échelle des flèches pour qu'elles soient visibles
    # On adapte la longueur max des flèches à ~20% de l'étendue des docs
    span = (Xd.max(axis=0) - Xd.min(axis=0))
    if np.any(span == 0):
        span[span == 0] = 1.0
    scale = 0.20 * span  # 20% du span
    scaled_vecs = vecs * (scale / (np.max(np.abs(vecs), axis=0) + 1e-12))

    plt.figure(figsize=(9, 6))
    plt.scatter(Xd[:, 0], Xd[:, 1], s=8, alpha=0.15)
    plt.title(title)
    plt.xlabel("Component 1"); plt.ylabel("Component 2")

    origin = np.zeros(2)
    for (dx, dy), label in zip(scaled_vecs, terms):
        plt.arrow(origin[0], origin[1], dx, dy, head_width=0.02*span.max(), length_includes_head=True)
        # position du label au bout de la flèche
        plt.text(dx * 1.05, dy * 1.05, label, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"[OK] biplot → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="features/config.toml")
    ap.add_argument("--out", default="results/figures/biplot_b2.png")
    ap.add_argument("--top-terms", type=int, default=30)
    ap.add_argument("--sample", type=int, default=12000, help="docs max affichés")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg.get("paths", {})
    X_train = pd.read_csv(paths.get("x_train_csv", "data/X_train_update.csv"), index_col=0)

    # 1) Reconstruire corpus + TF-IDF
    tc, tfidf = build_text_pipeline_from_cfg(cfg)
    corpus = (X_train["designation"].fillna("") + " " + X_train["description"].fillna("")).map(tc.clean_text)

    Xtf = tfidf.fit_transform(corpus)
    feat_names = np.array(tfidf.get_feature_names_out())

    # 2) SVD 2D pour documents et loadings termes
    svd = TruncatedSVD(n_components=2, random_state=42)
    XY_docs = svd.fit_transform(Xtf)  # (n_docs, 2)
    # loadings 2D pour les features = composantes transposées
    # (équivalent biplot simple ; pour d'autres échelles on pourrait multiplier par singular_values_)
    loadings_2d = svd.components_.T  # (n_features, 2)

    make_biplot(
        XY_docs=XY_docs,
        loadings_2d=loadings_2d,
        feat_names=feat_names,
        top_terms=args.top_terms,
        sample_docs=args.sample,
        out_path=args.out,
        title="Biplot (texte — TF-IDF + SVD 2D)",
    )

if __name__ == "__main__":
    main()
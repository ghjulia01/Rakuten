#!/usr/bin/env python3

"""
Trace une matrice de confusion (normalisée) pour une ligne de base choisie ou le modèle multimodal complet.

Ce script recalcule les prédictions hors pli sur X_train via cross_val_predict,
il ne dépend donc pas des prédictions précédemment enregistrées.

Examples de scripts:
# Texte seul (B2) – 3 folds, normalisé, top 25 classes les plus fréquentes
python tools/plot_confusion_matrix.py --config features/config.toml --baseline b2

# Multimodal (B4) – 5 folds, non normalisé, top 30 classes
python tools/plot_confusion_matrix.py --config features/config.toml --baseline b4 --splits 5 --normalize false --topk 30

# Image seule (B3) – sortie personnalisée
python tools/plot_confusion_matrix.py --config features/config.toml --baseline b3 --out results/figures/cm_b3.png

Baselines:
  b0 = Dummy(most_frequent)
  b1 = Dummy(stratified)
  b2 = Text only  (TextCleaner + TF-IDF -> LR)
  b3 = Image only (ImageLoader -> flatten -> PCA -> LR)
  b4 = Multimodal (your full pipeline with sampling)

Sorties générées

results/figures/cm_<baseline>.png (matrice de confusion top-k)
results/figures/cm_<baseline>_full.csv (toutes les classes)
results/figures/report_<baseline>_cv.txt (classification_report)


  - PNG figure at results/figures/cm_<baseline>.png (unless overridden by --out)
  - CSV with the full (non-topk) confusion matrix: results/figures/cm_<baseline>_full.csv
  - TXT classification report: results/figures/report_<baseline>_cv.txt

Notes

B4 utilise le pipeline complet (texte + image + sampling) 
en réimportant create_combined_pipeline et make_sampling_strategies.

La normalisation "true" affiche des pourcentages par classe réelle 
(chaque ligne somme à 1).

--topk garde la figure lisible en ne traçant que les classes les plus fréquentes ; 
le CSV complet contient toutes les classes.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline, Pipeline as SkPipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression

# ----------- TOML loader -----------
try:
    import tomllib  # Py 3.11+
except Exception:
    import tomli as tomllib  # pip install tomli for Py < 3.11


# ----------- Project imports -----------
# Expect project layout with packages:
# - models.text_pipeline, models.image_pipeline
# - main.train_model (for create_combined_pipeline, make_sampling_strategies when baseline=b4)
from models.text_pipeline import create_text_pipeline
from models.image_pipeline import create_image_pipeline
from main.train_model import create_combined_pipeline, make_sampling_strategies


def load_cfg(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_pipeline(kind: str, cfg: dict, y_train: pd.Series | None = None):
    """
    Return (pipeline, needed_cols) for the requested baseline/model.
    kind in {'b0','b1','b2','b3','b4'}
    """
    need_cols = ["designation", "description", "productid", "imageid"]

    if kind == "b0":
        return DummyClassifier(strategy="most_frequent"), ["designation"]

    if kind == "b1":
        return DummyClassifier(strategy="stratified", random_state=42), ["designation"]

    if kind == "b2":
        text_cfg = cfg.get("text", {})
        text_branch = create_text_pipeline(
            max_features=text_cfg.get("max_features", 5000),
            translate_map_path=text_cfg.get("translate_map_path", None),
            use_stem=bool(text_cfg.get("use_stem", True)),
            min_df=text_cfg.get("min_df", 0.0),
            max_df=text_cfg.get("max_df", 1.0),
        )
        clf = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced")
        return SkPipeline([("text", text_branch), ("clf", clf)]), need_cols

    if kind == "b3":
        img_size = tuple(cfg["images"].get("size", [64, 64]))
        img_dir  = cfg["images"]["train_dir"]
        img_branch = create_image_pipeline(
            image_dir=img_dir,
            image_size=img_size,
            dim_reduction={"enabled": False},
        )
        pca_n = int(cfg.get("images", {}).get("dim_reduction", {}).get("n_components", 100))
        img_pca = make_pipeline(img_branch, PCA(n_components=pca_n, random_state=42))
        clf = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced")
        return SkPipeline([("img", img_pca), ("clf", clf)]), need_cols

    if kind == "b4":
        if y_train is None:
            raise ValueError("y_train is required to build the multimodal pipeline (b4).")
        under, over = make_sampling_strategies(
            y_train,
            major_class=cfg["sampling"]["major_class"],
            major_cap=cfg["sampling"]["major_cap"],
            tail_min=cfg["sampling"]["tail_min"],
        )
        pipe = create_combined_pipeline(cfg, under, over)
        return pipe, need_cols

    raise ValueError(f"Unknown baseline kind: {kind}")


def plot_confmat(cm: np.ndarray, labels: list[str], title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, aspect='auto')  # no explicit colormap to follow style guide
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[ok] Confusion matrix saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.toml")
    ap.add_argument("--baseline", choices=["b0","b1","b2","b3","b4"], required=True)
    ap.add_argument("--splits", type=int, default=3, help="CV splits for OOF predictions")
    ap.add_argument("--normalize", choices=["true","false"], default="true",
                    help="Row-normalize confusion matrix (percent per true class)")
    ap.add_argument("--topk", type=int, default=25, help="Limit plot to top-k classes by support (full CSV still saved)")
    ap.add_argument("--out", default=None, help="Output PNG path (default: results/figures/cm_<baseline>.png)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    X_train = pd.read_csv(cfg["paths"]["x_train_csv"], index_col=0)
    y_train = pd.read_csv(cfg["paths"]["y_train_csv"], index_col=0).squeeze()

    pipe, need_cols = build_pipeline(args.baseline, cfg, y_train=y_train if args.baseline=="b4" else None)

    cv = StratifiedKFold(n_splits=int(args.splits), shuffle=True, random_state=42)
    print(f"[i] Computing OOF predictions for {args.baseline.upper()} with {args.splits}-fold CV ...")
    y_pred = cross_val_predict(pipe, X_train[need_cols], y_train, cv=cv, n_jobs=1, method="predict")

    # Full confusion matrix with all labels in training set
    labels = np.unique(y_train)
    cm = confusion_matrix(y_train, y_pred, labels=labels)

    # Save the full (non-topk) matrix as CSV
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    full_csv = os.path.join("results", "figures", f"cm_{args.baseline}_full.csv")
    full_df = pd.DataFrame(cm, index=labels, columns=labels)
    full_df.to_csv(full_csv, index=True)
    print(f"[ok] Full confusion matrix (all classes) saved: {full_csv}")

    # Classification report
    report_txt = os.path.join("results", "figures", f"report_{args.baseline}_cv.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(classification_report(y_train, y_pred, digits=4))
    print(f"[ok] Classification report saved: {report_txt}")

    # Optionally normalize by true-class row
    if args.normalize == "true":
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, where=row_sums!=0)

    # Decide which labels to plot (top-k by support in y_train)
    support = pd.Series(y_train).value_counts().sort_values(ascending=False)
    top_labels = support.index[:args.topk].to_numpy()

    # Slice cm to top-k order (by true label rows and predicted label columns)
    # We'll also keep predicted labels in the same top-k order for readability
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    top_idx = [label_to_idx[lab] for lab in top_labels if lab in label_to_idx]

    cm_top = cm[np.ix_(top_idx, top_idx)]
    out_path = args.out or os.path.join("results", "figures", f"cm_{args.baseline}.png")
    plot_confmat(cm_top, labels=[str(l) for l in top_labels], title=f"Confusion Matrix — {args.baseline.upper()}", out_path=out_path)


if __name__ == "__main__":
    main()

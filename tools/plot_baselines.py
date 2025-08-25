#!/usr/bin/env python3

"""
Plot baseline results from results/baseline_results_summary.csv

Exemples de scripts:
# F1 macro (par défaut)
python tools/plot_baselines.py

# Choisir le CSV et la métrique
python tools/plot_baselines.py --csv results/baseline_results_summary.csv --metric f1_weighted

# Choisir le chemin de sortie de l’image
python tools/plot_baselines.py --out results/figures/baseline_f1_weighted.png


Notes:
- Attend un fichier CSV contenant au moins : baseline, f1_macro, f1_weighted, cv_splits, train_infer_time_sec
- Si plusieurs exécutions par baseline existent, le script trace la moyenne avec un écart-type de +/- 1 comme barres d'erreur.
- Enregistre à la fois une figure PNG et un fichier CSV agrégé à côté.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_CSV = os.path.join("results", "baseline_results_summary.csv")
DEFAULT_OUT = os.path.join("results", "figures", "baseline_f1_macro.png")

def load_and_aggregate(csv_path: str, metric: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"baseline", metric}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}; found {list(df.columns)}")

    # Normalize baseline names to upper-case (B0, B1, ...)
    df["baseline"] = df["baseline"].astype(str).str.upper()

    # aggregate: mean, std, count per baseline
    agg = (
        df.groupby("baseline")[metric]
          .agg(["mean", "std", "count"])
          .reset_index()
    )

    # Preferred ordering if present
    order = ["B0", "B1", "B2", "B3", "B4"]
    agg["__order__"] = agg["baseline"].apply(lambda b: order.index(b) if b in order else 999)
    agg = agg.sort_values(["__order__", "mean"], ascending=[True, False]).drop(columns="__order__")
    return agg

def make_plot(agg: pd.DataFrame, metric: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    labels = agg["baseline"].tolist()
    means  = agg["mean"].to_numpy()
    stds   = agg["std"].fillna(0.0).to_numpy()
    counts = agg["count"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))

    bars = ax.bar(x, means, yerr=stds, capsize=6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(1.0, means.max() + 0.05))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Baselines — {metric.replace('_', ' ').title()} (mean ± std)")

    # Annotate bars with value and (n)
    for i, b in enumerate(bars):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2.0, h + 0.01, f"{h:.3f}\n(n={counts[i]})",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Graph: {out_path}")

def save_aggregated_csv(agg: pd.DataFrame, out_path_img: str, metric: str):
    base_dir = os.path.dirname(out_path_img)
    out_csv  = os.path.join(base_dir, f"baseline_{metric}_aggregated.csv")
    agg.to_csv(out_csv, index=False)
    print(f"CSV sauvé: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to baseline_results_summary.csv")
    ap.add_argument("--metric", choices=["f1_macro", "f1_weighted"], default="f1_macro", help="Metric to plot")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output PNG path")
    args = ap.parse_args()

    agg = load_and_aggregate(args.csv, args.metric)
    print(agg)

    make_plot(agg, args.metric, args.out)
    save_aggregated_csv(agg, args.out, args.metric)

if __name__ == "__main__":
    main()

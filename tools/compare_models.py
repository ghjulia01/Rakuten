import argparse
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("compare_models")

DEFAULT_CSV = Path("results") / "baseline_results_summary.csv"
OUT_PNG1    = Path("results") / "figures" / "compare_f1_macro.png"
OUT_PNG2    = Path("results") / "figures" / "compare_f1_weighted.png"

def load_agg(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    latest = df.groupby("baseline", as_index=False).tail(1)
    latest.to_csv("results/baseline_results_summary_latest.csv", index=False)
    df["baseline"] = df["baseline"].astype(str).str.upper()
    agg = (
        df.groupby("baseline")[["f1_macro", "f1_weighted"]]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    # aplatir les colonnes MultiIndex
    agg.columns = ["baseline",
                   "f1_macro_mean", "f1_macro_std", "f1_macro_n",
                   "f1_weighted_mean", "f1_weighted_std", "f1_weighted_n"]
    return agg

def barplot(agg: pd.DataFrame, col_mean: str, col_std: str, col_n: str, out_png: Path, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    order = ["B0","B1","B2","B3","B4"]
    agg = agg.copy()
    agg["__o__"] = agg["baseline"].apply(lambda b: order.index(b) if b in order else 999)
    agg = agg.sort_values(["__o__", col_mean], ascending=[True, False]).drop(columns="__o__")

    ax = agg.plot(kind="bar", x="baseline", y=col_mean, yerr=col_std, capsize=6, legend=False)
    ax.set_ylim(0, max(1.0, agg[col_mean].max() + 0.05))
    ax.set_ylabel(col_mean.replace("_", " ").title())
    ax.set_title(title)
    for i, (m, n) in enumerate(zip(agg[col_mean], agg[col_n])):
        ax.text(i, m + 0.01, f"{m:.3f}\n(n={int(n)})", ha="center", va="bottom", fontsize=9)
    ax.figure.tight_layout()
    ax.figure.savefig(out_png, dpi=150)
    log.info("Graphique sauvegardé: %s", out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    agg = load_agg(csv_path)
    barplot(agg, "f1_macro_mean", "f1_macro_std", "f1_macro_n", OUT_PNG1, "Baselines — F1 macro")
    barplot(agg, "f1_weighted_mean", "f1_weighted_std", "f1_weighted_n", OUT_PNG2, "Baselines — F1 weighted")

def compare_all_models(csv_path: str | Path = DEFAULT_CSV) -> dict:
    """
    API function used by main.train_model --compare-all.
    Loads the aggregated results CSV and writes two comparison plots.
    Returns the output paths for convenience.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    agg = load_agg(csv_path)

    # Ensure output directory exists
    OUT_PNG1.parent.mkdir(parents=True, exist_ok=True)

    barplot(agg, "f1_macro_mean",    "f1_macro_std",    "f1_macro_n",    OUT_PNG1, "Baselines — F1 macro")
    barplot(agg, "f1_weighted_mean", "f1_weighted_std", "f1_weighted_n", OUT_PNG2, "Baselines — F1 weighted")

    return {"f1_macro_png": str(OUT_PNG1), "f1_weighted_png": str(OUT_PNG2)}

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Analyse fine des contributions des features 'image_stats_combined' dans B4.

Sorties :
- results/reports/b4_stats_global_contribs.csv
    cols: feature, signed_mean, abs_mean
- results/reports/b4_stats_contribs_per_class.csv
    cols: class_id, class_name, feature, signed_mean, abs_mean
- results/figures/b4_stats_topnegpos.png
    barplot des 10 plus négatives / 10 plus positives (signed_mean)

Usage:
  python tools/stats_contribs_b4.py --model artifacts/b4.joblib \
    --x data/X_train_update.csv --y data/Y_train_CVw08PX.csv \
    --labels features/labels_map.json --top 10
"""
import argparse, json, os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import joblib
import matplotlib.pyplot as plt

        

try:
    from models.cnn_features import ToFloat32, CNNFeaturizer  # noqa: F401
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.pipeline")

# --- ajoute le repo root au PYTHONPATH pour importer des modules ---
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # repo root = parent of tools/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# pour importer main.train_model 
import main.train_model as _tm          
sys.modules['train_model'] = _tm

def _as_dense(m):
    return m.toarray() if sparse.issparse(m) else np.asarray(m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--x", "--x-csv", dest="x_csv", required=True)
    ap.add_argument("--y", "--y-csv", dest="y_csv", required=True)
    ap.add_argument("--labels", "--labels-map", dest="labels_map", default=None)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    out_reports = Path("results/reports"); out_reports.mkdir(parents=True, exist_ok=True)
    out_fig = Path("results/figures"); out_fig.mkdir(parents=True, exist_ok=True)

    # 1) charge données et modèle
    X = pd.read_csv(args.x_csv)
    y = pd.read_csv(args.y_csv).iloc[:, 0].values  # 1ère colonne = prdtypecode
    pipe = joblib.load(args.model)

    # 2) récupère la FeatureUnion et le scaler
    #    (noms attendus d'après ton pipeline)
    fu = pipe.named_steps.get("features") or pipe.named_steps.get("union")
    scaler = pipe.named_steps.get("scaler")

    # 3) slices par bloc + noms des colonnes du bloc stats
    trans_dict = dict(fu.transformer_list)
    if "image_stats_combined" not in trans_dict:
        raise RuntimeError("Bloc 'image_stats_combined' introuvable dans la fusion.")
    stats_tr = trans_dict["image_stats_combined"]

    # calcule les tailles de chaque bloc pour créer les slices
    offset = 0
    slices = {}
    for name, tr in fu.transformer_list:
        Xi = tr.transform(X)
        ncols = Xi.shape[1]
        slices[name] = slice(offset, offset + ncols)
        offset += ncols

    sl_stats = slices["image_stats_combined"]
    # noms de colonnes définis dans ton featurizer
    if hasattr(stats_tr, "columns_"):
        stat_names = list(stats_tr.columns_)
    else:
        # re-fit pour avoir columns_ si nécessaire (léger)
        stats_tr.fit(X)
        stat_names = list(stats_tr.columns_)

    # 4) features après SCALER (comme vues par le modèle)
    X_fused = fu.transform(X)                # hstack
    X_scaled = scaler.transform(X_fused)     # standardisation
    # isole le bloc stats (19 colonnes) et densifie
    X_stats = _as_dense(X_scaled[:, sl_stats])

    # 5) récupère le classifieur et ses poids
    clf = pipe.named_steps["model"]
    W = clf.coef_                # (n_classes, n_features_totales)
    W_stats = _as_dense(W[:, sl_stats])  # (n_classes, 19)

    classes = clf.classes_
    # mapping éventuel id->nom lisible
    labels = json.load(open(args.labels_map)) if args.labels_map else {}
    id2name = {int(k): str(v) for k, v in (labels or {}).items()}

    # 6) contributions globales et par classe
    signed_sum = np.zeros(X_stats.shape[1], dtype=np.float64)
    abs_sum    = np.zeros(X_stats.shape[1], dtype=np.float64)
    n_total = 0

    rows = []  # per-class
    for k, cls in enumerate(classes):
        mask = (y == cls)
        if mask.sum() == 0:
            continue
        Xk = X_stats[mask, :]                  # (nk, 19)
        wk = W_stats[k, :][None, :]            # (1, 19) pour broadcast
        contrib = Xk * wk                      # (nk, 19), signé
        signed_mean = contrib.mean(axis=0)
        abs_mean = np.abs(contrib).mean(axis=0)

        # cumul global
        signed_sum += signed_mean * mask.sum()
        abs_sum    += abs_mean * mask.sum()
        n_total    += mask.sum()

        # lignes par classe
        for j, fname in enumerate(stat_names):
            rows.append({
                "class_id": int(cls),
                "class_name": id2name.get(int(cls), str(cls)),
                "feature": fname,
                "signed_mean": float(signed_mean[j]),
                "abs_mean": float(abs_mean[j])
            })

    # 7) DataFrames et sauvegardes
    per_class = pd.DataFrame(rows)
    per_class.to_csv(out_reports / "b4_stats_contribs_per_class.csv", index=False, encoding="utf-8")

    global_df = pd.DataFrame({
        "feature": stat_names,
        "signed_mean": (signed_sum / max(1, n_total)),
        "abs_mean":    (abs_sum / max(1, n_total))
    }).sort_values("signed_mean")
    global_df.to_csv(out_reports / "b4_stats_global_contribs.csv", index=False, encoding="utf-8")

    # 8) Figure top négatives / positives (signed)
    top = args.top
    neg = global_df.nsmallest(top, "signed_mean")
    pos = global_df.nlargest(top,  "signed_mean")

    fig, ax = plt.subplots(figsize=(11, 6))
    ylabels = list(neg["feature"]) + list(pos["feature"][::-1])
    yvals   = list(neg["signed_mean"]) + list(pos["signed_mean"][::-1])
    y_pos = np.arange(len(ylabels))
    ax.barh(y_pos, yvals)
    ax.set_yticks(y_pos, ylabels)
    ax.set_xlabel("Impact moyen signé (x·w)")
    ax.set_title("B4 — Stats images : top négatives vs positives (impact signé)")
    plt.tight_layout()
    fig.savefig(out_fig / "b4_stats_topnegpos.png", dpi=200)
    print("[OK] Global per-stat saved:",
          out_reports / "b4_stats_global_contribs.csv",
          out_reports / "b4_stats_contribs_per_class.csv",
          out_fig / "b4_stats_topnegpos.png")

    # --- Top-K stats par classe (optionnel) ---
    import pandas as pd

    pc = pd.read_csv(out_reports / "b4_stats_contribs_per_class.csv")
    K = int(args.top or 10)  # réutilise --top si fourni

    topk = (pc.sort_values(["class_id", "abs_mean"], ascending=[True, False])
            .groupby("class_id")
            .head(K))

    top_path = out_reports / f"b4_top{K}_stats_per_class.csv"
    topk.to_csv(top_path, index=False)
    print("[OK] Top-K per-class saved:", top_path)

if __name__ == "__main__":
    main()
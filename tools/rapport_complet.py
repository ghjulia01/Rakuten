# 
# tools/rapport_complet.py
# Génère un rapport (Markdown + HTML) à partir d'un CSV de prédictions,
# ou en fusionnant preds (id,y_pred) + truth (id,y_true).
# Exemples :
# 1) OOF direct (contient y_true,y_pred) :
#    python tools/rapport_complet.py --preds results/preds_oof_b4.csv --labels-map features/labels_map.json --out-md reports/oof_b4.md
# 2) Fusion test (preds_b4.csv + df.csv) :
#    python tools/rapport_complet.py --preds results/preds_b4.csv --truth notebooks/df.csv --id-col productid --true-col prdtypecode --labels-map features/labels_map.json --out-md reports/test_b4.md

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def load_json(p):
    if not p: return None
    p = Path(p)
    if not p.exists(): return None
    with open(p, "r", encoding="utf-8") as f:
        return {str(k): v for k, v in json.load(f).items()}

def top_confusions(y_true, y_pred, k=20):
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: str(x))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)  # focus erreurs
    rows, cols = np.where(cm > 0)
    records = [(labels[r], labels[c], int(cm[r, c])) for r, c in zip(rows, cols)]
    df = pd.DataFrame(records, columns=["true", "pred", "count"]).sort_values("count", ascending=False)
    return df.head(k), labels, cm

def summarize_by_theme(per_class_df, labels_map, theme_map):
    if theme_map is None: return None
    inv = {}
    for theme, ids in theme_map.items():
        for cid in ids:
            inv[str(cid)] = theme
    out = per_class_df.copy()
    out["theme"] = out["class_id"].map(inv).fillna("Autres")
    grp = (out.groupby("theme")[["support","f1","precision","recall"]]
              .agg({"support":"sum", "f1":"mean", "precision":"mean", "recall":"mean"})
              .reset_index()
              .sort_values("support", ascending=False))
    return grp

def autodetect_id_col(df_preds, df_truth):
    candidates = ["productid", "id", "sku", "uid"]
    for c in candidates:
        if c in df_preds.columns and c in df_truth.columns:
            return c
    # dernier recours : intersection de colonnes
    inter = [c for c in df_preds.columns if c in df_truth.columns]
    return inter[0] if inter else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV with y_pred or (y_true,y_pred)")
    ap.add_argument("--truth", default=None, help="CSV with ground truth if --preds has no y_true")
    ap.add_argument("--id-col", default=None, help="Join key (auto-detected if omitted)")
    ap.add_argument("--true-col", default="y_true", help="Ground-truth column name (e.g. prdtypecode)")
    ap.add_argument("--pred-col", default="y_pred", help="Prediction column name")
    ap.add_argument("--labels-map", default="features/labels_map.json")
    ap.add_argument("--theme-map", default="features/theme_map.json")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-html", default=None)
    ap.add_argument("--topK", type=int, default=20)
    args = ap.parse_args()

    preds = pd.read_csv(args.preds)

    # 1) Obtenir y_true / y_pred
    has_ytrue = args.true_col in preds.columns
    has_ypred = args.pred_col in preds.columns

    if not has_ypred:
        raise ValueError(f"--preds doit contenir la colonne '{args.pred_col}' (ex: y_pred).")

    if not has_ytrue:
        if args.truth is None:
            raise ValueError(f"--preds ne contient pas '{args.true_col}'. Fournis --truth et (optionnel) --id-col/--true-col.")
        truth = pd.read_csv(args.truth)
        id_col = args.id_col or autodetect_id_col(preds, truth)
        if id_col is None:
            raise ValueError("Impossible de détecter la clé de jointure. Spécifie --id-col.")
        if args.true_col not in truth.columns:
            raise ValueError(f"--truth ne contient pas la colonne '{args.true_col}'.")
        merged = preds.merge(truth[[id_col, args.true_col]], on=id_col, how="inner")
        if merged.empty:
            raise ValueError("La fusion preds/truth est vide. Vérifie --id-col et les fichiers.")
        y_true = merged[args.true_col].astype(str).values
        y_pred = merged[args.pred_col].astype(str).values
    else:
        y_true = preds[args.true_col].astype(str).values
        y_pred = preds[args.pred_col].astype(str).values

    # 2) Maps (facultatives)
    lbl = load_json(args.labels_map) or {}
    themap = load_json(args.theme_map)

    # 3) Métriques par classe
    labels = np.unique(y_true)
    P,R,F,S = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class_id": labels, "precision":P, "recall":R, "f1":F, "support":S})
    per_class["class_name"] = per_class["class_id"].map(lbl).fillna(per_class["class_id"])
    per_class = per_class.sort_values("support", ascending=False)

    # 4) Macro / weighted
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_macro = report["macro avg"]["f1-score"]
    f1_weighted = report["weighted avg"]["f1-score"]

    # 5) Top confusions
    top_err, label_order, cm = top_confusions(y_true, y_pred, k=args.topK)
    top_err["true_name"] = top_err["true"].map(lbl).fillna(top_err["true"])
    top_err["pred_name"] = top_err["pred"].map(lbl).fillna(top_err["pred"])

    # 6) Synthèse par thématique (optionnelle)
    theme_summary = summarize_by_theme(per_class, lbl, themap)

    # 7) Exports
    outdir = Path("results/reports"); outdir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(outdir / "per_class_metrics.csv", index=False)
    top_err.to_csv(outdir / "top_confusions.csv", index=False)
    if theme_summary is not None:
        theme_summary.to_csv(outdir / "themes_summary.csv", index=False)

    # 8) Markdown
    md = []
    md += [f"# Rapport — {Path(args.preds).stem}"]
    md += [f"- **F1-macro**: {f1_macro:.4f}"]
    md += [f"- **F1-weighted**: {f1_weighted:.4f}", ""]
    md += ["## Top confusions",
           "| vrai | prédit | count |", "|:----|:-------|------:|"]
    for _,r in top_err.iterrows():
        md.append(f"| {r['true_name']} ({r['true']}) | {r['pred_name']} ({r['pred']}) | {r['count']} |")
    md += ["", "## Top 30 classes par support",
           "| id | nom | support | precision | recall | f1 |",
           "|---:|:-----|-------:|----------:|-------:|----:|"]
    for _,r in per_class.head(30).iterrows():
        md.append(f"| {r['class_id']} | {r['class_name']} | {int(r['support'])} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    if theme_summary is not None:
        md += ["", "## Synthèse par thématique",
               "| thématique | support | f1 | precision | recall |",
               "|:-----------|--------:|---:|----------:|-------:|"]
        for _,r in theme_summary.iterrows():
            md.append(f"| {r['theme']} | {int(r['support'])} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |")

    out_md = Path(args.out_md) if args.out_md else outdir / f"{Path(args.preds).stem}_report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    out_html = Path(args.out_html) if args.out_html else outdir / f"{Path(args.preds).stem}_report.html"
    out_html.write_text("<html><body><pre>"+("\n".join(md))+"</pre></body></html>", encoding="utf-8")

    print(f"[OK] Wrote:\n- {out_md}\n- {out_html}\n- {outdir/'per_class_metrics.csv'}\n- {outdir/'top_confusions.csv'}" + (f"\n- {outdir/'themes_summary.csv'}" if theme_summary is not None else ""))

if __name__ == "__main__":
    main()
# tools/rapport_complet.py
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def load_json(p):
    if not p: return None
    p = Path(p)
    if not p.exists(): return None
    with open(p, "r", encoding="utf-8") as f:
        return {str(k): v for k, v in json.load(f).items()}

def top_confusions(y_true, y_pred, k=20):
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: str(x))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # zero diagonal to focus on errors
    np.fill_diagonal(cm, 0)
    rows, cols = np.where(cm > 0)
    records = []
    for r, c in zip(rows, cols):
        records.append((labels[r], labels[c], int(cm[r, c])))
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV with y_true,y_pred")
    ap.add_argument("--labels-map", default="features/labels_map.json")
    ap.add_argument("--theme-map", default="features/theme_map.json")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-html", default=None)
    ap.add_argument("--topK", type=int, default=20)
    args = ap.parse_args()

    preds = pd.read_csv(args.preds)
    if not {"y_true","y_pred"}.issubset(preds.columns):
        raise ValueError("CSV must contain y_true,y_pred")
    y_true = preds["y_true"].astype(str).values
    y_pred = preds["y_pred"].astype(str).values

    lbl = load_json(args.labels_map) or {}
    themap = load_json(args.theme_map)

    # per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    labels = np.unique(y_true)
    P,R,F,S = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class_id": labels, "precision":P, "recall":R, "f1":F, "support":S})
    per_class["class_name"] = per_class["class_id"].map(lbl).fillna(per_class["class_id"])
    per_class = per_class.sort_values("support", ascending=False)

    # macro/weighted
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1_macro = report["macro avg"]["f1-score"]
    f1_weighted = report["weighted avg"]["f1-score"]

    top_err, label_order, cm = top_confusions(y_true, y_pred, k=args.topK)
    top_err["true_name"] = top_err["true"].map(lbl).fillna(top_err["true"])
    top_err["pred_name"] = top_err["pred"].map(lbl).fillna(top_err["pred"])

    theme_summary = summarize_by_theme(per_class, lbl, themap)

    # write side CSVs
    outdir = Path("results/reports"); outdir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(outdir / "per_class_metrics.csv", index=False)
    top_err.to_csv(outdir / "top_confusions.csv", index=False)
    if theme_summary is not None:
        theme_summary.to_csv(outdir / "themes_summary.csv", index=False)

    # Build Markdown
    md = []
    md += [f"# Rapport — {Path(args.preds).stem}"]
    md += [f"- **F1-macro**: {f1_macro:.4f}"]
    md += [f"- **F1-weighted**: {f1_weighted:.4f}", ""]
    md += ["## Top confusions"]
    md += ["| vrai | prédit | count |", "|:----|:-------|------:|"]
    for _,r in top_err.iterrows():
        md.append(f"| {r['true_name']} ({r['true']}) | {r['pred_name']} ({r['pred']}) | {r['count']} |")
    md += ["", "## Top 30 classes par support", "| id | nom | support | precision | recall | f1 |", "|---:|:-----|-------:|----------:|-------:|----:|"]
    for _,r in per_class.head(30).iterrows():
        md.append(f"| {r['class_id']} | {r['class_name']} | {int(r['support'])} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    if theme_summary is not None:
        md += ["", "## Synthèse par thématique", "| thématique | support | f1 | precision | recall |", "|:-----------|--------:|---:|----------:|-------:|"]
        for _,r in theme_summary.iterrows():
            md.append(f"| {r['theme']} | {int(r['support'])} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |")

    out_md = Path(args.out_md) if args.out_md else outdir / f"{Path(args.preds).stem}_report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    # Very light HTML (optional)
    out_html = Path(args.out_html) if args.out_html else outdir / f"{Path(args.preds).stem}_report.html"
    out_html.write_text("<html><body><pre>"+("\n".join(md))+"</pre></body></html>", encoding="utf-8")

    print(f"[OK] Wrote:\n- {out_md}\n- {out_html}\n- {outdir/'per_class_metrics.csv'}\n- {outdir/'top_confusions.csv'}" + (f"\n- {outdir/'themes_summary.csv'}" if theme_summary is not None else ""))

if __name__ == "__main__":
    main()
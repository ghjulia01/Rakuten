# tools/plot_confusion_from_csv.py
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_labels_mapping(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Labels mapping not found: {p}")
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # keys in JSON may be strings; normalize to str
        return {str(k): v for k, v in mapping.items()}
    elif p.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(p)
        # Expect columns: id, name
        if not {"id", "name"}.issubset(df.columns):
            raise ValueError("CSV mapping must have columns: id,name")
        return {str(r["id"]): r["name"] for _, r in df.iterrows()}
    else:
        raise ValueError("Unsupported labels mapping format (use .json or .csv)")

def infer_prob_columns(df):
    # Accept columns like proba_<class_id> or p_<class_id>
    cols = [c for c in df.columns if c.startswith("proba_") or c.startswith("p_")]
    return cols

def select_topN_classes_by_support(y_true, y_pred, N):
    # support = occurrences in y_true; ensure all predicted classes are present too
    vals, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)  # descending
    keep = set(vals[order[:N]].tolist())
    # also keep any class that appears in y_pred among top predictions in case it's outside topN
    keep |= set(np.unique(y_pred).tolist())
    return sorted(list(keep), key=lambda x: str(x))

def remap_labels(arr, keep_set):
    mask = np.array([a in keep_set for a in arr])
    return arr[mask], mask

def normalize_option(opt):
    if opt is None or opt.lower() == "none":
        return None
    opt = opt.lower()
    if opt in {"true", "pred", "all"}:
        return opt
    raise ValueError("--normalize must be one of: none, true, pred, all")

def main():
    ap = argparse.ArgumentParser(description="Plot confusion matrix from a CSV of predictions (no refit).")
    ap.add_argument("--csv", required=True, help="Path to CSV with at least columns: y_true,y_pred")
    ap.add_argument("--labels-map", default=None, help="Optional labels mapping file (.json with {id:name} or .csv with id,name)")
    ap.add_argument("--normalize", default="none", help="Normalization: none|true|pred|all (sklearn semantics)")
    ap.add_argument("--topN", type=int, default=None, help="If set, restrict plot to the N classes with highest support")
    ap.add_argument("--include-classes", default=None, help="Optional text/CSV file listing class ids to include (one per line)")
    ap.add_argument("--title", default=None, help="Figure title")
    ap.add_argument("--output", default=None, help="Output PNG path (default: results/figures/confusion_from_csv.png)")
    ap.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    ap.add_argument("--figsize", default=None, help='Matplotlib figsize, e.g. "14x12"')
    ap.add_argument("--topk", type=int, default=0, help="If >0, compute Top-k accuracy from probability columns (proba_*) if available")
    ap.add_argument("--save-matrix-csv", default=None, help="If set, also saves the confusion matrix as CSV at this path")
    args = ap.parse_args()

    out_png = args.output or "results/figures/confusion_from_csv.png"
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: y_true, y_pred")

    y_true = df["y_true"].astype(str).to_numpy()
    y_pred = df["y_pred"].astype(str).to_numpy()

    # Optionally compute Top-k from proba columns
    if args.topk and args.topk > 0:
        proba_cols = infer_prob_columns(df)
        if len(proba_cols) == 0:
            print("[WARN] --topk requested but no probability columns (proba_* or p_*) found. Skipping Top-k.")
        else:
            # class ids are suffixes after last underscore, maintain same ordering for both y and proba
            # Example columns: proba_1180, proba_1300, ...
            class_ids = [c.split("_", 1)[1] for c in proba_cols]
            probs = df[proba_cols].to_numpy()
            # argsort descending
            topk_idx = np.argsort(-probs, axis=1)[:, : args.topk]
            topk_classes = np.array(class_ids)[topk_idx]
            topk_hit = np.array([yt in set(topk_classes[i]) for i, yt in enumerate(y_true)])
            topk_acc = topk_hit.mean()
            print(f"[INFO] Top-{args.topk} accuracy (overall): {topk_acc:.4f}")

            # Per-class Top-k hit rate
            per_class = []
            for cls in np.unique(y_true):
                m = (y_true == cls)
                if m.any():
                    per_class.append({"class": cls, "support": int(m.sum()), f"top{args.topk}_hit_rate": float(topk_hit[m].mean())})
            topk_df = pd.DataFrame(per_class).sort_values("support", ascending=False)
            topk_csv = Path(out_png).with_name(Path(out_png).stem + f"_top{args.topk}_per_class.csv")
            topk_df.to_csv(topk_csv, index=False)
            print(f"[INFO] Saved per-class Top-{args.topk} hit rates → {topk_csv}")

    # Restrict classes if requested
    keep_classes = None
    if args.include_classes:
        incl = pd.read_csv(args.include_classes, header=None, squeeze=True).astype(str).tolist()
        keep_classes = set(incl)
    elif args.topN:
        keep_classes = set(select_topN_classes_by_support(y_true, y_pred, args.topN))

    if keep_classes is not None:
        y_true, mask_true = remap_labels(y_true, keep_classes)
        y_pred = y_pred[mask_true]
        # Also filter rows where prediction not in keep to avoid showing stray labels
        mask_pred = np.array([p in keep_classes for p in y_pred])
        y_true, y_pred = y_true[mask_pred], y_pred[mask_pred]

    labels = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))), key=lambda x: str(x))

    # Optional human-readable mapping
    mapping = load_labels_mapping(args.labels_map)
    if mapping:
        display_labels = [mapping.get(l, l) for l in labels]
    else:
        display_labels = labels

    norm = normalize_option(args.normalize)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)

    # Save matrix as CSV if requested
    if args.save_matrix_csv:
        mat_df = pd.DataFrame(cm, index=labels, columns=labels)
        Path(args.save_matrix_csv).parent.mkdir(parents=True, exist_ok=True)
        mat_df.to_csv(args.save_matrix_csv)
        print(f"[INFO] Saved confusion matrix CSV → {args.save_matrix_csv}")

    # Figure size
    if args.figsize:
        try:
            w, h = args.figsize.lower().split("x")
            figsize = (float(w), float(h))
        except Exception:
            raise ValueError('--figsize must be like "14x12"')
    else:
        # Heuristic for large class counts
        n = len(labels)
        base = max(6, min(24, int(n * 0.5)))  # scale with classes but cap
        figsize = (base, base)

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f" if norm else "d")
    ax.set_title(args.title or f"Confusion matrix ({'normalized: '+norm if norm else 'counts'})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Saved figure → {out_png}")

if __name__ == "__main__":
    main()

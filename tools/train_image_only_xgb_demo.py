# tools/train_image_only_xgb_demo.py
# python tools\train_image_only_xgb_demo.py --labels-csv data\demo_labels.csv
# Exemples :
#   python tools\train_image_only_xgb_demo.py --labels-csv data\demo_labels.csv
#   python tools\train_image_only_xgb_demo.py               # (fallback: label = nom du dossier parent)

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

NPZ = Path("data/demo_images_embeddings.npz")
INDEX = Path("data/demo_images_index.json")
OUT_JOBLIB = Path("artifacts/demo_image_only_xgb.joblib")

def label_from_path(p: str) -> str:
    # fallback: demo_images/<label>/img.jpg
    return Path(p).parent.name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", default=None,
                    help="CSV avec colonnes path,label (sinon label=nom du dossier parent).")
    args = ap.parse_args()

    # 1) Chargement embeddings + ordre des chemins
    X = np.load(NPZ)["X"].astype("float32", copy=False)  # (N, 2048)
    paths = pd.Series(json.loads(INDEX.read_text(encoding="utf-8"))["paths"])

    # 2) Construction des labels (robuste aux différences de chemins)
    def _norm_abs(p: str) -> str:
        # normalise en chemin absolu canonique + lower (Windows tolerant)
        try:
            return str(Path(p).resolve()).lower()
        except Exception:
            return str(Path(p)).lower()

    def _basename(p: str) -> str:
        return Path(p).name.lower()

    paths = paths.astype(str)
    p_abs = paths.map(_norm_abs)
    p_base = paths.map(_basename)

    if args.labels_csv:
        lab = pd.read_csv(args.labels_csv)
        if "path" not in lab.columns or "label" not in lab.columns:
            raise KeyError(f"{args.labels_csv} doit contenir les colonnes 'path' et 'label'.")

        lab["path"] = lab["path"].astype(str)
        lab["path_abs"] = lab["path"].map(_norm_abs)
        lab["fname"] = lab["path"].map(_basename)

        # 2.a) essai par chemin absolu
        map_abs = dict(zip(lab["path_abs"], lab["label"]))
        y_str = p_abs.map(map_abs)

        matched_abs = int(y_str.notna().sum())
        if matched_abs < len(paths):
            # 2.b) fallback par nom de fichier (basename)
            map_base = dict(zip(lab["fname"], lab["label"]))
            y2 = p_base.map(map_base)
            # priorité au match par chemin; remplit les trous avec le match par basename
            y_str = y_str.where(y_str.notna(), y2)
            matched_base = int(y2.notna().sum())
        else:
            matched_base = matched_abs  # tout matché

        # masque final
        mask = y_str.notna().values
        dropped = int((~mask).sum())
        if dropped:
            print(f"[warn] {dropped} image(s) sans label après normalisation — exclusion.")
            # petit diagnostic utile
            missing_examples = paths[~mask].head().tolist()
            print("[dbg] Exemples sans label (index.json):", missing_examples[:3])

        X = X[mask]
        paths = paths[mask].reset_index(drop=True)
        y_str = y_str[mask].astype(str).values
        print(f"[info] Labels appariés: {int(mask.sum())}/{len(mask)} "
              f"(match_abs={matched_abs}, match_basename≈{matched_base})")
    else:
        # fallback: label = nom du dossier parent
        y_str = p_base.map(lambda fn: label_from_path(fn)).astype(str).values

    # 3) Encodage labels & sanity checks
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    n_classes = len(le.classes_)
    if n_classes < 2:
        raise ValueError(
            "Il faut au moins 2 classes pour entraîner un classif. "
            f"Trouvé {n_classes}. Vérifie tes labels (--labels-csv) ou l'arborescence demo_images/<classe>/..."
        )
    if len(X) != len(y):
        raise RuntimeError(f"Incohérence tailles: X={len(X)} vs y={len(y)}")

    # Diagnostic répartition
    classes_idx, counts = np.unique(y, return_counts=True)
    print("Répartition des classes (encodées):", dict(zip(classes_idx.tolist(), counts.tolist())))
    print("n_classes =", n_classes, "| n_samples =", len(y))

    # 4) Split
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5) Modèle XGBoost compact — num_class requis avec multi:softprob (xgboost 3.0.5)
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=int(n_classes),
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        tree_method="hist",
        n_jobs=1,
        eval_metric="mlogloss",
    )

    # 6) Entraînement (sans early stopping pour compat max)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])

    # 7) Évaluation
    yhat = clf.predict(Xva)
    print("F1-macro (val):", f1_score(yva, yhat, average="macro"))

    # 8) Export
    payload = {
        "model": clf,
        "label_encoder_classes_": le.classes_.tolist(),
        "emb_dim": X.shape[1],
        "backbone": "resnet50-imagenet",
    }
    OUT_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUT_JOBLIB, compress=3)
    print("Saved ->", OUT_JOBLIB)

if __name__ == "__main__":
    main()
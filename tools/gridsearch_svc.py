# tools/gridsearch_svc.py
import argparse, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

# === Rendre importable main/train_model.py ===
ROOT = Path(__file__).resolve().parents[1]          # .../rakuten-main
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "main"))

try:
    # on importe ce qui existe réellement dans main/train_model.py
    from train_model import (
        load_config, init_seeds, build_baseline_pipeline
    )
except ModuleNotFoundError:
    from main.train_model import (
        load_config, init_seeds, build_baseline_pipeline
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="features/config.toml")
    p.add_argument("--baseline", choices=["b2","b4"], default="b2",
                   help="b2=texte seul, b4=multimodal (texte+image)")
    p.add_argument("--scoring", default="f1_weighted",
                   choices=["f1_weighted","f1_macro"])
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=60000)
    p.add_argument("--out", default="artifacts/best_svc.joblib")
    args = p.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config(args.config)
    init_seeds(42)

    # --- Charger les données comme dans train_model.main() ---
    x_path = cfg["paths"]["x_train_csv"]
    y_path = cfg["paths"]["y_train_csv"]
    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0).squeeze()
    # colonnes nécessaires (train_model les attend)
    needed = ["designation","description","productid","imageid"]
    for c in needed:
        if c not in X.columns:
            raise ValueError(f"Colonne manquante dans X: '{c}'")

    # downsample (optionnel)
    if len(X) > args.max_samples > 0:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X), size=args.max_samples, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx]

    # --- Construire la pipeline baseline ---
    # Remplace le modèle par SVC côté config pour être sûr
    cfg_local = {**cfg, "model": {**cfg.get("model", {}), "name": "svc"}}
    seed = int(cfg.get("random", {}).get("seed", 42))

    # B2 n’a pas besoin de y_train, B4 oui (pour les stratégies d’échantillonnage)
    pipe, need_cols = build_baseline_pipeline(
        args.baseline, cfg_local, seed, y_train=y if args.baseline=="b4" else None
    )

    # Étape finale : 'clf' (B2) ou 'model' (B4)
    if "clf" in pipe.named_steps:
        est = "clf"
    elif "model" in pipe.named_steps:
        est = "model"
    else:
        raise RuntimeError("Étape finale inconnue dans la pipeline (ni 'clf' ni 'model').")

    # --- Grille LinearSVC ---
    param_grid = {
        f"{est}__loss": ["hinge", "squared_hinge"],
        f"{est}__dual": [True, False],
        f"{est}__C":    [0.5, 1.0, 2.0],
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=args.scoring,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    gs.fit(X[need_cols], y)

    print("\n=== BEST ===")
    print(gs.best_params_)
    print("best_score:", gs.best_score_)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, out)
    print(f"[OK] Modèle sauvegardé → {out}")

if __name__ == "__main__":
    main()
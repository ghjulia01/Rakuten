# tools/train_b3_and_save.py
import argparse
from pathlib import Path
import sys

import pandas as pd
import joblib

# Lancer avec :
# python tools/train_b3_and_save.py --config features/config.toml --out artifacts/b3.joblib

# --- Rendre le repo et le dossier 'main/' importables ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))           # importer à la racine
sys.path.append(str(ROOT / "main"))  # importer 'train_model' si dans main/

# d'abord à la racine, sinon via le package 'main'
try:
    from train_model import load_config, init_seeds, build_baseline_pipeline
except ModuleNotFoundError:
    from main.train_model import load_config, init_seeds, build_baseline_pipeline


def read_y(path: Path) -> pd.Series:
    """Lecture robuste de y_train (colonne prdtypecode, label ou 1ère colonne)."""
    dfy = pd.read_csv(path)
    if dfy.shape[1] == 1:
        return dfy.iloc[:, 0].squeeze()
    for c in ("prdtypecode", "label", "y"):
        if c in dfy.columns:
            return dfy[c].squeeze()
    return dfy.iloc[:, 0].squeeze()


def main():
    p = argparse.ArgumentParser(description="Entraîne B3 (images) et sauvegarde le pipeline en .joblib")
    p.add_argument("--config", default="features/config.toml")
    p.add_argument("--out", default="artifacts/b3.joblib")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("random", {}).get("seed", 42))
    init_seeds(seed)

    x_path = Path(cfg["paths"]["x_train_csv"]).resolve()
    y_path = Path(cfg["paths"]["y_train_csv"]).resolve()

    if not x_path.exists():
        raise FileNotFoundError(f"X_train introuvable : {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"y_train introuvable : {y_path}")

    X_train = pd.read_csv(x_path)
    y_train = read_y(y_path)

    # Construit la pipeline baseline B3 (branche images)
    pipe, need_cols = build_baseline_pipeline("b3", cfg, seed)

    # Certaines pipelines attendent des colonnes spécifiques (ex. image_path/imageid…)
    X_fit = X_train[need_cols] if need_cols else X_train

    # Entraînement
    pipe.fit(X_fit, y_train)

    # Sauvegarde
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"[OK] Modèle B3 sauvegardé → {out}")

    # Info rapide
    final_est = getattr(pipe, "named_steps", {}).get("model") or getattr(pipe, "named_steps", {}).get("clf")
    if final_est is not None:
        print(f"Estimator final: {final_est.__class__.__name__}")

if __name__ == "__main__":
    main()
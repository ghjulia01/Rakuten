# tools/train_b4_and_save.py
# python tools/train_b4_and_save.py --config features/config.toml --out artifacts/b4.joblib 
# Entraîne B4 (multimodal) et sauvegarde le pipeline en .joblib
#
import argparse
from pathlib import Path
import sys
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "main"))

try:
    from train_model import load_config, init_seeds, build_baseline_pipeline
except ModuleNotFoundError:
    from main.train_model import load_config, init_seeds, build_baseline_pipeline

def main():
    p = argparse.ArgumentParser(description="Entraîne B4 (multimodal) et sauvegarde le pipeline en .joblib")
    p.add_argument("--config", default="features/config.toml")
    p.add_argument("--out", default="artifacts/b4.joblib")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("random", {}).get("seed", 42))
    init_seeds(seed)

    X = pd.read_csv(cfg["paths"]["x_train_csv"])
    y_df = pd.read_csv(cfg["paths"]["y_train_csv"])
    for col in ["prdtypecode", "label", "y"]:
        if col in y_df.columns:
            y = y_df[col]
            break
    else:
        y = y_df.squeeze()
        if getattr(y, "ndim", 2) != 1:
            raise ValueError(
                "Le fichier y_train_csv contient plusieurs colonnes. "
                "Spécifie la colonne label (ex: prdtypecode)."
            )

    pipe, need_cols = build_baseline_pipeline("b4", cfg, seed, y_train=y)
    pipe.fit(X[need_cols], y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"[OK] Modèle B4 sauvegardé → {out}")

if __name__ == "__main__":
    main()
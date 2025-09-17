# tools/optuna_tune.py
# $env:RAKUTEN_MAX_N=8000; python tools/optuna_tune.py --config features/config.toml --baseline b4 --trials 30 --seed 42 --storage sqlite:///optuna.db

import os
import argparse
import optuna
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# imports résilients
try:
    from main.train_model import load_config, build_baseline_pipeline, init_seeds
except Exception:
    from train_model import load_config, build_baseline_pipeline, init_seeds

def _read_y(p: Path) -> pd.Series:
    df = pd.read_csv(p)
    for c in ("prdtypecode","label","y"):
        if c in df.columns: return df[c].squeeze()
    return df.iloc[:,0].squeeze()

def objective(trial, cfg, baseline, seed):
    # Espace de recherche (exemple simple)
    model = trial.suggest_categorical("model", ["lr","svc","xgb","lgbm"])
    cfg["model"]["name"] = model

    if model == "lr":
        cfg["model"]["penalty"] = trial.suggest_categorical("penalty", ["l2"])
        cfg["model"]["C"]       = trial.suggest_float("C", 1e-2, 5.0, log=True)
        cfg["text"]["svd"]["l2norm"] = True
    elif model == "svc":
        cfg["model"]["C"]       = trial.suggest_float("C", 1e-2, 10.0, log=True)
        cfg["text"]["svd"]["l2norm"] = True
    elif model == "xgb":
        cfg["model"]["learning_rate"] = trial.suggest_float("lr", 1e-3, 0.3, log=True)
        cfg["model"]["max_depth"]     = trial.suggest_int("max_depth", 4, 10)
        cfg["model"]["n_estimators"]  = trial.suggest_int("n_estimators", 200, 700)
        cfg["model"]["subsample"]     = trial.suggest_float("subsample", 0.6, 1.0)
        cfg["model"]["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        cfg["text"]["svd"]["l2norm"] = False
    else: # lgbm
        cfg["model"]["learning_rate"] = trial.suggest_float("lr", 1e-3, 0.3, log=True)
        cfg["model"]["num_leaves"]    = trial.suggest_int("num_leaves", 31, 255)
        cfg["model"]["n_estimators"]  = trial.suggest_int("n_estimators", 200, 800)
        cfg["model"]["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 60)
        cfg["text"]["svd"]["l2norm"] = False

    # charge données
    X = pd.read_csv(cfg["paths"]["x_train_csv"])
    y = _read_y(Path(cfg["paths"]["y_train_csv"]))
    max_n = int(os.getenv("RAKUTEN_MAX_N", "8000"))
    if len(X) > max_n:
        X = X.iloc[:max_n].copy()
        y = y.iloc[:max_n].copy()

    # construit pipeline baseline
    pipe, need_cols = build_baseline_pipeline(baseline, cfg, seed=seed, y_train=y)
    Xs = X[need_cols] if need_cols else X

    # CV rapide
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(Xs, y):
        Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xva)
        scores.append(f1_score(yva, yhat, average="macro"))

    return float(np.mean(scores))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="features/config.toml")
    ap.add_argument("--baseline", choices=["b2","b3","b4"], default="b4")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--storage", default=None, help="ex: sqlite:///optuna.db")
    args = ap.parse_args()

    cfg = load_config(args.config)
    init_seeds(args.seed)
    if "model" not in cfg: cfg["model"] = {}
    if "text" not in cfg: cfg["text"] = {}
    if "svd" not in cfg["text"]: cfg["text"]["svd"] = {}

    study = optuna.create_study(direction="maximize", storage=args.storage, study_name=f"{args.baseline}-study", load_if_exists=True)
    study.optimize(lambda t: objective(t, cfg, args.baseline, args.seed), n_trials=args.trials)

    print("\nBest value:", study.best_value)
    print("Best params:", study.best_params)

if __name__ == "__main__":
    main()
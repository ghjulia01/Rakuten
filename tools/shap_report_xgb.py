# tools/shap_report_xgb.py
# Usage:
#   python -m tools.shap_report_xgb --model artifacts/b4.joblib --data-csv notebooks/df.csv --target prdtypecode --max-n 3000 --outdir results/shap_b4 --topn 30
#     

import argparse, os, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import shap
import matplotlib.pyplot as plt
from scipy import sparse

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Utils ----------


def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# --- shim pour l’unpickling des modèles qui référencent ToFloat32 ---
try:
    from main.train_model import ToFloat32 as _ToFloat32
except Exception:
    # fallback neutre si import impossible
    class _ToFloat32:
        def fit(self, X, y=None): return self
        def transform(self, X): return X
ToFloat32 = _ToFloat32

# --- shim pour LabelEncodingClassifier (visible au niveau module) ---
try:
    # s'il existe côté projet
    from models.text_pipeline import LabelEncodingClassifier as LabelEncodingClassifier  # noqa: F401
except Exception:
    try:
        # autre emplacement possible (train)
        from main.train_model import LabelEncodingClassifier as LabelEncodingClassifier  # noqa: F401
    except Exception:
        # fallback neutre pour satisfaire l'unpickling
        class LabelEncodingClassifier:
            def __init__(self, *args, **kwargs): pass
            def fit(self, X, y=None, **k): return self
            def predict(self, X): 
                try:
                    import numpy as np
                    return np.zeros(len(X))
                except Exception:
                    return []
            def predict_proba(self, X): return None

def _walk_steps(est):
    """Iterateur DFS sur steps transformeurs d'un objet sklearn (Pipeline, FeatureUnion, ColumnTransformer)."""
    if hasattr(est, "steps"):  # Pipeline
        for name, sub in est.steps:
            yield name, sub
            yield from _walk_steps(sub)
    if hasattr(est, "transformer_list"):  # FeatureUnion
        for name, sub in est.transformer_list:
            yield name, sub
            yield from _walk_steps(sub)
    if hasattr(est, "transformers_"):  # ColumnTransformer (fitted)
        for name, sub, cols in est.transformers_:
            yield name, sub
            yield from _walk_steps(sub)

def _get_feature_names_from_block(block):
    """Tente d'extraire les noms de features d'un transformeur de texte/tabulaire."""
    names = None
    # cas TfidfVectorizer
    try:
        if hasattr(block, "get_feature_names_out"):
            names = block.get_feature_names_out()
        elif hasattr(block, "get_feature_names"):
            names = block.get_feature_names()
    except Exception:
        pass
    # cas pipeline simple: chercher un TfidfVectorizer dedans
    if names is None and hasattr(block, "steps"):
        for _, sub in block.steps:
            try:
                if hasattr(sub, "get_feature_names_out"):
                    names = sub.get_feature_names_out()
                elif hasattr(sub, "get_feature_names"):
                    names = sub.get_feature_names()
                if names is not None:
                    break
            except Exception:
                continue
    # fallback: nombre de colonnes seulement (ex: stats, CNN embeddings)
    if names is None and hasattr(block, "n_features_"):
        names = np.array([f"f{idx}" for idx in range(block.n_features_)])
    return names

def _get_classifier(pipe):
    """
    Retourne l’estimateur de base (XGBClassifier/LGBMClassifier, pas le Pipeline).
    Gère le wrapper LabelEncodingClassifier (attribut est_).
    """
    est = pipe
    if hasattr(pipe, "named_steps"):
        for key in ("model", "clf"):
            if key in pipe.named_steps:
                est = pipe.named_steps[key]
                break
    # unwrap LabelEncodingClassifier / wrappers similaires
    if hasattr(est, "est_"):       # cas LabelEncodingClassifier(est_=...)
        est = est.est_
    if hasattr(est, "clf"):        # par prudence si un autre wrapper a .clf
        est = est.clf
    return est

def _predict_input(pipe, Xdf, target_col=None, max_n=None):
    """Applique la partie 'pre' du pipeline pour obtenir la matrice finale de features + mapping indices->group/feature."""
    # Détermine la step 'pre' (ou 'features' / défaut au pipe si pas de nommage)
    pre = None
    if hasattr(pipe, "named_steps"):
        for k in ("pre", "features"):
            if k in pipe.named_steps:
                pre = pipe.named_steps[k]
                break
    if pre is None:
        # parfois le pipe est juste ('pre', 'clf')
        pre = pipe.named_steps.get(list(pipe.named_steps.keys())[0], pipe)

    X = Xdf.copy()
    if target_col and target_col in X.columns:
        X = X.drop(columns=[target_col])

    if max_n:
        X = X.iloc[:max_n].copy()

    # Fit déjà fait → transform direct
    Xmat = pre.transform(X)

    # Construire la cartographie indices -> (group_name, feature_name)
    groups = []
    feat_names = []
    col_offset = 0
    # On re-parcourt 'pre' et on reconstruit la largeur de chaque sous-branche
    if hasattr(pre, "transformer_list"):  # FeatureUnion/ColumnTransformer
        for name, tr in pre.transformer_list:
            # 1) privilégier un attribut de largeur déjà appris
            ncols = getattr(tr, "n_features_", None)
            if ncols is None and hasattr(tr, "steps"):
                # essayer la dernière step (souvent SVD/Normalizer) pour récupérer n_features_
                for _nm, _st in reversed(tr.steps):
                    ncols = getattr(_st, "n_features_", None)
                    if ncols is not None:
                        break
            # 2) dernier recours: valeur sentinelle + warning (mais NE PAS transform)
            if ncols is None:
                ncols = 100
                print(f"[WARN] largeur inconnue pour bloc '{name}' → fallback {ncols} (pas de transform).")
            # 3) noms
            names = _get_feature_names_from_block(tr)
            if names is None or len(names) != ncols:
                names = np.array([f"{name}__{i}" for i in range(ncols)])
            groups.extend([name] * ncols)
            feat_names.extend(list(names))
    else:
        # 'pre' n'est pas une union: tout dans un seul groupe
        ncols = Xmat.shape[1]
        groups = ["features"]*ncols
        feat_names = [f"feat_{i}" for i in range(ncols)]

    groups = np.array(groups)
    feat_names = np.array(feat_names)

    return Xmat, groups, feat_names

def _mean_abs_shap(shap_values):
    """Agrège en moyenne absolue; gère multi-classes (liste d’array)."""
    if isinstance(shap_values, list):
        # multi-classes → moyenne sur classes de la magnitude
        mags = [np.abs(sv).mean(axis=0) for sv in shap_values]  # par classe
        return np.mean(np.vstack(mags), axis=0)
    else:
        return np.abs(shap_values).mean(axis=0)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--max-n", type=int, default=3000)
    ap.add_argument("--outdir", default="results/shap_report")
    ap.add_argument("--topn", type=int, default=20, help="Top-N features sur le barplot global")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) Charger le pipeline (les shims top-level suffisent pour l’unpickling)
    pipe = joblib.load(args.model)

    # 2) Charger les données et construire X transformé + mapping features→groupes
    df = pd.read_csv(args.data_csv)
    Xmat, groups, feat_names = _predict_input(pipe, df, target_col=args.target, max_n=args.max_n)

    # 3) Récupérer le classifieur de base (XGB/LGBM…) et construire l’explainer
    clf = _get_classifier(pipe)
    if clf is None:
        raise RuntimeError("Classifieur introuvable dans le pipeline.")
    explainer = shap.TreeExplainer(clf)

    # 4) SHAP values
    shap_values = explainer.shap_values(Xmat)

    # 5) Importances par feature (mean |SHAP|)
    imp_feat = _mean_abs_shap(shap_values)
    df_feat = pd.DataFrame({
        "feature": feat_names,
        "group": groups,
        "mean_abs_shap": imp_feat
    }).sort_values("mean_abs_shap", ascending=False)
    df_feat.to_csv(Path(args.outdir, "shap_per_feature.csv"), index=False, encoding="utf-8")

    # 6) Agrégation par groupe
    df_group = df_feat.groupby("group", as_index=False)["mean_abs_shap"].sum() \
                      .sort_values("mean_abs_shap", ascending=False)
    df_group.to_csv(Path(args.outdir, "shap_per_group.csv"), index=False, encoding="utf-8")

    # 7) Barplots
    topn = min(args.topn, len(df_feat))
    fig = plt.figure(figsize=(7, 5.5))
    plt.barh(df_feat["feature"].iloc[:topn][::-1], df_feat["mean_abs_shap"].iloc[:topn][::-1])
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
    plt.tight_layout()
    plt.savefig(Path(args.outdir, "shap_top_features.png"), dpi=150)
    plt.close(fig)

    fig2 = plt.figure(figsize=(7, 5.5))
    plt.barh(df_group["group"].iloc[::-1], df_group["mean_abs_shap"].iloc[::-1])
    plt.xlabel("Sum of mean(|SHAP|) per group")
    plt.tight_layout()
    plt.savefig(Path(args.outdir, "shap_groups.png"), dpi=150)
    plt.close(fig2)

    # 8) Récap
    Path(args.outdir, "summary.json").write_text(json.dumps({
        "n_samples": int(Xmat.shape[0]),
        "n_features": int(Xmat.shape[1]),
        "topn": int(topn),
        "outdir": str(args.outdir)
    }, indent=2), encoding="utf-8")

    print("[DONE] SHAP XGBoost terminé.",
          "\n- shap_per_feature.csv",
          "\n- shap_per_group.csv",
          "\n- shap_top_features.png",
          "\n- shap_groups.png", sep="")

    # récap JSON
    recap = {
        "n_samples": int(Xmat.shape[0]),
        "n_features": int(Xmat.shape[1]),
        "topn": int(topn),
        "outdir": str(args.outdir)
    }
    Path(args.outdir, "summary.json").write_text(json.dumps(recap, indent=2), encoding="utf-8")
    print("[DONE] SHAP XGBoost terminé.",
          "\n- shap_per_feature.csv",
          "\n- shap_per_group.csv",
          "\n- shap_top_features.png",
          "\n- shap_groups.png", sep="")
if __name__ == "__main__":
    main()
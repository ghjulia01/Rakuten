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

def _get_pre_to_model(pipe):
    """
    Retourne une Pipeline qui applique TOUTES les steps jusqu'à 'model' (exclue),
    pour garantir que la matrice produite correspond exactement à celle vue par le classifieur.
    """
    from sklearn.pipeline import Pipeline
    if not hasattr(pipe, "steps"):
        return pipe  # pas une pipeline: improbable ici
    steps = pipe.steps
    # chercher l'index de 'model' (ou 'clf')
    idx = None
    for i, (nm, _) in enumerate(steps):
        if nm == "model":
            idx = i
            break
    if idx is None:
        for i, (nm, _) in enumerate(steps):
            if nm == "clf":
                idx = i
                break
    if idx is None or idx == 0:
        # pas trouvé / rien avant → retourner pipe tel quel (dégradé)
        return pipe
    return Pipeline(steps[:idx])

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
    """
    Transforme X via la sous-pipeline *jusqu'à* 'model' (exclue), puis construit
    le mapping (groups, feat_names) en parcourant CETTE sous-pipeline.
    """
    from sklearn.pipeline import Pipeline

    # Sous-pipeline exacte avant le modèle
    pre_to_model = _get_pre_to_model(pipe)

    # X source
    X = Xdf.copy()
    if target_col and target_col in X.columns:
        X = X.drop(columns=[target_col])
    if max_n:
        X = X.iloc[:max_n].copy()

    # Matrice finale exactement vue par le modèle
    Xmat = pre_to_model.transform(X)

    # Construire la cartographie indices -> (group_name, feature_name)
    groups = []
    feat_names = []

    # On parcourt la sous-pipeline pour reconstituer les blocs
    # Cherche l’objet union/coltrans de plus haut niveau s’il y en a un
    union = None
    if hasattr(pre_to_model, "named_steps"):
        # heuristique: dernière step de pre_to_model est souvent l’union/coltrans
        for _nm, _st in reversed(pre_to_model.steps):
            if hasattr(_st, "transformer_list") or hasattr(_st, "transformers_"):
                union = _st
                break
    # Si pas d’union détectée, on utilisera la largeur totale de Xmat comme un seul bloc
    if union is None:
        ncols = Xmat.shape[1]
        groups = ["features"] * ncols
        feat_names = [f"feat_{i}" for i in range(ncols)]
        groups = np.asarray(groups, dtype=str)
        feat_names = np.asarray(feat_names, dtype=str)
        return Xmat, groups, feat_names

    # Sondage fiable bloc-par-bloc sur 1 ligne (rapide)
    X_probe = X.iloc[:1].copy()
    total = 0
    for name, tr in getattr(union, "transformer_list", []):
        try:
            Xpart = tr.transform(X_probe)
            ncols = Xpart.shape[1] if hasattr(Xpart, "shape") else len(Xpart[0])
        except Exception:
            ncols = getattr(tr, "n_features_", None)
            if ncols is None:
                ncols = 0
                print(f"[WARN] largeur inconnue pour bloc '{name}' → 0 (fallback).")

        names = _get_feature_names_from_block(tr)
        if names is None or (hasattr(names, "__len__") and len(names) != ncols):
            names = np.array([f"{name}__{i}" for i in range(int(ncols))])

        groups.extend([name] * int(ncols))
        feat_names.extend(list(names))
        total += int(ncols)

    # Ajustement strict pour coller à Xmat
    if total != Xmat.shape[1]:
        print(f"[WARN] Somme colonnes blocs={total} ≠ Xmat={Xmat.shape[1]} → ajustement dernier bloc.")
        delta = Xmat.shape[1] - total
        if delta > 0:
            last = union.transformer_list[-1][0]
            groups.extend([last] * delta)
            feat_names.extend([f"{last}__extra_{i}" for i in range(delta)])
        elif delta < 0:
            keep = Xmat.shape[1]
            groups = groups[:keep]
            feat_names = feat_names[:keep]

    groups = np.asarray(groups, dtype=str)
    feat_names = np.asarray(feat_names, dtype=str)

    # Sanity check final
    assert len(groups) == Xmat.shape[1] == len(feat_names), \
        f"Mapping features non aligné: groups={len(groups)}, feat_names={len(feat_names)}, Xmat={Xmat.shape[1]}"

    return Xmat, groups, feat_names

def _mean_abs_shap(shap_values, n_features_expected=None, clf=None):
    """
    Retourne un vecteur (n_features,).
    Gère les 3 cas:
    - liste par classe -> moyenne des |SHAP| sur classes puis sur samples,
    - array 2D (N, C*F) -> on recompose via n_features_expected,
    - array 2D (N, F) -> direct.
    """
    import numpy as np

    # Cas standard: liste par classe
    if isinstance(shap_values, list):
        mags = [np.abs(sv).mean(axis=0) for sv in shap_values]  # (F,) par classe
        return np.mean(np.vstack(mags), axis=0)                 # (F,)

    sv = np.asarray(shap_values)
    if sv.ndim != 2:
        return np.abs(sv).mean(axis=tuple(range(sv.ndim - 1)))

    n_samples, m = sv.shape

    # Essai 1: si on connaît F attendu (plus fiable)
    if n_features_expected and m % int(n_features_expected) == 0:
        n_classes = m // int(n_features_expected)
        if n_classes >= 2:  # multi-classe plausible
            sv_reshaped = sv.reshape(n_samples, n_classes, int(n_features_expected))  # (N, C, F)
            return np.abs(sv_reshaped).mean(axis=0).mean(axis=0)  # (F,)

    # Essai 2: on tente via clf (selon versions)
    n_classes = None
    try:
        if hasattr(clf, "classes_"): n_classes = len(clf.classes_)
        elif hasattr(clf, "n_classes_"): n_classes = int(clf.n_classes_)
    except Exception:
        pass
    if n_classes and m % n_classes == 0:
        F = m // n_classes
        sv_reshaped = sv.reshape(n_samples, n_classes, F)
        return np.abs(sv_reshaped).mean(axis=0).mean(axis=0)

    # Sinon: on considère m == F
    return np.abs(sv).mean(axis=0)

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

    print(f"[INFO] Matrix for model: Xmat shape = {Xmat.shape}")

    # 3) Récupérer le classifieur et construire l’explainer "unifié"
    clf = _get_classifier(pipe)
    if clf is None:
        raise RuntimeError("Classifieur introuvable dans le pipeline.")

    # Utilise l'API Explainer (plus robuste que TreeExplainer pour les cas multi-classes)
    exp = shap.Explainer(clf, Xmat, algorithm="tree")
    sh = exp(Xmat)  # objet Explanation

    # Extraction des valeurs sous forme numpy
    sv = sh.values  # peut être (N, F) ou (N, C, F)


    # 5) Importances par feature (mean |SHAP|) avec gestion (N,F) et (N,C,F)
    if sv.ndim == 3:
        # (N, C, F) -> moyenne des magnitudes sur N, puis sur C
        imp_feat = np.abs(sv).mean(axis=0).mean(axis=0)  # (F,)
    elif sv.ndim == 2 and sv.shape[1] == Xmat.shape[1]:
        # (N, F)
        imp_feat = np.abs(sv).mean(axis=0)               # (F,)
    else:
        # dernier recours: on tente de reconstituer (N, C, F) via F attendu
        N, M = sv.shape[0], int(np.prod(sv.shape[1:])) if sv.ndim > 1 else 0
        F = Xmat.shape[1]
        if sv.ndim == 2 and M % F == 0 and M // F >= 2:
            C = M // F
            imp_feat = np.abs(sv.reshape(N, C, F)).mean(axis=0).mean(axis=0)
        else:
            raise RuntimeError(f"Forme SHAP inattendue: sv.shape={sv.shape}, Xmat.shape={Xmat.shape}")

    # Normalisation défensive
    imp_feat   = np.asarray(imp_feat).reshape(-1)
    feat_names = np.asarray(feat_names, dtype=str).reshape(-1)
    groups     = np.asarray(groups, dtype=str).reshape(-1)
    assert len(imp_feat) == len(feat_names) == len(groups) == Xmat.shape[1], \
        f"Tailles incohérentes: imp={len(imp_feat)}, names={len(feat_names)}, groups={len(groups)}, X={Xmat.shape[1]}"

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
    
if __name__ == "__main__":
    main()
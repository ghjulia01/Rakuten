# -*- coding: utf-8 -*-
"""
Diagnostics ACP & SHAP + Importances par blocs (B2/B4)
- ACP 2D via SVD preview (rapide) sinon PCA sur OOF
- Importances par blocs globales et par classe
- Option "fidèle" pour B4 : rétro-projection du TEXTE (pré-SVD) afin
  d'obtenir le détail TF-IDF/Stats comme en B2, y compris en "par classe, magnitude signée".

Usage typiques
--------------
B4 — ACP + confusions + importances par classe (magnitude signée, tri total)
python -m tools.diagnostics_acp_shap \
  --kind b4 \
  --model artifacts/b4.joblib \
  --data-csv notebooks/df.csv \
  --blocks-b4-per-class-signed-mag \
  --sort-by total

B4 — *fidèle* (texte rétro-projeté avant SVD) :
python -m tools.diagnostics_acp_shap \
  --kind b4 \
  --model artifacts/b4.joblib \
  --data-csv notebooks/df.csv \
  --blocks-b4-per-class-signed-mag-backproj-text \
  --sort-by total
python -m tools.diagnostics_acp_shap --kind b4 --model artifacts/b4.joblib --data-csv notebooks/df.csv --blocks-b4-per-class-signed-mag-backproj-text --sort-by total               
tools/diagnostics_acp_shap.py
Harmonisé avec le schéma d'export train_model.py :
   preds_{kind}_{phase}.csv  (ex: preds_b4_test.csv)
   model_{kind}_{phase}.joblib (ex: model_b4_final.joblib)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Résolution du chemin racine du repo (si besoin d'importer des modules locaux)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Utils
# --- shim pour l’unpickling des modèles qui référencent ToFloat32 ---
try:
    from main.train_model import ToFloat32 as _ToFloat32
except Exception:
    # fallback neutre si import impossible
    class _ToFloat32:
        def fit(self, X, y=None): return self
        def transform(self, X): return X
ToFloat32 = _ToFloat32

# --- shim pour LabelEncodingClassifier (présent côté entraînement) ---
try:
    # essaie d'abord là où il vit probablement
    from models.text_pipeline import LabelEncodingClassifier as _LEC
except Exception:
    try:
        # autre emplacement possible
        from main.train_model import LabelEncodingClassifier as _LEC
    except Exception:
        # fallback neutre: laisse l'unpickling remettre les attributs (clf, classes_, etc.)
        class _LEC:
            def __init__(self, *args, **kwargs): pass
            def fit(self, X, y=None): return self
            def predict(self, X):
                # si l'objet unpicklé a un attribut 'clf', on délègue
                if hasattr(self, "clf") and hasattr(self.clf, "predict"):
                    return self.clf.predict(X)
                # sinon, fallback silencieux
                try:
                    import numpy as np
                    return np.zeros(len(X))
                except Exception:
                    return []
            def predict_proba(self, X):
                if hasattr(self, "clf") and hasattr(self.clf, "predict_proba"):
                    return self.clf.predict_proba(X)
                return None
            def decision_function(self, X):
                if hasattr(self, "clf") and hasattr(self.clf, "decision_function"):
                    return self.clf.decision_function(X)
                return None
LabelEncodingClassifier = _LEC
# ----------------------------------------
def _format_out_path(tmpl: str, *, kind: str, phase: str) -> Path:
    """
    Formate un chemin en appliquant tmpl.format(kind=..., phase=...).
    Si tmpl ne contient pas ces placeholders, on suffixe proprement.
    """
    try:
        return Path(tmpl.format(kind=kind, phase=phase))
    except Exception:
        base, ext = os.path.splitext(tmpl)
        return Path(f"{base}_{kind}_{phase}{ext}")

def ensure_dirs():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/reports").mkdir(parents=True, exist_ok=True)

def _load_preds_csv(preds_csv: Path) -> Optional[pd.DataFrame]:
    if not preds_csv.exists():
        print(f"[WARN] Fichier de prédictions introuvable: {preds_csv}")
        return None
    try:
        df = pd.read_csv(preds_csv)
        # Certaines versions exportent avec index en 1ère colonne
        if "Unnamed: 0" in df.columns and "y_pred" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    except Exception as e:
        print(f"[WARN] Impossible de lire {preds_csv}: {e}")
        return None
# ------------------------------------------------------------------
def blocks_b4_per_class_signed_magnitude_backproj_text(args, pipe, df):
    """
    B4 fidèle : rétro-projection du texte (pré-SVD) pour obtenir des importances
    par terme TF-IDF, avec magnitude signée par classe.

    Hypothèse robuste :
    - Le pipeline final est de type Pipeline([... ('pre', pre_feat), ('clf', clf) ...])
      où 'pre' (ou équivalent) contient une FeatureUnion/ColumnTransformer
      avec une branche texte de la forme Pipeline([('tfidf', TfidfVectorizer), ('svd', TruncatedSVD)])
      et un classifieur linéaire en sortie (LogReg/LinearSVC/SAG/SAGA, etc.) produisant coef_.

    Sorties :
    - results/reports/b4_backproj_text_total.csv  (somme des |poids| par terme, toutes classes)
    - results/reports/b4_backproj_text_per_class__<label>.csv (top termes par classe)
    """
    ensure_dirs()
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    # --------- Helpers pour retrouver les briques dans le pipeline ---------
    def _find_step_by_type(obj, cls):
        """Retourne la première instance de type 'cls' trouvée en parcourant pipelines/union/coltrans."""
        visited = set()
        def _walk(x):
            xid = id(x)
            if xid in visited:
                return None
            visited.add(xid)
            # correspondance directe
            if isinstance(x, cls):
                return x
            # Pipeline-like
            if hasattr(x, "steps"):  # sklearn.pipeline.Pipeline
                for _, t in x.steps:
                    r = _walk(t)
                    if r is not None:
                        return r
            # FeatureUnion-like
            if hasattr(x, "transformer_list"):  # FeatureUnion  ou ColumnTransformer
                for _, t in x.transformer_list:
                    r = _walk(t)
                    if r is not None:
                        return r
            # ColumnTransformer 'transformers_' fitted
            if hasattr(x, "transformers_"):
                for _, t, _cols in x.transformers_:
                    r = _walk(t)
                    if r is not None:
                        return r
            # sous-attribut éventuel (backbone)
            for name in ("preprocessor", "features", "feature_union"):
                if hasattr(x, name):
                    r = _walk(getattr(x, name))
                    if r is not None:
                        return r
            return None
        return _walk(obj)

    def _find_chain_tfidf_svd(obj):
        """
        Essaie de retrouver (tfidf, svd) dans une même sous-chaine.
        Retourne (tfidf, svd, container_pipeline) ou (None, None, None).
        """
        # Cherche une SVD
        svd = _find_step_by_type(obj, TruncatedSVD)
        if svd is None:
            return (None, None, None)
        # Remonte pour trouver le TfidfVectorizer "en amont" de cette SVD :
        # on reparcourt mais on mémorise la relation parent -> enfant.
        parent = {}

        def _walk_parent(x):
            if hasattr(x, "steps"):
                for _, t in x.steps:
                    parent[id(t)] = x
                    _walk_parent(t)
            if hasattr(x, "transformer_list"):
                for _, t in x.transformer_list:
                    parent[id(t)] = x
                    _walk_parent(t)
            if hasattr(x, "transformers_"):
                for _, t, _c in x.transformers_:
                    parent[id(t)] = x
                    _walk_parent(t)

        _walk_parent(obj)

        # Dans le parent de svd (souvent un Pipeline), regarde la/les steps précédentes
        cont = parent.get(id(svd), None)
        tfidf = None
        if cont is not None and hasattr(cont, "steps"):
            steps = cont.steps
            for idx, (nm, est) in enumerate(steps):
                if est is svd and idx > 0:
                    # Parcours en arrière pour trouver un TfidfVectorizer juste avant
                    for j in range(idx - 1, -1, -1):
                        prev_est = steps[j][1]
                        if isinstance(prev_est, TfidfVectorizer):
                            tfidf = prev_est
                            break
                    break
        return (tfidf, svd, cont)

    def _get_linear_clf(pipe):
        """Retourne (clf, coef_, classes_) si disponible ; sinon (None, None, None)."""
        # Cas 1 : pipeline de forme ('clf', clf)
        clf = None
        if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
            clf = pipe.named_steps["clf"]
        # Sinon, l'objet peut être déjà le classifieur
        if clf is None:
            clf = pipe if hasattr(pipe, "fit") and (hasattr(pipe, "predict") or hasattr(pipe, "predict_proba")) else None
        if clf is None:
            return (None, None, None)
        # Récup coef (LogReg/LR one-vs-rest, LinearSVC, SGDClassifier…)
        coef_ = getattr(clf, "coef_", None)
        classes_ = getattr(clf, "classes_", None)
        if coef_ is None or classes_ is None:
            return (None, None, None)
        return (clf, coef_, classes_)

    # ---------- Garde-fous ----------
    if pipe is None:
        print("[WARN] Modèle non chargé — abandon.")
        return
    tfidf, svd, chain = _find_chain_tfidf_svd(pipe)
    if tfidf is None or svd is None:
        print("[WARN] Impossible de retrouver la chaîne (TFIDF → SVD) dans le pipeline — abandon.")
        return

    clf, coef_all, classes_ = _get_linear_clf(pipe)
    if clf is None:
        print("[WARN] Classifieur linéaire introuvable ou sans coef_ — abandon.")
        return

    # ---------- Récupération des noms TF-IDF ----------
    try:
        terms = tfidf.get_feature_names_out()
    except Exception:
        # Anciennes versions
        terms = tfidf.get_feature_names()
    terms = np.asarray(terms)

    k = svd.n_components
    if coef_all.shape[1] < k:
        # Le classifieur voit plus de colonnes (autres features concaténées après SVD)
        # ou, inversement, moins (sélection de features). On tente d’estimer l’offset du bloc SVD.
        # Heuristique: on balaye toutes les positions possibles et on choisit celle maximisant la norme.
        C = coef_all  # (n_classes, n_features_total)
        n_total = C.shape[1]
        best_pos, best_norm = 0, -1
        for start in range(0, n_total - k + 1):
            seg = C[:, start:start + k]
            nrm = float(np.linalg.norm(seg, ord="fro"))
            if nrm > best_norm:
                best_norm, best_pos = nrm, start
        coef_svd = C[:, best_pos:best_pos + k]
        offset = best_pos
        print(f"[INFO] Heuristique: bloc SVD détecté à l'offset {offset} (k={k}).")
    else:
        # Hypothèse simple : le classifieur ne voit que la SVD texte
        coef_svd = coef_all

    # ---------- Backprojection : w_text = V * w_svd, où V = svd.components_.T ----------
    # svd.components_ shape: (k, n_terms). On transpose pour obtenir V.
    V = svd.components_.T  # (n_terms, k)
    # coef_svd shape: (n_classes, k)
    W_text = coef_svd @ V.T  # (n_classes, n_terms)

    # ---------- Agrégation & exports ----------
    sort_by = getattr(args, "sort_by", "total")
    max_n = int(getattr(args, "max_n", 8000))

    # 1) Global (somme des magnitudes)
    abs_sum = np.sum(np.abs(W_text), axis=0)  # (n_terms,)
    df_total = pd.DataFrame({
        "term": terms,
        "magnitude_total": abs_sum
    }).sort_values("magnitude_total", ascending=False)
    if max_n:
        df_total = df_total.head(max_n)
    out_total = Path("results/reports") / "b4_backproj_text_total.csv"
    df_total.to_csv(out_total, index=False, encoding="utf-8")
    print(f"[OK] Global (total |poids|) → {out_total}")

    # 2) Par classe (signé + magnitude)
    Path("results/reports/b4_per_class").mkdir(parents=True, exist_ok=True)
    for i, cls in enumerate(classes_):
        w = W_text[i, :]  # (n_terms,)
        df_c = pd.DataFrame({
            "term": terms,
            "weight": w,
            "abs_weight": np.abs(w)
        })
        if sort_by in ("total", "abs", "magnitude"):
            df_c = df_c.sort_values("abs_weight", ascending=False)
        else:
            df_c = df_c.sort_values("weight", ascending=False)
        if max_n:
            df_c = df_c.head(max_n)
        out_c = Path("results/reports/b4_per_class") / f"b4_backproj_text_per_class__{cls}.csv"
        df_c.to_csv(out_c, index=False, encoding="utf-8")
    print(f"[OK] Par classe → dossier results/reports/b4_per_class")

    # 3) Petit récap JSON
    recap = {
        "n_terms": int(V.shape[0]),
        "n_components": int(k),
        "n_classes": int(len(classes_)),
        "classes": [str(c) for c in classes_],
        "sort_by": sort_by,
        "max_n": max_n
    }
    Path("results/reports").joinpath("b4_backproj_text_summary.json").write_text(
        json.dumps(recap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[DONE] Rétro-projection texte B4 terminée.")
# ------------------------------------------------------------------
# ACP & Confusions légères (basées sur preds CSV)

def do_acp(kind: str, preds_path: Optional[Path] = None, max_n: int = 8000):
    """
    Version légère : charge le CSV de prédictions pour vérifier format et
    produire de petits résumés (sans plots lourds).
    """
    ensure_dirs()
    preds_csv = preds_path if preds_path is not None else (Path("results") / f"preds_{kind}.csv")
    df = _load_preds_csv(preds_csv)
    if df is None:
        return

    # On vérifie la présence des colonnes standards
    cols = df.columns.tolist()
    summary = {
        "path": str(preds_csv),
        "columns": cols,
        "n_rows": len(df),
        "head_y_pred": df["y_pred"].head(5).tolist() if "y_pred" in df.columns else [],
        "has_y_true": "y_true" in df.columns
    }
    if "y_true" in df.columns:
        # Petit aperçu des classes
        summary["y_true_unique"] = len(pd.unique(df["y_true"]))
        summary["sample_true_counts"] = (
            df["y_true"].astype(str).value_counts().head(10).to_dict()
        )

    out_json = Path("results/reports") / f"acp_{kind}_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ACP] Résumé écrit → {out_json}")

def _confusion_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    ctab = pd.crosstab(y_true.astype(str), y_pred.astype(str), dropna=False)
    ctab.index.name = "true"
    ctab.columns.name = "pred"
    return ctab

def top_confusions(kind: str, topk: int = 20, labels_map: Optional[Dict[str, str]] = None,
                   preds_path: Optional[Path] = None):
    """
    Construit un top des confusions à partir du CSV (y_true / y_pred).
    Sauvegarde deux CSV :
      - confusion_matrix_{kind}.csv
      - top_confusions_{kind}.csv
    """
    ensure_dirs()
    preds_csv = preds_path if preds_path is not None else (Path("results") / f"preds_{kind}.csv")
    df = _load_preds_csv(preds_csv)
    if df is None:
        return

    if not {"y_true", "y_pred"}.issubset(df.columns):
        print(f"[WARN] Colonnes y_true/y_pred absentes dans {preds_csv.name}.")
        return

    ctab = _confusion_table(df["y_true"], df["y_pred"])
    conf_path = Path("results/reports") / f"confusion_matrix_{kind}.csv"
    ctab.to_csv(conf_path)
    print(f"[CONF] Matrice sauvegardée → {conf_path}")

    # Top paires confondues (hors diagonale)
    stacked = ctab.stack().reset_index(name="count")
    stacked = stacked[stacked["true"] != stacked["pred"]]
    stacked = stacked.sort_values("count", ascending=False).head(topk).reset_index(drop=True)

    if labels_map:
        stacked["true_label"] = stacked["true"].map(lambda k: labels_map.get(str(k), str(k)))
        stacked["pred_label"] = stacked["pred"].map(lambda k: labels_map.get(str(k), str(k)))

    top_path = Path("results/reports") / f"top_confusions_{kind}.csv"
    stacked.to_csv(top_path, index=False)
    print(f"[CONF] Top confusions sauvegardé → {top_path}")

# ------------------------------------------------------------------
# Wrappers “blocs” B2/B4 (tentent d’appeler les fonctions existantes)

def _try_import_blocks():
    """
    Essaie d'importer les fonctions 'blocs' si elles existent dans ce projet.
    Adapte ici si tes fonctions résident dans d'autres modules.
    """
    funcs = {}
    candidates = [
        # (module_path, [names...])
        ("tools.blocks_b2", [
            "blocks_b2", "blocks_b2_per_class",
            "blocks_b2_per_class_signed_mag"
        ]),
        ("tools.blocks_b4", [
            "blocks_b4", "blocks_b4_per_class",
            "blocks_b4_fine_global", "blocks_b4_per_class_signed",
            "blocks_b4_per_class_signed_magnitude_backproj_text",
        ]),
        # Si tes fonctions sont dans ce même fichier à l'origine,
        # remplace ici par ("tools.diagnostics_acp_shap", [...])
    ]
    for mod, names in candidates:
        try:
            m = __import__(mod, fromlist=names)
            for nm in names:
                if hasattr(m, nm):
                    funcs[nm] = getattr(m, nm)
        except Exception:
            # module non trouvé ou partiellement manquant : on ignore
            pass
        # Fallback: prendre des fonctions définies dans CE module si présentes
    local = sys.modules.get(__name__)
    if local is not None:
        for nm in [
            "blocks_b4_per_class_signed_magnitude_backproj_text",
            "blocks_b4", "blocks_b4_per_class", "blocks_b4_fine_global",
            "blocks_b4_per_class_signed",
            "blocks_b2", "blocks_b2_per_class", "blocks_b2_per_class_signed_mag",
        ]:
            if nm not in funcs and hasattr(local, nm):
                funcs[nm] = getattr(local, nm)
    return funcs

# ------------------------------------------------------------------
# CLI

def main():
    parser = argparse.ArgumentParser("Diagnostics ACP & SHAP + importances (B2/B4)")
    parser.add_argument("--kind", default="b4", help="b2, b3, b4… (pour nommage)")
    parser.add_argument("--model", dest="model_path", required=False,
                        help="Chemin du .joblib à diagnostiquer. Si omis, on essaie --model-template/{kind}/{phase}.")
    parser.add_argument("--data-csv", dest="data_csv", required=False,
                        help="Chemin du CSV source (nécessaire pour certains diagnostics blocs).")
    parser.add_argument("--label-col", default="prdtypecode")
    parser.add_argument("--label-map", dest="label_map_json", default=None)

    # ACP / Confusions
    parser.add_argument("--acp", action="store_true")
    parser.add_argument("--top-confusions", action="store_true")
    parser.add_argument("--topk", type=int, default=20, help="Top K confusions à lister")

    # Templates d'E/S alignés avec train_model.py
    parser.add_argument("--pred-template", default="results/preds_{kind}_{phase}.csv",
                        help="Modèle des prédictions (défaut results/preds_{kind}_{phase}.csv)")
    parser.add_argument("--pred-phase", default="test", help="Phase des prédictions (ex: test)")
    parser.add_argument("--model-template", default="artifacts/model_{kind}_{phase}.joblib",
                        help="Modèle du .joblib (défaut artifacts/model_{kind}_{phase}.joblib)")
    parser.add_argument("--model-phase", default="final", help="Phase du modèle (ex: final)")

    # Options communes blocs
    parser.add_argument("--max-n", type=int, default=8000)
    parser.add_argument("--shared-scale", action="store_true")
    parser.add_argument("--sort-by", default="magnitude",
                        help="Critère de tri (si pertinent dans tes fonctions blocs)")

    # B2
    parser.add_argument("--blocks-b2", action="store_true")
    parser.add_argument("--blocks-b2-per-class", action="store_true")
    parser.add_argument("--blocks-b2-per-class-signed-mag", action="store_true")

    # B4
    parser.add_argument("--blocks-b4", action="store_true")
    parser.add_argument("--blocks-b4-per-class", action="store_true")
    parser.add_argument("--blocks-b4-fine", action="store_true", dest="blocks_b4_fine")
    parser.add_argument("--blocks-b4-per-class-signed", action="store_true")
    parser.add_argument("--blocks-b4-per-class-signed-mag-backproj-text", action="store_true")

    args = parser.parse_args()

    # Construit chemins par défaut cohérents si non fournis
    preds_path = _format_out_path(args.pred_template, kind=args.kind, phase=args.pred_phase)
    model_path_auto = _format_out_path(args.model_template, kind=args.kind, phase=args.model_phase)
    # Si --model explicite est fourni, il prime
    effective_model_path = Path(args.model_path) if args.model_path else model_path_auto

    # ACP / Confusions
    if args.acp:
        do_acp(args.kind, preds_path=preds_path, max_n=args.max_n)

    if args.top_confusions:
        labels_map = None
        if args.label_map_json and Path(args.label_map_json).exists():
            try:
                labels_map = json.loads(Path(args.label_map_json).read_text(encoding="utf-8"))
            except Exception:
                labels_map = None
        top_confusions(args.kind, topk=args.topk, labels_map=labels_map, preds_path=preds_path)

    # Si aucune option “blocs” n’est utilisée, on s’arrête là
    wants_blocks = any([
        args.blocks_b2, args.blocks_b2_per_class, args.blocks_b2_per_class_signed_mag,
        args.blocks_b4, args.blocks_b4_per_class, args.blocks_b4_fine,
        args.blocks_b4_per_class_signed, args.blocks_b4_per_class_signed_mag_backproj_text
    ])
    if not wants_blocks:
        return

    # Pour les blocs, on a besoin du modèle et (souvent) des données
    if effective_model_path is None or not effective_model_path.exists():
        print(f"[WARN] Modèle absent : {effective_model_path} — impossible d'exécuter les blocs.")
        return
    if not args.data_csv or not Path(args.data_csv).exists():
        print("[WARN] --data-csv est requis pour exécuter les blocs (fichier introuvable).")
        return

    # Chargement du modèle (si les fonctions blocs en ont besoin)
    try:
        import joblib
        pipe = joblib.load(str(effective_model_path))
    except Exception as e:
        print(f"[WARN] Impossible de charger le modèle {effective_model_path}: {e}")
        pipe = None

    df = None
    try:
        df = pd.read_csv(args.data_csv)
    except Exception as e:
        print(f"[WARN] Impossible de lire {args.data_csv}: {e}")

    funcs = _try_import_blocks()

    def _call_or_warn(name: str, *fargs, **fkwargs):
        if name not in funcs:
            print(f"[WARN] Fonction '{name}' non disponible (module non importé).")
            return
        try:
            return funcs[name](*fargs, **fkwargs)
        except Exception as ex:
            print(f"[WARN] Erreur lors de l'appel '{name}': {ex}")

    # ---- B2
    if args.blocks_b2:
        _call_or_warn("blocks_b2", str(effective_model_path), args.data_csv,
                      max_n=args.max_n, normalize="abs")
    if args.blocks_b2_per_class:
        _call_or_warn("blocks_b2_per_class", str(effective_model_path), args.data_csv,
                      label_col=args.label_col, max_n=args.max_n)
    if args.blocks_b2_per_class_signed_mag:
        _call_or_warn("blocks_b2_per_class_signed_mag", str(effective_model_path), args.data_csv,
                      label_col=args.label_col, max_n=args.max_n,
                      label_map_json=args.label_map_json, shared_scale=True, sort_by=args.sort_by)

    # ---- B4
    if args.blocks_b4:
        _call_or_warn("blocks_b4", str(effective_model_path), args.data_csv,
                      max_n=args.max_n, normalize="abs")
    if args.blocks_b4_per_class:
        _call_or_warn("blocks_b4_per_class", str(effective_model_path), args.data_csv,
                      label_col=args.label_col, max_n=args.max_n)
    if args.blocks_b4_fine:
        _call_or_warn("blocks_b4_fine_global", str(effective_model_path), args.data_csv,
                      label_col=args.label_col, max_n=args.max_n)
    if args.blocks_b4_per_class_signed:
        _call_or_warn("blocks_b4_per_class_signed", str(effective_model_path), args.data_csv,
                      label_col=args.label_col, max_n=args.max_n,
                      label_map_json=args.label_map_json, shared_scale=args.shared_scale)
    if args.blocks_b4_per_class_signed_mag_backproj_text:
        # Si ta fonction a besoin d'objets pipe/df, on les passe
        _call_or_warn("blocks_b4_per_class_signed_magnitude_backproj_text", args, pipe, df)

if __name__ == "__main__":
    main()
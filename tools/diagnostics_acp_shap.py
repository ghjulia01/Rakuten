# -*- coding: utf-8 -*-
"""
Diagnostics ACP & SHAP pour baselines B2/B3/B4
- ACP 2D: SVD preview si disponible, sinon PCA sur la matrice OOF
- SHAP: LinearExplainer (LogReg) ou KernelExplainer (fallback, plus lent)
"""

import os
import argparse
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet",
    category=FutureWarning,
)

# chemins pour importer des modules de main/ et features/
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "main", ROOT / "features"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# === Helpers robustes pour retrouver les branches dans les pipelines ===

def _get_text_union_from_model(model):
    """
    Retourne l'objet FeatureUnion/Transformer de la branche texte quelle que soit la baseline :
    - B2 : step top-level 'text'
    - B4 : FeatureUnion 'features' qui contient un step 'text'
    """
    # Cas B2 : Pipeline([("text", <FeatureUnion texte>), ("clf", ...)])
    if hasattr(model, "named_steps"):
        text = model.named_steps.get("text")
        if text is not None:
            return text

        # Cas B4 : Pipeline([("features", <FeatureUnion>), ("sampler", ...), ("scaler", ...), ("clf", ...)])
        feat_union = model.named_steps.get("features")
        if feat_union is not None and hasattr(feat_union, "transformer_list"):
            for name, tr in feat_union.transformer_list:
                if name == "text":
                    return tr
            # recherche en profondeur si jamais la branche texte est nichée
            for _, tr in feat_union.transformer_list:
                if hasattr(tr, "transformer_list"):
                    for subname, subtr in tr.transformer_list:
                        if subname == "text":
                            return subtr
    return None


def _get_img_union_from_model(model):
    """
    Retourne la FeatureUnion image pour B3 (step 'img') ou B4 (step 'features' puis 'image_*').
    Ici on vise B3 pour le nouveau bloc (--blocks-b3).
    """
    if hasattr(model, "named_steps"):
        img = model.named_steps.get("img")  # B3: Pipeline([("img", FeatureUnion([...])) , ("clf", ...)])
        if img is not None:
            return img
    return None

def _find_feat_scaler_clf(model):
    """
    Retourne (feat_union, scaler, clf, names) depuis un Pipeline, 
    en scannant les étapes au lieu de supposer des noms fixes.
    """
    feat_union = None
    scaler = None
    clf = None
    names = {"features": None, "scaler": None, "clf": None}

    # 1) accès direct si les noms standards existent
    if hasattr(model, "named_steps"):
        feat_union = model.named_steps.get("features")
        scaler = model.named_steps.get("scaler")
        clf = (model.named_steps.get("clf")
               or model.named_steps.get("model")
               or model.named_steps.get("final_estimator"))
        if feat_union is not None: names["features"] = "features"
        if scaler is not None: names["scaler"] = "scaler"
        if clf is not None: names["clf"] = "clf/model/final_estimator"

    # 2) si l’un manque, on scanne toutes les étapes
    if isinstance(getattr(model, "steps", None), list):
        for nm, step in model.steps:
            # chercher la feature union (ou un ColumnTransformer)
            if feat_union is None and (
                isinstance(step, FeatureUnion) or isinstance(step, ColumnTransformer) or hasattr(step, "transformer_list")
            ):
                feat_union = step
                names["features"] = nm
            # scaler
            if scaler is None and nm.lower() in ("scaler", "standardscaler"):
                scaler = step
                names["scaler"] = nm
            # classifieur : quelque chose qui a coef_ ou decision_function/predict_proba
            if clf is None and (
                hasattr(step, "coef_") or hasattr(step, "classes_") or
                hasattr(step, "decision_function") or hasattr(step, "predict_proba")
            ):
                clf = step
                names["clf"] = nm

    return feat_union, scaler, clf, names

def _load_ok_mask(preds_csv: str, index_like=None):
    """
    Charge un CSV de prédictions OOF avec colonnes y_true / y_pred.
    Retourne un masque booléen ok=(y_true==y_pred) aligné à index_like si fourni.
    """
    if not preds_csv or not os.path.exists(preds_csv):
        return None
    df = pd.read_csv(preds_csv, index_col=0)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        return None
    ok = (df["y_true"].astype(str) == df["y_pred"].astype(str))
    if index_like is not None:
        # réaligne sur l’index cible si nécessaire
        ok = ok.reindex(index_like, fill_value=False)
    return ok.values

def ensure_dirs():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/reports").mkdir(parents=True, exist_ok=True)

def load_labels_map(path="features/labels_map.json"):
    p = Path(path)
    if not p.exists():
        return None
    try:
        import json
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): v for k, v in raw.items()}
    except Exception:
        return None

def try_load_oof_features(kind):
    """Retourne (Z, is_sparse, src) où Z peut être sparse ou dense, ou None si introuvable."""
    from pathlib import Path
    npz_path = Path("results") / f"features_{kind}_oof.npz"
    npy_path = Path("results") / f"features_{kind}_oof.npy"

    if npz_path.exists():
        from scipy.sparse import load_npz  # type: ignore
        return load_npz(npz_path), True, str(npz_path)
    if npy_path.exists():
        return np.load(npy_path), False, str(npy_path)
    return None, False, None

def do_acp(kind, max_n=8000):
    """
    Fait une ACP 2D:
    - Priorité au CSV SVD preview (rapide) -> results/features_<kind>_svd100_preview.csv
    - Sinon PCA 2D sur la matrice OOF (npz/npy)
    Génère:
      - results/figures/acp_<kind>_2d.png (SVD preview ou PCA OOF)
      - results/figures/acp_<kind>_ok_error.png (si y_true/y_pred dispo)
    """
    ensure_dirs()

    preds_csv = Path("results") / f"preds_{kind}.csv"
    preds = None
    if preds_csv.exists():
        preds = pd.read_csv(preds_csv, index_col=0)
        if {"y_true", "y_pred"}.issubset(preds.columns):
            preds["y_true"] = preds["y_true"].astype(str)
            preds["y_pred"] = preds["y_pred"].astype(str)
            preds["ok"] = (preds["y_true"] == preds["y_pred"]).astype(int)
        else:
            preds = None
            print(f"[WARN] {preds_csv} ne contient pas y_true,y_pred → pas de coloration OK/Erreur.")

    svd_csv = Path("results") / f"features_{kind}_svd100_preview.csv"
    fig_svd = Path("results/figures") / f"acp_{kind}_2d.png"
    fig_ok  = Path("results/figures") / f"acp_{kind}_ok_error.png"

    # --- Cas 1 : SVD preview disponible (rapide) ---
    if svd_csv.exists():
        print(f"[INFO] SVD preview trouvé: {svd_csv}")
        df = pd.read_csv(svd_csv, index_col=0)

        # Figure simple (tous docs)
        plt.figure(figsize=(9,6))
        plt.scatter(df["svd_1"], df["svd_2"], s=8, alpha=0.20)
        plt.xlabel("svd_1"); plt.ylabel("svd_2")
        plt.title(f"ACP (SVD preview) — {kind.upper()}")
        plt.tight_layout()
        plt.savefig(fig_svd, dpi=180)
        plt.close()
        print(f"[OK] ACP (SVD preview) → {fig_svd}")

        # Figure OK/Erreur si on a y_true/y_pred (dans le SVD ou via preds_*.csv)
        ok_mask = None
        if {"y_true","y_pred"}.issubset(df.columns):
            ok_mask = (df["y_true"].astype(str) == df["y_pred"].astype(str)).values
        elif preds is not None:
            # réaligne au besoin sur l’index du SVD
            ok_mask = preds.reindex(df.index).assign(
                ok=lambda d: (d["y_true"].astype(str) == d["y_pred"].astype(str)).astype(int)
            )["ok"].fillna(0).astype(bool).values

        if ok_mask is not None:
            XY = df[["svd_1","svd_2"]].values
            plt.figure(figsize=(9,6))
            plt.scatter(XY[ok_mask,0], XY[ok_mask,1], s=8, alpha=0.15, label="OK (y_true = y_pred)")
            plt.scatter(XY[~ok_mask,0], XY[~ok_mask,1], s=8, alpha=0.35, label="Erreur")
            plt.xlabel("svd_1"); plt.ylabel("svd_2")
            plt.title(f"ACP (SVD preview) — {kind.upper()} : OK vs Erreur")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_ok, dpi=180)
            plt.close()
            print(f"[OK] ACP colorée → {fig_ok}")

        return True  # on s'arrête là si SVD preview

    # --- Cas 2 : pas de SVD preview → PCA 2D sur OOF ---
    Z, is_sparse, src = try_load_oof_features(kind)
    if Z is None:
        print("[WARN] Aucune feature trouvée (ni SVD preview, ni OOF).")
        return False

    print(f"[INFO] OOF features: {src} (sparse={is_sparse})")
    n = Z.shape[0]
    take = min(max_n, n)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=take, replace=False))
    if hasattr(Z, "tocsr"):
        Zs = Z[idx]
    else:
        Zs = Z[idx, :]

    if hasattr(Zs, "toarray"):
        Zs = Zs.toarray()

    # Standardisation simple (with_mean=False si déjà centré/normalisé)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler(with_mean=False)
    Zs_std = scaler.fit_transform(Zs)

    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Zs_std)

    plt.figure(figsize=(9,6))
    plt.scatter(XY[:, 0], XY[:, 1], s=8, alpha=0.20)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"ACP (PCA 2D) — {kind.upper()} — {take}/{n}")
    plt.tight_layout()
    plt.savefig(fig_svd, dpi=180)
    plt.close()
    print(f"[OK] ACP (PCA 2D) → {fig_svd}")
    return True

def top_confusions(kind, topk=20, labels_map=None):
    """Calcule le top des confusions et écrit un CSV lisible."""
    ensure_dirs()
    preds_csv = Path("results") / f"preds_{kind}.csv"
    if not preds_csv.exists():
        print("[WARN] Pas de preds CSV pour calculer les confusions.")
        return

    preds = pd.read_csv(preds_csv)
    preds["y_true"] = preds["y_true"].astype(str)
    preds["y_pred"] = preds["y_pred"].astype(str)

    from sklearn.metrics import confusion_matrix
    labels = sorted(set(preds["y_true"]) | set(preds["y_pred"]), key=lambda x: str(x))
    cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=labels)
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)

    rows, cols = np.where(cm2 > 0)
    records = []
    for r, c in zip(rows, cols):
        t, p = labels[r], labels[c]
        tn = labels_map.get(t, t) if labels_map else t
        pn = labels_map.get(p, p) if labels_map else p
        records.append((t, p, int(cm2[r, c]), tn, pn))

    records = sorted(records, key=lambda x: x[2], reverse=True)[:topk]
    df = pd.DataFrame(records, columns=["true_id", "pred_id", "count", "true_name", "pred_name"])
    out = Path("results/reports") / f"top_confusions_{kind}.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Top confusions → {out}")

######
#   SHAP helpers (pour B2/B3/B4)
def _walk_estimators(est):
    yield est
    # Pipelines
    if hasattr(est, "steps"):
        for _, s in getattr(est, "steps", []):
            yield from _walk_estimators(s)
    # FeatureUnion / ColumnTransformer-like
    if hasattr(est, "transformer_list"):
        for _, t in getattr(est, "transformer_list", []):
            yield from _walk_estimators(t)
    # OneVsRest / wrappers
    for attr in ("estimator", "base_estimator"):
        if hasattr(est, attr):
            yield from _walk_estimators(getattr(est, attr))

def _find_text_union(est):
    # on cherche un FeatureUnion issu de models.text_pipeline
    for obj in _walk_estimators(est):
        if obj.__class__.__name__ == "FeatureUnion":
            # heuristique: contient un transformer "tfidf" ou "tfidf_word"
            names = [n for n, _ in getattr(obj, "transformer_list", [])]
            if any(n in ("tfidf", "tfidf_word") for n in names):
                return obj
    return None

def _find_vectorizer(est):
    # un objet ayant get_feature_names_out + transform
    for obj in _walk_estimators(est):
        if hasattr(obj, "get_feature_names_out") and hasattr(obj, "transform"):
            try:
                _ = obj.get_feature_names_out()
                return obj
            except Exception:
                continue
    return None

def _get_coef_matrix(clf):
    # supporte LogisticRegression, LinearSVC et OneVsRestClassifier
    import numpy as np
    if hasattr(clf, "coef_"):
        C = clf.coef_
        return C if hasattr(C, "toarray") is False else C.toarray()
    # OneVsRest
    if hasattr(clf, "estimators_"):
        mats = []
        for est in clf.estimators_:
            mats.append(_get_coef_matrix(est))
        return np.vstack(mats)
    return None

def _feature_names_from_text_union(text_union):
    # on récupère le vectorizer interne pour avoir les noms de features
    vec = _find_vectorizer(text_union)
    if vec is not None:
        try:
            return vec.get_feature_names_out()
        except Exception:
            pass
    # fallback: indices muets
    return None

def _ensure_input_df(df):
    # Les transformeurs texte maison s’attendent souvent à ces colonnes
    needed = ["designation", "description", "productid", "imageid"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    return df[needed]

def do_shap(kind, model_path, data_csv=None, text_col="designation",
            label_col=None, max_n=3000, topk=30):
    """
    Génère: 
      - results/figures/shap_global_bar_<kind>.png
      - results/reports/top_words_<kind>.csv
    """
    ensure_dirs()
    if model_path is None or not os.path.exists(model_path):
        print("[WARN] Pas de modèle .joblib → SHAP ignoré.")
        return False

    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    model = joblib.load(model_path)

    # Localiser la branche texte (TF-IDF) et le classifieur
    text_union = _find_text_union(model)
    if text_union is None:
        print("[WARN] Branche texte introuvable dans la pipeline → SHAP ignoré.")
        return False

    # Charger des textes d'exemple
    if data_csv is None or not os.path.exists(data_csv):
        print("[WARN] data_csv introuvable. Fournissez --data-csv pour les explications texte.")
        return False

    df = pd.read_csv(data_csv)
    if text_col not in df.columns:
        print(f"[WARN] Colonne '{text_col}' absente de {data_csv} → SHAP ignoré.")
        return False

    # échantillonner pour rester léger
    if len(df) > max_n:
        df = df.sample(max_n, random_state=42).reset_index(drop=True)

    # 1) localiser la "text union" puis le sous-pipeline TF-IDF
    text_union = _find_text_union(model)  # ta fonction utilitaire
    if text_union is None:
        print("[WARN] Branche texte introuvable → skip SHAP.")
        return False

    # text_union peut être:
    #  - create_text_pipeline()  -> contient ('tfidf', 'has_desc', ...)
    #  - create_text_pipeline_from_cfg() -> nommée 'tfidf_word' qui elle-même contient 'tfidf'
    tfidf_pipe = None
    # cas direct: ('tfidf', pipeline)
    for name, tr in getattr(text_union, "transformer_list", []):
        if name == "tfidf":
            tfidf_pipe = tr
            break

    # cas imbriqué: ('tfidf_word', FeatureUnion(... 'tfidf' ...))
    if tfidf_pipe is None:
        for name, tr in getattr(text_union, "transformer_list", []):
            if name == "tfidf_word" and hasattr(tr, "transformer_list"):
                for n2, tr2 in tr.transformer_list:
                    if n2 == "tfidf":
                        tfidf_pipe = tr2
                        break

    if tfidf_pipe is None:
        print("[WARN] Sous-pipeline 'tfidf' introuvable → skip SHAP.")
        return False

    # 2) features TF-IDF
    X_tfidf = tfidf_pipe.transform(_ensure_input_df(df))  # sparse OK
    vec = getattr(tfidf_pipe, "named_steps", {}).get("texttfidfvectorizer", None)
    feat_names = vec.get_feature_names_out() if vec else None
    n_tfidf = X_tfidf.shape[1]
    if feat_names is None or len(feat_names) != n_tfidf:
        feat_names = np.array([f"w_{i}" for i in range(n_tfidf)])

    # 3) récupérer le classifieur linéaire final
    clf = None
    for obj in _walk_estimators(model):
        if hasattr(obj, "predict"):
            clf = obj
    if clf is None:
        print("[WARN] Classifieur final introuvable → SHAP ignoré.")
        return False

    used_fallback = False
    try:
        import shap
        if hasattr(clf, "predict_proba"):
            expl = shap.LinearExplainer(clf, X_tfidf, feature_perturbation="interventional")
            sv = expl.shap_values(X_tfidf)
            if isinstance(sv, list):
                abs_mean = np.mean([np.mean(np.abs(s), axis=0) for s in sv], axis=0)
                per_class = [np.mean(np.abs(s), axis=0) for s in sv]
            else:
                abs_mean = np.mean(np.abs(sv), axis=0)
                per_class = [abs_mean]
        else:
            used_fallback = True
    except Exception as e:
        print(f"[INFO] SHAP non disponible/incompatible ({e}) → fallback coef×mean(tfidf).")
        used_fallback = True

    if used_fallback:
        C = _get_coef_matrix(clf)
        if C is None:
            print("[WARN] Ni SHAP ni coef_ disponibles → SHAP ignoré.")
            return False

        # --- Detect if the TEXT branch uses SVD (reduction -> components_)
        # Locate the "text" branch inside the global FeatureUnion('features')
        features_union = getattr(model.named_steps.get("features", None), "transformer_list", None)
        text_branch = None
        if features_union is not None:
            for name, tr in features_union:
                if name == "text":
                    text_branch = tr
                    break

        has_svd = bool(getattr(getattr(text_branch, "named_steps", {}), "get", lambda *_: None)("svd"))

        # Helper: count output dims of a fitted transformer without refitting
        def _ncols_fitted(tr):
            Xtmp = _ensure_input_df(df.head(min(5, len(df))))
            Xt = tr.transform(Xtmp)
            return int(Xt.shape[1])

        if has_svd:
            # ----- SVD CASE: slice classifier weights on TEXT block, then
            # back-project and keep ONLY the TF-IDF word slice.
            # 1) find global slice [a:b] for the TEXT block in the full FeatureUnion
            a = 0
            text_width = None
            for name, tr in model.named_steps["features"].transformer_list:
                # helper to get fitted output width without refit
                Xtmp = _ensure_input_df(df.head(min(5, len(df))))
                w = tr.transform(Xtmp).shape[1]
                if name == "text":
                    text_width = w
                    break
                a += w
            if text_width is None:
                print("[WARN] Impossible de localiser la tranche TEXT → SHAP ignoré.")
                return False
            b = a + text_width

            # 2) weights on SVD components for TEXT
            C_text_comp = np.asarray(C[:, a:b])    # (n_classes, k)

            # 3) SVD components on the whole pre-SVD TEXT branch
            svd = text_branch.named_steps["svd"]
            Vk  = np.asarray(svd.components_)      # (k, d_text_pre)

            # 4) find the TF-IDF word slice inside the pre-SVD TEXT branch
            pre_union = text_branch.named_steps["text"]  # FeatureUnion before SVD
            df_small  = df.head(min(200, len(df))).copy()
            slices_pre = _block_slices_text(pre_union, df_small)
            # look for the entry corresponding to TF-IDF words
            a_w, b_w = None, None
            for lbl, aa, bb in slices_pre:
                if "tf-idf word" in lbl.lower() or "tfidf" == lbl.lower() or lbl.endswith("/tfidf"):
                    a_w, b_w = aa, bb
                    break
            if a_w is None:
                print("[WARN] Tranche TF-IDF introuvable dans la branche texte → SHAP ignoré.")
                return False

            # 5) mean TF-IDF on words (same dimension as b_w-a_w)
            mu = np.asarray(X_tfidf.mean(axis=0)).ravel()   # (n_words,)

            # 6) back-project per class, then slice to words only
            per_class = []
            for i in range(C_text_comp.shape[0]):
                w_comp   = C_text_comp[i, :].ravel()       # (k,)
                w_full   = Vk.T @ w_comp                   # (d_text_pre,)
                w_words  = w_full[a_w:b_w]                 # (n_words,)
                per_class.append(np.abs(w_words) * mu)     # (n_words,)

            abs_mean = np.mean(np.vstack(per_class), axis=0)  # (n_words,)

    # === Sorties ===
    order = np.argsort(abs_mean)[::-1][:topk]
    top_df = pd.DataFrame({
        "feature": feat_names[order],
        "importance": abs_mean[order]
    })
    out_csv = Path("results/reports") / f"top_words_{kind}.csv"
    top_df.to_csv(out_csv, index=False)

    # par classe (si possible)
    if hasattr(clf, "classes_"):
        classes = list(map(str, getattr(clf, "classes_")))
        rows = []
        for i, cls in enumerate(classes):
            imp = per_class[i]
            idx = np.argsort(imp)[::-1][:topk]
            for r in range(len(idx)):
                rows.append((cls, feat_names[idx[r]], float(imp[idx[r]]), int(r+1)))
        pd.DataFrame(rows, columns=["class", "feature", "score", "rank"])\
          .to_csv(Path("results/reports") / f"top_words_per_class_{kind}.csv", index=False)

    # barplot global
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1])
    plt.xlabel("Importance" + ("" if not used_fallback else " (coef×mean tf-idf)"))
    plt.title(f"Top {topk} caractéristiques globales — {kind.upper()}")
    plt.tight_layout()
    fig_path = Path("results/figures") / f"shap_global_bar_{kind}.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

    print(f"[OK] SHAP global → {fig_path}")
    print(f"[OK] Top mots → {out_csv}")
    return True
######
def _ensure_input_df(df):
    # garde les colonnes pertinentes si elles existent
    keep = [c for c in ("designation","description","productid","imageid") if c in df.columns]
    return df[keep] if keep else df

def _ncols(transformer, df_small):
    Xs = transformer.transform(_ensure_input_df(df_small))
    return Xs.shape[1]

def _coef_matrix(clf):
    C = getattr(clf, "coef_", None)
    if C is None:
        return None
    return np.asarray(C)

def _winner_per_row(clf, X):
    # prédire la classe "gagnante" (pour OvR)
    if hasattr(clf, "decision_function"):
        S = clf.decision_function(X)
        if S.ndim == 1:
            S = S[:, None]
        return np.argmax(S, axis=1)
    elif hasattr(clf, "predict"):
        pred = clf.predict(X)
        # map classes -> indices
        if hasattr(clf, "classes_"):
            inv = {c:i for i, c in enumerate(clf.classes_)}
            return np.array([inv.get(p, 0) for p in pred])
    return np.zeros(X.shape[0], dtype=int)

def _block_slices_text(text_union, df_small):
    """
    Retourne [(label, start, end)] pour la branche texte B2.
    Détaille les sous-éléments d'un éventuel sous-FeatureUnion 'tfidf_word'.
    """
    label_map = {
        "tfidf": "TF-IDF word",
        "tfidf_char": "TF-IDF char",
        "has_desc": "HasDescription",
        "title_len": "TitleLength",
        "text_stats": "TextStats",
        "language": "Language",
        "text_stats_pro": "TextStatsPro",
        "lexicon": "Lexicon",
    }

    slices = []
    start = 0

    for name, tr in getattr(text_union, "transformer_list", []):
        # Cas: sous-union imbriquée nommée 'tfidf_word'
        if name == "tfidf_word" and hasattr(tr, "transformer_list"):
            inner_start = start
            for n2, tr2 in tr.transformer_list:
                n = _ncols(tr2, df_small)
                lbl = label_map.get(n2, f"tfidf_word/{n2}")
                slices.append((lbl, inner_start, inner_start + n))
                inner_start += n
            start = inner_start
            continue

        # Cas: objets au 1er niveau (tfidf, tfidf_char, has_desc, ...)
        n = _ncols(tr, df_small)
        lbl = label_map.get(name, name)
        slices.append((lbl, start, start + n))
        start += n

    return slices

def _block_slices_b4(features_union, df_small):
    """
    Retourne des blocs haut niveau pour B4.
    Noms présents dans create_combined_pipeline(): 'text','image_pixels','image_cnn','image_stats'.
    """
    pretty = {"text":"Texte","image_pixels":"Pixels",
              "image_cnn":"CNN","image_stats":"Stats","image_stats_combined": "Stats images"}
    slices = []
    start = 0
    for name, tr in getattr(features_union, "transformer_list", []):
        n = _ncols(tr, df_small)
        slices.append((pretty.get(name, name), start, start+n))
        start += n
    return slices

def _block_importance_from_coef(C, X, slices, clf):
    """
    Agrège les contributions par bloc sur la classe gagnante de chaque ligne.
    Renvoie un DataFrame [block, imp_abs, imp_pos, imp_neg, imp_signed].
    """
    # assure un format efficace pour les découpes de lignes
    if hasattr(X, "tocsr"):
        X = X.tocsr()

    winners = _winner_per_row(clf, X)
    rows = []

    for (label, a, b) in slices:
        Xb = X[:, a:b]  # sous-matrice du bloc
        imp_abs, imp_pos, imp_neg = [], [], []

        for i in range(X.shape[0]):
            k = winners[i]
            w = C[k, a:b]  # coefficients du bloc pour la classe gagnante

            # ligne i du bloc (sparse ou dense)
            xi = Xb[i]
            # produit scalaire; cast en float quoi qu'il arrive (1x1, array(), matrix…)
            v = xi.dot(w)
            v = float(np.asarray(v).ravel()[0])

            imp_abs.append(abs(v))
            imp_pos.append(max(v, 0.0))
            imp_neg.append(max(-v, 0.0))

        rows.append({
            "block": label,
            "imp_abs": float(np.mean(imp_abs)),
            "imp_pos": float(np.mean(imp_pos)),
            "imp_neg": float(np.mean(imp_neg)),
            "imp_signed": float(np.mean(np.array(imp_pos) - np.array(imp_neg))),
        })

    df_imp = pd.DataFrame(rows).sort_values("imp_abs", ascending=False)
    return df_imp

# --------------------------- slicing helpers ---------------------------
def _safe_dim(transformer, df):
    """Retourne la dimension de sortie du transformer (déjà fit) sans refit."""
    from scipy import sparse
    Xb = transformer.transform(_ensure_input_df(df.head(5)))
    if hasattr(Xb, "shape"):
        return int(Xb.shape[1])
    # au cas improbable d'un vecteur 1D
    return int(np.array(Xb).reshape(1, -1).shape[1])

def _block_slices_b4_fine(feat_union, df):
    """
    Retourne une liste [(name, start, stop), ...] pour la FeatureUnion 'features'
    avec un découpage fin :
      - text : TF-IDF word, TF-IDF char, HasDescription, TitleLength, TextStats, Language (si présents)
      - pixels : 1 bloc
      - cnn : 1 bloc
      - img_stats : width, height, occupancy, white_ratio, black_ratio (ou stat_i sinon)
    """
    slices = []
    start = 0

    for name, tr in feat_union.transformer_list:
        if tr is None:
            continue

        # --- branche texte : on sous-découpe en reprenant la logique B2
        if name.lower() in ("text", "txt", "text_features"):
            sub = _block_slices_text(tr, df)  # [(subname, a, b) ...] relatifs à la branche texte
            # décaler chaque sous-bloc au bon offset global
            for (subname, a, b) in sub:
                slices.append((subname, start + a, start + b))
            # avancer le curseur global de la dimension totale de la branche texte
            start += _safe_dim(tr, df)
            continue

        # --- pixels (flatten + éventuel SVD)
        if name.lower() in ("pixels", "pix", "img_pixels", "image_flat"):
            w = _safe_dim(tr, df)
            if w > 0:
                slices.append(("Pixels", start, start + w))
                start += w
            continue

        # --- CNN embeddings (ResNet + éventuel SVD)
        if "cnn" in name.lower():
            w = _safe_dim(tr, df)
            if w > 0:
                slices.append(("CNN", start, start + w))
                start += w
            continue

        # --- Image stats (on éclate chaque dimension)
        if name.lower() in ("stats", "img_stats", "image_stats"):
            w = _safe_dim(tr, df)
            if w > 0:
                # noms attendus ; fallback générique si dimension différente
                base = ["Img:width", "Img:height", "Img:occupancy", "Img:white_ratio", "Img:black_ratio"]
                names = base if w == 5 else [f"Img:stat_{i}" for i in range(w)]
                for i in range(w):
                    slices.append((names[i], start + i, start + i + 1))
                start += w
            continue
        # --- Image stats (combined: basic + pro), explode each dimension
        if name.lower() in ("stats_combined", "image_stats_combined"):
            w = _safe_dim(tr, df)
            if w > 0:
                # Expected 5 (basic) + 14 (pro) = 19 features from ImageStatsCombinedFeaturizer
                names19 = [
                    # BASIC (5)
                    "Img:width", "Img:height", "Img:occupancy", "Img:white_ratio", "Img:black_ratio",
                    # PRO (14)
                    "Pro:gray_mean", "Pro:gray_std", "Pro:p10", "Pro:p90", "Pro:dyn_range",
                    "Pro:entropy", "Pro:lap_var", "Pro:edge_density", "Pro:aspect_ratio",
                    "Pro:bbox_center_offset", "Pro:sat_mean", "Pro:colorfulness",
                    "Pro:border_white_ratio", "Pro:file_bpp",
                ]
                names = names19 if w == 19 else [f"ImgPro:stat_{i}" for i in range(w)]
                for i in range(w):
                    slices.append((names[i], start + i, start + i + 1))
                start += w
            continue

        # --- au pire : bloc inconnu -> on met un nom générique
        w = _safe_dim(tr, df)
        if w > 0:
            slices.append((f"{name}", start, start + w))
            start += w

    return slices
# --------------------------- plotting helpers ---------------------------

def _save_barplot(df_imp, title, out_png, value_col="imp_abs"):
    plt.figure(figsize=(9, 5))
    dfp = df_imp.sort_values(value_col, ascending=False)
    y = np.arange(len(dfp))
    plt.barh(y, dfp[value_col].values)
    plt.yticks(y, dfp["block"].values)
    xlabel = {
        "imp_abs": "Importance moyenne |x·w|",
        "imp_signed": "Impact net moyen (signé)",
        "imp_pos": "Impact positif moyen",
        "imp_neg": "Impact négatif moyen",
    }.get(value_col, value_col)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.gca().invert_yaxis()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] {title} -> {out_png}")


# --------------------------- importances (général) ---------------------------

def _block_importance_from_coef(C, X, slices, clf):
    """
    Agrège les contributions par bloc, sur la classe gagnante de chaque ligne.
    Renvoie un DataFrame [block, imp_abs, imp_pos, imp_neg, imp_signed].
    """
    if hasattr(X, "tocsr"):
        X = X.tocsr()

    winners = _winner_per_row(clf, X)
    rows = []

    for (label, a, b) in slices:
        Xb = X[:, a:b]
        imp_abs, imp_pos, imp_neg = [], [], []

        for i in range(X.shape[0]):
            k = winners[i]
            w = C[k, a:b]
            xi = Xb[i]
            v = xi.dot(w)
            v = float(np.asarray(v).ravel()[0])

            imp_abs.append(abs(v))
            imp_pos.append(max(v, 0.0))
            imp_neg.append(max(-v, 0.0))

        rows.append({
            "block": label,
            "imp_abs": float(np.mean(imp_abs)),
            "imp_pos": float(np.mean(imp_pos)),
            "imp_neg": float(np.mean(imp_neg)),
            "imp_signed": float(np.mean(np.array(imp_pos) - np.array(imp_neg))),
        })

    df_imp = pd.DataFrame(rows).sort_values("imp_abs", ascending=False)
    return df_imp


# --------------------------- B2 (texte) global ---------------------------

def blocks_b2(model_path, data_csv, max_n=3000, normalize="abs"):
    model = joblib.load(model_path)
    text_union = model.named_steps.get("text")
    clf = model.named_steps.get("clf")
    if text_union is None or clf is None:
        print("[WARN] pipeline B2 inattendue -> ignoré.")
        return False

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    df = df.sample(n=min(len(df), max_n), random_state=0)
    X = text_union.transform(_ensure_input_df(df))
    slices = _block_slices_text(text_union, df)

    df_imp = _block_importance_from_coef(C, X, slices, clf)

    col_map = {"abs": "imp_abs", "signed": "imp_signed", "pos": "imp_pos", "neg": "imp_neg"}
    vcol = col_map[normalize]
    suffix = normalize

    out_csv = Path(f"results/reports/block_importance_b2_{suffix}.csv")
    out_png = Path(f"results/figures/block_importance_b2_{suffix}.png")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(out_csv, index=False)
    _save_barplot(df_imp, f"Importance par bloc — B2 ({normalize})", out_png, value_col=vcol)
    return True


# --------------------------- utilitaire par classe ---------------------------

def _block_contrib_per_class(C, X, y_true, slices, clf, normalize="abs"):
    """
    Pour chaque classe (ordre clf.classes_), calcule la contribution moyenne par bloc
    (abs/pos/neg/signed). Retourne un DataFrame (index=classes, colonnes=blocs)
    avec des pourcentages qui somment à 100% par classe.
    """
    if hasattr(X, "tocsr"):
        X = X.tocsr()

    classes = list(getattr(clf, "classes_", range(C.shape[0])))
    blocks = [lbl for (lbl, _, _) in slices]

    rows = []
    for k, cls in enumerate(classes):
        mask = (y_true == cls)
        if mask.sum() == 0:
            rows.append([0.0] * len(blocks))
            continue

        vals = []
        for (lbl, a, b) in slices:
            xb = X[mask, a:b]
            w = C[k, a:b]
            v = xb.dot(w)
            v = np.asarray(v).ravel()
            if normalize == "abs":
                s = float(np.mean(np.abs(v)))
            elif normalize == "pos":
                s = float(np.mean(np.clip(v, 0, None)))
            elif normalize == "neg":
                s = float(np.mean(np.clip(-v, 0, None)))
            else:
                s = float(np.mean(v))
            vals.append(s)

        total = sum(x for x in vals if np.isfinite(x))
        if total <= 1e-12:
            pct = [0.0] * len(vals)
        else:
            pct = [100.0 * x / total for x in vals]
        rows.append(pct)

    df_wide = pd.DataFrame(rows, columns=blocks, index=classes)
    return df_wide


# --------------------------- B2 (texte) par classe ---------------------------

def blocks_b2_per_class(
    model_path,
    data_csv,
    label_col="prdtypecode",
    max_n=3000,
    normalize="abs",
    label_map_json=None,
    out_csv="results/reports/block_importance_b2_per_class.csv",
    out_png="results/figures/block_importance_b2_per_class.png",
    topk_classes=30
):
    model = joblib.load(model_path)
    text_union = model.named_steps.get("text")
    clf = model.named_steps.get("clf")
    if text_union is None or clf is None:
        print("[WARN] pipeline B2 inattendue -> ignoré.")
        return False

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Colonne label absente: {label_col}")
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=0)

    X = text_union.transform(_ensure_input_df(df))
    slices = _block_slices_text(text_union, df)
    y = df[label_col].values

    df_wide = _block_contrib_per_class(C, X, y, slices, clf, normalize=normalize)

    counts = pd.Series(y).value_counts()
    keep = counts.index[:min(topk_classes, len(counts))].tolist()
    df_wide = df_wide.loc[[c for c in df_wide.index if c in keep]]

    label_map = None
    if label_map_json and Path(label_map_json).exists():
        try:
            import json
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None
    if label_map:
        df_wide.index = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in df_wide.index]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(out_csv, index=True)
    print(f"[OK] Par-classe B2 (CSV) -> {out_csv}")

    plt.figure(figsize=(10, max(6, 0.35 * len(df_wide))))
    left = np.zeros(len(df_wide))
    colors = plt.cm.tab20.colors
    for j, col in enumerate(df_wide.columns):
        plt.yticks(range(len(df_wide)), df_wide.index.tolist())
        vals = df_wide[col].values
        plt.barh(range(len(df_wide)), vals, left=left, label=col, color=colors[j % len(colors)])
        left += vals
    plt.gca().invert_yaxis()
    plt.xlabel("Part de contribution par bloc (%)")
    plt.ylabel("Classe (top par effectif)")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.title(f"Importance par bloc — B2 (par classe, {normalize})")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] Par-classe B2 (figure) -> {out_png}")
    return True


# --------------------------- B4 (multimodal) global ---------------------------

def blocks_b4(model_path, data_csv, max_n=3000, normalize="abs"):
    model = joblib.load(model_path)
    feat_union, scaler, clf, names = _find_feat_scaler_clf(model)
    if feat_union is None or clf is None:
        print("[WARN] pipeline B4 inattendue -> ignoré.")
        return False
    else:
        print(f"[INFO] Étapes détectées → features='{names['features']}', scaler='{names['scaler']}', clf='{names['clf']}'")

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    df = df.sample(n=min(len(df), max_n), random_state=0)

    X = feat_union.transform(_ensure_input_df(df))
    if scaler is not None:
        X = scaler.transform(X)

    slices = _block_slices_b4(feat_union, df)
    df_imp = _block_importance_from_coef(C, X, slices, clf)

    col_map = {"abs": "imp_abs", "signed": "imp_signed", "pos": "imp_pos", "neg": "imp_neg"}
    vcol = col_map[normalize]
    suffix = normalize

    out_csv = Path(f"results/reports/block_importance_b4_{suffix}.csv")
    out_png = Path(f"results/figures/block_importance_b4_{suffix}.png")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(out_csv, index=False)
    _save_barplot(df_imp, f"Importance par bloc — B4 ({normalize})", out_png, value_col=vcol)
    return True


# --------------------------- B4 (multimodal) par classe ---------------------------

def blocks_b4_per_class(
    model_path,
    data_csv,
    label_col="prdtypecode",
    max_n=3000,
    normalize="abs",
    label_map_json=None,
    out_csv="results/reports/block_importance_b4_per_class.csv",
    out_png="results/figures/block_importance_b4_per_class.png",
    topk_classes=30
):
    model = joblib.load(model_path)
    feat_union, scaler, clf, names = _find_feat_scaler_clf(model)
    if feat_union is None or clf is None:
        print("[WARN] pipeline B4 inattendue -> ignoré.")
        return False
    else:
        print(f"[INFO] Étapes détectées → features='{names['features']}', scaler='{names['scaler']}', clf='{names['clf']}'")

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Colonne label absente: {label_col}")
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=0)

    X = feat_union.transform(_ensure_input_df(df))
    if scaler is not None:
        X = scaler.transform(X)

    slices = _block_slices_b4(feat_union, df)
    y = df[label_col].values

    df_wide = _block_contrib_per_class(C, X, y, slices, clf, normalize=normalize)

    counts = pd.Series(y).value_counts()
    keep = counts.index[:min(topk_classes, len(counts))].tolist()
    df_wide = df_wide.loc[[c for c in df_wide.index if c in keep]]

    label_map = None
    if label_map_json and Path(label_map_json).exists():
        try:
            import json
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None
    if label_map:
        df_wide.index = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in df_wide.index]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(out_csv, index=True)
    print(f"[OK] Par-classe B4 (CSV) -> {out_csv}")

    plt.figure(figsize=(10, max(6, 0.35 * len(df_wide))))
    left = np.zeros(len(df_wide))
    colors = plt.cm.Paired.colors  # autre palette
    for j, col in enumerate(df_wide.columns):
        plt.yticks(range(len(df_wide)), df_wide.index.tolist())
        vals = df_wide[col].values
        plt.barh(range(len(df_wide)), vals, left=left, label=col, color=colors[j % len(colors)])
        left += vals
    plt.gca().invert_yaxis()
    plt.xlabel("Part de contribution par bloc (%)")
    plt.ylabel("Classe (top par effectif)")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.title(f"Importance par bloc — B4 (par classe, {normalize})")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] Par-classe B4 (figure) -> {out_png}")
    return True

def blocks_b4_fine_global(
    model_path, data_csv, label_col="prdtypecode", max_n=3000,
    out_dir="results"
):
    import pandas as pd
    from pathlib import Path

    model = joblib.load(model_path)
    feat_union, scaler, clf, names = _find_feat_scaler_clf(model)
    if feat_union is None or clf is None:
        print("[WARN] pipeline B4 inattendue -> ignoré.")
        return False
    else:
        print(f"[INFO] Étapes détectées → features='{names['features']}', scaler='{names['scaler']}', clf='{names['clf']}'")

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Colonne label absente: {label_col}")
    if max_n and len(df) > max_n:
        df = df.sample(n=max_n, random_state=0)

    # features concaténées
    X = feat_union.transform(_ensure_input_df(df))
    if scaler is not None:
        X = scaler.transform(X)

    slices = _block_slices_b4_fine(feat_union, df)

    # importance moyenne par bloc via coef × x
    df_imp = _block_importance_from_coef(C, X, slices, clf)  # imp_abs / imp_pos / imp_neg

    # tri décroissant sur abs
    df_imp = df_imp.sort_values("imp_abs", ascending=False)

    Path(out_dir, "figures").mkdir(parents=True, exist_ok=True)
    out_png = str(Path(out_dir, "figures", "block_importance_b4_fine.png"))

    _save_barplot(
        df_imp,
        title="Importance par bloc — B4 (multimodal, découpage fin)",
        out_png=out_png,
        value_col="imp_abs")   # on trace l’importance moyenne |x·w|
    print(f"[OK] Importance B4 (fin) → {out_png}")
    return True

def _plot_per_class_signed(df_pos, df_neg, title, out_png):
    """Dessine des barres horizontales divergeantes (+ à droite, − à gauche)."""
    import matplotlib.pyplot as plt
    import numpy as np
    n = len(df_pos)
    classes = df_pos.index.tolist()
    blocks = df_pos.columns.tolist()

    plt.figure(figsize=(10, max(6, 0.35 * n)))

    colors = plt.cm.tab20.colors
    # cumul séparé pour + et -
    left_pos = np.zeros(n)
    left_neg = np.zeros(n)

    for j, b in enumerate(blocks):
        vp = df_pos[b].values            # pourcentage +
        vn = df_neg[b].values            # pourcentage −
        c  = colors[j % len(colors)]

        # positif → à droite
        plt.barh(range(n), vp, left=left_pos, color=c, label=b)
        left_pos += vp

        # négatif → à gauche (valeur négative), plus clair
        plt.barh(range(n), -vn, left=-left_neg, color=c, alpha=0.35)
        left_neg += vn

    plt.axvline(0, color="#334155", lw=1)
    plt.yticks(range(n), classes)        # noms de classes
    plt.gca().invert_yaxis()
    plt.xlabel("Part de contribution par bloc (%) — + à droite, − à gauche")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.title(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] {title} -> {out_png}")

# --------------------------- B2 (texte) par classe, signé (+ et − séparés) ---------------------------
def blocks_b2_per_class_signed(
    model_path,
    data_csv,
    label_col="prdtypecode",
    max_n=3000,
    label_map_json=None,
    out_dir="results",
    topk_classes=30,
    shared_scale=False,
):
    """
    B2 — Contributions SIGNÉES (+ à droite / − à gauche) par classe,
    agrégées par sous-blocs texte (TF-IDF word, TF-IDF char, stats, …).
    Génère :
      - results/reports/block_importance_b2_per_class_pos.csv
      - results/reports/block_importance_b2_per_class_neg.csv
      - results/figures/block_importance_b2_per_class_signed.png
    """
    import json
    from pathlib import Path

    model = joblib.load(model_path)
    # B2 = Pipeline([("text", <FeatureUnion texte>), ("clf", ...)])
    text_union = getattr(model, "named_steps", {}).get("text")
    clf        = getattr(model, "named_steps", {}).get("clf")
    if text_union is None or clf is None:
        print("[WARN] pipeline B2 inattendue -> ignoré.")
        return False

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Colonne label absente: {label_col}")
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=0)

    # features texte + slices de blocs internes (tfidf_word, tfidf_char, …)
    X = text_union.transform(_ensure_input_df(df))
    slices = _block_slices_text(text_union, df)
    y = df[label_col].values

    # % de contribution + / − par classe et par bloc
    df_pos = _block_contrib_per_class(C, X, y, slices, clf, normalize="pos")
    df_neg = _block_contrib_per_class(C, X, y, slices, clf, normalize="neg")

    # garder les classes les + fréquentes (comme B4)
    counts = pd.Series(y).value_counts()
    keep = counts.index[:min(topk_classes, len(counts))].tolist()
    df_pos = df_pos.loc[[c for c in df_pos.index if c in keep]]
    df_neg = df_neg.loc[df_pos.index]

    # labels lisibles optionnels
    label_map = None
    if label_map_json and Path(label_map_json).exists():
        try:
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None
    if label_map:
        new_index = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in df_pos.index]
        df_pos.index = new_index
        df_neg.index = new_index

    Path(out_dir, "reports").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "figures").mkdir(parents=True, exist_ok=True)
    df_pos.to_csv(Path(out_dir, "reports", "block_importance_b2_per_class_pos.csv"))
    df_neg.to_csv(Path(out_dir, "reports", "block_importance_b2_per_class_neg.csv"))

    _plot_per_class_signed(
        df_pos, df_neg,
        "Importance par bloc — B2 (par classe, impact signé)",
        str(Path(out_dir, "figures", "block_importance_b2_per_class_signed.png")),
        shared_scale=shared_scale
    )
    return True


def blocks_b2_per_class_signed_mag(
    model_path,
    data_csv,
    label_col="prdtypecode",
    max_n=3000,
    label_map_json=None,
    out_dir="results",
    topk_classes=30,
    shared_scale=True,
    sort_by="none",
):
    """
    B2 — Contributions signées en MAGNITUDE (pas en %) par classe,
    agrégées par sous-blocs texte. Échelle linéaire du modèle (x·w).
    Génère :
      - results/reports/block_importance_b2_per_class_pos_mag.csv
      - results/reports/block_importance_b2_per_class_neg_mag.csv
      - results/figures/block_importance_b2_per_class_signed_mag.png
    """
    import json
    from pathlib import Path

    model = joblib.load(model_path)
    text_union = getattr(model, "named_steps", {}).get("text")
    clf        = getattr(model, "named_steps", {}).get("clf")
    if text_union is None or clf is None:
        print("[WARN] pipeline B2 inattendue -> ignoré.")
        return False

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Colonne label absente: {label_col}")
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=0)

    X = text_union.transform(_ensure_input_df(df))
    slices = _block_slices_text(text_union, df)
    y = df[label_col].values

    # magnitudes (pas de normalisation en %)
    dfp, dfn = _block_contrib_per_class_magnitude(C, X, y, slices, clf)

    counts = pd.Series(y).value_counts()
    keep = counts.index[:min(topk_classes, len(counts))].tolist()
    dfp = dfp.loc[[c for c in dfp.index if c in keep]]
    dfn = dfn.loc[dfp.index]

    # tri optionnel (par pos, neg, net, total)
    dfp, dfn = _sort_per_class_frames(dfp, dfn, sort_by=sort_by)

    label_map = None
    if label_map_json and Path(label_map_json).exists():
        try:
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None

    Path(out_dir, "reports").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "figures").mkdir(parents=True, exist_ok=True)
    dfp.to_csv(Path(out_dir, "reports", "block_importance_b2_per_class_pos_mag.csv"))
    dfn.to_csv(Path(out_dir, "reports", "block_importance_b2_per_class_neg_mag.csv"))

    _plot_per_class_signed_magnitude(
        dfp, dfn,
        "Importance par bloc — B2 (par classe, impact signé, magnitude)",
        str(Path(out_dir, "figures", "block_importance_b2_per_class_signed_mag.png")),
        label_map=label_map, shared_scale=shared_scale
    )
    return True

# --------------------------- B3 (image seule) global ---------------------------

def blocks_b3(model_path, data_csv, out_csv="results/reports/blocks_b3.csv"):
    """
    Importance globale des blocs pour B3 (image seule) :
    - 'img' (pixels ou CNN)
    - + éventuellement 'image_stats_combined' si activé dans B3
    """
    import joblib, pandas as pd, numpy as np, os
    model = joblib.load(model_path)
    img_union = _get_img_union_from_model(model)
    if img_union is None or not hasattr(img_union, "transformer_list"):
        print("[WARN] Modèle B3 inattendu (pas de step 'img' FeatureUnion) — ignoré.")
        return

    df = _ensure_input_df(data_csv)
    need_cols = ["productid", "imageid"]
    X_img = img_union.transform(df[need_cols])

    # Construire les slices bloc par bloc
    slices = {}
    start = 0
    for name, tr in img_union.transformer_list:
        Xi = tr.transform(df[need_cols])
        n = Xi.shape[1]
        slices[name] = slice(start, start + n)
        start += n

    C, _ = _coef_matrix(model)  # (n_classes, n_features_total)
    imp = _block_importance_from_coef(C, slices)  # déjà dispo dans ton script

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.Series(imp).sort_values(ascending=False).to_csv(out_csv, index_label="block", header=["importance"])
    print(f"[OK] Blocks B3 → {out_csv}")

# --------------------------- B4 (multimodal) par classe, signé (+ et − séparés) ---------------------------
def blocks_b4_per_class_signed(
    model_path,
    data_csv,
    label_col="prdtypecode",
    max_n=3000,
    label_map_json=None,
    out_dir="results",
    topk_classes=30,
    shared_scale=False
):
    model = joblib.load(model_path)
    feat_union, scaler, clf, names = _find_feat_scaler_clf(model)
    if feat_union is None or clf is None:
        print("[WARN] pipeline B4 inattendue -> ignoré.")
        return False
    else:
        print(f"[INFO] Étapes détectées → features='{names['features']}', scaler='{names['scaler']}', clf='{names['clf']}'")

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Colonne label absente: {label_col}")
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=0)

    X = feat_union.transform(_ensure_input_df(df))
    if scaler is not None:
        X = scaler.transform(X)

    slices = _block_slices_b4(feat_union, df)
    y = df[label_col].values

    df_pos = _block_contrib_per_class(C, X, y, slices, clf, normalize="pos")
    df_neg = _block_contrib_per_class(C, X, y, slices, clf, normalize="neg")

    counts = pd.Series(y).value_counts()
    keep = counts.index[:min(topk_classes, len(counts))].tolist()
    df_pos = df_pos.loc[[c for c in df_pos.index if c in keep]]
    df_neg = df_neg.loc[df_pos.index]

    label_map = None
    if label_map_json and Path(label_map_json).exists():
        import json
        try:
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None
    if label_map:
        new_index = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in df_pos.index]
        df_pos.index = new_index
        df_neg.index = new_index

    Path(out_dir, "reports").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "figures").mkdir(parents=True, exist_ok=True)
    df_pos.to_csv(Path(out_dir, "reports", "block_importance_b4_per_class_pos.csv"))
    df_neg.to_csv(Path(out_dir, "reports", "block_importance_b4_per_class_neg.csv"))
    _plot_per_class_signed(
        df_pos, df_neg,
        "Importance par bloc — B4 (par classe, impact signé)",
        str(Path(out_dir, "figures", "block_importance_b4_per_class_signed.png")),
        shared_scale=shared_scale
    )
    return True

# --------------------------- plotting signed per class helper ---------------------------
def _plot_per_class_signed(df_pos, df_neg, title, out_png, shared_scale=False):
    import matplotlib.pyplot as plt
    import numpy as np
    n = len(df_pos)
    classes = df_pos.index.tolist()
    blocks = df_pos.columns.tolist()

    plt.figure(figsize=(10, max(6, 0.35 * n)))

    colors = plt.cm.tab20.colors
    left_pos = np.zeros(n)   # cumul côté +
    left_neg = np.zeros(n)   # cumul côté −

    for j, b in enumerate(blocks):
        vp = df_pos[b].values            # pourcentage +
        vn = df_neg[b].values            # pourcentage −
        c  = colors[j % len(colors)]

        plt.barh(range(n), vp,  left=left_pos,      color=c, label=b)   # + à droite
        left_pos += vp

        plt.barh(range(n), -vn, left=-left_neg,     color=c, alpha=0.35) # − à gauche
        left_neg += vn

    plt.axvline(0, color="#334155", lw=1)
    if shared_scale:
        # on travaille en % donc on peut figer l’axe à [-100, +100]
        plt.xlim(-100, 100)

    plt.yticks(range(n), classes)
    plt.gca().invert_yaxis()
    plt.xlabel("Part de contribution par bloc (%) — + à droite, − à gauche")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.title(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] {title} -> {out_png}")

# --- NEW: per-class block contributions, absolute magnitudes (not normalized) ---
def _block_contrib_per_class_magnitude(C, X, y, slices, clf):
    """
    Returns two DataFrames (index=classes, columns=block names):
      df_pos[b,k] = mean( max(x_b · w_b, 0) ) for class k
      df_neg[b,k] = mean( max(-(x_b · w_b), 0) ) for class k
    No normalization -> values are on the model's linear score scale.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    classes = list(clf.classes_)
    class_to_row = {c: i for i, c in enumerate(classes)}
    block_names = [name for name, _, _ in slices]

    df_pos = pd.DataFrame(0.0, index=classes, columns=block_names)
    df_neg = pd.DataFrame(0.0, index=classes, columns=block_names)

    # Make sure X is CSR for fast slicing/dot
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    for k in classes:
        kidx = np.where(y == k)[0]
        if kidx.size == 0:
            continue
        row = class_to_row[k]
        w = np.asarray(C[row, :]).ravel()
        for (name, a, b) in slices:
            xb = X[kidx, a:b]                 # (nk, d_b)
            wb = w[a:b]                       # (d_b,)
            contrib = xb.dot(wb)              # (nk,)
            contrib = np.asarray(contrib).ravel()
            pos = np.clip(contrib, 0, None).mean()
            neg = np.clip(-contrib, 0, None).mean()
            df_pos.loc[k, name] = float(pos)
            df_neg.loc[k, name] = float(neg)

    return df_pos, df_neg

# --------------------------- B2 (texte) par classe, signé (magnitude) ---------------------------
def _plot_per_class_signed_magnitude(df_pos, df_neg, title, out_png, label_map=None, shared_scale=True):
    """
    Mirror stacked bars where width equals absolute magnitude (not %).
    Left = negative, Right = positive. Colors = blocks.
    If shared_scale=True, x-axis is symmetric [-M, +M] with M=max total magnitude across classes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Optional human labels
    if label_map:
        new_idx = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in df_pos.index]
        df_pos.index = new_idx
        df_neg.index = new_idx

    n = len(df_pos)
    blocks = list(df_pos.columns)

    total_pos = df_pos.sum(axis=1).values
    total_neg = df_neg.sum(axis=1).values
    M = max(float(total_pos.max()), float(total_neg.max()), 1e-9)

    colors = plt.cm.tab20.colors
    plt.figure(figsize=(10, max(6, 0.35 * n)))
    left_pos = np.zeros(n)
    left_neg = np.zeros(n)

    for j, b in enumerate(blocks):
        vp = df_pos[b].values
        vn = df_neg[b].values
        c = colors[j % len(colors)]

        # + to the right
        plt.barh(range(n), vp, left=left_pos, color=c, label=b)
        left_pos += vp

        # - to the left
        plt.barh(range(n), -vn, left=-left_neg, color=c, alpha=0.35)
        left_neg += vn

    plt.axvline(0, color="#334155", lw=1)
    if shared_scale:
        plt.xlim(-M * 1.05, M * 1.05)

    plt.yticks(range(n), df_pos.index.tolist())
    plt.gca().invert_yaxis()
    plt.xlabel("Contribution moyenne au score linéaire (|x·w|)")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.title(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] {title} -> {out_png}")

#   
def _sort_per_class_frames(dfp, dfn, sort_by="none"):
    """
    Trie dfp/dfn (index=classes, colonnes=blocs) selon la métrique choisie.
    - 'pos'  : somme des contributions positives
    - 'neg'  : somme des contributions négatives
    - 'net'  : pos - neg
    - 'total': pos + neg
    - 'none' : pas de tri
    """
    if sort_by == "none":
        return dfp, dfn

    s_pos = dfp.sum(axis=1)
    s_neg = dfn.sum(axis=1)
    metric = {
        "pos":   s_pos,
        "neg":   s_neg,
        "net":   (s_pos - s_neg),
        "total": (s_pos + s_neg),
    }[sort_by]

    order = metric.sort_values(ascending=False).index
    return dfp.loc[order], dfn.loc[order]

# --------------------------- B4 (multimodal) par classe, signé (magnitude) ---------------------------
def blocks_b4_per_class_signed_magnitude(
    model_path, data_csv, label_col="prdtypecode", max_n=3000,
    label_map_json=None, out_dir="results", topk_classes=30, shared_scale=True, sort_by="none"
):
    import pandas as pd, json
    from pathlib import Path
    model = joblib.load(model_path)
    feat_union, scaler, clf, names = _find_feat_scaler_clf(model)
    if feat_union is None or clf is None:
        print("[WARN] pipeline B4 inattendue -> ignoré.")
        return False
    else:
        print(f"[INFO] Étapes détectées → features='{names['features']}', scaler='{names['scaler']}', clf='{names['clf']}'")

    C = _coef_matrix(clf)
    if C is None:
        print("[WARN] Classifieur sans coef_ -> ignoré.")
        return False

    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Colonne label absente: {label_col}")
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=0)

    X = feat_union.transform(_ensure_input_df(df))
    if scaler is not None:
        X = scaler.transform(X)

    slices = _block_slices_b4_fine(feat_union, df)
    y = df[label_col].values

    dfp, dfn = _block_contrib_per_class_magnitude(C, X, y, slices, clf)

    counts = pd.Series(y).value_counts()
    keep = counts.index[:min(topk_classes, len(counts))].tolist()
    dfp = dfp.loc[[c for c in dfp.index if c in keep]]
    dfn = dfn.loc[dfp.index]

    dfp, dfn = _sort_per_class_frames(dfp, dfn, sort_by=sort_by)

    label_map = None
    if label_map_json and Path(label_map_json).exists():
        try:
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None

    Path(out_dir, "reports").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "figures").mkdir(parents=True, exist_ok=True)
    dfp.to_csv(Path(out_dir, "reports", "block_importance_b4_per_class_pos_mag.csv"))
    dfn.to_csv(Path(out_dir, "reports", "block_importance_b4_per_class_neg_mag.csv"))

    _plot_per_class_signed_magnitude(
        dfp, dfn,
        "Importance par bloc — B4 (par classe, impact signé, magnitude)",
        str(Path(out_dir, "figures", "block_importance_b4_per_class_signed_mag.png")),
        label_map=label_map, shared_scale=shared_scale
    )
    return True

# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["b2", "b3", "b4"],
                        help="Baseline à analyser")
    parser.add_argument("--model", default=None,
                        help="Chemin du pipeline .joblib (optionnel, requis pour SHAP et importances)")
    parser.add_argument("--data-csv", default="data/X_train.csv",
                        help="CSV contenant au moins les colonnes texte")
    parser.add_argument("--text-col", default="designation",
                        help="Nom de la colonne texte principale (info)")
    parser.add_argument("--label-col", default="prdtypecode",
                        help="Nom de la colonne label (pour rapports par classe)")
    parser.add_argument("--max-sample", type=int, default=8000,
                        help="Taille d’échantillon max pour ACP/SHAP")
    parser.add_argument("--topk", type=int, default=20,
                        help="Nombre de confusions à reporter")
    parser.add_argument("--blocks-b3", action="store_true", help="Importance des blocs pour B3 (image seule)")

    # Nouveaux switches d'importances
    parser.add_argument("--normalize", choices=["abs", "signed", "pos", "neg"],
                        default="abs",
                        help="Agrégation des contributions par bloc.")
    parser.add_argument("--both-global", action="store_true",
                        help="Génère deux graphes globaux (abs et signed).")
    parser.add_argument("--blocks-b2", action="store_true",
                        help="Importance par bloc — B2 (global).")
    parser.add_argument("--blocks-b4", action="store_true",
                        help="Importance par bloc — B4 (global).")
    parser.add_argument("--blocks-b2-per-class", action="store_true",
                        help="Importance par bloc — B2 (par classe).")
    parser.add_argument("--blocks-b4-per-class", action="store_true",
                        help="Importance par bloc — B4 (par classe).")
    parser.add_argument("--label-map", default="labels_map.json",
                        help="(optionnel) JSON pour libellés lisibles.")
    parser.add_argument("--blocks-b2-per-class-signed", action="store_true",
                    help="Importance par bloc — B2 (par classe, signé).")
    parser.add_argument("--blocks-b4-per-class-signed", action="store_true",
                    help="Importance par bloc — B4 (par classe, signé).")
    parser.add_argument("--signed-shared-scale", action="store_true",
                        help="Pour les graphes signés par classe, force une échelle symétrique partagée (ex. -100%..+100%).")
    parser.add_argument("--blocks-b2-per-class-signed-mag", action="store_true",
                        help="B2: par classe, barres divergentes avec magnitude réelle (axe partagé).")
    parser.add_argument("--blocks-b4-per-class-signed-mag", action="store_true",
                        help="B4: par classe, barres divergentes avec magnitude réelle (axe partagé).")
    parser.add_argument("--sort-by", choices=["none", "pos", "neg", "net", "total"], default="none", help=("Tri des classes sur les graphes signés (magnitude réelle) : "
                                                                                                           "'pos' (somme des impacts +), 'neg' (somme des impacts −), "
                                                                                                           "'net' (pos−neg), 'total' (pos+neg), 'none' (ordre original)."))
    parser.add_argument("--blocks-b4-fine", action="store_true", help="Importance globale par bloc (B4) avec découpage fin (texte détaillé + pixels + CNN + stats détaillées).")
    parser.add_argument("--blocks-b4-per-class-signed-mag-fine", action="store_true", help="B4 par classe, barres divergentes (magnitude réelle) avec découpage fin.")

    args = parser.parse_args()

    ensure_dirs()
    lblmap = load_labels_map(args.label_map)

    # ACP + confusions (toujours)
    do_acp(args.kind, max_n=args.max_sample)
    top_confusions(args.kind, topk=args.topk, labels_map=lblmap)

    # SHAP si modèle fourni (texte global)
    if args.model:
        do_shap(args.kind, args.model, data_csv=args.data_csv,
                text_col=args.text_col, label_col=args.label_col,
                max_n=min(args.max_sample, 30000), topk=30)
    else:
        print("[INFO] --model non fourni -> skip SHAP.")

    # Importances par bloc (B2)
    if args.blocks_b2 and args.model:
        if args.both_global:
            blocks_b2(args.model, args.data_csv, max_n=min(args.max_sample, 30000), normalize="abs")
            blocks_b2(args.model, args.data_csv, max_n=min(args.max_sample, 30000), normalize="signed")
        else:
            blocks_b2(args.model, args.data_csv, max_n=min(args.max_sample, 30000), normalize=args.normalize)

    # Importances par bloc (B4)
    if args.blocks_b4 and args.model:
        if args.both_global:
            blocks_b4(args.model, args.data_csv, max_n=min(args.max_sample, 30000), normalize="abs")
            blocks_b4(args.model, args.data_csv, max_n=min(args.max_sample, 30000), normalize="signed")
        else:
            blocks_b4(args.model, args.data_csv, max_n=min(args.max_sample, 30000), normalize=args.normalize)

    # Par classe
    if args.blocks_b2_per_class and args.model:
        blocks_b2_per_class(args.model, args.data_csv, label_col=args.label_col,
                            max_n=min(args.max_sample, 30000),
                            normalize=args.normalize,
                            label_map_json=args.label_map)

    if args.blocks_b4_per_class and args.model:
        blocks_b4_per_class(args.model, args.data_csv, label_col=args.label_col,
                            max_n=min(args.max_sample, 30000),
                            normalize=args.normalize,
                            label_map_json=args.label_map)
    if args.blocks_b2_per_class_signed and args.model:
        blocks_b2_per_class_signed(args.model, args.data_csv,
                                   label_col=args.label_col, max_n=min(args.max_sample, 30000),
                                   label_map_json=args.label_map)

    if args.blocks_b4_per_class_signed and args.model:
        blocks_b4_per_class_signed(args.model, args.data_csv, label_col=args.label_col, 
                                   max_n=min(args.max_sample, 30000), label_map_json=args.label_map)
    if args.blocks_b2_per_class_signed and args.model: blocks_b2_per_class_signed(
        args.model, args.data_csv,
        label_col=args.label_col,
        max_n=min(args.max_sample, 30000),
        label_map_json=args.label_map,
        shared_scale=args.signed_shared_scale
    )
    if args.blocks_b4_per_class_signed and args.model: blocks_b4_per_class_signed(
        args.model, args.data_csv,
        label_col=args.label_col,
        max_n=min(args.max_sample, 30000),
        label_map_json=args.label_map,
        shared_scale=args.signed_shared_scale
    )
        
    if args.blocks_b2_per_class_signed_mag and args.model:
        blocks_b2_per_class_signed_mag(
        args.model, args.data_csv,
        label_col=args.label_col,
        max_n=min(args.max_sample, 30000),
        label_map_json=args.label_map,
        shared_scale=True,                 # axe [-M, +M]
        sort_by=args.sort_by               # <<< NOUVEAU
    )

    if args.blocks_b4_per_class_signed_mag and args.model:
        blocks_b4_per_class_signed_magnitude(
        args.model, args.data_csv,
        label_col=args.label_col,
        max_n=min(args.max_sample, 30000),
        label_map_json=args.label_map,
        shared_scale=True,
        sort_by=args.sort_by
    )
        
    if args.blocks_b4_fine and args.model:
        blocks_b4_fine_global(
        args.model, args.data_csv,
        label_col=args.label_col,
        max_n=args.max_sample
    )
        
    if args.kind == "b3" and args.blocks_b3:
        blocks_b3(args.model, args.data_csv)
        
    if getattr(args, "blocks_b4_per_class_signed_mag_fine", False) and args.model:
        blocks_b4_per_class_signed_magnitude(
        args.model, args.data_csv,
        label_col=args.label_col,
        max_n=args.max_sample,
        label_map_json=args.label_map,
        shared_scale=True,
        sort_by=args.sort_by
    )


if __name__ == "__main__":
    main()
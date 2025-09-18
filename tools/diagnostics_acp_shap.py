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
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

# ------------------------------------------------------------------
# SysPath pour importer éventuellement main/ et features/
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "main", ROOT / "features"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# ------------------------------------------------------------------
# Utils de base

def ensure_dirs():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/reports").mkdir(parents=True, exist_ok=True)


def _ensure_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les colonnes pertinentes si elles existent (évite KeyError)."""
    keep = [c for c in ("designation", "description", "productid", "imageid") if c in df.columns]
    return df[keep] if keep else df


def _ncols(transformer, df_small: pd.DataFrame) -> int:
    """Nombre de colonnes en sortie d'un transformer déjà *fit* (sans refit)."""
    Xs = transformer.transform(_ensure_input_df(df_small.head(5)))
    return int(Xs.shape[1])


def _safe_dim(transformer, df_small: pd.DataFrame) -> int:
    """Alias _ncols (compat)."""
    return _ncols(transformer, df_small)


def _coef_matrix(clf):
    """Retourne la matrice des poids (n_classes, n_features) pour modèle linéaire / OneVsRest."""
    if hasattr(clf, "coef_"):
        C = clf.coef_
        try:
            return np.asarray(C) if not hasattr(C, "toarray") else C.toarray()
        except Exception:
            return np.asarray(C)
    # OneVsRestClassifier
    if hasattr(clf, "estimators_"):
        mats = []
        for est in clf.estimators_:
            if hasattr(est, "coef_"):
                CC = est.coef_
                CC = np.asarray(CC) if not hasattr(CC, "toarray") else CC.toarray()
                mats.append(CC[0] if CC.ndim == 2 and CC.shape[0] == 1 else CC)
        if len(mats):
            return np.vstack(mats)
    return None


def _get_linear_weights_and_bias(clf):
    """(W, b) où W shape=(n_classes, n_features)."""
    W = _coef_matrix(clf)
    b = getattr(clf, "intercept_", None)
    if b is None and hasattr(clf, "estimators_"):
        # OneVsRest
        bs = []
        for est in clf.estimators_:
            bs.append(getattr(est, "intercept_", np.zeros((1,))))
        try:
            b = np.array(bs).ravel()
        except Exception:
            b = None
    return W, b


def _winner_per_row(clf, X):
    """Indice de la classe gagnante par ligne, via decision_function si dispo."""
    if hasattr(X, "tocsr"):
        X = X.tocsr()
    if hasattr(clf, "decision_function"):
        S = clf.decision_function(X)
        if S.ndim == 1:
            S = S[:, None]
        return np.argmax(S, axis=1)
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(X)
        if isinstance(P, list):  # OvR style
            P = np.column_stack(P)
        return np.argmax(P, axis=1)
    pred = clf.predict(X)
    if hasattr(clf, "classes_"):
        inv = {c: i for i, c in enumerate(clf.classes_)}
        return np.array([inv.get(p, 0) for p in pred])
    return np.zeros(X.shape[0], dtype=int)


def _walk_estimators(est):
    """Itère récursivement dans les objets sklearn (pipelines, unions, wrappers)."""
    yield est
    if hasattr(est, "steps"):
        for _, s in getattr(est, "steps", []):
            yield from _walk_estimators(s)
    if hasattr(est, "transformer_list"):
        for _, t in getattr(est, "transformer_list", []):
            yield from _walk_estimators(t)
    for attr in ("estimator", "base_estimator"):
        if hasattr(est, attr):
            yield from _walk_estimators(getattr(est, attr))


def _find_feat_scaler_clf(model):
    """
    Retourne (features_union, scaler, clf, names_dict)
    Recherche robuste par nom & type plutôt que de supposer des noms fixes.
    """
    feat_union = None
    scaler = None
    clf = None
    names = {"features": None, "scaler": None, "clf": None}

    if hasattr(model, "named_steps"):
        feat_union = model.named_steps.get("features")
        scaler = model.named_steps.get("scaler")
        clf = model.named_steps.get("clf") or model.named_steps.get("model") or model.named_steps.get("final_estimator")
        if feat_union is not None: names["features"] = "features"
        if scaler is not None: names["scaler"] = "scaler"
        if clf is not None: names["clf"] = "clf/model/final_estimator"

    if isinstance(getattr(model, "steps", None), list):
        for nm, step in model.steps:
            if feat_union is None and (isinstance(step, FeatureUnion) or isinstance(step, ColumnTransformer) or hasattr(step, "transformer_list")):
                feat_union = step; names["features"] = nm
            if scaler is None and nm.lower() in ("scaler", "standardscaler"):
                scaler = step; names["scaler"] = nm
            if clf is None and (hasattr(step, "coef_") or hasattr(step, "classes_") or hasattr(step, "decision_function") or hasattr(step, "predict_proba")):
                clf = step; names["clf"] = nm

    return feat_union, scaler, clf, names


def _unwrap_feature_union(obj):
    """Retourne récursivement une FeatureUnion si obj est enveloppé (Pipeline, etc.)."""
    if hasattr(obj, "transformer_list"):
        return obj
    if hasattr(obj, "named_steps"):
        for key in ("text", "features", "union", "tfidf_word", "tfidf"):
            step = obj.named_steps.get(key)
            if step is not None:
                fu = _unwrap_feature_union(step)
                if hasattr(fu, "transformer_list"):
                    return fu
    return obj


def _block_slices_text(text_union, df_small: pd.DataFrame):
    """
    Détaille la branche texte en sous-blocs lisibles :
    TF-IDF word, TF-IDF char, HasDescription, TitleLength, TextStats, TextStatsPro, Lexicon, …
    Gère le cas ('tfidf_word', FeatureUnion(...)).
    """
    text_union = _unwrap_feature_union(text_union)

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
        # Sous-union 'tfidf_word' (elle-même une FeatureUnion)
        if name == "tfidf_word" and hasattr(tr, "transformer_list"):
            inner = start
            for n2, tr2 in tr.transformer_list:
                w = _ncols(tr2, df_small)
                lbl = label_map.get(n2, f"tfidf_word/{n2}")
                slices.append((lbl, inner, inner + w))
                inner += w
            start = inner
            continue

        # Niveau 1
        w = _ncols(tr, df_small)
        lbl = label_map.get(name, name)
        slices.append((lbl, start, start + w))
        start += w

    return slices


def _block_slices_b4(feat_union, df_small: pd.DataFrame):
    """Blocs *coarse* pour B4 : Texte / CNN / Pixels / Stats(images)."""
    pretty = {"text": "Texte", "image_pixels": "Pixels", "image_cnn": "CNN",
              "image_stats": "Stats", "image_stats_combined": "Stats images"}
    slices = []
    start = 0
    for name, tr in getattr(feat_union, "transformer_list", []):
        if tr is None:
            continue
        w = _ncols(tr, df_small)
        if w <= 0:
            continue
        slices.append((pretty.get(name, name), start, start + w))
        start += w
    return slices


def _block_slices_b4_fine(feat_union, df_small: pd.DataFrame):
    """
    Découpage *fin* multi-modal :
      - texte : sous-blocs de _block_slices_text
      - pixels : 1 bloc
      - cnn    : 1 bloc
      - stats  : chaque dimension (nommée si connu, sinon stat_i)
    """
    slices = []
    start = 0
    for name, tr in getattr(feat_union, "transformer_list", []):
        if tr is None:
            continue

        # TEXTE → sous-découpe
        if "text" in name.lower():
            sub = _block_slices_text(tr, df_small)
            for subname, a, b in sub:
                slices.append((subname, start + a, start + b))
            start += _safe_dim(tr, df_small)
            continue

        # PIXELS
        if name.lower() in ("pixels", "pix", "img_pixels", "image_flat"):
            w = _safe_dim(tr, df_small)
            if w > 0:
                slices.append(("Pixels", start, start + w))
                start += w
            continue

        # CNN
        if "cnn" in name.lower():
            w = _safe_dim(tr, df_small)
            if w > 0:
                slices.append(("CNN", start, start + w))
                start += w
            continue

        # STATS basiques
        if name.lower() in ("stats", "img_stats", "image_stats"):
            w = _safe_dim(tr, df_small)
            if w > 0:
                base = ["Img:width", "Img:height", "Img:occupancy", "Img:white_ratio", "Img:black_ratio"]
                names = base if w == 5 else [f"Img:stat_{i}" for i in range(w)]
                for i in range(w):
                    slices.append((names[i], start + i, start + i + 1))
                start += w
            continue

        # STATS combinées (basic+pro)
        if name.lower() in ("stats_combined", "image_stats_combined"):
            w = _safe_dim(tr, df_small)
            if w > 0:
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

        # fallback : bloc inconnu
        w = _safe_dim(tr, df_small)
        if w > 0:
            slices.append((f"{name}", start, start + w))
            start += w

    return slices


def _block_importance_from_coef(C, X, slices, clf):
    """
    Contribution moyenne par bloc, sur la *classe gagnante* de chaque ligne.
    Renvoie DataFrame : block, imp_abs, imp_pos, imp_neg, imp_signed.
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


def _block_contrib_per_class(C, X, y_true, slices, clf, normalize="abs"):
    """
    Pour chaque classe (ordre clf.classes_), contribution moyenne par bloc :
      - normalize="abs"/"pos"/"neg"/"signed"
    Sortie : DataFrame (index=classes, colonnes=blocs) avec des % (somme 100) si normalize != "mag".
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

    return pd.DataFrame(rows, columns=blocks, index=classes)


def _block_contrib_per_class_magnitude(C, X, y_true, slices, clf):
    """
    Par classe : renvoie deux DataFrames (pos, neg) avec contributions
    **en magnitude** (échelle linéaire) séparées par signe.
    """
    if hasattr(X, "tocsr"):
        X = X.tocsr()
    classes = list(getattr(clf, "classes_", range(C.shape[0])))
    blocks = [lbl for (lbl, _, _) in slices]

    rows_pos, rows_neg = [], []

    for k, cls in enumerate(classes):
        mask = (y_true == cls)
        if mask.sum() == 0:
            rows_pos.append([0.0] * len(blocks))
            rows_neg.append([0.0] * len(blocks))
            continue

        pos_vals, neg_vals = [], []
        for (lbl, a, b) in slices:
            xb = X[mask, a:b]
            w = C[k, a:b]
            v = np.asarray(xb.dot(w)).ravel()
            pos_vals.append(float(np.mean(np.clip(v, 0, None))))
            neg_vals.append(float(np.mean(np.clip(-v, 0, None))))
        rows_pos.append(pos_vals)
        rows_neg.append(neg_vals)

    dfp = pd.DataFrame(rows_pos, columns=blocks, index=classes)
    dfn = pd.DataFrame(rows_neg, columns=blocks, index=classes)
    return dfp, dfn


def _sort_per_class_frames(dfp, dfn, sort_by="none"):
    """
    Trie (df_pos, df_neg) par 'pos' / 'neg' / 'net' / 'total' en magnitude.
    """
    if sort_by not in {"pos", "neg", "net", "total", "none"}:
        return dfp, dfn
    if sort_by == "none":
        return dfp, dfn

    def _key(row):
        if sort_by == "pos":
            return np.sum(row.values)
        if sort_by == "neg":
            return np.sum(row.values)
        if sort_by == "net":
            return np.sum(dfp.loc[row.name].values) - np.sum(dfn.loc[row.name].values)
        if sort_by == "total":
            return np.sum(dfp.loc[row.name].values) + np.sum(dfn.loc[row.name].values)
        return 0.0

    order = sorted(dfp.index.tolist(), key=lambda c: _key(dfp.loc[c]), reverse=True)
    return dfp.loc[order], dfn.loc[order]


# ------------------------------------------------------------------
# Plot helpers

def _save_barplot(df_imp, title, out_png, value_col="imp_abs"):
    plt.figure(figsize=(9, 5))
    dfp = df_imp.sort_values(value_col, ascending=False)
    y = np.arange(len(dfp := dfp.reset_index(drop=True)))
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


def _plot_per_class_signed(df_pos, df_neg, title, out_png, shared_scale=False):
    n = len(df_pos)
    classes = df_pos.index.tolist()
    blocks = df_pos.columns.tolist()

    plt.figure(figsize=(10, max(6, 0.35 * n)))
    colors = plt.cm.tab20.colors

    left_pos = np.zeros(n)
    left_neg = np.zeros(n)

    for j, b in enumerate(blocks):
        vp = df_pos[b].values
        vn = df_neg[b].values
        c = colors[j % len(colors)]
        plt.barh(range(n), vp, left=left_pos, color=c, label=b)    # + à droite
        left_pos += vp
        plt.barh(range(n), -vn, left=-left_neg, color=c, alpha=0.35)  # − à gauche
        left_neg += vn

    plt.axvline(0, color="#334155", lw=1)
    if shared_scale:
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


def _plot_per_class_signed_magnitude(dfp, dfn, title, out_png, label_map=None, shared_scale=True):
    """
    Barres divergentes en *magnitude* (échelle linéaire du modèle).
    """
    n = len(dfp)
    classes = dfp.index.tolist()
    if label_map:
        classes = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in classes]

    blocks = dfp.columns.tolist()
    plt.figure(figsize=(10, max(6, 0.35 * n)))
    colors = plt.cm.tab20.colors

    left_pos = np.zeros(n)
    left_neg = np.zeros(n)

    for j, b in enumerate(blocks):
        vp = dfp[b].values
        vn = dfn[b].values
        c = colors[j % len(colors)]
        plt.barh(range(n), vp, left=left_pos, color=c, label=b)       # + à droite
        left_pos += vp
        plt.barh(range(n), -vn, left=-left_neg, color=c, alpha=0.35)  # − à gauche
        left_neg += vn

    plt.axvline(0, color="#334155", lw=1)
    if shared_scale:
        # bornes symétriques sur la base des extrêmes observés
        xmax = max(left_pos.max(), left_neg.max())
        plt.xlim(-xmax, xmax)

    plt.yticks(range(n), classes)
    plt.gca().invert_yaxis()
    plt.xlabel("Contribution moyenne au score linéaire (|x·w|) — + à droite, − à gauche")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.title(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ------------------------------------------------------------------
# B2 (texte) : global + par classe

def blocks_b2(model_path, data_csv, max_n=3000, normalize="abs"):
    import joblib
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

    out_csv = Path("results/reports") / f"block_importance_b2_{suffix}.csv"
    out_png = Path("results/figures") / f"block_importance_b2_{suffix}.png"
    ensure_dirs()
    df_imp.to_csv(out_csv, index=False)
    _save_barplot(df_imp, f"Importance par bloc — B2 ({normalize})", str(out_png), value_col=vcol)
    print(f"[OK] Importance B2 ({normalize}) → {out_png}")
    return True


def blocks_b2_per_class(model_path, data_csv, label_col="prdtypecode",
                        max_n=3000, normalize="abs",
                        label_map_json=None,
                        out_csv="results/reports/block_importance_b2_per_class.csv",
                        out_png="results/figures/block_importance_b2_per_class.png",
                        topk_classes=30):
    import joblib
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
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None
    if label_map:
        df_wide.index = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in df_wide.index]

    ensure_dirs()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(out_csv, index=True)
    print(f"[OK] Par-classe B2 (CSV) → {out_csv}")

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
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] Par-classe B2 (figure) → {out_png}")
    return True


def blocks_b2_per_class_signed_mag(model_path, data_csv, label_col="prdtypecode",
                                   max_n=3000, label_map_json=None, out_dir="results",
                                   topk_classes=30, shared_scale=True, sort_by="none"):
    import joblib
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

    ensure_dirs()
    pd_path_pos = Path(out_dir, "reports", "block_importance_b2_per_class_pos_mag.csv")
    pd_path_neg = Path(out_dir, "reports", "block_importance_b2_per_class_neg_mag.csv")
    pd_path_pos.parent.mkdir(parents=True, exist_ok=True)
    dfp.to_csv(pd_path_pos)
    dfn.to_csv(pd_path_neg)

    _plot_per_class_signed_magnitude(
        dfp, dfn,
        "Importance par bloc — B2 (par classe, impact signé, magnitude)",
        str(Path(out_dir, "figures", "block_importance_b2_per_class_signed_mag.png")),
        label_map=label_map, shared_scale=shared_scale
    )
    return True


# ------------------------------------------------------------------
# B4 (multimodal) : global + par classe (coarse)

def blocks_b4(model_path, data_csv, max_n=3000, normalize="abs"):
    import joblib
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

    ensure_dirs()
    out_csv = Path("results/reports") / f"block_importance_b4_{normalize}.csv"
    out_png = Path("results/figures") / f"block_importance_b4_{normalize}.png"
    df_imp.to_csv(out_csv, index=False)
    _save_barplot(df_imp, f"Importance par bloc — B4 ({normalize})", str(out_png), value_col=vcol)
    print(f"[OK] Importance B4 ({normalize}) → {out_png}")
    return True


def blocks_b4_per_class(model_path, data_csv, label_col="prdtypecode", max_n=3000,
                        normalize="abs", label_map_json=None,
                        out_csv="results/reports/block_importance_b4_per_class.csv",
                        out_png="results/figures/block_importance_b4_per_class.png",
                        topk_classes=30):
    import joblib
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
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None
    if label_map:
        df_wide.index = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in df_wide.index]

    ensure_dirs()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(out_csv, index=True)
    print(f"[OK] Par-classe B4 (CSV) → {out_csv}")

    plt.figure(figsize=(10, max(6, 0.35 * len(df_wide))))
    left = np.zeros(len(df_wide))
    colors = plt.cm.Paired.colors
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
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] Par-classe B4 (figure) → {out_png}")
    return True


def blocks_b4_fine_global(model_path, data_csv, label_col="prdtypecode", max_n=3000, out_dir="results"):
    import joblib
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

    X = feat_union.transform(_ensure_input_df(df))
    if scaler is not None:
        X = scaler.transform(X)

    slices = _block_slices_b4_fine(feat_union, df)
    df_imp = _block_importance_from_coef(C, X, slices, clf)
    df_imp = df_imp.sort_values("imp_abs", ascending=False)

    ensure_dirs()
    out_png = str(Path(out_dir, "figures", "block_importance_b4_fine.png"))
    _save_barplot(df_imp, "Importance par bloc — B4 (multimodal, découpage fin)", out_png, value_col="imp_abs")
    print(f"[OK] Importance B4 (fin) → {out_png}")
    return True


def blocks_b4_per_class_signed(model_path, data_csv, label_col="prdtypecode", max_n=3000,
                               label_map_json=None, out_dir="results", topk_classes=30, shared_scale=False):
    import joblib
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
        try:
            label_map = json.loads(Path(label_map_json).read_text(encoding="utf-8"))
        except Exception:
            label_map = None
    if label_map:
        new_index = [label_map.get(str(c), label_map.get(int(c), str(c))) for c in df_pos.index]
        df_pos.index = new_index
        df_neg.index = new_index

    ensure_dirs()
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


def _norm_l2_rows(X):
    """Normalise L2 ligne par ligne; retourne (X_normé, norms)."""
    norms = np.linalg.norm(X, axis=1)
    norms_safe = np.where(norms == 0, 1.0, norms)
    return (X / norms_safe[:, None]), norms_safe


def _compute_b4_image_blocks_signed_mag(pipe, df_small, features, scaler, clf, coarse_slices):
    """
    Calcule, pour chaque *classe*, la contribution en magnitude signée des blocs
    IMAGES (CNN/Pixels/Stats) dans l'espace *après scaler*.
    Retourne deux listes de DataFrames (pos_list, neg_list) indexés par block -> value.
    """
    # X complet (concat)
    X_full = features.transform(_ensure_input_df(df_small))
    if scaler is not None:
        X_full = scaler.transform(X_full)

    C, _ = _get_linear_weights_and_bias(clf)
    classes = list(getattr(clf, "classes_", range(C.shape[0])))

    # On prépare les résultats : par classe, un DF block->value
    pos_dfs, neg_dfs = [], []
    for ci, cls in enumerate(classes):
        rows = []
        for (label, a, b) in coarse_slices:
            # on ignore texte (traité séparément dans la fonction principale)
            if str(label).lower().startswith("text"):
                continue
            xb = X_full[:, a:b]
            w  = C[ci, a:b]
            v  = np.asarray(xb.dot(w)).ravel()
            # magnitude signée = signe(moyenne) * moyenne(|.|)
            sign = np.sign(v.mean()) if v.size else 0.0
            mag  = float(np.mean(np.abs(v))) if v.size else 0.0
            val  = float(sign * mag)
            rows.append((label, val))
        df_c = pd.DataFrame(rows, columns=["block", "value"]).set_index("block")
        pos_dfs.append(df_c[df_c["value"] > 0])
        neg_dfs.append(df_c[df_c["value"] < 0])
    return pos_dfs, neg_dfs


def blocks_b4_per_class_signed_magnitude_backproj_text(args, pipe, df):
    """
    B4 par classe (magnitude signée) avec rétro-projection du TEXTE (pré-SVD)
    pour retrouver le détail TF-IDF/stats, combiné avec les blocs image.
    Écrit une figure par classe + un résumé moyen.
    """
    ensure_dirs()
    features, scaler, clf, names = _find_feat_scaler_clf(pipe)
    if features is None or clf is None:
        raise RuntimeError("Pipeline B4 inattendue : 'features' ou classifieur manquant.")
    print(f"[INFO] Étapes détectées → features='{names['features']}', scaler='{names['scaler']}', clf='{names['clf']}'")

    C_full, bias = _get_linear_weights_and_bias(clf)
    classes = list(getattr(clf, "classes_", range(C_full.shape[0])))

    df_small = df.copy()
    df_small = _ensure_input_df(df_small)

    # COARSE slices pour localiser la tranche TEXT_SVD et les blocs image
    coarse = _block_slices_b4(features, df_small)
    text_coarse = [t for t in coarse if str(t[0]).lower().startswith("texte") or str(t[0]).lower().startswith("text")]
    if not text_coarse:
        raise RuntimeError("Tranche TEXTE introuvable dans FeatureUnion('features').")
    _, a_text, b_text = text_coarse[0]

    # scaler.scale_ éventuellement (colonne-par-colonne)
    if scaler is not None and hasattr(scaler, "scale_"):
        scale_text = np.asarray(scaler.scale_[a_text:b_text])
    else:
        scale_text = None

    # Récupère la branche texte : pre-union + SVD + éventuel L2
    text_pipeline = dict(features.transformer_list).get("text")
    if text_pipeline is None or not hasattr(text_pipeline, "named_steps"):
        raise RuntimeError("Branche 'text' manquante ou inattendue (pas un Pipeline).")

    pre_union = text_pipeline.named_steps.get("text") or _unwrap_feature_union(text_pipeline)
    svd = text_pipeline.named_steps.get("svd")
    if svd is None or not hasattr(svd, "components_"):
        raise RuntimeError("Pas de SVD entraîné dans la branche texte.")
    has_l2 = "l2" in text_pipeline.named_steps

    # Espace texte pré-SVD + slices détaillés
    X_text_pre = pre_union.transform(df_small)             # (n, D_pre)
    text_slices_pre = _block_slices_text(pre_union, df_small)

    # Espace texte SVD (+ normalisation L2 si présente) pour calculer alpha_i
    X_text_svd = svd.transform(X_text_pre)                 # (n, K)
    if has_l2:
        X_text_svd_norm, norms = _norm_l2_rows(X_text_svd)
        alpha = 1.0 / np.where(norms == 0, 1.0, norms)     # alpha_i = 1/||X_svd_i||
    else:
        X_text_svd_norm = X_text_svd
        alpha = np.ones(X_text_svd.shape[0], dtype=np.float64)

    # Poids du classifieur pour la tranche TEXT_SVD (corrigés du scaler éventuel)
    W_text_svd = C_full[:, a_text:b_text]                  # (n_classes, K)
    if scale_text is not None:
        W_text_svd_eff = W_text_svd / scale_text[None, :]
    else:
        W_text_svd_eff = W_text_svd

    # Rétro-projection des poids vers l'espace pré-SVD : w_pre = C^T @ w_comp
    Vk = np.asarray(svd.components_)                        # (K, D_pre)
    W_text_pre = (Vk.T @ W_text_svd_eff.T).T                # (n_classes, D_pre)

    # Contributions TEXTE par sous-bloc (magnitude signée)
    contrib_text_pos, contrib_text_neg = [], []
    n = X_text_pre.shape[0]
    for ci in range(len(classes)):
        w_pre_c = W_text_pre[ci]                            # (D_pre,)
        block_rows = []
        for label, s, e in text_slices_pre:
            x_b = X_text_pre[:, s:e]                        # (n, d_b)
            w_b = w_pre_c[s:e]                              # (d_b,)
            dot_i = np.asarray(x_b.dot(w_b)).ravel()        # (n,)
            signed = alpha * dot_i                          # tient compte du L2 après SVD
            sign = np.sign(signed.mean()) if signed.size else 0.0
            mag  = float(np.mean(np.abs(signed))) if signed.size else 0.0
            val  = float(sign * mag)
            block_rows.append((label, val))
        df_c = pd.DataFrame(block_rows, columns=["block", "value"]).set_index("block")
        contrib_text_pos.append(df_c[df_c["value"] > 0])
        contrib_text_neg.append(df_c[df_c["value"] < 0])

    # Ajout des blocs IMAGE (CNN/Pixels/Stats) calculés directement dans l'espace concaténé
    img_pos_list, img_neg_list = _compute_b4_image_blocks_signed_mag(pipe, df_small, features, scaler, clf, coarse)

    # Fusion texte + image et sorties
    outdir_fig = Path("results/figures"); outdir_fig.mkdir(parents=True, exist_ok=True)
    outdir_csv = Path("results/reports"); outdir_csv.mkdir(parents=True, exist_ok=True)

    label_map = None
    if getattr(args, "label_map_json", None):
        p = Path(args.label_map_json)
        if p.exists():
            try:
                label_map = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                label_map = None

    agg = []
    for ci, lab in enumerate(classes):
        txt_pos = contrib_text_pos[ci].copy()
        txt_neg = contrib_text_neg[ci].copy()
        img_pos = img_pos_list[ci].copy()
        img_neg = img_neg_list[ci].copy()

        df_all = txt_pos.combine_first(txt_neg).combine_first(img_pos).combine_first(img_neg).fillna(0.0)

        # tri (pos/neg/net/total)
        mode = (getattr(args, "sort_by", "total") or "total").lower()
        if mode == "pos":
            order = df_all.sort_values("value", ascending=False)
        elif mode == "neg":
            order = df_all.sort_values("value", ascending=True)
        elif mode == "net":
            # déjà signé
            order = df_all
        else:  # total : tri par |value|
            ord_df = df_all.assign(abs=lambda d: d["value"].abs()).sort_values("abs", ascending=False).drop(columns=["abs"])
            order = ord_df

        # CSV + figure par classe
        csv_path = outdir_csv / f"block_importance_b4_backproj_text_{lab}.csv"
        order.to_csv(csv_path, index=True)

        title_lab = label_map.get(str(lab), label_map.get(int(lab), str(lab))) if label_map else str(lab)
        plt.figure(figsize=(9, max(5, 0.35 * len(order))))
        vals = order["value"].values
        y = np.arange(len(order))
        colors = np.where(vals >= 0, "#4c78a8", "#f58518")
        plt.barh(y, vals, align="center", color=colors)
        plt.yticks(y, order.index.tolist())
        plt.axvline(0, color="#334155", lw=1)
        plt.xlabel("Contribution moyenne au score linéaire (|x·w|)")
        plt.title(f"Importance par bloc — B4 (texte rétro-projeté, classe {title_lab})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        out_png = outdir_fig / f"block_importance_b4_per_class_signed_mag_backproj_text_{lab}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()

        # pour la moyenne
        df_all_named = order.copy()
        df_all_named.columns = [str(lab)]
        agg.append(df_all_named)

    # Vue moyenne (toutes classes)
    agg_df = pd.concat(agg, axis=1).fillna(0.0)
    agg_df["mean"] = agg_df.mean(axis=1)
    agg_sorted = agg_df.sort_values("mean", ascending=False)
    out_csv = outdir_csv / "block_importance_b4_per_class_signed_mag_backproj_text_summary.csv"
    agg_sorted.drop(columns=["mean"]).to_csv(out_csv)

    # figure moyenne
    plt.figure(figsize=(9, max(5, 0.35 * len(agg_sorted))))
    series = agg_sorted["mean"]
    y = np.arange(len(series))
    colors = np.where(series.values >= 0, "#4c78a8", "#f58518")
    plt.barh(y, series.values, color=colors)
    plt.yticks(y, series.index.tolist())
    plt.axvline(0, color="#334155", lw=1)
    plt.xlabel("Contribution moyenne au score linéaire (|x·w|)")
    plt.title("Importance par bloc — B4 (texte rétro-projeté, moyenne classes)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out_png_mean = outdir_fig / "block_importance_b4_per_class_signed_mag_backproj_text_mean.png"
    plt.savefig(out_png_mean, dpi=160)
    plt.close()

    print(f"[OK] Back-proj TEXTE B4 → figures par classe + {out_png_mean}")
    print(f"[OK] CSV résumé → {out_csv}")
    return True


# ------------------------------------------------------------------
# ACP + confusions (facultatif, inchangé)

def try_load_oof_features(kind):
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

    svd_csv = Path("results") / f"features_{kind}_svd100_preview.csv"
    fig_svd = Path("results/figures") / f"acp_{kind}_2d.png"
    fig_ok  = Path("results/figures") / f"acp_{kind}_ok_error.png"

    if svd_csv.exists():
        print(f"[INFO] SVD preview trouvé: {svd_csv}")
        df = pd.read_csv(svd_csv, index_col=0)
        plt.figure(figsize=(9, 6))
        plt.scatter(df["svd_1"], df["svd_2"], s=8, alpha=0.20)
        plt.xlabel("svd_1"); plt.ylabel("svd_2")
        plt.title(f"ACP (SVD preview) — {kind.upper()}")
        plt.tight_layout(); plt.savefig(fig_svd, dpi=180); plt.close()
        print(f"[OK] ACP (SVD preview) → {fig_svd}")

        ok_mask = None
        if {"y_true", "y_pred"}.issubset(df.columns):
            ok_mask = (df["y_true"].astype(str) == df["y_pred"].astype(str)).values
        elif preds is not None:
            ok_mask = preds.reindex(df.index).assign(
                ok=lambda d: (d["y_true"].astype(str) == d["y_pred"].astype(str)).astype(int)
            )["ok"].fillna(0).astype(bool).values

        if ok_mask is not None:
            XY = df[["svd_1", "svd_2"]].values
            plt.figure(figsize=(9, 6))
            plt.scatter(XY[ok_mask,0], XY[ok_mask,1], s=8, alpha=0.15, label="OK (y_true=y_pred)")
            plt.scatter(XY[~ok_mask,0], XY[~ok_mask,1], s=8, alpha=0.35, label="Erreur")
            plt.xlabel("svd_1"); plt.ylabel("svd_2")
            plt.title(f"ACP (SVD preview) — {kind.upper()} : OK vs Erreur")
            plt.legend(); plt.tight_layout(); plt.savefig(fig_ok, dpi=180); plt.close()
            print(f"[OK] ACP colorée → {fig_ok}")
        return True

    Z, is_sparse, src = try_load_oof_features(kind)
    if Z is None:
        print("[WARN] Aucune feature trouvée (ni SVD preview, ni OOF).")
        return False

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print(f"[INFO] OOF features: {src} (sparse={is_sparse})")
    n = Z.shape[0]; take = min(max_n, n)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=take, replace=False))
    Zs = Z[idx] if hasattr(Z, "tocsr") else Z[idx, :]
    Zs = Zs.toarray() if hasattr(Zs, "toarray") else Zs

    scaler = StandardScaler(with_mean=False)
    Zs_std = scaler.fit_transform(Zs)
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Zs_std)

    plt.figure(figsize=(9, 6))
    plt.scatter(XY[:, 0], XY[:, 1], s=8, alpha=0.20)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"ACP (PCA 2D) — {kind.upper()} — {take}/{n}")
    plt.tight_layout(); plt.savefig(fig_svd, dpi=180); plt.close()
    print(f"[OK] ACP (PCA 2D) → {fig_svd}")
    return True


def top_confusions(kind, topk=20, labels_map=None):
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
    cm2 = cm.copy(); np.fill_diagonal(cm2, 0)

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


# ------------------------------------------------------------------
# CLI

def main():
    parser = argparse.ArgumentParser("Diagnostics ACP & SHAP + importances (B2/B4)")
    parser.add_argument("--kind", default="b4", help="b2, b3, b4… (pour nommage des sorties ACP/confusions)")
    parser.add_argument("--model", dest="model_path", required=False)
    parser.add_argument("--data-csv", dest="data_csv", required=False)
    parser.add_argument("--label-col", default="prdtypecode")
    parser.add_argument("--label-map", dest="label_map_json", default=None)

    # ACP / Confusions
    parser.add_argument("--acp", action="store_true")
    parser.add_argument("--top-confusions", action="store_true")

    # B2
    parser.add_argument("--blocks-b2", action="store_true")
    parser.add_argument("--blocks-b2-per-class", action="store_true")
    parser.add_argument("--blocks-b2-per-class-signed", action="store_true")
    parser.add_argument("--blocks-b2-per-class-signed-mag", action="store_true")

    # B4
    parser.add_argument("--blocks-b4", action="store_true")
    parser.add_argument("--blocks-b4-per-class", action="store_true")
    parser.add_argument("--blocks-b4-fine", action="store_true")
    parser.add_argument("--blocks-b4-per-class-signed", action="store_true")
    parser.add_argument("--blocks-b4-per-class-signed-mag", action="store_true")
    # Option fidèle (texte rétro-projeté)
    parser.add_argument("--blocks-b4-per-class-signed-mag-backproj-text", action="store_true")

    # options communes
    parser.add_argument("--max-n", type=int, default=3000)
    parser.add_argument("--sort-by", default="none", help="pos|neg|net|total|none")
    parser.add_argument("--shared-scale", action="store_true", help="échelle commune pour les graphes signés")

    args = parser.parse_args()

    # ACP
    if args.acp:
        do_acp(args.kind)

    # Confusions
    if args.top_confusions:
        labels_map = None
        if args.label_map_json and Path(args.label_map_json).exists():
            try:
                labels_map = json.loads(Path(args.label_map_json).read_text(encoding="utf-8"))
            except Exception:
                labels_map = None
        top_confusions(args.kind, labels_map=labels_map)

    # Besoin du modèle et des données pour les blocs
    if not args.model_path or not args.data_csv:
        return

    import joblib
    pipe = joblib.load(args.model_path)
    df = pd.read_csv(args.data_csv)

    # B2
    if args.blocks_b2:
        blocks_b2(args.model_path, args.data_csv, max_n=args.max_n, normalize="abs")
    if args.blocks_b2_per_class:
        blocks_b2_per_class(args.model_path, args.data_csv, label_col=args.label_col, max_n=args.max_n)
    if args.blocks_b2_per_class_signed:
        # par-classe signé (%) avec échelle partagée
        # (réutilise blocks_b2_per_class avec normalize pos/neg au plotting)
        # Ici, on préfère renvoyer vers la version magnitude si besoin
        blocks_b2_per_class(args.model_path, args.data_csv, label_col=args.label_col, max_n=args.max_n)
    if args.blocks_b2_per_class_signed_mag:
        blocks_b2_per_class_signed_mag(args.model_path, args.data_csv, label_col=args.label_col,
                                       max_n=args.max_n, label_map_json=args.label_map_json,
                                       shared_scale=True, sort_by=args.sort_by)

    # B4
    if args.blocks_b4:
        blocks_b4(args.model_path, args.data_csv, max_n=args.max_n, normalize="abs")
    if args.blocks_b4_per_class:
        blocks_b4_per_class(args.model_path, args.data_csv, label_col=args.label_col, max_n=args.max_n)
    if args.blocks_b4_fine:
        blocks_b4_fine_global(args.model_path, args.data_csv, label_col=args.label_col, max_n=args.max_n)
    if args.blocks_b4_per_class_signed:
        blocks_b4_per_class_signed(args.model_path, args.data_csv, label_col=args.label_col,
                                   max_n=args.max_n, label_map_json=args.label_map_json,
                                   shared_scale=args.shared_scale)
    if args.blocks_b4_per_class_signed_mag:
        # version "agrégée": texte reste compact si SVD (pas de rétro-proj)
        # → on conserve pour compat
        # on ré-utilise la vue "fine" + magnitude sur X concaténé
        # (déjà disponible via blocks_b4_fine_global / b4_per_class)
        blocks_b4_fine_global(args.model_path, args.data_csv, label_col=args.label_col, max_n=args.max_n)

    if args.blocks_b4_per_class_signed_mag_backproj_text:
        # NOUVEAU : fidèle B4 avec détail texte (pré-SVD) via rétro-projection
        blocks_b4_per_class_signed_magnitude_backproj_text(args, pipe, df)


if __name__ == "__main__":
    main()
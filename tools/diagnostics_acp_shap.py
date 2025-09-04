# -*- coding: utf-8 -*-
"""
Diagnostics ACP & SHAP pour baselines B2/B3/B4
- ACP 2D: SVD preview si disponible, sinon PCA sur la matrice OOF
- SHAP: LinearExplainer (LogReg) ou KernelExplainer (fallback, plus lent)
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    Génère results/figures/acp_<kind>_2d.png
    """
    ensure_dirs()

    preds_csv = Path("results") / f"preds_{kind}.csv"
    if not preds_csv.exists():
        print(f"[WARN] {preds_csv} introuvable (y_true/y_pred). L'ACP colorée ne sera pas annotée.")
        preds = None
    else:
        preds = pd.read_csv(preds_csv)
        assert {"y_true", "y_pred"}.issubset(preds.columns), "preds CSV doit contenir y_true,y_pred."
        preds["y_true"] = preds["y_true"].astype(str)
        preds["y_pred"] = preds["y_pred"].astype(str)
        preds["ok"] = (preds["y_true"] == preds["y_pred"]).astype(int)

    svd_csv = Path("results") / f"features_{kind}_svd100_preview.csv"
    fig_path = Path("results/figures") / f"acp_{kind}_2d.png"

    if svd_csv.exists():
        print(f"[INFO] SVD preview trouvé: {svd_csv}")
        df = pd.read_csv(svd_csv, index_col=0)
        if {"y_true","y_pred"}.issubset(df.columns):
            base = df
        else:
            if preds is None:
                base = df
                base["y_true"] = ""
                base["y_pred"] = ""
                base["ok"] = 0
            else:
                base = df.join(preds, how="left")

        # Scatter SVD 2D
        plt.figure()
        plt.scatter(base["svd_1"], base["svd_2"])
        plt.xlabel("svd_1"); plt.ylabel("svd_2")
        plt.title(f"ACP (SVD preview) — {kind.upper()}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        print(f"[OK] ACP (SVD preview) → {fig_path}")
        return True

    # Sinon: PCA sur OOF
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

    plt.figure()
    plt.scatter(XY[:, 0], XY[:, 1])
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"ACP (PCA 2D) — {kind.upper()} — {take}/{n}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    print(f"[OK] ACP (PCA 2D) → {fig_path}")
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

def do_shap(kind, model_path, bg_size=2000, explain_size=300):
    """
    Démo SHAP:
    - charge le pipeline .joblib
    - prend un échantillon des features OOF comme background + points à expliquer
    - LinearExplainer si LogisticRegression, sinon KernelExplainer (plus lent)
    """
    try:
        import joblib, shap  # shap est optionnel, installer si besoin
    except Exception as e:
        print("[WARN] SHAP indisponible (installe 'shap'): ", e)
        return False

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[WARN] Modèle introuvable: {model_path}")
        return False

    Z, _, src = try_load_oof_features(kind)
    if Z is None:
        print("[WARN] Pas de features OOF pour SHAP.")
        return False

    print(f"[INFO] Chargement du pipeline: {model_path}")
    pipe = joblib.load(model_path)

    # Estimator final dans le pipeline
    clf = getattr(pipe, "named_steps", {}).get("model", None)
    if clf is None:
        print("[WARN] Étape 'model' introuvable dans le pipeline.")
        return False

    n = Z.shape[0]
    bg = min(bg_size, n)
    ex = min(explain_size, n)
    rng = np.random.default_rng(0)
    bg_idx = np.sort(rng.choice(n, size=bg, replace=False))
    ex_idx = np.sort(rng.choice(n, size=ex, replace=False))

    X_bg = Z[bg_idx] if not hasattr(Z, "tocsr") else Z[bg_idx]
    X_ex = Z[ex_idx] if not hasattr(Z, "tocsr") else Z[ex_idx]

    # LinearExplainer pour LogisticRegression, sinon Kernel (lent)
    name = clf.__class__.__name__
    print(f"[INFO] Estimator final: {name}")
    try:
        if name == "LogisticRegression":
            explainer = shap.LinearExplainer(clf, X_bg)
            sv = explainer.shap_values(X_ex)
            print("[OK] SHAP LinearExplainer calculé (LogReg). Ouvre un notebook et fais shap.summary_plot(sv, X_ex).")
        else:
            f = clf.decision_function if hasattr(clf, "decision_function") else clf.predict_proba
            # background plus petit pour accélérer
            if hasattr(X_bg, "tocsr"):
                X_bg_small = X_bg[:200]
            else:
                X_bg_small = X_bg[:200]
            explainer = shap.KernelExplainer(f, X_bg_small)
            sv = explainer.shap_values(X_ex[:50])
            print("[OK] SHAP KernelExplainer calculé (estimation non linéaire). "
                  "Ouvre un notebook et fais shap.summary_plot(sv, X_ex[:50]).")
        return True
    except Exception as e:
        print("[WARN] SHAP a échoué:", e)
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["b2","b3","b4"], help="Baseline à analyser")
    parser.add_argument("--model", default=None, help="Chemin du pipeline .joblib pour SHAP (optionnel)")
    parser.add_argument("--max-sample", type=int, default=8000, help="Échantillon max pour PCA 2D sur OOF")
    parser.add_argument("--topk", type=int, default=20, help="Nombre de confusions à reporter")
    args = parser.parse_args()

    ensure_dirs()
    lblmap = load_labels_map()

    ok_acp = do_acp(args.kind, max_n=args.max_sample)
    top_confusions(args.kind, topk=args.topk, labels_map=lblmap)

    if args.model:
        do_shap(args.kind, args.model)

if __name__ == "__main__":
    main()

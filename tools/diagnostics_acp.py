# -*- coding: utf-8 -*-
"""
Diagnostics ACP & SHAP pour baselines B2/B3/B4
- ACP 2D: SVD preview si dispo, sinon PCA 2D sur la matrice OOF
- Coloration: OK/Erreur (par défaut) ou thématique (theme_map.json)
- Export: figures + top confusions CSV (lisibles via labels_map.json)
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
    import json
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}

def load_theme_map(path="features/theme_map.json"):
    p = Path(path)
    if not p.exists():
        return None
    import json
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # attendu: {"10":"Livres & presse", "40":"Jeux & gaming", ...}
    return {str(k): v for k, v in raw.items()}

def try_load_oof_features(kind):
    npz = Path("results") / f"features_{kind}_oof.npz"
    npy = Path("results") / f"features_{kind}_oof.npy"
    if npz.exists():
        from scipy.sparse import load_npz
        return load_npz(npz), True, str(npz)
    if npy.exists():
        return np.load(npy), False, str(npy)
    return None, False, None

def _color_by_theme(ax, XY, y_true_str, theme_map):
    # map classe -> theme ; palette discrète
    themes = pd.Series(y_true_str).map(theme_map).fillna("Autres").values
    uniq = pd.Index(themes).unique().tolist()
    # limite à 10 thèmes affichés + "Autres"
    top10 = pd.Series(themes).value_counts().index[:10].tolist()
    keep = set(top10 + (["Autres"] if "Autres" in uniq else []))
    colors = plt.cm.tab10.colors
    cdict = {t: colors[i % len(colors)] for i, t in enumerate(top10)}
    if "Autres" in keep:
        cdict["Autres"] = (0.7, 0.7, 0.7, 0.35)

    for t in keep:
        mask = (themes == t)
        ax.scatter(XY[mask,0], XY[mask,1], s=8, alpha=0.25, label=t, c=[cdict[t]])
    ax.legend(loc="best", fontsize=8)

def do_acp(kind, color_mode="okerr", max_n=8000):
    """
    Génère:
      - results/figures/acp_<kind>_2d.png
      - results/figures/acp_<kind>_<mode>.png  (ok_error | themes)
    """
    ensure_dirs()
    preds_csv = Path("results") / f"preds_{kind}.csv"
    preds = None
    if preds_csv.exists():
        preds = pd.read_csv(preds_csv, index_col=0)
        for c in ("y_true", "y_pred"):
            if c in preds: preds[c] = preds[c].astype(str)
        if {"y_true","y_pred"}.issubset(preds.columns):
            preds["ok"] = (preds["y_true"] == preds["y_pred"]).astype(bool)
        else:
            preds = None

    svd_csv = Path("results") / f"features_{kind}_svd100_preview.csv"
    fig_svd = Path("results/figures") / f"acp_{kind}_2d.png"

    def _plot_okerr(ax, XY, ok_mask):
        ax.scatter(XY[ok_mask,0], XY[ok_mask,1], s=8, alpha=0.18, label="OK")
        ax.scatter(XY[~ok_mask,0], XY[~ok_mask,1], s=8, alpha=0.35, label="Erreur")
        ax.legend()

    # --- Cas SVD preview (rapide) ---
    if svd_csv.exists():
        df = pd.read_csv(svd_csv, index_col=0)
        XY = df[["svd_1","svd_2"]].values
        plt.figure(figsize=(9,6))
        plt.scatter(XY[:,0], XY[:,1], s=8, alpha=0.20)
        plt.xlabel("svd_1"); plt.ylabel("svd_2")
        plt.title(f"ACP (SVD preview) — {kind.upper()}")
        plt.tight_layout(); plt.savefig(fig_svd, dpi=180); plt.close()

        # couleur avancée
        if color_mode == "okerr" and preds is not None:
            ok_mask = preds.reindex(df.index)["ok"].fillna(False).values
            fig_ok = Path("results/figures") / f"acp_{kind}_ok_error.png"
            fig, ax = plt.subplots(figsize=(9,6))
            _plot_okerr(ax, XY, ok_mask)
            ax.set_xlabel("svd_1"); ax.set_ylabel("svd_2")
            ax.set_title(f"ACP (SVD preview) — {kind.upper()} : OK vs Erreur")
            fig.tight_layout(); fig.savefig(fig_ok, dpi=180); plt.close(fig)

        elif color_mode == "theme" and preds is not None:
            theme_map = load_theme_map() or {}
            if theme_map:
                y_true = preds.reindex(df.index)["y_true"].fillna("").values
                fig_th = Path("results/figures") / f"acp_{kind}_themes.png"
                fig, ax = plt.subplots(figsize=(9,6))
                _color_by_theme(ax, XY, y_true, theme_map)
                ax.set_xlabel("svd_1"); ax.set_ylabel("svd_2")
                ax.set_title(f"ACP (SVD preview) — {kind.upper()} : Thématiques")
                fig.tight_layout(); fig.savefig(fig_th, dpi=180); plt.close(fig)
        return True

    # --- Sinon PCA 2D sur OOF ---
    Z, is_sparse, src = try_load_oof_features(kind)
    if Z is None:
        print("[WARN] Aucune feature trouvée (ni SVD preview, ni OOF).")
        return False
    n = Z.shape[0]; take = min(max_n, n)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=take, replace=False))
    Zs = Z[idx] if hasattr(Z, "tocsr") else Z[idx, :]
    if hasattr(Zs, "toarray"): Zs = Zs.toarray()

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    XY = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler(with_mean=False).fit_transform(Zs)
    )
    plt.figure(figsize=(9,6))
    plt.scatter(XY[:,0], XY[:,1], s=8, alpha=0.2)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"ACP (PCA 2D) — {kind.upper()} — {take}/{n}")
    plt.tight_layout(); plt.savefig(fig_svd, dpi=180); plt.close()
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
    labels = sorted(set(preds["y_true"]) | set(preds["y_pred"]), key=str)
    cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=labels)
    cm2 = cm.copy(); np.fill_diagonal(cm2, 0)
    rows, cols = np.where(cm2 > 0)
    recs = []
    for r, c in zip(rows, cols):
        t, p = labels[r], labels[c]
        tn = labels_map.get(t, t) if labels_map else t
        pn = labels_map.get(p, p) if labels_map else p
        recs.append((t, p, int(cm2[r, c]), tn, pn))
    df = pd.DataFrame(sorted(recs, key=lambda x: x[2], reverse=True)[:topk],
                      columns=["true_id","pred_id","count","true_name","pred_name"])
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["b2","b3","b4"])
    ap.add_argument("--model", default=None, help="Chemin du pipeline .joblib pour SHAP (optionnel)")
    ap.add_argument("--max-sample", type=int, default=8000)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--color", choices=["okerr","theme","none"], default="okerr",
                    help="Schéma de couleur pour l’ACP")
    args = ap.parse_args()

    ensure_dirs()
    lblmap = load_labels_map()
    do_acp(args.kind, color_mode=args.color, max_n=args.max_sample)
    top_confusions(args.kind, topk=args.topk, labels_map=lblmap)

    if args.model:
        do_shap(args.kind, args.model)

if __name__ == "__main__":
    main()
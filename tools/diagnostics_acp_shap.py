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
# Endroit pour ajouter Shap si besoin
######

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


if __name__ == "__main__":
    main()

# tools/peek_features.py
# Inspecter la taille/nnz/mémoire des features par branche avant la fusion
# Échantillon rapide (8k) pour ne pas exploser la RAM
# le script à lancer: RAKUTEN_MAX_N=8000 python tools/peek_features.py
# Windows PowerShell
# $env:RAKUTEN_MAX_N=8000; python tools/peek_features.py
import os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
    sys.path.append(str(REPO / "main"))
os.chdir(REPO)

import numpy as np
import pandas as pd
from scipy import sparse

from main.train_model import load_config, init_seeds
from models.text_pipeline import create_text_pipeline_from_cfg
from models.image_pipeline import create_image_pipeline_from_cfg
from models.cnn_features import create_cnn_branch_from_cfg
from features.image_stats import ImageStatsCombinedFeaturizer

# ---------- utilitaires ----------
def describe_sparse(name, X, float_bytes=8, index_bytes=4):
    """
    Estime la mémoire si CSR: data(float) + indices(int) + indptr(int).
    float_bytes: 8 pour float64, 4 pour float32. index_bytes: 4 (int32) ou 8 (int64).
    """
    if sparse.issparse(X):
        nnz = X.nnz
        rows, cols = X.shape
        bytes_data = nnz * float_bytes
        bytes_indices = nnz * index_bytes
        bytes_indptr = (rows + 1) * index_bytes
        total_mb = (bytes_data + bytes_indices + bytes_indptr) / (1024**2)
        print(f"[{name}] SPARSE {rows}x{cols} | nnz={nnz:,} | ~{total_mb:,.1f} MB (CSR est.)")
    else:
        rows, cols = X.shape
        nbytes = X.nbytes if hasattr(X, "nbytes") else rows * cols * 8
        print(f"[{name}] DENSE  {rows}x{cols} | ~{nbytes/(1024**2):,.1f} MB")

def timer(fn, *a, **k):
    t0 = time.time()
    out = fn(*a, **k)
    dt = time.time() - t0
    return out, dt

# ---------- paramètres ----------
CFG_PATH = "features/config.toml"
MAX_N = int(os.environ.get("RAKUTEN_MAX_N", "8000"))  # 0 pour full (à éviter)

def main():
    cfg = load_config(CFG_PATH)
    seed = int(cfg.get("random", {}).get("seed", 42))
    init_seeds(seed)

    # Charger les données comme train_model
    x_path = cfg["paths"]["x_train_csv"]
    y_path = cfg["paths"]["y_train_csv"]
    X = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0).squeeze()

    if MAX_N > 0 and len(X) > MAX_N:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X), size=MAX_N, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx]

    need_cols = ["designation", "description", "productid", "imageid"]
    for c in need_cols:
        if c not in X.columns:
            raise ValueError(f"Colonne manquante: {c}")

    print(f"Échantillon: {len(X)} lignes")

    # --------- BRANCHE TEXTE ----------
    print("\n=== TEXTE — vue détaillée par sous-branche ===")
    text_union = create_text_pipeline_from_cfg(cfg.get("text", {}))
    # text_union est un FeatureUnion de [('tfidf_word', FeatureUnion(...)), ('tfidf_char', Pipeline(...?))]
    # 1) Déplier le top-level
    for name, trans in text_union.transformer_list:
        if hasattr(trans, "fit_transform"):
            if name == "tfidf_word" and hasattr(trans, "transformer_list"):
                # C'est le FeatureUnion interne: tfidf + has_desc + title_len + text_stats + language + lexicon...
                print("\n  -- tfidf_word (FeatureUnion interne) --")
                inner_list = trans.transformer_list
                for in_name, in_trans in inner_list:
                    try:
                        X_in, dt = timer(in_trans.fit_transform, X[need_cols], y)
                        # devine un type raisonnable pour l'estimation mémoire :
                        if sparse.issparse(X_in):
                            # TF-IDF: souvent float64 -> 8 bytes; indices int32 -> 4 bytes (ajuste si besoin)
                            describe_sparse(f"    [word/{in_name}]", X_in, float_bytes=8, index_bytes=4)
                        else:
                            describe_sparse(f"    [word/{in_name}]", X_in)
                        print(f"       ↳ fit_transform: {dt:.1f}s")
                    except Exception as e:
                        print(f"    [word/{in_name}] SKIP ({e})")
            else:
                # Autres top-level: typiquement tfidf_char (Pipeline) si activée
                try:
                    X_top, dt = timer(trans.fit_transform, X[need_cols], y)
                    if sparse.issparse(X_top):
                        describe_sparse(f"  [{name}]", X_top, float_bytes=8, index_bytes=4)
                    else:
                        describe_sparse(f"  [{name}]", X_top)
                    print(f"     ↳ fit_transform: {dt:.1f}s")
                except Exception as e:
                    print(f"  [{name}] SKIP ({e})")

    # Et calculer la **branche texte complète** telle qu'utilisée dans le training :
    X_text, dt = timer(text_union.fit_transform, X[need_cols], y)
    # si le pipeline caste en float32 après TF-IDF, mets float_bytes=4 pour une meilleure estimation
    describe_sparse("text_branch (TOTAL)", X_text, float_bytes=8, index_bytes=4)
    print(f"   ↳ fit_transform texte (TOTAL): {dt/60:.2f} min")

    # --------- BRANCHE PIXELS ----------
    print("\n=== PIXELS ===")
    img_pipe = create_image_pipeline_from_cfg(cfg.get("images", {}), use_test_dir=False)
    X_pix, dt = timer(img_pipe.fit_transform, X[need_cols], y)
    if sparse.issparse(X_pix):
        describe_sparse("pixels_branch", X_pix, float_bytes=8, index_bytes=4)
    else:
        describe_sparse("pixels_branch", X_pix)
    print(f"   ↳ fit_transform pixels: {dt/60:.2f} min")

    # --------- BRANCHE CNN (si activée) ----------
    X_cnn = None
    cnn_cfg = cfg.get("images", {}).get("cnn", {})
    if bool(cnn_cfg.get("enabled", False)):
        print("\n=== CNN ===")
        try:
            cnn_pipe = create_cnn_branch_from_cfg(cfg.get("images", {}))
            X_cnn, dt = timer(cnn_pipe.fit_transform, X[need_cols], y)
            if sparse.issparse(X_cnn):
                describe_sparse("cnn_branch", X_cnn, float_bytes=8, index_bytes=4)
            else:
                describe_sparse("cnn_branch", X_cnn)
            print(f"   ↳ fit_transform cnn: {dt/60:.2f} min")
        except Exception as e:
            print(f"[WARN] CNN non disponible: {e}")
    else:
        print("\n=== CNN désactivée dans la config ===")

    # --------- BRANCHE STATS IMAGE (si activée) ----------
    X_stats = None
    stats_c = cfg.get("images", {}).get("stats_combined", {})
    if bool(stats_c.get("enabled", False)):
        print("\n=== IMAGE STATS ===")
        stats = ImageStatsCombinedFeaturizer(
            image_dir=cfg["images"]["train_dir"],
            imgid_col="imageid", pid_col="productid",
            white_threshold=int(stats_c.get("white_threshold", 230)),
            black_threshold=int(stats_c.get("black_threshold", 25)),
            min_area=int(stats_c.get("min_area", 16)),
            prefix_basic=str(stats_c.get("prefix_basic", "img_")),
            prefix_pro=str(stats_c.get("prefix_pro", "pro_")),
        )
        X_stats, dt = timer(stats.fit_transform, X[need_cols], y)
        describe_sparse("img_stats_branch", X_stats)
        print(f"   ↳ fit_transform stats: {dt/60:.2f} min")
    else:
        print("\n=== IMAGE STATS désactivées dans la config ===")

    # --------- HSTACK MANUEL (estimation fusion) ----------
    print("\n=== FUSION (hstack) — estimation ===")
    blocks = [b for b in [X_text, X_pix, X_cnn, X_stats] if b is not None]
    blocks_csr = [b.tocsr() if sparse.issparse(b) else sparse.csr_matrix(b) for b in blocks]
    X_all = sparse.hstack(blocks_csr).tocsr()
    describe_sparse("FUSION_total", X_all, float_bytes=8, index_bytes=4)
    print("OK.")

if __name__ == "__main__":
    main()
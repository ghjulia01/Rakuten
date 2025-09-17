# tools/peek_features.py
# Inspecter la taille/nnz/mémoire des features par branche avant la fusion
# Échantillon rapide (3k) pour ne pas exploser la RAM
# le script à lancer: RAKUTEN_MAX_N=83000 python tools/peek_features.py
# Windows PowerShell
# $env:RAKUTEN_MAX_N=3000; python tools/peek_features.py
# $env:RAKUTEN_MAX_N=3000; python tools/peek_features.py --try-model xgb
# $env:RAKUTEN_MAX_N=3000; python tools/peek_features.py --try-model lgbm
# $env:RAKUTEN_MAX_N=3000; python tools/peek_features.py --try-model lr
# $env:RAKUTEN_MAX_N=3000; python tools/peek_features.py --try-model svc

import os, sys, time
from pathlib import Path
import argparse

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
try:
    from models.cnn_features import create_cnn_branch_from_cfg
except ImportError:
    from main.train_model import create_cnn_branch_from_cfg
from features.image_stats import ImageStatsCombinedFeaturizer

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning
)

# ---------- utilitaires ----------
TIMES = {}  # collecte des durées par étape

def remember(key, dt):
    TIMES[key] = TIMES.get(key, 0.0) + dt

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

    fusion_w = (cfg.get("fusion", {}) or {}).get("weights", {}) or {}
    want_pixels = not (fusion_w.get("image_pixels", None) == 0)
    want_cnn = not (fusion_w.get("image_cnn", None) == 0)
    want_stats  = not (fusion_w.get("image_stats_combined", None) == 0)
    print(f"Config: {CFG_PATH} | seed={seed} | max_n={MAX_N} | pixels={want_pixels} | cnn={want_cnn} | stats={want_stats}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--try-model", choices=["lr","svc","xgb","lgbm"], default=None,
                        help="Optionnel: entraîne vite un modèle sur l'échantillon et affiche son nom/params")
    args = parser.parse_args()
    
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
                        remember(f"text/{in_name}", dt)
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
                    remember(f"text/{name}", dt)
                    if sparse.issparse(X_top):
                        describe_sparse(f"  [{name}]", X_top, float_bytes=8, index_bytes=4)
                    else:
                        describe_sparse(f"  [{name}]", X_top)
                    print(f"     ↳ fit_transform: {dt:.1f}s")
                except Exception as e:
                    print(f"  [{name}] SKIP ({e})")

    # Et calculer la **branche texte complète** telle qu'utilisée dans le training :
    X_text, dt = timer(text_union.fit_transform, X[need_cols], y)
    remember("text/total", dt)
    # si le pipeline caste en float32 après TF-IDF, mets float_bytes=4 pour une meilleure estimation
    describe_sparse("text_branch (TOTAL)", X_text, float_bytes=8, index_bytes=4)
    print(f"   ↳ fit_transform texte (TOTAL): {dt/60:.2f} min")

    svd_cfg = (cfg.get("text", {}).get("svd", {}) or {})
    if bool(svd_cfg.get("enabled", False)):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer
        n_comp = int(svd_cfg.get("n_components", 600))
        rs = int(svd_cfg.get("random_state", 42))
        use_l2_cfg = bool(svd_cfg.get("l2norm", True))
        use_l2 = use_l2_cfg and (args.try_model not in ("xgb", "lgbm"))
        print(f"\n=== TEXTE → SVD({n_comp}){' + L2' if use_l2 else ''} ===")
        X_text = TruncatedSVD(n_components=n_comp, random_state=rs).fit_transform(X_text)
        if use_l2:
            from sklearn.preprocessing import Normalizer
        X_text = Normalizer(copy=False).fit_transform(X_text)
        # re-CSR pour l’estimation hstack
        from scipy import sparse as sp
        X_text = sp.csr_matrix(X_text)
        describe_sparse("text_branch (POST-SVD)", X_text)

    # --------- BRANCHE PIXELS ----------
    print("\n=== PIXELS ===")
    if want_pixels:
        img_pipe = create_image_pipeline_from_cfg(cfg.get("images", {}), use_test_dir=False)
        X_pix, dt = timer(img_pipe.fit_transform, X[need_cols], y)
        remember("pixels", dt)
        describe_sparse("pixels_branch", X_pix if sparse.issparse(X_pix) else X_pix)
        print(f"   ↳ fit_transform pixels: {dt/60:.2f} min")
    else:
        X_pix = None
        print("Pixels SKIPPED (poids=0 dans fusion.weights)")

    # --------- BRANCHE CNN (si activée) ----------
    X_cnn = None
    cnn_cfg = cfg.get("images", {}).get("cnn", {})
    if want_cnn and bool(cnn_cfg.get("enabled", False)):
        print("\n=== CNN ===")
        try:
            cnn_pipe = create_cnn_branch_from_cfg(cfg.get("images", {}))
            X_cnn, dt = timer(cnn_pipe.fit_transform, X[need_cols], y)
            remember("cnn", dt)
            describe_sparse("cnn_branch", X_cnn if sparse.issparse(X_cnn) else X_cnn)
            print(f"   ↳ fit_transform cnn: {dt/60:.2f} min")
            try:
                cstep = getattr(cnn_pipe, "named_steps", {}).get("cnn", None)
                if cstep is not None and hasattr(cstep, "get_diagnostics"):
                    diag = cstep.get_diagnostics()
                    print(f"   ↳ CNN raw feat_dim: {diag.get('feat_dim')} | device: {diag.get('device')} "
                        f"| batch_size: {diag.get('batch_size')} | imagenet_norm: {diag.get('use_imagenet_norm')}")
            except Exception as e:
                print(f"[WARN] Impossible de lire les diag CNN: {e}")
        except Exception as e:
            print(f"[WARN] CNN non disponible: {e}")
    else:
        print("\n=== CNN SKIPPED (poids=0 ou disabled) ===")

    # --------- BRANCHE STATS IMAGE (si activée) ----------
    X_stats = None
    stats_cfg = cfg.get("images", {}).get("stats_combined", {})
    if want_stats and bool(stats_cfg.get("enabled", False)):
        print("\n=== IMAGE STATS ===")
        stats = ImageStatsCombinedFeaturizer(
            image_dir=cfg["images"]["train_dir"],   # ou test_dir selon le contexte
        imgid_col="imageid", pid_col="productid",
        white_threshold=stats_cfg.get("white_threshold", 230),
        black_threshold=stats_cfg.get("black_threshold", 25),
        min_area=stats_cfg.get("min_area", 16),
        prefix_basic="img_", prefix_pro="pro_",
        fast=bool(stats_cfg.get("fast", False)),
        fast_size=int(stats_cfg.get("fast_size", 96)),
        entropy_bins=int(stats_cfg.get("entropy_bins", 256)),
        )
        X_stats, dt = timer(stats.fit_transform, X[need_cols], y)
        remember("image_stats", dt)
        describe_sparse("img_stats_branch", X_stats)
        print(f"   ↳ fit_transform stats: {dt/60:.2f} min")
    else:
        print("\n=== IMAGE STATS SKIPPED (poids=0 ou disabled) ===")

    print("\n=== RÉCAP (comme en training) ===")
    present_branches = ["text"]
    if X_pix is not None:
        present_branches.append("image_pixels")
    if X_cnn is not None:
        present_branches.append("image_cnn")
    if X_stats is not None:
        present_branches.append("image_stats_combined")

    # Poids effectifs = uniquement ceux des branches présentes
    fusion_w = (cfg.get("fusion", {}) or {}).get("weights", {}) or {}
    effective_weights = {k: v for k, v in fusion_w.items() if k in present_branches} or None

    # Dimensions (après éventuelle réduction texte, etc.)
    def ncols(arr):
        return arr.shape[1] if arr is not None else 0

    dim_text  = ncols(X_text)   # si tu as appliqué SVD texte, c'est déjà la taille réduite
    dim_pix   = ncols(X_pix)
    dim_cnn   = ncols(X_cnn)
    dim_stats = ncols(X_stats)
    dim_total = dim_text + dim_pix + dim_cnn + dim_stats

    print(f"Branches fusionnées: {present_branches}")
    print(f"Weights effectifs  : {effective_weights}")
    print(f"Dimensions         : text={dim_text}, pixels={dim_pix}, cnn={dim_cnn}, stats={dim_stats}")
    print(f"Dimension totale attendue ≈ {dim_total}")

    print("\n=== RÉCAP TEMPS (fit_transform) ===")
    if TIMES:
        n = len(X)
        total = sum(TIMES.values())
        for k, v in sorted(TIMES.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{k:20s} : {v:6.1f}s  (~{v/60:.2f} min)  | {1000*v/n:5.1f} s / 1k échant.")
        print(f"{'-'*20}\nTOTAL{' '*15}: {total:6.1f}s  (~{total/60:.2f} min)")
    else:
        print("Aucune durée collectée (TIMES est vide).")

    # --------- HSTACK MANUEL (estimation fusion) ----------
    print("\n=== FUSION (hstack) — estimation ===")
    blocks = [b for b in [X_text, X_pix, X_cnn, X_stats] if b is not None]
    blocks_csr = [b.tocsr() if sparse.issparse(b) else sparse.csr_matrix(b) for b in blocks]
    X_all = sparse.hstack(blocks_csr).tocsr()
    describe_sparse("FUSION_total", X_all, float_bytes=8, index_bytes=4)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder

    # 1) Encoder les labels pour tous les modèles (sécurise XGB/LGBM)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    Xtr, Xva, ytr, yva = train_test_split(
        X_all, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # 2) Choisir le modèle
    if args.try_model == "lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)  # multi_class="auto"
    elif args.try_model == "svc":
        from sklearn.svm import LinearSVC
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(LinearSVC())
    elif args.try_model == "xgb":
        from xgboost import XGBClassifier
        n_classes = len(le.classes_)
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.2, max_depth=8,
            subsample=0.9, colsample_bytree=0.8, tree_method="hist",
            objective="multi:softprob", num_class=n_classes,
            n_jobs=-1
        )
    else:  # lgbm
        from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.2,
        num_leaves=127,           # ↑ plus de liberté de split
        min_child_samples=10,     # ↓ autorise des feuilles plus petites
        feature_fraction=0.8,     # sous-échantillonnage de colonnes
        bagging_fraction=0.9,     # sous-échantillonnage de lignes
        bagging_freq=1,
        colsample_bytree=0.8,     # redondant si feature_fraction, ok
        objective="multiclass",
        force_row_wise=True,      # mieux pour CSR
        verbosity=-1              # coupe le spam de logs
    )

    # 3) Fit + score
    model.fit(Xtr, ytr)
    yhat = model.predict(Xva)
    f1m = f1_score(yva, yhat, average="macro")

    print("\n=== TRY-MODEL ===")
    print(f"Model: {model.__class__.__name__} | f1_macro={f1m:.4f}")

    # 4) Afficher quelques hyperparamètres clés
    getp = getattr(model, "get_params", None)
    if getp:
        params = getp()
        keys = ["n_estimators","learning_rate","max_depth","subsample","colsample_bytree",
            "reg_alpha","reg_lambda","tree_method","C","loss","penalty","max_iter","tol",
            "num_leaves","objective","num_class"]
        short = {k: params[k] for k in keys if k in params}
    print(f"Params: {short}")

    print("OK.")
    
if __name__ == "__main__":
    main()
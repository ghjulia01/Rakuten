# tools/shap_on_embeddings_xgb.py  -- backend XGBoost (pas de dépendance shap/numba)
from __future__ import annotations
import argparse, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import xgboost as xgb

# --- shims pour unpickle des artefacts (features, main.*) ---
def _install_unpickle_shims():
    import sys, types
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # rendre 'features' importable
    if "features" not in sys.modules:
        feat_pkg = types.ModuleType("features")
        feat_pkg.__path__ = [str(ROOT / "features")]
        sys.modules["features"] = feat_pkg
    # stubs utilisés pendant l'entraînement
    main = sys.modules.setdefault("main", types.ModuleType("main"))
    if not hasattr(main, "profiling_tools"):
        pt = types.ModuleType("main.profiling_tools")
        pt.profile_func = (lambda f: f)
        pt.list_debug_add = (lambda *a, **k: None)
        sys.modules["main.profiling_tools"] = pt
        setattr(main, "profiling_tools", pt)
    tm = sys.modules.setdefault("main.train_model", types.ModuleType("main.train_model"))

    def _to32(X):
        try:
            import numpy as _np
            from scipy import sparse as sp
            return X.astype("float32") if sp.issparse(X) else _np.asarray(X, dtype=_np.float32)
        except Exception:
            return X

    ToFloat32 = type("ToFloat32", (), {
        "__module__": "main.train_model",
        "fit": lambda self, X, y=None: self,
        "transform": lambda self, X: _to32(X),
    })
    LabelEncodingClassifier = type("LabelEncodingClassifier", (), {
        "__module__": "main.train_model",
        "__init__": lambda self, estimator=None, base_estimator=None, **k:
            setattr(self, "base_estimator", estimator or base_estimator or k.get("model")),
        "fit": lambda self, X, y=None, **k: (self.base_estimator.fit(X, y, **k), setattr(self, "est_", self.base_estimator)) and self,
        "predict": lambda self, X: self.est_.predict(X),
        "predict_proba": lambda self, X: self.est_.predict_proba(X),
    })
    tm.ToFloat32 = ToFloat32
    tm.LabelEncodingClassifier = LabelEncodingClassifier
    # quelques pickles pointent vers __main__
    m = sys.modules.get("__main__")
    if m is not None:
        if not hasattr(m, "ToFloat32"): m.ToFloat32 = ToFloat32
        if not hasattr(m, "LabelEncodingClassifier"): m.LabelEncodingClassifier = LabelEncodingClassifier

def _densify(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def _abs_mean_over_classes_from_contrib(contrib: np.ndarray, d: int) -> np.ndarray:
    """
    Agrège les contributions XGBoost en importance globale par feature.
    - multi-classe: (n, d+1, K) ou (n, (d+1)*K) -> moyenne(|.|) sur n et K, drop bias
    - binaire/régression: (n, d+1) -> moyenne(|.|) sur n, drop bias
    """
    A = contrib
    if A.ndim == 3:                      # (n, d+1, K)
        A = A[:, :-1, :]                 # drop biais
        vals = np.mean(np.abs(A), axis=(0, 2))  # (d,)
    elif A.ndim == 2:
        dplus = A.shape[1]
        if dplus == d + 1:               # (n, d+1)
            A = A[:, :-1]
            vals = np.mean(np.abs(A), axis=0)
        else:                            # (n, (d+1)*K) aplati
            K = dplus // (d + 1)
            A = A.reshape(A.shape[0], d + 1, K)[:, :-1, :]
            vals = np.mean(np.abs(A), axis=(0, 2))
    else:
        raise ValueError(f"contrib ndim inattendu: {A.ndim}")
    return vals.astype(np.float32, copy=False)

# ============== MODE 1 — IMAGE ONLY (ancien mini-modèle) ==============
def run_image_only(npz_path: Path, job_path: Path, out_dir: Path, n_sample: int = 400, seed: int = 42):
    X = np.load(npz_path)["X"].astype("float32", copy=False)  # (N, d_img)
    pack = joblib.load(job_path)
    clf  = pack["model"]  # XGBClassifier fitted
    bst  = clf.get_booster()

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    Xs  = X[idx]
    dm  = xgb.DMatrix(Xs)

    contrib = bst.predict(dm, pred_contribs=True)  # SHAP natif
    vals = _abs_mean_over_classes_from_contrib(contrib, d=Xs.shape[1])  # (d_img,)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "global_importance_dims.csv", vals, delimiter=",")
    print("[ok] image-only SHAP →", out_dir / "global_importance_dims.csv")
    print("Top10 dims:", np.argsort(-vals)[:10].tolist())

# -------- helper : charger booster depuis estimator ou meta --------
def _load_booster_from_est_or_meta(est_path: Path, meta_path: Path):
    est = joblib.load(est_path)
    bst = None
    if hasattr(est, "get_booster"):
        try:
            bst = est.get_booster()
        except Exception:
            bst = None
    if bst is None:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        bpath = meta.get("booster_path") or "artifacts/xgb_full.ubj"
        bp = Path(bpath)
        if not bp.is_absolute():
            bp = Path.cwd() / bp
        if not bp.exists():
            raise RuntimeError(
                "Estimator non fitted et booster introuvable. "
                "Ré-exporte le booster :\n"
                "python tools\\export_booster_text_only.py --model artifacts\\b4.joblib "
                "--out-booster artifacts\\xgb_full.ubj --out-estimator artifacts\\final_estimator.joblib "
                "--out-text artifacts\\text_preproc.joblib --meta artifacts\\demo_meta.json"
            )
        bst = xgb.Booster()
        bst.load_model(str(bp))
    return est, bst

# ============== MODE 2 — LITE (texte + image bank) ====================
def run_lite(
    text_ct_path: Path,
    est_path: Path,
    img_npz_path: Path,
    df_csv_path: Path,
    meta_path: Path,
    out_dir: Path,
    n_sample: int = 400,
    seed: int = 42,
):
    text_ct = joblib.load(text_ct_path)
    est, bst = _load_booster_from_est_or_meta(est_path, meta_path)

    Ximg    = np.load(img_npz_path)["X_img"].astype("float32", copy=False)
    df      = pd.read_csv(df_csv_path)
    meta    = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    order   = meta.get("concat_order", ["text","image"])

    cols = [c for c in ["designation", "description"] if c in df.columns]
    if not cols:
        raise ValueError("Colonnes texte manquantes (attendues: 'designation' et/ou 'description').")
    Xt = text_ct.transform(df[cols])
    Xt = _densify(Xt).astype("float32", copy=False)

    if len(Xt) != len(Ximg):
        raise ValueError(f"Tailles mismatch: Xt={len(Xt)} vs Ximg={len(Ximg)}.")

    tw = (meta.get("transformer_weights") or {})
    w_txt = float(tw.get("text", 1.0))
    w_img = float(tw.get("image", 1.0))

    Xt  = Xt  * w_txt
    Ximg = Ximg * w_img
    parts, slices, start = [], {}, 0
    for p in order:
        if p == "text":
            parts.append(Xt);  slices["text"]  = slice(start, start + Xt.shape[1]);  start += Xt.shape[1]
        elif p == "image":
            parts.append(Ximg); slices["image"] = slice(start, start + Ximg.shape[1]); start += Ximg.shape[1]
        else:
            raise ValueError(f"Part inconnue dans concat_order: {p}")
    X = np.hstack(parts)                  # (N, d_total)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    Xs  = X[idx]
    dm  = xgb.DMatrix(Xs)

    contrib = bst.predict(dm, pred_contribs=True)           # SHAP natif
    vals = _abs_mean_over_classes_from_contrib(contrib, d=Xs.shape[1])  # (d_total,)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "global_importance_dims.csv", vals, delimiter=",")
    print("[ok] dims →", out_dir / "global_importance_dims.csv")

    # agrégation texte vs image
    rows = []
    for name, sl in slices.items():
        rows.append({"block": name, "importance": float(vals[sl].sum())})
    df_blocks = pd.DataFrame(rows)
    tot = df_blocks["importance"].sum()
    if tot > 0:
        df_blocks["share"] = df_blocks["importance"] / tot
    df_blocks.to_csv(out_dir / "global_importance_blocks.csv", index=False)
    print("[ok] blocks →", out_dir / "global_importance_blocks.csv")
    print(df_blocks)

def main():
    _install_unpickle_shims()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["image_only", "lite"], required=True)
    # image-only
    ap.add_argument("--npz", default="data/demo_images_embeddings.npz")
    ap.add_argument("--job", default="artifacts/demo_image_only_xgb.joblib")
    # lite
    ap.add_argument("--text",   default="artifacts/text_preproc.joblib")
    ap.add_argument("--est",    default="artifacts/final_estimator.joblib")
    ap.add_argument("--npz-img",default="data/demo_image_features.npz")
    ap.add_argument("--df",     default="data/demo_df_for_predict.csv")
    ap.add_argument("--meta",   default="artifacts/demo_meta.json")
    # commun
    ap.add_argument("--out", default="results/shap_demo")
    ap.add_argument("--n-sample", type=int, default=400)
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.mode == "image_only":
        run_image_only(Path(args.npz), Path(args.job), out_dir, n_sample=args.n_sample)
    else:
        run_lite(Path(args.text), Path(args.est), Path(args.npz_img),
                 Path(args.df), Path(args.meta), out_dir, n_sample=args.n_sample)

if __name__ == "__main__":
    main()
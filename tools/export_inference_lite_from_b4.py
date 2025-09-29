# tools/export_inference_lite_from_b4.py
# Usage :
#   (raku01) PS> python tools\export_inference_lite_from_b4.py ^
#       --model artifacts\b4.joblib ^
#       --images-dir streamlit_app\demo_images ^
#       --df-csv data\demo_df_for_predict.csv ^
#       --out artifacts\b4_inference_lite.joblib

from __future__ import annotations
import argparse, os, sys, json, re, types, joblib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- Shims pour unpickle ----------
def _register_unpickle_shims():
    # ToFloat32
    try:
        from main.train_model import ToFloat32 as _ToFloat32
    except Exception:
        class _ToFloat32:
            def __init__(self, *a, **k): pass
            def fit(self, X, y=None): return self
            def transform(self, X):
                import numpy as _np
                try: return _np.asarray(X, dtype=_np.float32)
                except Exception: return X
    # LabelEncodingClassifier
    try:
        from main.train_model import LabelEncodingClassifier as _LEC
    except Exception:
        class _LEC:
            def __init__(self, estimator=None, base_estimator=None, **k):
                self.estimator = estimator or base_estimator or k.get("model", None)
            def fit(self, X, y=None):
                if hasattr(self.estimator, "fit"): self.estimator.fit(X, y)
                return self
            def predict(self, X):
                if hasattr(self.estimator, "predict"): return self.estimator.predict(X)
                raise AttributeError("Stub LEC: predict indisponible.")
            def predict_proba(self, X):
                if hasattr(self.estimator, "predict_proba"): return self.estimator.predict_proba(X)
                raise AttributeError("Stub LEC: predict_proba indisponible.")
            def get_params(self, deep=True): return {"estimator": self.estimator}
            def set_params(self, **p):
                if "estimator" in p: self.estimator = p["estimator"]; return self
                return self

    # Attacher aux modules 'main' et '__main__' SANS écraser un vrai package main/
    try:
        import main as main_pkg
    except Exception:
        main_pkg = types.ModuleType("main")
        sys.modules["main"] = main_pkg
    setattr(main_pkg, "ToFloat32", _ToFloat32)
    setattr(main_pkg, "LabelEncodingClassifier", _LEC)

    running_main = sys.modules.get("__main__")
    if running_main is not None:
        if not hasattr(running_main, "ToFloat32"):
            setattr(running_main, "ToFloat32", _ToFloat32)
        if not hasattr(running_main, "LabelEncodingClassifier"):
            setattr(running_main, "LabelEncodingClassifier", _LEC)

# ---------- Utilitaires ----------
def _retarget_images_dir(obj, new_dir: str) -> int:
    """Remplace images_dir/images_base_dir/base_dir/img_dir partout dans l'objet sklearn."""
    touched, seen = 0, set()
    def _walk(o):
        nonlocal touched
        oid = id(o)
        if oid in seen: return
        seen.add(oid)

        for attr in ("images_dir", "images_base_dir", "base_dir", "img_dir"):
            if hasattr(o, attr):
                try:
                    setattr(o, attr, new_dir)
                    touched += 1
                except Exception:
                    pass

        if isinstance(o, (list, tuple, set)):
            for x in o: _walk(x)
        elif isinstance(o, dict):
            for x in o.values(): _walk(x)
        else:
            for name in ("steps", "transformer_list", "transformers"):
                if hasattr(o, name):
                    try:
                        for _, tr, _ in getattr(o, name, []): _walk(tr)
                    except Exception:
                        try:
                            for _, tr in getattr(o, name, []): _walk(tr)
                        except Exception:
                            pass
            for attr in ("transformer", "transformers_", "estimator", "base_estimator", "model"):
                if hasattr(o, attr):
                    try: _walk(getattr(o, attr))
                    except Exception: pass
    _walk(obj)
    return touched

def _strip_training_artifacts(obj) -> int:
    """
    Supprime les attributs volumineux 'cache' / 'precomputed' / 'paths_' / 'embeddings_' / 'X_' etc.
    → On ne touche PAS aux attributs nécessaires à l'inférence (poids, vocabulaires, idf_, etc.).
    """
    removed, seen = 0, set()
    big_attrs = {
        "cache", "cache_", "_cache", "precomputed", "precomputed_",
        "X_", "_X", "paths_", "all_paths", "filenames_", "indices_",
        "embeddings_", "features_", "features_cache_", "transform_cache_"
    }

    def _walk(o):
        nonlocal removed
        oid = id(o)
        if oid in seen: return
        seen.add(oid)

        # supprime attributs "gros" s'ils existent
        for a in list(getattr(o, "__dict__", {}).keys()):
            if a in big_attrs or any(a.endswith(suf) for suf in ("_cache", "_X", "_embeddings", "_features")):
                try:
                    setattr(o, a, None)
                    removed += 1
                except Exception:
                    pass

        # descendre
        if isinstance(o, (list, tuple, set)):
            for x in o: _walk(x)
        elif isinstance(o, dict):
            for x in o.values(): _walk(x)
        else:
            for name in ("steps", "transformer_list", "transformers"):
                if hasattr(o, name):
                    try:
                        for _, tr, _ in getattr(o, name, []): _walk(tr)
                    except Exception:
                        try:
                            for _, tr in getattr(o, name, []): _walk(tr)
                        except Exception:
                            pass
            for attr in ("transformer", "transformers_", "estimator", "base_estimator", "model"):
                if hasattr(o, attr):
                    try: _walk(getattr(o, attr))
                    except Exception: pass

    _walk(obj)
    return removed

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/b4.joblib", help="Chemin du gros modèle .joblib entraîné globalement")
    ap.add_argument("--images-dir", default="streamlit_app/demo_images", help="Dossier images pour l'inférence")
    ap.add_argument("--df-csv", default="data/demo_df_for_predict.csv", help="DF pour un test rapide (sanity check)")
    ap.add_argument("--out", default="artifacts/b4_inference_lite.joblib", help="Fichier .joblib de sortie (lite)")
    args = ap.parse_args()

    _register_unpickle_shims()

    # 1) Charger le gros modèle
    print(f"[info] chargement : {args.model}")
    big = joblib.load(args.model, mmap_mode="r")

    # 2) Retarget images_dir → démo
    new_dir = str(Path(args.images_dir).resolve())
    n = _retarget_images_dir(big, new_dir)
    print(f"[info] images_dir/base_dir redirigé {n} fois → {new_dir}")

    # 3) Purge des artefacts d'entraînement
    removed = _strip_training_artifacts(big)
    print(f"[info] attributs 'cache/precomputed/*' supprimés : {removed}")

    # 4) Sauvegarde "lite"
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(big, outp, compress=3)
    sz = outp.stat().st_size / (1024**2)
    print(f"[ok] écrit {outp} ({sz:.1f} MB)")

    # 5) (optionnel) Sanity check sur DF de démo
    dfp = Path(args.df_csv)
    if dfp.exists():
        print(f"[check] prédiction rapide sur {dfp} …")
        df = pd.read_csv(dfp)
        try:
            y = big.predict(df)
            uniq = pd.Series(y).nunique()
            print(f"[check] ok: {len(y)} prédictions, {uniq} classes prédites.")
        except Exception as e:
            print(f"[warn] échec du check prédiction: {e}")
    else:
        print(f"[warn] DF {dfp} absent — check sauté.")

if __name__ == "__main__":
    # limiter BLAS si besoin
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
# tools/make_demo_labels_from_model.py
# Usage :
#   1) Avec ton DF prêt pour la prédiction (recommandé) :
#      python tools\make_demo_labels_from_model.py ^
#          --model artifacts\b4.joblib ^
#          --df-csv data\demo_df_for_predict.csv ^
#          --images-dir streamlit_app\demo_images ^
#          --out data\demo_labels.csv
#
#   2) À partir de l'index JSON (fallback) :
#      python tools\make_demo_labels_from_model.py ^
#          --model artifacts\b4.joblib ^
#          --index data\demo_images_index.json ^
#          --images-dir streamlit_app\demo_images ^
#          --out data\demo_labels.csv

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --- S'assure que la racine du repo est sur sys.path ---------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Utils ---------------------------------------------------------------------------

def _normalize_dropbox(url: str) -> str:
    if "/scl/fo/" in url or "/sh/" in url:
        raise ValueError("Lien Dropbox = dossier. Fournir un lien FICHIER .joblib.")
    url = url.replace("www.dropbox.com", "dl.dropboxusercontent.com")
    if "dl=" in url:
        import re as _re
        url = _re.sub(r"dl=\d", "dl=1", url)
    else:
        url += ("&" if "?" in url else "?") + "dl=1"
    return url


def _download_to_tmp(url: str) -> str:
    # Téléchargement robuste avec retries
    from requests import Session
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import tempfile

    s = Session()
    retries = Retry(
        total=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))

    if "dropbox.com" in url:
        url = _normalize_dropbox(url)

    r = s.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        for chunk in r.iter_content(chunk_size=2**20):
            if chunk:
                tmp.write(chunk)
        return tmp.name


def _extract_ids_from_path(p: str):
    # attend .../image_<imgid>_product_<prodid>.jpg
    p_norm = p.replace("\\", "/")
    m = re.search(r"image_(\d+)_product_(\d+)\.", p_norm, flags=re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _retarget_images_dir(obj, new_dir: str) -> int:
    """
    Parcourt l'objet sklearn/pipeline et remplace les attributs images_dir/images_base_dir/base_dir/img_dir si présents.
    Retourne le nombre de remplacements effectués.
    """
    touched = 0
    seen = set()

    def _walk(o):
        nonlocal touched
        oid = id(o)
        if oid in seen:
            return
        seen.add(oid)

        # change known attrs
        for attr in ("images_dir", "images_base_dir", "base_dir", "img_dir"):
            if hasattr(o, attr):
                try:
                    setattr(o, attr, new_dir)
                    touched += 1
                except Exception:
                    pass

        # containers
        if isinstance(o, (list, tuple, set)):
            for x in o: _walk(x)
        elif isinstance(o, dict):
            for x in o.values(): _walk(x)
        else:
            # sklearn pipeline-like
            for name in ("steps", "transformer_list", "transformers"):
                if hasattr(o, name):
                    try:
                        for _, tr, _ in getattr(o, name, []):
                            _walk(tr)
                    except Exception:
                        try:
                            for _, tr in getattr(o, name, []):
                                _walk(tr)
                        except Exception:
                            pass
            for attr in ("transformer", "transformers_", "estimator", "base_estimator", "model"):
                if hasattr(o, attr):
                    try:
                        _walk(getattr(o, attr))
                    except Exception:
                        pass

    _walk(obj)
    return touched


def _register_unpickle_shims():
    """
    Assure la présence de ToFloat32 et LabelEncodingClassifier sur main.* et __main__.*,
    en important les vraies classes si dispo ; sinon stubs compatibles.
    """
    # ToFloat32
    try:
        from main.train_model import ToFloat32 as _ToFloat32
    except Exception:
        class _ToFloat32:
            def __init__(self, *args, **kwargs): pass
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
            def __init__(self, estimator=None, base_estimator=None, **kwargs):
                self.estimator = estimator or base_estimator or kwargs.get("model", None)
            def fit(self, X, y=None):
                if hasattr(self.estimator, "fit"):
                    self.estimator.fit(X, y)
                return self
            def predict(self, X):
                if hasattr(self.estimator, "predict"):
                    return self.estimator.predict(X)
                raise AttributeError("Stub LabelEncodingClassifier: predict() indisponible.")
            def predict_proba(self, X):
                if hasattr(self.estimator, "predict_proba"):
                    return self.estimator.predict_proba(X)
                raise AttributeError("Stub LabelEncodingClassifier: predict_proba() indisponible.")
            def get_params(self, deep=True):
                return {"estimator": self.estimator}
            def set_params(self, **params):
                if "estimator" in params: self.estimator = params["estimator"]
                return self

    # Attacher aux modules main et __main__
    import types as _types
    # ne pas écraser un vrai package 'main'
    try:
        import main as main_pkg
    except Exception:
        main_pkg = _types.ModuleType("main")
        sys.modules["main"] = main_pkg
    setattr(main_pkg, "ToFloat32", _ToFloat32)
    setattr(main_pkg, "LabelEncodingClassifier", _LEC)

    running_main = sys.modules.get("__main__")
    if running_main is not None:
        if not hasattr(running_main, "ToFloat32"):
            setattr(running_main, "ToFloat32", _ToFloat32)
        if not hasattr(running_main, "LabelEncodingClassifier"):
            setattr(running_main, "LabelEncodingClassifier", _LEC)


def _load_model(path_or_url: str):
    _register_unpickle_shims()
    import joblib
    if path_or_url.startswith(("http://", "https://")):
        path_or_url = _download_to_tmp(path_or_url)
    return joblib.load(path_or_url, mmap_mode="r")


# --- Main ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Chemin local ou URL vers le .joblib (gros modèle).")
    ap.add_argument("--df-csv", default=None, help="DF complet pour le pipeline (contient imageid/productid et image_rel).")
    ap.add_argument("--index", default="data/demo_images_index.json", help="Index JSON des images si --df-csv n’est pas fourni.")
    ap.add_argument("--images-dir", default="streamlit_app/demo_images", help="Répertoire contenant les images de démo.")
    ap.add_argument("--out", default="data/demo_labels.csv", help="CSV de sortie (colonnes path,label).")
    args = ap.parse_args()

    # 1) Construire le DataFrame d'entrée
    if args.df_csv:
        df = pd.read_csv(args.df_csv)
        if "image_rel" not in df.columns:
            raise KeyError("Le DF fourni doit contenir une colonne 'image_rel' pointant vers les fichiers dans --images-dir.")
        paths = df["image_rel"].astype(str).tolist()
    else:
        # fallback depuis index JSON
        idx = json.loads(Path(args.index).read_text(encoding="utf-8"))["paths"]
        rows = []
        for p in idx:
            imgid, prodid = _extract_ids_from_path(p)
            rows.append({
                "image_rel": p,
                "designation": "",
                "description": "",
                "imageid": imgid,
                "productid": prodid,
            })
        df = pd.DataFrame(rows)
        paths = [str(p) for p in idx]

    # 2) Charger le modèle (avec shims)
    mdl = _load_model(args.model)

    # 3) Rediriger le dossier d’images si nécessaire
    demo_dir = str(Path(args.images_dir).resolve())
    n = _retarget_images_dir(mdl, demo_dir)
    print(f"[info] images_dir retargeted {n} time(s) -> {demo_dir}")

    # 4) Prédire les labels
    y = mdl.predict(df)
    classes = getattr(mdl, "classes_", None)
    if classes is not None and len(classes) > 0:
        # classes peut être str ou int
        labels = [classes[int(i)] if isinstance(classes[0], (str, np.str_, int, np.integer)) else int(y[i])
                  for i in range(len(y))]
    else:
        labels = [int(v) for v in y]

    # 5) Sauvegarder
    out = pd.DataFrame({"path": paths, "label": labels})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved labels -> {out_path} | n={len(out)} | n_classes={out['label'].nunique()}")


if __name__ == "__main__":
    # Limiter le BLAS sur certaines machines
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
# tools/build_image_feature_bank.py  (FeatureUnion-friendly)
# Exemples:
#   python tools\build_image_feature_bank.py ^
#       --model artifacts\b4.joblib ^
#       --df-csv data\demo_df_for_predict.csv ^
#       --images-dir streamlit_app\demo_images ^
#       --out-npz data\demo_image_features.npz ^
#       --out-index data\demo_image_index.json ^
#       --meta artifacts\demo_meta.json

from __future__ import annotations
import argparse, sys, types, json
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- shims pour unpickle ----------
def _ensure_main_pkg():
    main_pkg = sys.modules.get("main")
    if main_pkg is None:
        main_pkg = types.ModuleType("main")
        main_pkg.__path__ = []
        sys.modules["main"] = main_pkg
    return main_pkg

def _install_profiling_tools(main_pkg):
    if "main.profiling_tools" not in sys.modules:
        pt = types.ModuleType("main.profiling_tools")
        def profile_func(f): return f
        def list_debug_add(*args, **kwargs): pass
        pt.profile_func = profile_func
        pt.list_debug_add = list_debug_add
        sys.modules["main.profiling_tools"] = pt
        setattr(main_pkg, "profiling_tools", pt)

def _install_train_model_stubs(main_pkg):
    tm = sys.modules.get("main.train_model")
    if tm is None:
        tm = types.ModuleType("main.train_model")
        sys.modules["main.train_model"] = tm
        setattr(main_pkg, "train_model", tm)

    # classes picklables (module-level)
    def _to_float32_safe(X):
        try:
            import numpy as _np
            from scipy import sparse as sp
            return X.astype("float32") if sp.issparse(X) else _np.asarray(X, dtype=_np.float32)
        except Exception:
            return X

    ToFloat32 = type("ToFloat32", (), {
        "__module__": "main.train_model",
        "__init__": lambda self, *a, **k: None,
        "fit":       lambda self, X, y=None: self,
        "transform": lambda self, X: _to_float32_safe(X),
    })
    LabelEncodingClassifier = type("LabelEncodingClassifier", (), {
        "__module__": "main.train_model",
        "__init__":   lambda self, estimator=None, base_estimator=None, **k:
                        setattr(self, "base_estimator", estimator or base_estimator or k.get("model", None)),
        "fit":        lambda self, X, y=None, **k: (self.base_estimator.fit(X, y, **k), setattr(self, "est_", self.base_estimator)) and self,
        "predict":    lambda self, X: self.est_.predict(X),
        "predict_proba": lambda self, X: self.est_.predict_proba(X),
    })
    tm.ToFloat32 = ToFloat32
    tm.LabelEncodingClassifier = LabelEncodingClassifier

def _register_unpickle_shims():
    import sys, types

    # -- Crée un "package" main importable --
    main_pkg = sys.modules.get("main")
    if main_pkg is None:
        main_pkg = types.ModuleType("main")
        main_pkg.__path__ = []  # le rendre package-like pour les sous-modules
        sys.modules["main"] = main_pkg

    # -- Sous-module: main.profiling_tools --
    if "main.profiling_tools" not in sys.modules:
        pt = types.ModuleType("main.profiling_tools")
        def profile_func(f): return f
        def list_debug_add(*args, **kwargs): pass
        pt.profile_func = profile_func
        pt.list_debug_add = list_debug_add
        sys.modules["main.profiling_tools"] = pt
        setattr(main_pkg, "profiling_tools", pt)

    # -- Sous-module: main.train_model (stubs pickle-safe au niveau module) --
    tm = sys.modules.get("main.train_model")
    if tm is None:
        tm = types.ModuleType("main.train_model")
        sys.modules["main.train_model"] = tm
        setattr(main_pkg, "train_model", tm)

    # Stubs avec __module__ fixé pour être sérialisables
    def _to_float32_safe(X):
        try:
            import numpy as _np
            from scipy import sparse as sp
            return X.astype("float32") if sp.issparse(X) else _np.asarray(X, dtype=_np.float32)
        except Exception:
            return X

    ToFloat32 = type("ToFloat32", (), {
        "__module__": "main.train_model",
        "__init__": lambda self, *a, **k: None,
        "fit":       lambda self, X, y=None: self,
        "transform": lambda self, X: _to_float32_safe(X),
    })
    LabelEncodingClassifier = type("LabelEncodingClassifier", (), {
        "__module__": "main.train_model",
        "__init__":   lambda self, estimator=None, base_estimator=None, **k:
                        setattr(self, "base_estimator", estimator or base_estimator or k.get("model", None)),
        "fit":        lambda self, X, y=None, **k: (self.base_estimator.fit(X, y, **k), setattr(self, "est_", self.base_estimator)) and self,
        "predict":    lambda self, X: self.est_.predict(X),
        "predict_proba": lambda self, X: self.est_.predict_proba(X),
    })
    tm.ToFloat32 = ToFloat32
    tm.LabelEncodingClassifier = LabelEncodingClassifier

    # -- Expose aussi les stubs sur __main__ (critique: l'objet picklé pointe vers __main__.ToFloat32) --
    running_main = sys.modules.get("__main__")
    if running_main is not None:
        if not hasattr(running_main, "ToFloat32"):
            running_main.ToFloat32 = ToFloat32
        if not hasattr(running_main, "LabelEncodingClassifier"):
            running_main.LabelEncodingClassifier = LabelEncodingClassifier

# ---------- utils de parcours ----------
def _walk(obj, seen: set, visit):
    oid = id(obj)
    if oid in seen: return
    seen.add(oid); visit(obj)
    if isinstance(obj, (list, tuple, set)):
        for x in obj: _walk(x, seen, visit)
    elif isinstance(obj, dict):
        for x in obj.values(): _walk(x, seen, visit)
    else:
        for attr in ("steps","transformer_list","transformers"):
            if hasattr(obj, attr):
                try:
                    for item in getattr(obj, attr):
                        if isinstance(item, (list, tuple)) and len(item)>=2:
                            _walk(item[-1], seen, visit)  # (name, transformer, [cols]?)
                except Exception: pass
        for attr in ("transformer","transformers_","estimator","base_estimator",
                     "model","features","featurizer","preprocessor","pipeline",
                     "best_estimator_","final_estimator","named_steps"):
            if hasattr(obj, attr):
                try:
                    _walk(getattr(obj, attr), seen, visit)
                except Exception: pass

def _get_features_union(model):
    """Retrouve l'union de features (FeatureUnion ou objet avec .transformer_list)."""
    # chemin direct : pipeline.named_steps['features'].named_steps['union']
    features = None
    if hasattr(model, "named_steps") and "features" in model.named_steps:
        features = model.named_steps["features"]
    else:
        found = []
        def visit(o):
            if hasattr(o, "named_steps") and "features" in getattr(o, "named_steps"):
                found.append(o.named_steps["features"])
        _walk(model, set(), visit)
        if found: features = found[-1]
    if features is None:
        raise RuntimeError("Step 'features' introuvable dans le modèle.")

    union = None
    if hasattr(features, "named_steps") and "union" in features.named_steps:
        union = features.named_steps["union"]
    elif hasattr(features, "transformer_list"):
        union = features
    if union is None or not hasattr(union, "transformer_list"):
        raise RuntimeError("Union de features introuvable ou invalide.")
    return union

def _get_image_branch(union):
    """Retourne la branche image complète (le transformer de l'union) :
       - priorise le nom exact 'image' si présent,
       - sinon, choisit le transformer dont l'arbre contient un objet de models.cnn_features.
       On NE renvoie PAS l'objet CNN seul, mais bien la branche entière (pour inclure SVD, normalisation, etc.)."""

    # 1) Nom exact 'image'
    for name, tr in getattr(union, "transformer_list", []):
        if name == "image":
            print(f"[info] branche image trouvée par nom: '{name}'")
            return tr

    # 2) Chercher, parmi les transformeurs de l'union, celui qui contient le CNN
    def _has_cnn_inside(obj):
        hit = False
        def visit(o):
            nonlocal hit
            if hit: return
            mod = getattr(o, "__module__", "")
            if "models.cnn_features" in mod:
                hit = True
        _walk(obj, set(), visit)
        return hit

    candidates = [(name, tr) for name, tr in getattr(union, "transformer_list", []) if _has_cnn_inside(tr)]
    if candidates:
        print(f"[info] candidats branche image dans l'union: {[n for n,_ in candidates]}")
        # on préfère une Pipeline (contient SVD/normalisation) si possible
        for name, tr in candidates:
            if hasattr(tr, "steps") or hasattr(tr, "named_steps"):
                print(f"[info] branche image retenue: '{name}' (pipeline)")
                return tr
        # sinon, on prend le premier candidat
        name, tr = candidates[0]
        print(f"[warn] branche image retenue: '{name}' (pas de pipeline détectée)")
        return tr

    # 3) Rien trouvé -> message explicite
    raise RuntimeError(
        "Branche image introuvable dans l’union. "
        f"Transformers disponibles: {', '.join(n for n,_ in getattr(union, 'transformer_list', []))}"
    )

def _retarget_images_dir(obj, new_dir: str) -> int:
    touched, seen = 0, set()
    def _walk2(o):
        nonlocal touched
        oid = id(o)
        if oid in seen: return
        seen.add(oid)
        for attr in ("images_dir","images_base_dir","base_dir","img_dir"):
            if hasattr(o, attr):
                try:
                    setattr(o, attr, new_dir); touched += 1
                except Exception: pass
        if isinstance(o, (list, tuple, set)):
            for x in o: _walk2(x)
        elif isinstance(o, dict):
            for x in o.values(): _walk2(x)
        else:
            for name in ("steps","transformer_list","transformers"):
                if hasattr(o, name):
                    try:
                        for item in getattr(o, name):
                            if isinstance(item, (list, tuple)) and len(item)>=2:
                                _walk2(item[-1])
                    except Exception: pass
            for attr in ("transformer","transformers_","estimator","base_estimator","model","named_steps"):
                if hasattr(o, attr):
                    try: _walk2(getattr(o, attr))
                    except Exception: pass
    _walk2(obj)
    return touched

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--df-csv", required=True)
    ap.add_argument("--images-dir", default="streamlit_app/demo_images")
    ap.add_argument("--out-npz", default="data/demo_image_features.npz")
    ap.add_argument("--out-index", default="data/demo_image_index.json")
    ap.add_argument("--meta", default="artifacts/demo_meta.json")
    args = ap.parse_args()

    _register_unpickle_shims()
    big = joblib.load(args.model, mmap_mode="r")

    # 1) retrouver l'union et la branche image
    union = _get_features_union(big)
    img_branch = _get_image_branch(union)

    # 2) rediriger le répertoire d'images
    demo_dir = str(Path(args.images_dir).resolve())
    n = _retarget_images_dir(img_branch, demo_dir)
    print(f"[info] images_dir/base_dir redirigé {n} fois -> {demo_dir}")

    # 3) charger le DF et transformer via la branche image (déjà fittée)
    df = pd.read_csv(args.df_csv)
    X_img = img_branch.transform(df)
    if hasattr(X_img, "toarray"):
        X_img = X_img.toarray()
    X_img = np.asarray(X_img, dtype=np.float32)
    d_img = X_img.shape[1]
    print(f"[ok] image features: shape={X_img.shape}")

    # 4) sauvegardes
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, X_img=X_img)

    # index des images → priorité à 'image_rel', sinon reconstruit depuis imageid/productid
    if "image_rel" in df.columns:
        paths = df["image_rel"].astype(str).tolist()
    else:
        paths = []
        for _, row in df.iterrows():
            try:
                fname = f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"
            except Exception:
                fname = str(row.get("image_name", ""))
            paths.append(str(Path(demo_dir) / fname))
    Path(args.out_index).write_text(json.dumps({"paths": paths}, ensure_ascii=False), encoding="utf-8")

    # meta
    meta_path = Path(args.meta)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["d_img"] = int(d_img)
    meta["concat_order"] = meta.get("concat_order", ["text","image"])
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] bank -> {args.out_npz}, index -> {args.out_index}, meta mis à jour -> {args.meta}")

if __name__ == "__main__":
    main()
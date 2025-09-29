# tools/export_booster_text_only.py
# Usage :
#   python tools\export_booster_text_only.py ^
#     --model artifacts\b4.joblib ^
#     --out-booster artifacts\xgb_full.ubj ^
#     --out-estimator artifacts\final_estimator.joblib ^
#     --out-text artifacts\text_preproc.joblib ^
#     --meta artifacts\demo_meta.json

from __future__ import annotations
import argparse, sys, types, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- helpers: safe module + stubs ----------
def _ensure_main_pkg():
    main_pkg = sys.modules.get("main")
    if main_pkg is None:
        main_pkg = types.ModuleType("main")
        main_pkg.__path__ = []  # le rendre "package-like"
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

    # Crée des classes au niveau module (pickle-safe)
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

    # Expose aussi sur __main__ (parfois référencé)
    running_main = sys.modules.get("__main__")
    if running_main is not None:
        if not hasattr(running_main, "ToFloat32"): running_main.ToFloat32 = ToFloat32
        if not hasattr(running_main, "LabelEncodingClassifier"): running_main.LabelEncodingClassifier = LabelEncodingClassifier

def _to_float32_safe(X):
    try:
        import numpy as _np
        from scipy import sparse as sp
        return X.astype("float32") if sp.issparse(X) else _np.asarray(X, dtype=_np.float32)
    except Exception:
        return X

def _register_unpickle_shims():
    main_pkg = _ensure_main_pkg()
    _install_profiling_tools(main_pkg)
    _install_train_model_stubs(main_pkg)

# ---------- walk utils ----------
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
                            _walk(item[-1], seen, visit)
                except Exception: pass
        for attr in ("transformer","transformers_","estimator","base_estimator",
                     "model","features","featurizer","preprocessor","pipeline",
                     "best_estimator_","final_estimator","est_"):
            if hasattr(obj, attr):
                try: _walk(getattr(obj, attr), seen, visit)
                except Exception: pass

# Re-binde les classes locales (__main__.<locals>.*) vers main.train_model pour qu'elles soient picklables
def _normalize_stub_classes(root):
    tm = sys.modules["main.train_model"]
    def visit(o):
        cls = getattr(o, "__class__", None)
        if cls is None: return
        mod = getattr(cls, "__module__", "")
        name = getattr(cls, "__name__", "")
        if mod.startswith("__main__") and name in {"ToFloat32", "_ToFloat32", "LabelEncodingClassifier", "_LabelEncodingClassifier"}:
            # Attache la classe au module main.train_model sous les deux noms (avec et sans _)
            setattr(tm, name.lstrip("_"), cls)
            setattr(tm, name, cls)
            try:
                cls.__module__ = "main.train_model"  # clef pour pickle
            except Exception:
                pass
    _walk(root, set(), visit)

# ---------- finders ----------
def _iter_xgb_candidates(root):
    cands = []
    def visit(o):
        if hasattr(o, "get_booster") and callable(o.get_booster):
            cands.append(o)
    _walk(root, set(), visit)
    for est in reversed(cands):
        yield est

def _find_fitted_xgb(root):
    for est in _iter_xgb_candidates(root):
        try:
            booster = est.get_booster()  # NotFittedError si non entraîné
            classes_ = getattr(est, "classes_", None)
            n_cls = getattr(est, "n_classes_", None) or (len(classes_) if classes_ is not None else None)
            return est, booster, classes_, n_cls
        except Exception:
            continue
    return None, None, None, None

def _iter_classifier_candidates(root):
    cands = []
    def visit(o):
        if hasattr(o, "predict") and not hasattr(o, "transformer_list"):
            cands.append(o)
    _walk(root, set(), visit)
    for est in reversed(cands):
        yield est

def _find_final_estimator(root):
    for est in _iter_classifier_candidates(root):
        mod = getattr(est, "__module__", "")
        if "pipeline" in mod.lower() or "compose" in mod.lower():
            continue
        classes_ = getattr(est, "classes_", None)
        n_cls = getattr(est, "n_classes_", None) or (len(classes_) if classes_ is not None else None)
        return est, classes_, n_cls
    return None, None, None

# ---------- text branch extraction ----------
def _get_text_branch_trained(big):
    # retrouver 'features'
    features = None
    if hasattr(big, "named_steps") and "features" in big.named_steps:
        features = big.named_steps["features"]
    else:
        found = []
        def visit(o):
            if hasattr(o, "named_steps") and "features" in getattr(o, "named_steps"):
                found.append(o.named_steps["features"])
        _walk(big, set(), visit)
        if found: features = found[-1]
    if features is None:
        raise RuntimeError("Step 'features' introuvable dans le pipeline.")

    # récupérer l’union
    union = None
    if hasattr(features, "named_steps") and "union" in features.named_steps:
        union = features.named_steps["union"]
    elif hasattr(features, "transformer_list"):
        union = features
    if union is None or not hasattr(union, "transformer_list"):
        raise RuntimeError("Union de features introuvable ou invalide.")

    # chercher ('text', ...)
    for name, tr in getattr(union, "transformer_list", []):
        if name == "text":
            return tr
    raise RuntimeError("Branche 'text' absente dans l’union.")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Chemin local du gros .joblib")
    ap.add_argument("--out-booster", default="artifacts/xgb_full.ubj")
    ap.add_argument("--out-estimator", default="artifacts/final_estimator.joblib")
    ap.add_argument("--out-text", default="artifacts/text_preproc.joblib")
    ap.add_argument("--meta", default="artifacts/demo_meta.json")
    args = ap.parse_args()

    _register_unpickle_shims()

    # Charger et normaliser les stubs éventuels
    big = joblib.load(args.model, mmap_mode="r")
    _normalize_stub_classes(big)

    # --- classif ---
    est, booster, classes_, n_cls = _find_fitted_xgb(big)
    meta_estimator_type = None
    if booster is not None:
        Path(args.out_booster).parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(args.out_booster)
        meta_estimator_type = "xgb_booster"
        print(f"[ok] XGBoost booster exporté -> {args.out_booster}")
    else:
        est2, classes2, n_cls2 = _find_final_estimator(big)
        est_to_dump = getattr(est2, "est_", est2)
        joblib.dump(est_to_dump, args.out_estimator, compress=3)
        classes_, n_cls = classes2, n_cls2
        meta_estimator_type = "sklearn_estimator"
        print(f"[ok] Estimateur sklearn exporté -> {args.out_estimator} ({est2.__class__.__name__})")

    # --- texte ---
    text_preproc = _get_text_branch_trained(big)
    _normalize_stub_classes(text_preproc)  # s'il contient ToFloat32/local
    # best-effort pour la dimension
    try:
        cols = ["designation","description"]
        Zmini = text_preproc.transform(pd.DataFrame({c: [""] for c in cols}))
        d_text = int(Zmini.shape[1])
    except Exception as e:
        print(f"[warn] Impossible d’estimer d_text: {e}")
        d_text = None

    Path(args.out_text).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(text_preproc, args.out_text, compress=3)
    print(f"[ok] text preproc exporté -> {args.out_text} (d_text={d_text})")

    # --- meta ---
    meta = {
        "estimator_type": meta_estimator_type,
        "booster_path": (args.out_booster if meta_estimator_type=="xgb_booster" else None),
        "estimator_path": (args.out_estimator if meta_estimator_type=="sklearn_estimator" else None),
        "d_text": (int(d_text) if d_text is not None else None),
        "num_class": (int(n_cls) if n_cls is not None else None),
        "classes_": (classes_.tolist() if classes_ is not None else None),
        "concat_order": ["text","image"]
    }
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] meta -> {args.meta}")

if __name__ == "__main__":
    main()
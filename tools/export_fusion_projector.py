# tools/export_fusion_projector.py
from __future__ import annotations
import sys, types, argparse, json
from pathlib import Path
import joblib
from pathlib import Path
import sys, types


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# fallback : assurer un package 'features' importable si besoin
if "features" not in sys.modules:
    feat_pkg = types.ModuleType("features")
    feat_pkg.__path__ = [str(ROOT / "features")]  # permet l'import de features.text_cleaner, etc.
    sys.modules["features"] = feat_pkg
def _shim():
    # minimal shims so joblib can unpickle
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
            import numpy as np
            from scipy import sparse as sp
            return X.astype("float32") if sp.issparse(X) else np.asarray(X, dtype=np.float32)
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
    # also expose on __main__ in case pickled path points there
    m = sys.modules.get("__main__")
    if m is not None:
        if not hasattr(m, "ToFloat32"): m.ToFloat32 = ToFloat32
        if not hasattr(m, "LabelEncodingClassifier"): m.LabelEncodingClassifier = LabelEncodingClassifier

def _walk(obj, seen, visit):
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
                    for it in getattr(obj, attr):
                        if isinstance(it, (list, tuple)) and len(it)>=2:
                            _walk(it[-1], seen, visit)
                except Exception: pass
        for attr in ("transformer","transformers_","estimator","base_estimator",
                     "model","features","featurizer","preprocessor","pipeline",
                     "best_estimator_","final_estimator","est_","named_steps"):
            if hasattr(obj, attr):
                try: _walk(getattr(obj, attr), seen, visit)
                except Exception: pass

def _find_main_pipeline(big):
    # try obvious: sklearn Pipeline at top
    if hasattr(big, "steps"): return big
    found = []
    _walk(big, set(), lambda o: hasattr(o, "steps") and found.append(o))
    return found[-1] if found else None

def _extract_chain_between_features_and_estimator(p):
    """
    Return a sklearn Pipeline containing every transformer AFTER 'features'
    and BEFORE the final estimator. If there is nothing in between, return None.
    """
    if not hasattr(p, "steps"): return None
    names = [n for n,_ in p.steps]
    if "features" not in names: return None
    i_feat = names.index("features")
    pre = p.steps[:i_feat+1]        # up to features
    post = p.steps[i_feat+1:]       # after features
    if not post: return None
    # drop the final estimator (last step with predict/proba)
    trans = []
    for name, step in post:
        is_est = any(hasattr(step, a) for a in ("predict","predict_proba","decision_function"))
        if is_est: break
        trans.append((name, step))
    if not trans: return None
    from sklearn.pipeline import Pipeline
    return Pipeline(trans)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-fusion", default="artifacts/fusion_projector.joblib")
    ap.add_argument("--meta", default="artifacts/demo_meta.json")
    args = ap.parse_args()

    _shim()
    big = joblib.load(args.model, mmap_mode="r")
    mp = _find_main_pipeline(big)
    if mp is None:
        raise RuntimeError("Pipeline principal introuvable.")
    fusion = _extract_chain_between_features_and_estimator(mp)
    if fusion is None:
        print("[info] aucun transformer post-fusion détecté. Rien à exporter.")
        # still write meta flag
        meta = {}
        mpath = Path(args.meta)
        if mpath.exists():
            import json
            meta = json.loads(mpath.read_text(encoding="utf-8"))
        meta["has_fusion_projector"] = False
        mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    Path(args.out_fusion).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fusion, args.out_fusion, compress=3)
    # meta
    import json
    mpath = Path(args.meta)
    meta = {}
    if mpath.exists():
        meta = json.loads(mpath.read_text(encoding="utf-8"))
    meta["has_fusion_projector"] = True
    meta["fusion_projector_path"] = args.out_fusion
    mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] fusion projector -> {args.out_fusion}")

if __name__ == "__main__":
    main()
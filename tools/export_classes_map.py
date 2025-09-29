# tools/export_classes_map.py
# python tools/export_classes_map.py --model artifacts/b4.joblib --out-json artifacts/classes_map.json --meta artifacts/demo_meta.json
from __future__ import annotations
import sys, types, argparse, json
from pathlib import Path
import joblib
import numpy as np

# --- rendre importables les modules du repo + shims d'unpickle ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# fallback 'features' pkg (certaines pickles l'importent)
if "features" not in sys.modules:
    feat_pkg = types.ModuleType("features"); feat_pkg.__path__ = [str(ROOT / "features")]
    sys.modules["features"] = feat_pkg

# shims 'main.*' utilisés pendant l'entraînement
main = sys.modules.setdefault("main", types.ModuleType("main"))
if not hasattr(main, "profiling_tools"):
    pt = types.ModuleType("main.profiling_tools")
    pt.profile_func = (lambda f: f)
    pt.list_debug_add = (lambda *a, **k: None)
    sys.modules["main.profiling_tools"] = pt
    setattr(main, "profiling_tools", pt)

def _to32(X):
    try:
        import numpy as np
        from scipy import sparse as sp
        return X.astype("float32") if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    except Exception:
        return X

tm = sys.modules.setdefault("main.train_model", types.ModuleType("main.train_model"))
ToFloat32 = type("ToFloat32", (), {
    "__module__": "main.train_model", "fit": lambda self, X, y=None: self, "transform": lambda self, X: _to32(X)
})
LabelEncodingClassifier = type("LabelEncodingClassifier", (), {
    "__module__": "main.train_model",
    "__init__": lambda self, estimator=None, base_estimator=None, **k: setattr(self, "base_estimator", estimator or base_estimator or k.get("model")),
    "fit": lambda self, X, y=None, **k: (self.base_estimator.fit(X, y, **k), setattr(self, "est_", self.base_estimator)) and self,
    "predict": lambda self, X: self.est_.predict(X),
    "predict_proba": lambda self, X: self.est_.predict_proba(X),
})
tm.ToFloat32 = ToFloat32
tm.LabelEncodingClassifier = LabelEncodingClassifier
m = sys.modules.get("__main__")
if m is not None:
    if not hasattr(m, "ToFloat32"): m.ToFloat32 = ToFloat32
    if not hasattr(m, "LabelEncodingClassifier"): m.LabelEncodingClassifier = LabelEncodingClassifier

# --- utils ---
def _walk(obj, seen, visit):
    oid = id(obj)
    if oid in seen: return
    seen.add(oid); visit(obj)
    if isinstance(obj, (list, tuple, set)):
        for x in obj: _walk(x, seen, visit)
    elif isinstance(obj, dict):
        for x in obj.values(): _walk(x, seen, visit)
    else:
        # sklearn common attributes
        for attr in ("steps","transformer_list","transformers"):
            if hasattr(obj, attr):
                try:
                    for it in getattr(obj, attr):
                        if isinstance(it, (list, tuple)) and len(it)>=2:
                            _walk(it[-1], seen, visit)
                except Exception: pass
        for attr in ("transformer","transformers_","estimator","base_estimator",
                     "model","preprocessor","pipeline","best_estimator_",
                     "final_estimator","est_","named_steps"):
            if hasattr(obj, attr):
                try: _walk(getattr(obj, attr), seen, visit)
                except Exception: pass

def _looks_like_prdtypecode(arr) -> bool:
    try:
        a = np.array(arr)
        if a.ndim != 1 or len(a) < 5 or len(a) > 200: return False
        # prdtypecode: des entiers >= 0, souvent >= 10, et pas trop serrés
        if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.str_):
            vals = a.astype(int)
            return (vals.min() >= 0) and (vals.max() >= 10)
        return False
    except Exception:
        return False

def extract_classes(big):
    """Retourne la liste ordonnée des prdtypecode (labels réels)."""
    candidates = []

    def visit(o):
        # LabelEncoder, wrapper encodant y, ou tout objet avec classes_ plausibles
        if hasattr(o, "classes_"):
            try:
                c = getattr(o, "classes_")
                if _looks_like_prdtypecode(c):
                    candidates.append(np.array(c).astype(int).tolist())
            except Exception:
                pass
        # Certains wrappers stockent label_encoder_classes_ / classes_real_
        for key in ("label_encoder_classes_", "classes_real_", "target_classes_"):
            if hasattr(o, key):
                try:
                    c = getattr(o, key)
                    if _looks_like_prdtypecode(c):
                        candidates.append(np.array(c).astype(int).tolist())
                except Exception:
                    pass

    _walk(big, set(), visit)
    if not candidates:
        raise RuntimeError("Impossible de retrouver les classes réelles (LabelEncoder).")
    # Heuristique : on prend la plus “riche” (valeurs max plus grandes)
    candidates.sort(key=lambda L: (len(L), max(L)), reverse=True)
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="artifacts/b4.joblib")
    ap.add_argument("--meta", default="artifacts/demo_meta.json")
    ap.add_argument("--out-json", default="artifacts/classes_map.json")
    args = ap.parse_args()

    big = joblib.load(args.model, mmap_mode="r")
    classes = extract_classes(big)  # liste ordonnée des prdtypecode

    # write json simple
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"classes_": classes}, indent=2), encoding="utf-8")
    print(f"[ok] classes -> {args.out_json} ({len(classes)} labels)")

    # update meta
    meta = {}
    mpath = Path(args.meta)
    if mpath.exists():
        try: meta = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception: meta = {}
    meta["classes_"] = classes
    mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] meta updated -> {args.meta}")

if __name__ == "__main__":
    main()
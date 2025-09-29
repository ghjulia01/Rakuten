# streamlit_app/inference_lite.py
import json, joblib, numpy as np, pandas as pd
from pathlib import Path

META="artifacts/demo_meta.json"
TEXT="artifacts/text_preproc.joblib"
EST="artifacts/final_estimator.joblib"
NPZ="data/demo_image_features.npz"
IDX="data/demo_image_index.json"

_text=None; _est=None; _Ximg=None; _idx=None; _meta=None

def _load():
    global _text, _est, _Ximg, _idx, _meta
    if _meta is None:
        _meta=json.loads(Path(META).read_text(encoding="utf-8"))
        _text=joblib.load(TEXT)
        _est=joblib.load(EST)
        _Ximg=np.load(NPZ)["X_img"].astype("float32", copy=False)
        _idx=json.loads(Path(IDX).read_text(encoding="utf-8"))["paths"]
    return _text, _est, _Ximg, _idx, _meta

def predict(text_row: dict, image_path: str):
    text, est, Ximg, idx, meta = _load()
    df=pd.DataFrame([text_row])
    Xt=text.transform(df)
    if hasattr(Xt,"toarray"): Xt=Xt.toarray()
    # retrouver l'index image
    try:
        k=idx.index(image_path)
    except ValueError:
        base=Path(image_path).name.lower()
        k=next((i for i,p in enumerate(idx) if Path(p).name.lower()==base), None)
        if k is None: raise KeyError("Image non trouvée dans la bank.")
    X=np.hstack([Xt.astype("float32"), Ximg[k:k+1]])
    proba=est.predict_proba(X)[0]
    labels=[str(c) for c in (meta.get("classes_") or range(len(proba)))]
    return labels[int(np.argmax(proba))], dict(zip(labels, map(float, proba)))
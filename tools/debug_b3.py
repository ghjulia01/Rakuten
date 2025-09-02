# tools/debug_b3.py
from __future__ import annotations
import os, sys
from pathlib import Path
from inspect import signature
import numpy as np                   # <-- import en haut (global)
from scipy import sparse             # <-- import en haut (global)
import pandas as pd
from tomllib import load as toml_load  # Python 3.11+

# S’assurer que la racine du projet est dans sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CFG = ROOT / "features" / "config.toml"

def _resolve_base(cfg: dict) -> Path:
    paths = cfg.get("paths", {}) or {}
    base = paths.get("base_path", None)
    if base is None:
        return ROOT
    p = Path(base)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()

def _path_from_cfg(cfg: dict, base: Path, key: str, fallback: str) -> Path:
    rel = (cfg.get("paths", {}) or {}).get(key, None)
    if rel is None:
        rel = fallback
    p = Path(rel)
    if not p.is_absolute():
        p = base / p
    return p.resolve()

def _build_imageloader(img_dir: Path):
    """Créer un ImageLoader compatible quelle que soit sa signature."""
    from features.image_loader import ImageLoader
    sig = signature(ImageLoader.__init__)
    params = sig.parameters.keys()

    kwargs = {}
    if "image_dir" in params:
        kwargs["image_dir"] = str(img_dir)
    # colonnes selon version
    if "imgid_col" in params:
        kwargs["imgid_col"] = "imageid"
    elif "img_col" in params:
        kwargs["img_col"] = "imageid"
    elif "image_col" in params:
        kwargs["image_col"] = "imageid"

    if "pid_col" in params:
        kwargs["pid_col"] = "productid"
    elif "prod_col" in params:
        kwargs["prod_col"] = "productid"
    elif "product_col" in params:
        kwargs["product_col"] = "productid"

    if "target_size" in params:
        kwargs["target_size"] = (128, 128)
    elif "size" in params:
        kwargs["size"] = (128, 128)

    if "quiet" in params:
        kwargs["quiet"] = True
    if "safe_mode" in params:
        kwargs["safe_mode"] = True
    if "flatten" in params:
        kwargs["flatten"] = True

    print("[debug] ImageLoader kwargs résolus:", kwargs)
    return ImageLoader(**kwargs)

def main():
    with open(CFG, "rb") as f:
        cfg = toml_load(f)

    base = _resolve_base(cfg)
    X_train = _path_from_cfg(cfg, base, "X_train", "data/X_train_update.csv")
    y_train = _path_from_cfg(cfg, base, "y_train", "data/Y_train_CVw08PX.csv")
    img_dir = _path_from_cfg(cfg, base, "img_train", "data/images/images/image_train")

    print("[debug] base    :", base)
    print("[debug] X_train :", X_train)
    print("[debug] y_train :", y_train)
    print("[debug] img_dir :", img_dir)

    X = pd.read_csv(X_train, index_col=0).reset_index(drop=True)
    y = pd.read_csv(y_train, index_col=0).iloc[:, 0].reset_index(drop=True)
    X = X[["productid", "imageid"]].copy()

    n = min(2000, len(X))
    idx = np.random.choice(len(X), size=n, replace=False)   # <- np OK ici
    Xs = X.loc[idx].reset_index(drop=True)
    ys = y.loc[idx].reset_index(drop=True)

    loader = _build_imageloader(img_dir)
    loader.fit(None)
    imgs = loader.transform(Xs)

    # Conversion en array pour stats
    if isinstance(imgs, np.ndarray):
        arr = imgs.reshape(len(Xs), -1)
    elif hasattr(imgs, "toarray"):
        arr = imgs.toarray()
    else:
        arr = np.asarray(imgs).reshape(len(Xs), -1)

    row_norms = np.linalg.norm(arr, axis=1)
    hit_rate  = float((row_norms > 0).mean())
    mean_var  = float(arr.var(axis=0).mean())

    print(f"[résumé] images non-nulles : {hit_rate*100:.1f}%")
    print(f"[résumé] variance moyenne  : {mean_var:.6f}")
    print(f"[résumé] classes (échant.) : {int(ys.nunique())}")

if __name__ == "__main__":
    main()
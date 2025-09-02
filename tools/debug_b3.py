# tools/debug_b3.py
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# S'assurer que la racine du projet est dans sys.path si on lance sans -m
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomllib import load as toml_load  # Python 3.11+
from inspect import signature

# === Résolution des chemins depuis le TOML =====================================

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

# === Construction d'un ImageLoader compatible quelle que soit la signature =====

def _build_imageloader(img_dir: Path):
    """Créer un ImageLoader en s'adaptant aux noms de paramètres de __init__."""
    from features.image_loader import ImageLoader  # import après sys.path
    sig = signature(ImageLoader.__init__)
    params = sig.parameters.keys()

    kwargs = {}
    # obligatoire
    if "image_dir" in params:
        kwargs["image_dir"] = str(img_dir)

    # colonnes id: plusieurs variantes possibles selon ta version
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

    # taille
    if "target_size" in params:
        kwargs["target_size"] = (128, 128)
    elif "size" in params:
        kwargs["size"] = (128, 128)

    # options de confort si disponibles
    if "quiet" in params:
        kwargs["quiet"] = True
    if "safe_mode" in params:
        kwargs["safe_mode"] = True
    if "flatten" in params:
        # on préfère récupérer des vecteurs (compatible stats ci-dessous)
        kwargs["flatten"] = True

    print("[debug] ImageLoader kwargs résolus:", kwargs)
    return ImageLoader(**kwargs)

# === Programme principal =======================================================

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

    # Charger un petit échantillon
    X = pd.read_csv(X_train, index_col=0).reset_index(drop=True)
    y = pd.read_csv(y_train, index_col=0).iloc[:, 0].reset_index(drop=True)
    X = X[["productid", "imageid"]].copy()

    n = min(2000, len(X))
    idx = np.random.choice(len(X), size=n, replace=False)
    Xs = X.loc[idx].reset_index(drop=True)
    ys = y.loc[idx].reset_index(drop=True)

    # Construire un loader compatible
    loader = _build_imageloader(img_dir)
    loader.fit(None)

    imgs = loader.transform(Xs)

    # Normaliser la sortie pour calculer les stats
    try:
        import numpy as np
        from scipy import sparse
        if isinstance(imgs, np.ndarray):
            arr = imgs.reshape(len(Xs), -1)
        elif 'sparse' in str(type(imgs)) or hasattr(imgs, 'toarray'):
            arr = imgs.toarray()
        else:
            # dernier recours : convertir en numpy
            arr = np.asarray(imgs).reshape(len(Xs), -1)
    except Exception as e:
        print("[erreur] Conversion features -> array:", e)
        return

    # Stats simples
    row_norms = np.linalg.norm(arr, axis=1)
    hit_rate  = float((row_norms > 0).mean())
    mean_var  = float(arr.var(axis=0).mean())

    print(f"[résumé] images non-nulles : {hit_rate*100:.1f}%")
    print(f"[résumé] variance moyenne  : {mean_var:.6f}")
    print(f"[résumé] classes (échant.) : {int(ys.nunique())}")

if __name__ == "__main__":
    main()

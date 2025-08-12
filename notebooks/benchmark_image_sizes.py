# notebooks/benchmark_image_sizes_cv.py
# Benchmark k-fold des tailles d'images (64, 128, 224) sur le split TRAIN officiel Rakuten
# - Branche image uniquement (pixels) + LogisticRegression
# - CV stratifiée -> F1-macro (moyenne, std)
# - Chemins reproductibles via os.path

from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

# Bootstrapping import (lancé depuis notebooks/)
ROOT = os.path.abspath("..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.image_pipeline import create_image_pipeline

# TOML loader
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# === Paths (os.path) ===
base_path = os.path.abspath("..")
cfg_candidates = [
    os.path.join(base_path, "config.toml"),
    os.path.join(base_path, "features", "config.toml"),
]
cfg = None
for p in cfg_candidates:
    if os.path.exists(p):
        with open(p, "rb") as f:
            cfg = tomllib.load(f)
        print(f"[ok] Config chargée: {p}")
        break
if cfg is None:
    raise FileNotFoundError("Aucun config.toml trouvé.")

paths = cfg.get("paths", {})
images_cfg = cfg.get("images", {})
dimred_cfg = images_cfg.get("dim_reduction", {})

X_TRAIN_CSV = os.path.join(base_path, paths.get("x_train_csv", "data/X_train_update.csv"))
X_TEST_CSV  = os.path.join(base_path, paths.get("x_test_csv", "data/X_test_update.csv"))
Y_TRAIN_CSV = os.path.join(base_path, paths.get("y_train_csv", "data/Y_train_CVw08PX.csv"))

IMG_DIR_TRAIN = os.path.join(base_path, images_cfg.get("train_dir", "data/images/images/image_train"))
IMG_DIR_TEST  = os.path.join(base_path, images_cfg.get("test_dir",  "data/images/images/image_test"))

# === Params bench ===
DEFAULT_SIZE = tuple(images_cfg.get("size", [64, 64]))   # depuis TOML
IMAGE_SIZES  = [DEFAULT_SIZE, (128,128), (224,224)]      # tu peux adapter
SAMPLE_MAX   = 8000
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# === Chargement data ===
X_train = pd.read_csv(X_TRAIN_CSV, index_col=0)
y_train = pd.read_csv(Y_TRAIN_CSV, index_col=0).squeeze()
X_test  = pd.read_csv(X_TEST_CSV,  index_col=0)

NEEDED_COLS = ["productid", "imageid"]
for c in NEEDED_COLS:
    if c not in X_train.columns or c not in X_test.columns:
        raise ValueError(f"Colonne manquante '{c}' dans X_train/X_test.")

if SAMPLE_MAX and len(X_train) > SAMPLE_MAX:
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=SAMPLE_MAX, stratify=y_train, random_state=RANDOM_STATE
    )

print(f"[ok] X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"[ok] images_train_path: {IMG_DIR_TRAIN}")
print(f"[ok] images_test_path : {IMG_DIR_TEST}")
print(f"[ok] dim_reduction cfg: {dimred_cfg}")

def bench_one_size(image_size: tuple[int,int]) -> dict:
    # split interne train/val (labels seulement côté train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train[NEEDED_COLS], y_train, test_size=TEST_SIZE, stratify=y_train, random_state=RANDOM_STATE
    )

    # branche image avec réduction depuis TOML
    img_branch_train = create_image_pipeline(
        image_dir=IMG_DIR_TRAIN, image_size=image_size, dim_reduction=dimred_cfg
    )

    # pipeline image uniquement
    pipe = FeatureUnion([("image_pixels", img_branch_train)])

    start = time.perf_counter()

    # fit_transform TRAIN
    Xtr_img = pipe.fit_transform(X_tr)
    # Scale (compat sparse)
    scaler = StandardScaler(with_mean=False)
    Xtr_img = scaler.fit_transform(Xtr_img)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    clf.fit(Xtr_img, y_tr)

    # éval sur VAL (même dossier TRAIN)
    Xval_img = pipe.transform(X_val)
    Xval_img = scaler.transform(Xval_img)
    y_pred = clf.predict(Xval_img)

    duration = time.perf_counter() - start
    f1 = f1_score(y_val, y_pred, average="macro")

    # smoke test sur dossier TEST (sans labels)
    img_branch_test = create_image_pipeline(
        image_dir=IMG_DIR_TEST, image_size=image_size, dim_reduction=dimred_cfg
    )
    pipe_test = FeatureUnion([("image_pixels", img_branch_test)])
    _ = pipe_test.fit_transform(X_test[NEEDED_COLS].head(5))

    return {"size": f"{image_size[0]}x{image_size[1]}", "f1_macro": f1, "train_time_s": duration}

# run
results = []
for size in IMAGE_SIZES:
    print(f"\n[run] Taille image = {size[0]}x{size[1]}")
    try:
        res = bench_one_size(size)
        print(f" -> F1-macro={res['f1_macro']:.4f} | temps={res['train_time_s']:.1f}s")
        results.append(res)
    except Exception as e:
        print(f" !! échec pour {size}: {e}")

df = pd.DataFrame(results)
print("\n=== Résultats ===")
print(df)

# viz rapide
import matplotlib.pyplot as plt
if len(df):
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax2 = ax1.twinx()
    ax1.bar(df["size"], df["f1_macro"], alpha=0.6, label="F1-macro")
    ax2.plot(df["size"], df["train_time_s"], marker="o", label="Temps (s)")
    ax1.set_xlabel("Taille d'image (HxW)")
    ax1.set_ylabel("F1-macro")
    ax2.set_ylabel("Temps d'entraînement (s)")
    plt.title("Benchmark tailles d'image (branche image + LR)")
    fig.tight_layout()
    plt.show()

# save
out_csv = os.path.join(base_path, "notebooks", "benchmark_image_sizes_results.csv")
df.to_csv(out_csv, index=False)
print(f"[ok] Résultats sauvegardés: {out_csv}")
# notebooks/benchmark_image_grid.py
# Grid search "léger" sur la branche image :
# - tailles d’images (HxW) × n_components (SVD/PCA)
# - LogisticRegression (class_weight='balanced')
# - Métriques : F1-macro (val), temps d’entraînement
# - Sorties : CSV + figures (courbes & heatmap)

from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

# --- Bootstrapping imports projet quand on lance depuis notebooks/ ---

base_path = Path(__file__).resolve().parents[1] 
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

from models.image_pipeline import create_image_pipeline  

# --- TOML loader (Py>=3.11 tomllib sinon tomli) ---
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# =========================
# 0) Paramètres généraux
# =========================
RANDOM_STATE = 42
TEST_SIZE    = 0.2
NEEDED_COLS  = ["productid", "imageid"]

# Grille par défaut (tu peux changer ici)
IMAGE_SIZES      = [(64, 64), (128, 128), (224, 224)]
N_COMPONENTS_GRID = [100, 150, 200, 400]
DIMRED_METHOD     = "pca"  # "svd" (TruncatedSVD, recommandé) ou "pca" (dense)

# Sous-échantillonnage optionnel pour aller plus vite (None = tout garder)
SAMPLE_MAX = 10000  # ex: 10000 ou  None pour tout garder

# =========================
# 1) Lecture config + données (chemins robustes)
# =========================
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
    raise FileNotFoundError("Aucun config.toml trouvé (racine ou features/).")

paths      = cfg.get("paths", {})
images_cfg = cfg.get("images", {})

X_TRAIN_CSV = os.path.join(base_path, paths.get("x_train_csv", "data/X_train_update.csv"))
X_TEST_CSV  = os.path.join(base_path, paths.get("x_test_csv",  "data/X_test_update.csv"))
Y_TRAIN_CSV = os.path.join(base_path, paths.get("y_train_csv", "data/Y_train_CVw08PX.csv"))

IMG_DIR_TRAIN = os.path.join(base_path, images_cfg.get("train_dir", "data/images/images/image_train"))
IMG_DIR_TEST  = os.path.join(base_path, images_cfg.get("test_dir",  "data/images/images/image_test"))

print("[paths] base_path =", base_path)
print("[paths] X_train  =", X_TRAIN_CSV)
print("[paths] y_train  =", Y_TRAIN_CSV)
print("[paths] X_test   =", X_TEST_CSV)
print("[paths] img_train=", IMG_DIR_TRAIN)
print("[paths] img_test =", IMG_DIR_TEST)

X_train = pd.read_csv(X_TRAIN_CSV, index_col=0)
y_train = pd.read_csv(Y_TRAIN_CSV, index_col=0).squeeze()
X_test  = pd.read_csv(X_TEST_CSV,  index_col=0)

# --- Adapter 'imageid' au nom réel des fichiers: image_{imageid}_product_{productid}.jpg ---
from pathlib import Path

def _to_int_str(x):
    """Nettoie l'ID (ex: '00123'/'123.0' -> '123')."""
    try:
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()

def _compose_stem(row):
    # on génère le *stem* (sans extension)
    iid = _to_int_str(row["imageid"])
    pid = _to_int_str(row["productid"])
    return f"image_{iid}_product_{pid}"

# On ne garde que les colonnes utiles et on transforme 'imageid'
X_train = X_train[["productid", "imageid"]].copy()
X_test  = X_test[["productid", "imageid"]].copy()

X_train["imageid"] = X_train.apply(_compose_stem, axis=1)
X_test["imageid"]  = X_test.apply(_compose_stem,  axis=1)

# (debug rapide) vérifier que le tout premier chemin existe bien
from pathlib import Path
_sample = Path(IMG_DIR_TRAIN) / f"{X_train['imageid'].iloc[0]}.jpg"
print("[debug] sample image exists?:", _sample, _sample.exists())

# ----------------------------------------
for c in NEEDED_COLS:
    if c not in X_train.columns or c not in X_test.columns:
        raise ValueError(f"Colonne manquante '{c}' dans X_train/X_test.")

# Option: sous-échantillonnage
if SAMPLE_MAX is not None and len(X_train) > SAMPLE_MAX:
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=SAMPLE_MAX, stratify=y_train, random_state=RANDOM_STATE
    )

print(f"[info] X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")

# Fixer un split train/val unique pour comparer équitablement tailles & n_components
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train[NEEDED_COLS], y_train, test_size=TEST_SIZE, stratify=y_train, random_state=RANDOM_STATE
)
print(f"[info] split: n_train={len(X_tr)} | n_val={len(X_val)}")

# =========================
# 2) Boucle Grid (tailles × n_components)
# =========================
def bench_one_combo(image_size: tuple[int, int], n_components: int, method: str) -> dict:
    """
    Entraîne/évalue pour (image_size, n_components).
    Retourne un dict avec F1-macro, temps, etc.
    """
    H, W = image_size
    dimred_cfg = {
        "enabled": True,
        "method": method,           # "svd" (sparse-friendly) ou "pca" (dense)
        "n_components": int(n_components),
        "random_state": RANDOM_STATE,
    }

    print(f"   -> size={H}x{W} | n_comp={n_components} | method={method}")
    img_branch_train = create_image_pipeline(
        image_dir=IMG_DIR_TRAIN, image_size=(H, W), dim_reduction=dimred_cfg
    )
    pipe = FeatureUnion([("image_pixels", img_branch_train)])

    start = time.perf_counter()
    Xtr_img = pipe.fit_transform(X_tr)
    scaler = StandardScaler(with_mean=False)  # ok pour sparse
    Xtr_img = scaler.fit_transform(Xtr_img)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    clf.fit(Xtr_img, y_tr)

    Xval_img = pipe.transform(X_val)
    Xval_img = scaler.transform(Xval_img)
    y_pred = clf.predict(Xval_img)

    dur = time.perf_counter() - start
    f1  = f1_score(y_val, y_pred, average="macro")

    # Smoke test : repointer vers le dossier TEST (pas de scoring ici)
    img_branch_test = create_image_pipeline(
        image_dir=IMG_DIR_TEST, image_size=(H, W), dim_reduction=dimred_cfg
    )
    pipe_test = FeatureUnion([("image_pixels", img_branch_test)])
    _ = pipe.transform(X_test[NEEDED_COLS].head(5))  # juste pour vérifier que ça passe
    print(f"      F1-macro={f1:.4f} | temps={dur:.1f}s")
    return {
        "image_size": f"{H}x{W}",
        "H": H, "W": W,
        "n_components": int(n_components),
        "method": method,
        "f1_macro": f1,
        "train_time_s": dur,
        "n_train": len(X_tr),
        "n_val": len(X_val),
    }

import traceback  # <-- ajoute cet import en haut si absent

results = []
print("\n[run] Grid search tailles × n_components")
for (H, W) in IMAGE_SIZES:
    for n_comp in N_COMPONENTS_GRID:
        # --- enlever le try/except pour voir l'erreur réelle ---
        print(f"   -> size={H}x{W} | n_comp={n_comp} | method={DIMRED_METHOD}")
        res = bench_one_combo((H, W), n_comp, DIMRED_METHOD)
        results.append(res)

df = pd.DataFrame(results)
print("\n=== Résultats (aperçu) ===")
print(df.head())

# =========================
# 3) Sauvegardes CSV + Figures
# =========================
OUT_DIR = os.path.join(base_path, "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)

csv_path = os.path.join(
    OUT_DIR, f"img_grid_{DIMRED_METHOD}_{'-'.join([f'{h}x{w}' for h,w in IMAGE_SIZES])}.csv"
)
df.to_csv(csv_path, index=False)
print(f"[ok] CSV sauvegardé : {csv_path}")

# Courbes F1 et temps pour chaque taille
if len(df):
    fig1, ax1 = plt.subplots(figsize=(7,4))
    for size in sorted(df["image_size"].unique()):
        s = df[df["image_size"] == size].sort_values("n_components")
        ax1.plot(s["n_components"], s["f1_macro"], marker="o", label=size)
    ax1.set_xlabel("n_components")
    ax1.set_ylabel("F1-macro (val)")
    ax1.set_title(f"F1 vs n_components — méthode {DIMRED_METHOD.upper()}")
    ax1.legend(title="Taille")
    fig1.tight_layout()
    png1 = os.path.join(OUT_DIR, f"img_grid_f1_{DIMRED_METHOD}.png")
    fig1.savefig(png1, dpi=150)
    plt.show()
    print(f"[ok] Figure F1 sauvegardée : {png1}")

    fig2, ax2 = plt.subplots(figsize=(7,4))
    for size in sorted(df["image_size"].unique()):
        s = df[df["image_size"] == size].sort_values("n_components")
        ax2.plot(s["n_components"], s["train_time_s"], marker="s", label=size)
    ax2.set_xlabel("n_components")
    ax2.set_ylabel("Temps d'entraînement (s)")
    ax2.set_title(f"Temps vs n_components — méthode {DIMRED_METHOD.upper()}")
    ax2.legend(title="Taille")
    fig2.tight_layout()
    png2 = os.path.join(OUT_DIR, f"img_grid_time_{DIMRED_METHOD}.png")
    fig2.savefig(png2, dpi=150)
    plt.show()
    print(f"[ok] Figure Temps sauvegardée : {png2}")

# Heatmap F1 : lignes = tailles, colonnes = n_components
if len(df):
    pivot = df.pivot_table(
        index="image_size", columns="n_components", values="f1_macro", aggfunc="max"
    ).reindex(index=[f"{h}x{w}" for h,w in IMAGE_SIZES])
    fig3, ax3 = plt.subplots(figsize=(1.2*len(N_COMPONENTS_GRID)+2, 0.8*len(IMAGE_SIZES)+2))
    im = ax3.imshow(pivot.values, aspect="auto")
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels(list(pivot.columns))
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(list(pivot.index))
    ax3.set_xlabel("n_components")
    ax3.set_ylabel("Taille image (HxW)")
    ax3.set_title(f"Heatmap F1-macro — méthode {DIMRED_METHOD.upper()}")
    cbar = fig3.colorbar(im, ax=ax3)
    cbar.set_label("F1-macro")
    # annotations des cellules
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax3.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig3.tight_layout()
    png3 = os.path.join(OUT_DIR, f"img_grid_heatmap_{DIMRED_METHOD}.png")
    fig3.savefig(png3, dpi=150)
    plt.show()
    print(f"[ok] Heatmap sauvegardée : {png3}")

print("\nTerminé ")
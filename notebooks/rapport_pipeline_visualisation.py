# Rapport visuel — Pipeline Texte + Images
# Génère : histogrammes avant/après, top tokens, exemples images avant/après,
# boxplots RGB, et affiche un schéma ASCII du pipeline.

from __future__ import annotations
import os
from pathlib import Path
import re, string, datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Lecture TOML (Py>=3.11: tomllib ; sinon tomli) ---
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

# --- Imports projet ---
from features.text_cleaner import TextCleaner
from features.text_vectorizer import TextTfidfVectorizer
from PIL import Image

# =========================
# 0) En-tête + Schéma
# =========================
print(f"# Rapport visuel — Pipeline Texte + Images (généré le {datetime.date.today().isoformat()})\n")

pipeline_diagram = r"""
Données brutes
   ├─ Texte (designation, description)
   │    ├─ Nettoyage (HTML→remove, lower, ponctuation, stopwords, mots vagues, stemming, traduction map)
   │    ├─ Features simples (has_description, designation_length)
   │    └─ TF-IDF (1-2 grams)
   ├─ Images (image_{imageid}_product_{productid}.jpg)
   │    ├─ Resize (H×W), RGB
   │    ├─ Normalisation [0,1]
   │    └─ Stats objet (width, height, occupancy)
   └─ Fusion
        └─ FeatureUnion([texte, image_pixels, image_stats])
             → StandardScaler(with_mean=False) → Sampling (Under/Over) → Modèle (LR / LinearSVC)
             → Évaluation / Prédiction
"""
print(pipeline_diagram)

# =========================
# 1) Config + Données
# =========================
CFG_PATHS = [
    "config.toml",
    "features/config.toml",
    "features\\config.toml",
]

cfg = None
for p in CFG_PATHS:
    if Path(p).exists():
        with open(p, "rb") as f:
            cfg = tomllib.load(f)
        print(f"[ok] Config chargée: {p}")
        break

if cfg is None:
    raise FileNotFoundError("Aucun config.toml trouvé. Place le script à la racine du projet.")

paths = cfg.get("paths", {})
text_cfg = cfg.get("text", {})
images_cfg = cfg.get("images", {})

X_train_path = paths.get("x_train_csv", "data/X_train_update.csv")
y_train_path = paths.get("y_train_csv", "data/Y_train_update.csv")

X_train = pd.read_csv(X_train_path, index_col=0)
y_train = pd.read_csv(y_train_path, index_col=0).squeeze() if Path(y_train_path).exists() else None
print(f"[ok] X_train: {X_train.shape} | y_train: {None if y_train is None else y_train.shape}")

# =========================
# 2) Texte — Longueur AVANT/APRÈS
# =========================
print("[viz] Longueur des descriptions AVANT/APRÈS nettoyage…")

tc = TextCleaner(
    remove_html=True,
    translate_map={},   # laisser vide ici (ta branche texte charge le map si besoin)
    use_stem=bool(text_cfg.get("use_stem", True))
)

desc_raw = X_train["description"].fillna("").astype(str)
len_raw = desc_raw.str.len()
len_clean = desc_raw.map(tc.clean_text).str.len()

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.hist(len_raw, bins=50)
plt.title("Longueur description — AVANT nettoyage")
plt.xlabel("nb caractères")

plt.subplot(1,2,2)
plt.hist(len_clean, bins=50)
plt.title("Longueur description — APRÈS nettoyage")
plt.xlabel("nb caractères")

plt.tight_layout()
plt.show()

# =========================
# 3) Texte — Top tokens AVANT/APRÈS (proxy Wordcloud)
# =========================
print("[viz] Top tokens AVANT/APRÈS nettoyage…")

def basic_normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def top_tokens(series, topk=30):
    c = Counter()
    for line in series:
        for t in basic_normalize(line).split():
            c[t] += 1
    return pd.Series(dict(c.most_common(topk)))

top_raw = top_tokens(desc_raw, 30)
top_clean = top_tokens(desc_raw.map(tc.clean_text), 30)

fig, axes = plt.subplots(1, 2, figsize=(14,6))
top_raw.sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("Top tokens — AVANT nettoyage")
top_clean.sort_values().plot(kind="barh", ax=axes[1])
axes[1].set_title("Top tokens — APRÈS nettoyage")
plt.tight_layout()
plt.show()

# =========================
# 4) Images — Exemples AVANT/APRÈS resize + normalisation
# =========================
print("[viz] Exemples d’images AVANT/APRÈS (resize + normalisation)…")

img_dir = images_cfg.get("train_dir", "data/images/train")
size = images_cfg.get("size", [64,64])
H, W = size[0], size[1]

def path_for(row):
    return Path(img_dir) / f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"

def load_original(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))

def load_resized_norm(path: Path, size=(H, W)) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((W, H))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr

sample = X_train.dropna(subset=["imageid"]).sample(n=min(6, len(X_train.dropna(subset=['imageid']))), random_state=42)
n = len(sample)
if n > 0:
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols*2, figsize=(cols*4, rows*3))
    axes = np.atleast_2d(axes)

    for i, (_, row) in enumerate(sample.iterrows()):
        p = path_for(row)
        ax1 = axes[i//(cols)][(i%cols)*2]
        ax2 = axes[i//(cols)][(i%cols)*2 + 1]
        try:
            orig = load_original(p)
            resz = load_resized_norm(p, (H,W))
            ax1.imshow(orig); ax1.set_title("Original"); ax1.axis("off")
            ax2.imshow(resz); ax2.set_title(f"Resized {H}×{W} (norm)"); ax2.axis("off")
        except Exception as e:
            ax1.set_title("manquante"); ax1.axis("off")
            ax2.set_title("manquante"); ax2.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("Aucun échantillon image affichable (imageid manquant).")

# =========================
# 5) Images — Boxplot RGB AVANT/APRÈS
# =========================
print("[viz] Boxplots des canaux RGB AVANT (0..255) / APRÈS (0..1)…")

def rgb_flatten(arr):  # arr: (H,W,3)
    r = arr[...,0].ravel()
    g = arr[...,1].ravel()
    b = arr[...,2].ravel()
    return r,g,b

vals_before = {"R":[], "G":[], "B":[]}
vals_after  = {"R":[], "G":[], "B":[]}

for _, row in sample.iterrows():
    p = path_for(row)
    try:
        o = load_original(p)     # 0..255 uint8
        r,g,b = rgb_flatten(o.astype(np.float32))
        vals_before["R"].extend(r); vals_before["G"].extend(g); vals_before["B"].extend(b)

        z = load_resized_norm(p, (H,W))  # 0..1 float
        r,g,b = rgb_flatten(z)
        vals_after["R"].extend(r); vals_after["G"].extend(g); vals_after["B"].extend(b)
    except Exception:
        pass

if len(vals_before["R"]) and len(vals_after["R"]):
    df_before = pd.DataFrame(vals_before).assign(stage="Avant (0..255)")
    df_after  = pd.DataFrame(vals_after).assign(stage="Après (0..1)")
    df_rgb = pd.concat([df_before, df_after], ignore_index=True)

    fig, axes = plt.subplots(1,3, figsize=(12,4), sharey=False)
    for i, ch in enumerate(["R","G","B"]):
        df_rgb.boxplot(column=ch, by="stage", ax=axes[i])
        axes[i].set_title(f"Canal {ch}")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()
else:
    print("Pas assez d’images pour tracer les boxplots RGB.")

# =========================
# 6) Encadré “Code clé” (affiché en console)
# =========================
print("\n=== Code clé (extraits 1–3 lignes) ===\n")
print("Texte — nettoyage :")
print("  from features.text_cleaner import TextCleaner")
print('  cleaned = TextCleaner().clean_text("Belle robe <i>noire</i> en coton.")')
print("\nTexte — TF-IDF :")
print("  from features.text_vectorizer import TextTfidfVectorizer")
print("  Xv = TextTfidfVectorizer(ngram_range=(1,2)).fit_transform(cleaned_series)")
print("\nImages — resize + normalisation :")
print("  arr = np.asarray(Image.open(path).convert('RGB').resize((W,H)), np.float32) / 255.0")
print("\nStats images (width/height/occupancy) :")
print("  from features.image_stats import ImageStatsFeaturizer")
print("  feat = ImageStatsFeaturizer(image_dir=img_dir).fit_transform(X_train)")
print("\n[ok] Rapport visuel généré (affichage à l’écran). Pour l’export PNG/PDF, je peux ajouter une option savefig().")
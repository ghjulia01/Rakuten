# Rapport visuel — Pipeline Texte + Images (amélioré)
from __future__ import annotations
import os, re, string, datetime, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --- Rendez le paquet projet importable même lancé depuis notebooks/ ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    os.chdir(str(REPO_ROOT))
except Exception:
    pass

# --- Lecture TOML (Py>=3.11: tomllib ; sinon tomli) ---
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

# --- Imports projet ---
from features.text_cleaner import TextCleaner
from features.text_vectorizer import TextTfidfVectorizer
from models.text_pipeline import create_text_pipeline_from_cfg
from features.image_loader import ImageLoader
from features.image_stats import ImageStatsFeaturizer

# =========================
# 0) En-tête + Schéma
# =========================
print(f"# Rapport visuel — Pipeline Texte + Images (généré le {datetime.date.today().isoformat()})\n")
pipeline_diagram = r"""
Données brutes
   ├─ Texte (designation, description)
   │    ├─ Nettoyage (HTML→remove, lower, ponctuation, stopwords, stemming, translate_map)
   │    ├─ Features simples (has_desc, title_len, text_stats, langue)
   │    └─ TF-IDF (word 1–2 grams [+ char si actif])
   ├─ Images (image_{imageid}_product_{productid}.jpg)
   │    ├─ Resize (H×W), RGB
   │    ├─ Normalisation [0,1]
   │    └─ Stats brutes (width, height, occupancy, white_ratio, black_ratio)
   └─ Fusion (B4)
        └─ FeatureUnion([texte, image_pixels, image_stats])
             → StandardScaler(with_mean=False) → Sampling → Modèle (LR / LinearSVC)
"""
print(pipeline_diagram)

# =========================
# 1) Config + Données
# =========================
CFG_CANDIDATES = [
    "features/config.toml",
    "config.toml",
]
cfg = None
for p in CFG_CANDIDATES:
    if Path(p).exists():
        with open(p, "rb") as f:
            cfg = tomllib.load(f)
        print(f"[ok] Config chargée: {p}")
        break
if cfg is None:
    raise FileNotFoundError("Aucun config TOML trouvé (essai features/config.toml).")

paths = cfg.get("paths", {})
text_cfg = cfg.get("text", {})
images_cfg = cfg.get("images", {})

X_train_path = paths.get("x_train_csv", "data/X_train_update.csv")
y_train_path = paths.get("y_train_csv", "data/Y_train_update.csv")
X_train = pd.read_csv(X_train_path, index_col=0)
y_train = pd.read_csv(y_train_path, index_col=0).squeeze() if Path(y_train_path).exists() else None
print(f"[ok] X_train: {X_train.shape} | y_train: {None if y_train is None else y_train.shape}")

# Créer dossier sortie figures
out_dir = Path("results/figures/rapport")
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2) TEXTE — AVANT/APRÈS global
# =========================
print("[viz] TEXTE — longueurs AVANT/APRÈS nettoyage")

tc = TextCleaner(
    remove_html=True,
    translate_map_path=text_cfg.get("translate_map_path", None),
    use_stem=bool(text_cfg.get("use_stem", True)),
    clean_special=bool(text_cfg.get("clean_special", True)),
    handle_emojis=bool(text_cfg.get("handle_emojis", True)),
    remove_numbers=bool(text_cfg.get("remove_numbers", False)),
)

desc_raw = X_train["description"].fillna("").astype(str)
len_raw = desc_raw.str.len()
len_clean = desc_raw.map(tc.clean_text).str.len()

plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.hist(len_raw, bins=50); plt.title("Longueur description — AVANT"); plt.xlabel("nb caractères")
plt.subplot(1,2,2); plt.hist(len_clean, bins=50); plt.title("Longueur description — APRÈS"); plt.xlabel("nb caractères")
plt.tight_layout()
plt.savefig(out_dir / "text_length_before_after.png", dpi=160)
plt.close()

# Top tokens AVANT/APRÈS
print("[viz] TEXTE — top tokens AVANT/APRÈS")
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
top_raw.sort_values().plot(kind="barh", ax=axes[0]); axes[0].set_title("Top tokens — AVANT")
top_clean.sort_values().plot(kind="barh", ax=axes[1]); axes[1].set_title("Top tokens — APRÈS")
plt.tight_layout()
plt.savefig(out_dir / "text_top_tokens_before_after.png", dpi=160)
plt.close()

# Exemple pas-à-pas sur 1 ligne
print("[viz] TEXTE — exemple pas-à-pas sur 1 item")
ex_row = X_train[['designation','description']].fillna("").astype(str).iloc[0]
ex_raw = (ex_row['designation'] + " " + ex_row['description']).strip()
ex_clean = tc.clean_text(ex_raw)
with open(out_dir / "text_example_step_by_step.txt", "w", encoding="utf-8") as f:
    f.write("=== TEXTE EXEMPLE ===\n")
    f.write("[RAW]\n" + ex_raw + "\n\n")
    f.write("[CLEAN]\n" + ex_clean + "\n")

# TF-IDF : top features par poids moyen
print("[viz] TEXTE — top TF-IDF features (poids moyen)")
tfidf = TextTfidfVectorizer(
    analyzer="word",
    max_features=int(text_cfg.get("max_features", 100_000)),
    ngram_range=(int(text_cfg.get("ngram_min", 1)), int(text_cfg.get("ngram_max", 2))),
    min_df=float(text_cfg.get("min_df", 2)),
    max_df=float(text_cfg.get("max_df", 0.95)),
    sublinear_tf=bool(text_cfg.get("sublinear_tf", True)),
    norm=str(text_cfg.get("norm", "l2")),
    strip_accents=text_cfg.get("strip_accents", "unicode"),
    lowercase=False,
    dtype="float64",
)
corpus_clean = (X_train['designation'].fillna("") + " " + X_train['description'].fillna("")).map(tc.clean_text)
Xtf = tfidf.fit_transform(corpus_clean)
feat_names = np.array(tfidf.get_feature_names_out())
# poids moyen (approx visuelle)
weights = np.asarray(Xtf.mean(axis=0)).ravel()
top_idx = weights.argsort()[-30:]
top_feat = pd.Series(weights[top_idx], index=feat_names[top_idx]).sort_values()

plt.figure(figsize=(7,10))
top_feat.plot(kind="barh"); plt.title("Top TF-IDF (poids moyen)")
plt.tight_layout()
plt.savefig(out_dir / "text_tfidf_top_features.png", dpi=160)
plt.close()

# =========================
# 3) IMAGES — AVANT/APRÈS + stats
# =========================
print("[viz] IMAGES — exemples AVANT/APRÈS + stats")
img_dir = images_cfg.get("train_dir", "data/images/train")
size = images_cfg.get("size", [128, 128])
H, W = int(size[0]), int(size[1])

def img_path_for(row):
    return Path(img_dir) / f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"

def load_original(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))

def load_resized_norm(path: Path, size=(H, W)) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((W, H))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr

with_ids = X_train.dropna(subset=["imageid","productid"])
sample = with_ids.sample(n=min(6, len(with_ids)), random_state=42)
n = len(sample)
if n > 0:
    cols = 2  # original vs preprocess
    rows = n
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    if rows == 1:
        axes = np.array([axes])
    for i, (_, row) in enumerate(sample.iterrows()):
        p = img_path_for(row)
        try:
            orig = load_original(p)
            proc = load_resized_norm(p)
            axes[i,0].imshow(orig); axes[i,0].axis("off"); axes[i,0].set_title("Original")
            axes[i,1].imshow(proc); axes[i,1].axis("off"); axes[i,1].set_title(f"Resize {H}×{W} / [0,1]")
        except Exception:
            axes[i,0].text(0.5,0.5,"(image manquante)", ha="center")
            axes[i,0].axis("off")
            axes[i,1].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "image_examples_before_after.png", dpi=160)
    plt.close()

# Boxplots RGB sur images prétraitées (échantillon)
rgb_vals = []
for _, row in sample.iterrows():
    p = img_path_for(row)
    try:
        arr = load_resized_norm(p)
        rgb_vals.append(arr.reshape(-1,3))
    except Exception:
        pass
if rgb_vals:
    rgb = np.vstack(rgb_vals)
    plt.figure(figsize=(6,4))
    plt.boxplot([rgb[:,0], rgb[:,1], rgb[:,2]], labels=["R","G","B"])
    plt.title("Distribution RGB après normalisation [0,1]")
    plt.tight_layout()
    plt.savefig(out_dir / "image_rgb_boxplot.png", dpi=160)
    plt.close()

# Stats brutes via ImageStatsFeaturizer
if len(with_ids) > 0:
    stats = ImageStatsFeaturizer(
        image_dir=img_dir,
        white_threshold=int(images_cfg.get("stats", {}).get("white_threshold", 230)),
        black_threshold=int(images_cfg.get("stats", {}).get("black_threshold", 25)),
        min_area=int(images_cfg.get("stats", {}).get("min_area", 16)),
        out_prefix="auto",
        use_cache=False,
    ).fit(with_ids)
    vals = stats.transform(with_ids.iloc[:100])  # petit échantillon
    df_stats = pd.DataFrame(vals, columns=getattr(stats, "columns_", ["width","height","occupancy","white_ratio","black_ratio"]))
    df_stats.describe().to_csv(out_dir / "image_stats_describe.csv", index=True)

    plt.figure(figsize=(7,4))
    plt.hist(df_stats["occupancy"], bins=30)
    plt.title("Histogramme occupancy (100 échantillons)")
    plt.tight_layout()
    plt.savefig(out_dir / "image_occupancy_hist.png", dpi=160)
    plt.close()

print(f"Figures et fichiers écrits dans: {out_dir}")
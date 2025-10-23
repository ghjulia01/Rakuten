# -*- coding: utf-8 -*-
"""
Streamlit – Rakuten Multimodal Dashboard 
- Supprime toute dépendance à un labels_map JSON (mapping en dur)
- Force l'utilisation de streamlit_app/demo_images pour l'affichage des images
- Corrige les erreurs de DuplicateElementId et remplace use_container_width par width
- Ajoute des caches pour accélérer le rechargement
- Ajoute un wordcloud optionnel (si le package est installé)
- Corrige quelques bugs mineurs
 python -m streamlit run streamlit_app/config.py
"""

from __future__ import annotations
import glob 
import os
import sys
import types
import json
import random
import string
import re
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
from sklearn.metrics import f1_score, confusion_matrix
import requests
import tempfile
import joblib
import math



# ----------------------------
# Constantes démo
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
IMAGES_BASE_DIR = (APP_DIR / "demo_images").resolve()  # dossier d'images démo
DEMO_CSV = IMAGES_BASE_DIR / "demo_images.csv"         # CSV de démo
RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
REP_DIR = RESULTS_DIR / "reports"
DEMO_EMB_NPZ  = "data/demo_images_embeddings.npz"
DEMO_IDX_JSON = "data/demo_images_index.json"
DEMO_XGB_JOBLIB = "artifacts/demo_image_only_xgb.joblib"

# Mapping des labels en dur (exemple basé sur votre projet)
LABEL_MAP: dict[str, str] = {
    "10": "Livres et ouvrages culturels",
    "40": "Jeux vidéo et accessoires",
    "50": "Accessoires gaming",
    "60": "Consoles rétro",
    "1140": "Figurines & licences geek",
    "1160": "Cartes à collectionner",
    "1180": "Jeux de figurines & wargames",
    "1280": "Jouets enfants & bébés",
    "1281": "Jeux et loisirs enfants",
    "1300": "Drones & modèles réduits",
    "1301": "Chaussettes & accessoires enfants",
    "1302": "Jouets / loisirs créatifs",
    "1320": "Puériculture & équipement bébé",
    "1560": "Mobilier & articles de maison",
    "1920": "Linge de maison & déco textile",
    "1940": "Alimentation & boissons",
    "2060": "Décoration saisonnière",
    "2220": "Accessoires pour animaux",
    "2280": "Magazines & journaux anciens",
    "2403": "Livres / mangas / partitions",
    "2462": "Lots JV & consoles",
    "2522": "Fournitures de papeterie",
    "2582": "Mobilier & accessoires jardin",
    "2583": "Accessoires piscines/spas",
    "2585": "Outils & jardinage",
    "2705": "Essais & livres d’histoire",
    "2905": "Jeux PC & éditions spéciales",
}

def _nice_label(lab):
    """
    Convertit un code de catégorie en libellé lisible via LABEL_MAP.
    Accepte int/str et gère le fallback si absent du mapping.
    """
    s = str(lab)
    if s.isdigit():
        i = int(s)
        return LABEL_MAP.get(s, LABEL_MAP.get(i, s))
    return LABEL_MAP.get(s, s)
# ----------------------------
# Helpers & cache
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file_or_path: Optional[str | Path]) -> pd.DataFrame:
    """Charge le CSV depuis un uploader Streamlit OU un chemin local.
    Si rien n'est fourni, tente de charger la démo (demo_images/demo_images.csv).
    """
    try:
        if uploaded_file_or_path is None:
            if DEMO_CSV.exists():
                return pd.read_csv(DEMO_CSV)
            return pd.DataFrame()
        if hasattr(uploaded_file_or_path, "read"):
            return pd.read_csv(uploaded_file_or_path)
        p = Path(uploaded_file_or_path)
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


# --- Unpickle shim for models saved with main.ToFloat32 ---
try:
    # Try the real class from your repo if available
    from main.train_model import ToFloat32 as _ToFloat32
except Exception:
    # Fallback no-op (keeps compatibility if the real class isn't importable)
    class _ToFloat32:
        def fit(self, X, y=None): 
            return self
        def transform(self, X):
            # Be tolerant: try float32 without copying; otherwise return as-is
            try:
                import numpy as _np
                return _np.asarray(X, dtype=_np.float32)
            except Exception:
                return X

ToFloat32 = _ToFloat32

@st.cache_resource
def _resnet50_model_and_transform():
    import torch
    from torchvision import transforms, models
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.eval().to(device)
    return m, tfm, device

def _resnet50_embed_one(image_abs_path: str):
    import numpy as np
    from PIL import Image
    m, tfm, device = _resnet50_model_and_transform()
    img = Image.open(image_abs_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    import torch
    with torch.no_grad():
        f = m(x).cpu().numpy().astype("float32")
    return f

def plotly_auto(fig):
    try: st.plotly_chart(fig, width='stretch')
    except TypeError: st.plotly_chart(fig, use_container_width=True)

def image_auto(obj, caption=None):
    try: st.image(obj, caption=caption, width='content')
    except TypeError: st.image(obj, caption=caption, use_container_width=True)

# Ensure unpickler can resolve "main.ToFloat32"
sys.modules.setdefault("main", types.ModuleType("main"))
setattr(sys.modules["main"], "ToFloat32", ToFloat32)

@st.cache_data(show_spinner=False)
def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"col": [], "pct_missing": []})
    pct = df.isna().mean().sort_values(ascending=False) * 100
    return pct.rename_axis("col").reset_index(name="pct_missing")

@st.cache_data(show_spinner=False)
def text_length_series(s: pd.Series) -> np.ndarray:
    if s is None or len(s) == 0:
        return np.array([])
    return s.fillna("").astype(str).str.len().values

@st.cache_data(show_spinner=False)
def top_tokens(s: pd.Series, n: int = 30) -> pd.DataFrame:
    if s is None or len(s) == 0:
        return pd.DataFrame({"token": [], "count": []})
    # nettoyage très léger pour l'aperçu
    toks = re.sub(rf"[{re.escape(string.punctuation)}]", " ", " ".join(s.fillna("").astype(str))).lower().split()
    cnt = Counter([t for t in toks if len(t) > 2])
    top = cnt.most_common(n)
    return pd.DataFrame(top, columns=["token", "count"])

@st.cache_data(show_spinner=False)
def label_distribution(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if df.empty or label_col not in df.columns:
        return pd.DataFrame({"label": [], "count": []})
    vc = df[label_col].value_counts(dropna=False)
    return vc.rename_axis("label").reset_index(name="count")

def resolve_image_path(p: str, base_dir: str | Path) -> Optional[str]:
    """Résout un chemin image relatif/absolu/URL vers un chemin exploitable par st.image."""
    if not p:
        return None
    if isinstance(p, str) and (p.startswith("http://") or p.startswith("https://")):
        return p
    p0 = Path(str(p))
    if p0.exists():
        return str(p0)
    if base_dir:
        p1 = Path(base_dir) / p0
        if p1.exists():
            return str(p1)
    return None


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# --- Inference légère : estimator sklearn + text_preproc + bank image ---
import json, joblib, numpy as np
from pathlib import Path

TEXT_PREPROC_PATH = "artifacts/text_preproc.joblib"
ESTIMATOR_PATH    = "artifacts/final_estimator.joblib"
IMG_BANK_NPZ      = "data/demo_image_features.npz"
IMG_INDEX_JSON    = "data/demo_image_index.json"
META_JSON         = "artifacts/demo_meta.json"
FUSION_PROJECTOR_PATH = "artifacts/fusion_projector.joblib"


_estimator = None
_text_ct   = None
_X_img     = None
_img_idx   = None
_meta      = None
_labels    = None
_order     = None
_fusion = None

@st.cache_resource
def _load_inference_stack():
    global _estimator, _text_ct, _X_img, _img_idx, _meta, _labels, _order, _fusion
    if _meta is None:
        _meta    = json.loads(Path(META_JSON).read_text(encoding="utf-8"))
        _text_ct = joblib.load(TEXT_PREPROC_PATH)
        _X_img   = np.load(IMG_BANK_NPZ)["X_img"].astype("float32", copy=False)
        _img_idx = json.loads(Path(IMG_INDEX_JSON).read_text(encoding="utf-8"))["paths"]
        _labels  = [str(c) for c in (_meta.get("classes_") or [])] if _meta.get("classes_") else None
        _order   = _meta.get("concat_order", ["text","image"])
        if Path(FUSION_PROJECTOR_PATH).exists():
            _fusion = joblib.load(FUSION_PROJECTOR_PATH)
        else:
            _fusion = None
        # booster/estimator loading 
        booster_path = _meta.get("booster_path")
        if booster_path and Path(booster_path).exists():
            import xgboost as xgb
            clf = xgb.XGBClassifier()
            clf.load_model(booster_path)
            _estimator = clf
        else:
            _estimator = joblib.load(ESTIMATOR_PATH)
            try:
                if hasattr(_estimator, "get_booster"):
                    _ = _estimator.get_booster()
            except Exception as e:
                raise RuntimeError("L’estimateur chargé n’est pas entraîné.") from e
        if (_labels is None) and hasattr(_estimator, "classes_"):
            _labels = [int(c) if str(c).isdigit() else str(c) for c in _estimator.classes_]
    return _estimator, _text_ct, _X_img, _img_idx, _labels, _order
def predict_estimator(text_row: dict, image_path: str):
    """
    text_row: ex. {'designation': '...', 'description': '...'}
    image_path: chemin EXACT (ou juste le basename) présent dans data/demo_image_index.json
    """
    import pandas as pd
    est, text_ct, Ximg, idx, labels, order = _load_inference_stack()

    Xt = text_ct.transform(pd.DataFrame([text_row]))
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = Xt.astype(np.float32, copy=False)

    # retrouver l'index de l'image
    try:
        k = idx.index(image_path)
    except ValueError:
        base = Path(image_path).name.lower()
        k = next((i for i, p in enumerate(idx) if Path(p).name.lower() == base), None)
        if k is None:
            raise KeyError(f"Image non trouvée dans la bank: {image_path}")

    Xi = Ximg[k:k+1]

    # concat dans l'ordre demandé par meta.concat_order
    X = None
    for part in order:
        if part == "text":
            X = Xt if X is None else np.hstack([X, Xt])
        elif part == "image":
            X = Xi if X is None else np.hstack([X, Xi])
        else:
            raise ValueError(f"Part inconnue dans concat_order: {part}")

    try:
        import xgboost as xgb
        exp = None
        if hasattr(est, "get_booster"):
            try:
                exp = est.get_booster().num_features()
            except Exception:
                exp = None
        if _fusion is not None:
            X = _fusion.transform(X)
        # If still mismatched and we know expected, fail fast with a clearer message
        if exp is not None and X.shape[1] != exp:
            raise ValueError(f"Feature shape mismatch after fusion projector: expected {exp}, got {X.shape[1]}")
    except Exception:
        # let predict_proba raise if truly inconsistent
        pass

    proba = est.predict_proba(X)[0]
    if labels is None:
        labels = [str(i) for i in range(len(proba))]
    y = labels[int(np.argmax(proba))]
    return y, dict(zip(labels, map(float, proba)))

def predict_one_text_image(model, text: str, image_path: str | None):
    """
    Prévision sur 1 produit à partir du pipeline joblib.
    Retourne (classe_predite:int, proba:dict{classe->proba}).
    """
    row = {
        "designation": text or "",
        "description": "",
        "image_rel": image_path or "",
    }
    X = pd.DataFrame([row])

    ypred = model.predict(X)[0]

    proba = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", list(range(len(probs))))
        proba = {int(c): float(p) for c, p in zip(classes, probs)}
    elif hasattr(model, "decision_function"):
        z = model.decision_function(X)
        z = z[0] if getattr(z, "ndim", 1) > 1 else z
        z = z - np.max(z)
        e = np.exp(z)
        probs = e / e.sum()
        classes = getattr(model, "classes_", list(range(len(probs))))
        proba = {int(c): float(p) for c, p in zip(classes, probs)}

    return int(ypred), proba
   
# --- Mermaid (diagrammes) ---
def render_mermaid(mermaid_text: str, height: int = 700, theme: str = "neutral"):
    import json as _json
    code = _json.dumps(mermaid_text)  # protège les caractères spéciaux
    components.html(
        f"""
        <div id="mmd" style="height:{height}px; overflow:auto;"></div>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{
            startOnLoad: false,
            securityLevel: 'loose',
            theme: '{theme}',
            flowchart: {{ htmlLabels: true, curve: 'basis' }}
          }});
          const code = {code};
          mermaid.render('graphDiv', code)
            .then((res) => {{
              // res.svg contient le SVG rendu
              document.getElementById('mmd').innerHTML = res.svg;
            }})
            .catch((e) => {{
              document.getElementById('mmd').innerHTML =
                '<pre style="color:#b91c1c">Mermaid error: ' + e.message + '</pre>';
            }});
        </script>
        """,
        height=height,
    )
def _show_table(df_, hide_index=False):
    try:
        st.dataframe(df_, width='stretch', hide_index=hide_index)
    except TypeError:
        st.dataframe(df_.reset_index(drop=True) if hide_index else df_, width='stretch')

# ----------------------------
# Sidebar – Data & Config
# ----------------------------
st.set_page_config(page_title="Rakuten Multimodal Dashboard", layout="wide")
st.title("Rakuten – Dashboard Multimodal")


st.sidebar.header("Données")

# -> chemin par défaut notebooks/df.csv s'il existe
DEFAULT_DF = Path("notebooks/df.csv")
default_path_str = str(DEFAULT_DF) if DEFAULT_DF.exists() else ""

up = st.sidebar.file_uploader("CSV des données (optionnel)", type=["csv"], accept_multiple_files=False)
path_hint = st.sidebar.text_input("…ou chemin vers un CSV local", value=default_path_str)

# priorité : upload > chemin saisi > notebooks/df.csv > démo
source = up if up is not None else (path_hint if path_hint else (DEFAULT_DF if DEFAULT_DF.exists() else None))
df = load_csv(source)
if df.empty and DEMO_CSV.exists():
    df = pd.read_csv(DEMO_CSV)
    st.sidebar.success("Dataset démo chargé : streamlit_app/demo_images/demo_images.csv")

# Répertoire images (fixe pour la démo)
st.sidebar.subheader("Répertoire des images")
images_base_dir = str(IMAGES_BASE_DIR)
st.sidebar.code(images_base_dir)

# Labels : mapping en dur uniquement
label_map = LABEL_MAP

# Mapping des colonnes
st.sidebar.subheader("Mapping des colonnes")
cols = df.columns.tolist() if not df.empty else []

def _idx(name: Optional[str]) -> int:
    return (cols.index(name) + 1) if (name and name in cols) else 0

# Texte : on tente 'designation' puis 'description'
text_default = "designation" if "designation" in cols else ("description" if "description" in cols else None)
col_text = st.sidebar.selectbox("Colonne texte (description)", options=["(aucune)"] + cols, index=_idx(text_default))

# Image : on préfère 'image_rel' (démo), sinon 'image_path'
img_default = "image_rel" if "image_rel" in cols else ("image_path" if "image_path" in cols else None)
col_img = st.sidebar.selectbox("Colonne image (facultatif)", options=["(aucune)"] + cols, index=_idx(img_default))

# Label : on préfère 'prdtypecode', sinon 'label'
lbl_default = "prdtypecode" if "prdtypecode" in cols else ("label" if "label" in cols else None)
col_lbl = st.sidebar.selectbox("Colonne cible (facultatif)", options=["(aucune)"] + cols, index=_idx(lbl_default))

# Aperçu
st.sidebar.subheader("Aperçu")
n_show = st.sidebar.slider("Lignes à afficher", 0, 50, 10)

# ----------------------------
# Onglets
# ----------------------------
expl_tab, meth_tab, diag_tab, sim_tab = st.tabs(["Exploration", "Méthode","Diagnostics modèle", "Simulation"])
# ----------------------------
# Tab 1 – Exploration
# ----------------------------
with expl_tab:
    if not df.empty:
        # === Aperçu structuré des entrées/sorties (1 large + 2 compacts) ===
        st.subheader("Jeu d'entraînement — aperçu des entrées / sorties")

        def _short_text(series, n=160):
            return (
                series.fillna("")
                .astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.slice(0, n)
            )

        # colonnes dispo
        has = set(df.columns)

        # ► mise en page : gros tableau 1 (≈70%) + deux petits (≈15% + 15%)
        col_big, col_small = st.columns([7, 3])  # ajuste à 8/3 si tu veux encore plus large

        # ------------------ Tableau 1 (large) ------------------
        with col_big:
            c1 = st.container()
            with c1:
                st.markdown("**Données d'entrée X_train (produits)**")
                wanted = ["designation", "description", "productid", "imageid"]
                cols_1 = [c for c in wanted if c in has]
                if cols_1:
                    df1 = df[cols_1].head(n_show).copy()
                    for tc in ("designation", "description"):
                        if tc in df1.columns:
                            df1[tc] = _short_text(df1[tc])
                    # on laisse l'INDEX visible ici pour numéroter les produits
                    _show_table(df1, hide_index=False)
                else:
                    st.info("Colonnes attendues manquantes : designation, description, productid, imageid.")

        # ------------------ Tableaux 2 & 3 (compacts) ------------------
        with col_small:
            c2, c3 = st.columns([1, 1])

            # [3] Labellisation des images produits — uniquement image_name, SANS index
            with c2:
                st.markdown("**Labellisation des images**")
                if "image_name" in has:
                    df2 = df[["image_name"]].dropna().head(n_show)
                    _show_table(df2, hide_index=True)
                elif "image_rel" in has:
                    from pathlib import Path
                    tmp = df["image_rel"].dropna().astype(str).map(lambda p: Path(p).name)
                    df2 = pd.DataFrame({"image_name": tmp}).head(n_show)
                    _show_table(df2, hide_index=True)
                    st.caption("Info : 'image_name' manquant, dérivé depuis 'image_rel'.")
                else:
                    st.info("Colonne 'image_name' absente.")

            # [2] Y_train — cible (27 catégories) — uniquement prdtypecode, SANS index
            with c3:
                st.markdown("**Données cibles Y_train**")
                if "prdtypecode" in has:
                    df3 = df[["prdtypecode"]].head(n_show)
                    _show_table(df3, hide_index=True)
                else:
                    st.info("Colonne 'prdtypecode' absente.")

        # Valeurs manquantes
        st.markdown(
        """
        - **Problème** : sur la marketplace Rakuten (≈10 000 vendeurs), la variabilité des libellés et des pratiques de mise en ligne génère des **erreurs de catégorisation** et un **catalogue incohérent**.  
        - **Effets** : **recherche** dégradée, recommandations moins pertinentes et **expérience utilisateur** (UX) moins fluide.  
        - **Objectif** : **classifier automatiquement** chaque produit dans la **bonne catégorie** (notre colonne cible: `prdtypecode`).  
        - **Approche** : modèle **multimodal** (*texte + image*) : titres/descriptions & visuels produits, avec **fusion** de représentations. 
        - **Données** : ~**100 000** produits : **84 916** en entraînement, **13 812** en test ; ~5 colonnes ; **35 % de NaN** sur `description` ; images nommées `image_{imageid}_product_{productid}.jpg`.  
        - **Indicateur** : **F1 pondéré** pour mesurer la performance globale en tenant compte du déséquilibre des classes.  
        - **Impact business** : meilleure **visibilité** des produits, **réduction** des coûts de modération, **mise en ligne** accélérée → **conversion** et **fidélisation** accrues.  
        - **Intérêt scientifique** : cas réel à grande échelle, propice à évaluer la **robustesse** des approches multimodales et à comparer des stratégies.
        """
        )
        # KPIs visuels (démo + dataset)
        c1, c2, c3 = st.columns(3)
        c1.metric("X_train", "84 916")
        c2.metric("X_test", "13 812")
        c3.metric("NaN(description)", "35%")


        # === Classes de produits – analyses avancées (EDA) ===
        st.divider()
        st.header("Classes de produits – analyses avancées (EDA)")

        # 1) Comptage & pourcentage par catégorie
        if col_lbl != "(aucune)" and col_lbl in df.columns:
            st.subheader("Comptage & pourcentage par catégorie")
            counts_series = df[col_lbl].value_counts().sort_values(ascending=False)
            if not counts_series.empty:
                pct = (counts_series / counts_series.sum() * 100).round(1)
                show_labels = st.toggle("Afficher libellés lisibles", value=True, key="eda_show_labels")
                x_labels = [
                    f"{c} – {label_map.get(str(c), label_map.get(int(c), ''))}" if show_labels else str(c)
                    for c in counts_series.index
                ]
                figb = px.bar(
                    x=x_labels,
                    y=counts_series.values,
                    text=[f"{v:.1f}%" for v in pct.values],
                    labels={"x": "Catégorie", "y": "# produits"},
                    title="Nombre de produits par catégorie",
                )
                figb.update_traces(textposition="outside")
                figb.update_layout(xaxis_tickangle=-45, height=520, margin=dict(t=60, b=120))
                st.plotly_chart(figb, use_container_width=True)

        # 2) Galerie d'images par catégorie (EDA avancée)
        if (col_img != "(aucune)" and col_img in df.columns and
            col_lbl != "(aucune)" and col_lbl in df.columns):

            st.subheader("Galerie d'images par catégorie")
            nb_cats = int(df[col_lbl].nunique())
            if nb_cats == 0:
                st.info("Aucune catégorie détectée.")
            else:
                max_cats = min(30, nb_cats)
                n_cats = st.slider("Nombre de catégories (échantillon)", 1, max_cats, min(6, max_cats), key="eda_ncats")
                n_per = st.slider("Images par catégorie", 2, 12, 6, key="eda_nper")

                selected_cats = st.multiselect(
                    "Limiter à certaines catégories (optionnel)",
                    options=sorted(df[col_lbl].unique().tolist()),
                    default=[],
                    key="eda_selcats",
                )
                cats = selected_cats if selected_cats else random.sample(sorted(df[col_lbl].unique().tolist()), k=n_cats)

                for cat in cats:
                    st.markdown(f"**Catégorie {cat} – {label_map.get(str(cat), label_map.get(int(cat), ''))}**")
                    sub = df[df[col_lbl] == cat]
                    if sub.empty:
                        continue
                    sample = sub.sample(min(n_per, len(sub)), random_state=42)
                    cols_img = st.columns(min(6, n_per))
                    j = 0
                    for _, r in sample.iterrows():
                        rp = resolve_image_path(str(r[col_img]), IMAGES_BASE_DIR)
                        if not rp:
                            continue
                        try:
                            with cols_img[j % len(cols_img)]:
                                if rp.startswith("http"):
                                    st.image(rp, caption=os.path.basename(rp), width='content')
                                else:
                                    img = Image.open(rp).convert("RGB")
                                    st.image(img, caption=os.path.basename(rp), width='content')
                            j += 1
                        except Exception:
                            continue

        # 3) Top mots par catégorie (nettoyage rapide + mots-clés)
        if (col_text != "(aucune)" and col_text in df.columns and
            col_lbl != "(aucune)" and col_lbl in df.columns):

            st.subheader("Top mots par catégorie (à partir des désignations)")

            @st.cache_data(show_spinner=False)
            def _build_clean(df_in: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
                base_stop = {"le","la","les","de","des","du","un","une","et","pour","avec","sur","dans","aux","au","en","par","plus","sans","set"}
                mots_vagues = {
                    "lot","vie","magic","set","produit","produits","article","pièce","pièces","new","die","life","boite","boîte","pack",
                    "format","modèle","kit","assortiment","item","tome","import","accessoire","accessoires","ensemble","collection","gamme","série",
                    "version","volumes","volume","édition","edition","édition spéciale","édition limitée","série limitée","petit","petite","grand","grande",
                    "gros","grosse","mini","maxi","super","ultra","pcs","pcs.","pc","piece","pieces","der","dernier","dernière","nouveau","nouvelle",
                    "ancien","ancienne","original","originale","noir","noire","blanc","blanche","rouge","bleu","jaune","vert","rose","orange","gris",
                    "grise","marron","violet","violette","turquoise","argent","doré","or","cuivre","beige","ivoire","auucne","aucune","aucun","aucuns",
                    "aucunes","aucunement","und","magideal","allemand","allemande","deutsch","deutsche","german","japonais","japonaise","japonaises",
                    "français","française","francais","francaises","francophone","anglais","anglaise","english","complet","complete","completes","jap",
                    "japon","sans","intégré","intégrée","intégrés","intégrées","pvc","plastique","acier","aluminium","rare","commun","communes",
                    "neuf","neuve","neuves","neufs","occasion","occasions","occasionnel","occasionnelle","occasionnels","occasionnelles","occasionnellement",
                    "générique","génériques","anti","tout","toute","tous","toutes","stream","design","home","style","mode","fashion","vol","année",
                    "années","voir","largeur","longueur","hauteur","largeure","microns","comment","extension","extensions"
                }
                def clean_text(t: str) -> str:
                    if pd.isna(t):
                        return ""
                    t = t.lower()
                    t = re.sub(rf"[{re.escape(string.punctuation)}]", " ", t)
                    t = re.sub(r"\d+", " ", t)
                    words = [w for w in t.split() if len(w) > 2 and w not in base_stop and w not in mots_vagues]
                    return " ".join(words)
                out = df_in[[text_col, label_col]].copy()
                out["__clean__"] = out[text_col].astype(str).apply(clean_text)
                return out

            clean_df = _build_clean(df, col_text, col_lbl)

            # Top 3 mots par catégorie
            top3: dict = {}
            for cat, grp in clean_df.groupby(col_lbl):
                tokens = " ".join(grp["__clean__"].tolist()).split()
                cnt = Counter(tokens) if tokens else Counter()
                top3[cat] = [w for w, _ in cnt.most_common(3)]

            # Comptages locaux
            counts_series = df[col_lbl].value_counts().sort_values(ascending=False)

            # Table top-10 (optionnelle)
            if st.checkbox("Voir le top 10 mots par catégorie", value=False, key="eda_top10"):
                rows = []
                for cat, grp in clean_df.groupby(col_lbl):
                    cnt = Counter(" ".join(grp["__clean__"]).split())
                    rows.append({
                        "cat": cat,
                        "libellé": label_map.get(str(cat), label_map.get(int(cat), "")),
                        "mots_cles": ", ".join([w for w, _ in cnt.most_common(10)])
                    })
                st.dataframe(pd.DataFrame(rows).sort_values("cat"), width='stretch')

            # Treemap
            if not counts_series.empty:
                st.subheader("Treemap catégories + 3 mots-clés")
                labels = [
                    f"{label_map.get(str(cat), label_map.get(int(cat), cat))}<br>{' • '.join(top3.get(cat, []))}"
                    for cat in counts_series.index
                ]
                treemap_df = pd.DataFrame({"label": labels, "value": counts_series.values})
                figt = px.treemap(treemap_df, path=["label"], values="value", title="Produits par catégorie (avec 3 mots-clés)")
                plotly_auto(figt)

            # Wordcloud optionnel
            if st.checkbox("Afficher le wordcloud (optionnel)", value=False, key="eda_wc"):
                try:
                    from wordcloud import WordCloud
                    combined = [
                        f"{label_map.get(str(cat), label_map.get(int(cat), cat))} " + " ".join(top3.get(cat, []))
                        for cat in counts_series.index
                    ]
                    wtext = "\n".join(combined)
                    wc = WordCloud(width=1000, height=600, background_color="white", collocations=False).generate(wtext)
                    st.image(wc.to_array(), caption="Nuage de mots par catégorie", width='stretch')
                except Exception as e:
                    st.info(f"Module 'wordcloud' indisponible ({e}). Ajoute-le à requirements si besoin.")

# ----------------------------
# Tab 2 – Méthode & Pipeline
# ----------------------------
# === Parcours animé B2 (texte) ===============================================
def show_b2_walkthrough():

    # 1) Définition des étapes
    
    # Affichage du diagramme depuis GitHub
    st.subheader("Architecture du pipeline")
    # Chemins possibles vers le SVG (priorité: à côté de config.py, puis /assets, puis chemin absolu Windows)
    CANDIDATE_SVG_PATHS = [
        Path(__file__).parent / "mermaid-flow.svg",
        Path(__file__).parent / "assets" / "mermaid-flow.svg",
        Path(r"C:\Users\colle\Dropbox\10 Learning\01 LES MINES DATASCIENTEST\Rakuten\rakuten-main\streamlit_app\mermaid-flow.svg"),
    ]

    svg_path = next((p for p in CANDIDATE_SVG_PATHS if p.exists()), None)

    if svg_path:
        try:
            # Méthode simple : Streamlit sait afficher les SVG locaux
            st.image(str(svg_path), caption="Pipeline multimodal Rakuten", width="stretch")
        except Exception:
            # Fallback robuste : on inline le contenu SVG en HTML
            svg = svg_path.read_text(encoding="utf-8")
            st.markdown(svg, unsafe_allow_html=True)
    else:
        # Si le fichier est introuvable, on met un lien (évite d'appeler st.image sur une page HTML GitHub)
        st.info("Diagramme local introuvable.")
        st.markdown(
            "[Voir le diagramme sur GitHub (version RAW recommandée)]("
            "https://raw.githubusercontent.com/ghjulia01/Rakuten/main/streamlit_app/mermaid-flow.svg)"
        )

        # Fallback avec GraphViz si l'image ne charge pas
        st.subheader("Diagramme alternatif (GraphViz)")
        dot = """
        digraph Pipeline {
            rankdir=TB;
            node [shape=box, style=filled];
            
            // Texte
            subgraph cluster_text {
                label="Pipeline Texte";
                color=lightblue;
                style=filled;
                fillcolor=lightcyan;
                
                T1 [label="designation +\\ndescription"];
                T2 [label="TextCleaner"];
                T3 [label="TF-IDF mots"];
                T4 [label="TF-IDF chars"];
                T5 [label="Features Stats"];
                T6 [label="FeatureUnion"];
                T7 [label="SVD 700D"];
                
                T1 -> T2;
                T2 -> T3;
                T2 -> T4;
                T2 -> T5;
                T3 -> T6;
                T4 -> T6;
                T5 -> T6;
                T6 -> T7;
            }
            
            // Images
            subgraph cluster_images {
                label="Pipeline Visuel";
                color=lightpink;
                style=filled;
                fillcolor=mistyrose;
                
                I1 [label="Images\\n224x224"];
                I2 [label="ViT-Base"];
                I3 [label="Fine-tuning"];
                I4 [label="Embeddings\\n768D"];
                I5 [label="SVD 256D"];
                
                I1 -> I2;
                I2 -> I3;
                I3 -> I4;
                I4 -> I5;
            }
            
            // Fusion
            F1 [label="Fusion\\nPondérée", fillcolor=lightyellow];
            F2 [label="Sampling", fillcolor=lightgreen];
            F3 [label="XGBoost", fillcolor=lightgreen];
            F4 [label="F1=0.83", fillcolor=lightgreen];
            
            T7 -> F1 [label="poids 2.2"];
            I5 -> F1 [label="poids 1.3"];
            F1 -> F2;
            F2 -> F3;
            F3 -> F4;
        }
        """
        st.graphviz_chart(dot, use_container_width=True)

    st.divider()

    # 1) Définition des étapes TEXTE
    steps_text = [
        {
            "title": "Fusion des colonnes textuelles"
            ,
            "bullets": [
                "Entrée : designation + description des produits",
                "Sortie : Texte combiné pour analyse unifiée",
                "Implémentation : Concaténation simple avec gestion des valeurs nulles",
                "****Exemple**** :",
                "***avant*** :", 
                "**designation** : 'Powerbank <br> externe 10 000 mAh – câble INCLUS & \\garantie\\ 12mois USB-C fast charge!!!'\n",
                "**description** : 'USB-C, 18W. Câble inclus. Compat. iPhone/Android.'",
                "***après*** :", 
                "***texte_fusionné*** : 'Powerbank <br> externe 10 000 mAh – câble INCLUS & \\garantie\\ 12mois USB-C fast charge!!! USB-C 18W, câble inclus, compatible iPhone et Android.",
                "****Impact****: Meilleure couverture des cooccurrences (ex. 'USB-C')."
            ],
        },
        {
            "title": "Nettoyage et normalisation (TextCleaner)",
            "bullets": [
                "HTML/Balises : Suppression des tags <br>, <p>, entités HTML",
                "Traduction multilingue : Dictionnaire FR/EN/DE (500+ termes)",
                "Normalisation Unicode : Gestion des accents, caractères spéciaux",
                "Tokenisation et stemming : Snowball stemmer multilingue",
                "****Exemple**** :",
                "'Powerbank <br> externe 10 000 mAh – câble INCLUS & \\garantie\\ 12mois USB-C fast charge!!!'\n",
                "**nettoyé+normalisé** : 'powerbank externe 10 000 mah  fast charge cable inclus garantie 12 mois fast charge usb c'\n",
                "***lexique FR/EN/DE*** :'batterie externe 10 000 mah  charge rapide cable inclus garantie 12 mois charge rapide usb c'", 
                "***stemming*** : 'batteri extern 10000 mah charg rapid cabl inclu garanti 12 moi charg rapid usb c'",
                "****Impact****: Vocabulaire plus court et cohérent, meilleure généralisation inter-vendeurs/langues"
            ],
        },
        {
            "title": "Vectorisation TF-IDF (branche mots)",
            "bullets": [
                "N-grammes : Unigrammes (1) + Bigrammes (2)",
                "Paramètres : max_features=100k, min_df=5, max_df=0.95",
                "Sublinéarité : log(1 + tf) pour atténuer les termes très fréquents",
                "Normalisation L2 : Vecteurs unitaires pour la stabilité",
                "****Exemple**** :",
                "**Top n-grammes ** : 'charg rapid', 'usb c', 'batteri extern'\n",
                "***Poids *** : charg rapid=0.42, usb c=0.38, batterie externe=0.31\n", 
                "****Impact****: Capture des termes porteurs de sens pour séparer des catégories proches"

            ],
        },
        {
            "title": "Vectorisation caractères (branche char)",
            "bullets": [
                "Analyseur : char_wb (caractères au niveau des mots)",
                "N-grammes : 2 à 5 caractères consécutifs",
                "Robustesse : Gestion des fautes de frappe et variantes",
                "Complémentarité : Capture les patterns non captés par les mots"
                "****Exemple**** :",
                "**Top n-grammes ** : ch, ar, arg, rg, ge, er, re, es, se, ur, ra, at, te, ec, ca, ab, bl, le\n",
                "****Impact****: Amélioration de la robustesse aux erreurs typographiques et variantes orthographiques"
            ],
        },
        {
            "title": "Features additionnelles textuelles",
            "bullets": [
                "Présence description : Flag binaire has_description",
                "Longueur titre : designation_length (nombre de caractères)", 
                "Statistiques Pro : 25 features (ratios, patterns, domaines métier)",
                "Lexiques Chi2 : Top-80 mots discriminants par classe",
                "****Exemple**** :",
                " 'has_description=1, designation_length=34, ratio_digits=0.18,'\n",
                " 'lexicon_power_accessories=1, , gaming_flag=0, , baby_flag=0, , garden_flag=0, has_isbn=0 ...'\n",
                "****Impact****: Signaux différents des TF-IDF (chiffres, unités, motifs, thématiques) pour désambiguïser"


            ],
        },
        {
            "title": "Fusion pondérée texte (FeatureUnion)",
            "bullets": [
                "Branches combinées : tfidf_word=1.0 + tfidf_char=0.8 + has_desc=0.2+ title_len=0.2+ stats_pro=0.6 + lexicons=0.4",
                "Réduction SVD : 700 composantes + normalisation L2 afin que les textes, qu’ils soient courts ou longs, contribuent de manière équitable à la classification.",
                "Matrice sparse optimisée : ~700D pour classification efficace"
            ],
        },
    ]

    # 2) Définition des étapes VISION
    steps_vision = [
        {
            "title": "Chargement intelligent des images: Les images sont ouvertes avec PIL, converties en RGB, puis redimensionnées selon le modèle choisi (224x224 pour ViT).",
            "bullets": [
                "Reconstruction chemins : image_{imageid}_product_{productid}.jpg",
                "Preprocessing HuggingFace : Auto-resize selon le modèle ViT",
                "Fallback robuste : Vecteur zéro si image manquante/corrompue"
            ],
        },
        {
            "title": "Vision Transformer (ViT-Base)",
            "bullets": [
                "Modèle : google/vit-base-patch16-224 pré-entraîné",
                "Architecture : 12 couches, 768 dimensions, patches 16×16",
                "Token [CLS] : Extraction de l'embedding global de classification"
            ],
        },
        {
            "title": "Fine-tuning supervisé",
            "bullets": [
                "Couches dégelées : 2 derniers blocs Transformer",
                "Augmentations : RandomHorizontalFlip(0.2), MixUp(0.1)",
                "Optimisation : AdamW, early stopping, label smoothing(0.05)"
            ],
        },
        {
            "title": "Post-traitement des embeddings",
            "bullets": [
                "Normalisation L2 : Stabilisation des features ViT",
                "Réduction SVD : 768D → 256D pour compression",
                "Conversion sparse : Dense → CSR pour efficacité mémoire"
            ],
        },
    ]

    # 3) Fusion finale
    steps_fusion = [
        {
            "title": "Fusion multimodale pondérée",
            "bullets": [
                "Concaténation : [Features_texte_700D, Features_ViT_256D]",
                "Pondération : texte=2.2 (dominant), ViT=1.3 (complémentaire)",
                "Dimensions finales : ~956D combinés pour classification"
            ],
        },
        {
            "title": "Rééquilibrage des classes",
            "bullets": [
                "UnderSampling adaptatif : Classe 2583 limitée à 2500 échantillons", 
                "OverSampling ciblé : Classes <800 échantillons remontées à 900",
                "Conservation des patterns : Préservation de la distribution naturelle"
            ],
        },
        {
            "title": "Classification XGBoost finale",
            "bullets": [
                "Hyperparamètres : 2000 arbres, lr=0.05, early stopping",
                "Objectif : Maximisation F1-weighted (métrique cible Rakuten)",
                "Performance : F1-weighted = 0.84 (validation 3-fold)"
            ],
        },
    ]

    # Interface à onglets pour les 3 parties
    tab_text, tab_vision, tab_fusion = st.tabs(["Pipeline Texte", "Pipeline Visuel", "Fusion Multimodale"])
    
    with tab_text:
        st.markdown("### Pipeline Texte - 6 Étapes")
        for i, step in enumerate(steps_text, 1):
            with st.expander(f"Étape {i}: {step['title']}", expanded=(i==1)):
                for bullet in step["bullets"]:
                    st.markdown(f"• {bullet}")

    with tab_vision:  
        st.markdown("### Pipeline Visuel - Architecture ViT - 4 Étapes") 
        for i, step in enumerate(steps_vision, 1):
            with st.expander(f"Étape {i}: {step['title']}", expanded=(i==1)):
                for bullet in step["bullets"]:
                    st.markdown(f"• {bullet}")

    with tab_fusion:
        st.markdown("### Fusion Finale - Pipeline Multimodal Texte et Images - 3 Étapes")
        for i, step in enumerate(steps_fusion, 1):
            with st.expander(f"Étape {i}: {step['title']}", expanded=(i==1)):
                for bullet in step["bullets"]:
                    st.markdown(f"• {bullet}")

    st.divider()

    # ====== Méthode · Section "F1-weighted par baseline" ======
    def render_f1_weighted_overview():
        st.subheader("Comparatif F1-weighted par baseline (B0 → B4)")
        st.caption("F1-weighted = moyenne pondérée des F1 par classe, pondérée par le support (nombre d’instances).")

        # --- Helpers ---
        def _read_preds(path: Path):
            if not path or not path.exists():
                return None
            dfp = pd.read_csv(path)
            cols = {c.lower(): c for c in dfp.columns}
            y_true = next((cols[k] for k in cols if k in ("y_true","y","label","target","prdtypecode_true")), None)
            y_pred = next((cols[k] for k in cols if k in ("y_pred","pred","prediction","prdtypecode_pred")), None)
            if not (y_true and y_pred):
                st.warning(f"Colonnes y_true / y_pred absentes dans {path.name}.")
                return None
            return dfp[y_true].astype(str), dfp[y_pred].astype(str)

        # --- B0 & B1 : valeurs fournies ---
        rows = [
            {"Code": "B0", "Baseline": "Naïf (majoritaire)",
            "Idée principale": "Toujours prédire la classe la plus fréquente.",
            "F1-weighted (%)": 2.6},
            {"Code": "B1", "Baseline": "Naïf (stratifié)",
            "Idée principale": "Tirage aléatoire respectant la distribution des classes.",
            "F1-weighted (%)": 5.2},
        ]

        # --- B2/B3/B4 : calcul à partir des prédictions ---
        P_B2 = Path("results/preds_b2.csv")
        P_B3 = Path("results/preds_b3.csv")
        P_B4 = next((p for p in [Path("results/preds_oof_b4.csv"), Path("results/preds_b4.csv")] if p.exists()), None)

        for code, base, idea, path in [
            ("B2", "Pipeline Texte seule", "Texte uniquement", P_B2),
            ("B3", "Pipeline Image seule", "Image (ViT)", P_B3),
            ("B4", "Pipeline Multimodale", "Texte + Image", P_B4),
        ]:
            data = _read_preds(path) if path else None
            if data:
                y_true, y_pred = data
                f1w = f1_score(y_true, y_pred, average="weighted") * 100.0
                rows.append({"Code": code, "Baseline": base, "Idée principale": idea,
                         "F1-weighted (%)": round(f1w, 2)})
            else:
                # Affiche quand même la ligne (valeur manquante) pour transparence
                rows.append({"Code": code, "Baseline": base, "Idée principale": idea,
                            "F1-weighted (%)": None})

        df = pd.DataFrame(rows)
        order = ["B0","B1","B2","B3","B4"]
        df["Code"] = pd.Categorical(df["Code"], categories=order, ordered=True)
        df = df.sort_values("Code")

        # --- Tableau récapitulatif ---
        st.dataframe(df, use_container_width=True)

        # --- Bar chart ---
        df_plot = df.dropna(subset=["F1-weighted (%)"])
        if len(df_plot) == 0:
            st.info("Aucune valeur disponible pour tracer le graphique (vérifie les fichiers preds_b2/b3/b4).")
            return
        # --- Ticks personnalisés sur l'axe X : Code — Baseline (ou juste Baseline) ---
        df_plot["CodeBaseline"] = df_plot["Code"].astype(str) + " — " + df_plot["Baseline"]

        fig = px.bar(
            df_plot,
            x="CodeBaseline",
            y="F1-weighted (%)",
            hover_data=["Baseline", "Idée principale"],
            text=df_plot["F1-weighted (%)"].map(lambda v: f"{v:.1f}%"),
            title="F1-weighted (%) par baseline",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        ymax = max(10.0, float(df_plot["F1-weighted (%)"].max()) * 1.15)
        fig.update_yaxes(range=[0, ymax], title_text="F1-weighted (%)")
        fig.update_xaxes(title_text="Baseline")
        fig.update_layout(margin=dict(t=60,b=20,l=0,r=0), height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Petit rappel méthodo sous le graphe
        st.caption("Interprétation : plus le F1-weighted est élevé, meilleure est la performance globale en tenant compte des classes fréquentes.")
    # --- Afficher la section F1 B0→B4 ---
    st.divider()
    render_f1_weighted_overview()

    # 4) Résultats (F1 + matrice de confusion si dispo)
    preds_path = Path("results/preds_oof_b4.csv")  # Changé pour b4
    if preds_path.exists():
        try:
            dfp = pd.read_csv(preds_path)
            # on cherche y_true/y_pred
            y_true_col = next((c for c in dfp.columns if c.lower() in ("y_true","y","label","target","prdtypecode_true")), None)
            y_pred_col = next((c for c in dfp.columns if c.lower() in ("y_pred","pred","prediction","prdtypecode_pred")), None)
            if y_true_col and y_pred_col:
                f1 = f1_score(dfp[y_true_col], dfp[y_pred_col], average="weighted")
                st.metric("F1 pondéré (validation)", f"{f1:.3f}")
                # == Matrice de confusion (pourcentages & labels) ==
                # 1) On détermine les classes présentes et on trie par fréquence (y_true)
                counts = dfp[y_true_col].value_counts().sort_values(ascending=False)
                max_k = int(min(40, len(counts)))  # sécurité
                k = st.slider("Top classes à afficher", 10, max_k, min(20, max_k), key="b4_cm_topk")

                sel_classes = counts.index[:k].tolist()

                # 2) Matrice + normalisation par ligne (en %)
                cm = confusion_matrix(dfp[y_true_col], dfp[y_pred_col], labels=sel_classes)
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

                # 3) Libellés lisibles via LABEL_MAP
                def _nice(lab):
                    return LABEL_MAP.get(str(lab), LABEL_MAP.get(int(lab) if str(lab).isdigit() else lab, str(lab)))

                tick_labels = [_nice(c) for c in sel_classes]

                # 4) Plot Plotly avec annotations
                figcm = px.imshow(
                    cm_pct,
                    text_auto=".1f",               # affiche 1 décimale
                    aspect="auto",
                    color_continuous_scale="Blues",
                    zmin=0, zmax=100,
                    title=f"Matrice de confusion — Pipeline Multimodale (top {len(sel_classes)} classes)",
                )

                # Axes avec libellés lisibles + rotation des X
                figcm.update_xaxes(
                    title_text="Predicted label",
                    tickmode="array",
                    tickvals=list(range(len(sel_classes))),
                    ticktext=tick_labels,
                    tickangle=-45,
                )
                figcm.update_yaxes(
                    title_text="True label", 
                    tickmode="array",
                    tickvals=list(range(len(sel_classes))),
                    ticktext=tick_labels,
                )

                figcm.update_layout(height=900, margin=dict(t=60, b=160, l=0, r=0))
                st.plotly_chart(figcm, use_container_width=True)
            else:
                st.info("`results/preds_oof_b4.csv` trouvé, mais colonnes y_true / y_pred manquantes.")
        except Exception as e:
            st.warning(f"Impossible de lire `results/preds_oof_b4.csv` ({e}).")
    else:
        st.caption("Dépose `results/preds_oof_b4.csv` pour calculer F1 et tracer la matrice de confusion automatiquement.")

    # ============================
    # Gains par classe (Δ rappel diagonale) : B3−B2 et B4−B2
    # ============================
    st.divider()
    st.header("Gains par classe : apport de B3 (vision) et B4 (multimodal) vs B2 (texte)")


    # Fichiers de prédictions attendus
    P_B2 = Path("results/preds_b2.csv")
    P_B3 = Path("results/preds_b3.csv")
    # B4 : on tente d'abord OOF (cross-val), sinon preds_b4.csv
    P_B4_CANDS = [Path("results/preds_oof_b4.csv"), Path("results/preds_b4.csv")]
    P_B4 = next((p for p in P_B4_CANDS if p.exists()), None)

    def _read_preds(path: Path) -> tuple[pd.Series, pd.Series] | None:
        if not path or not path.exists():
            return None
        dfp = pd.read_csv(path)
        # nommage tolérant
        cols = {c.lower(): c for c in dfp.columns}
        y_true = next((cols[k] for k in cols if k in ("y_true","y","label","target","prdtypecode_true")), None)
        y_pred = next((cols[k] for k in cols if k in ("y_pred","pred","prediction","prdtypecode_pred")), None)
        if not (y_true and y_pred):
            st.warning(f"Colonnes y_true / y_pred introuvables dans {path.name}.")
            return None
        return dfp[y_true], dfp[y_pred]

    def _per_class_recall(y_true: pd.Series, y_pred: pd.Series, labels: list) -> pd.Series:
        """
        Rappel (sens ligne) en %, pour chaque classe de `labels`.
        """
        lab_to_idx = {l:i for i,l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            t = str(t); p = str(p)
            if t in lab_to_idx and p in lab_to_idx:
                cm[lab_to_idx[t], lab_to_idx[p]] += 1
        row_sum = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = np.where(row_sum>0, cm.diagonal().reshape(-1,1)/row_sum, 0.0).ravel() * 100.0
        return pd.Series(rec, index=labels, dtype=float)

    def _nice(lab):
        return LABEL_MAP.get(str(lab), LABEL_MAP.get(int(lab) if str(lab).isdigit() else lab, str(lab)))

    # Lecture des trois jeux (on continue même si B3 ou B4 manquent)
    b2 = _read_preds(P_B2)
    b3 = _read_preds(P_B3)
    b4 = _read_preds(P_B4) if P_B4 else None

    if not b2:
        st.error("Preds B2 introuvables (results/preds_b2.csv). Dépose le fichier pour activer cette section.")
    else:
        y2, p2 = b2
        # Ensemble des classes = union de celles observées dans les 3 prédictions
        labs = set(map(str, pd.unique(y2)))
        if b3: labs |= set(map(str, pd.unique(b3[0])))
        if b4: labs |= set(map(str, pd.unique(b4[0])))
        labels = sorted(labs, key=lambda x: (x.isdigit(), int(x) if str(x).isdigit() else x))

        rec2 = _per_class_recall(y2.astype(str), p2.astype(str), labels)

        rec3 = None
        if b3:
            y3, p3 = b3
            rec3 = _per_class_recall(y3.astype(str), p3.astype(str), labels)

        rec4 = None
        if b4:
            y4, p4 = b4
            rec4 = _per_class_recall(y4.astype(str), p4.astype(str), labels)

        # Tableau récapitulatif
        df_gain = pd.DataFrame({
            "classe": labels,
            "libellé": [ _nice(l) for l in labels ],
            "rappel_B2_%": rec2.round(2)
        })
        if rec3 is not None:
            df_gain["rappel_B3_%"] = rec3.round(2)
        if rec4 is not None:
            df_gain["rappel_B4_%"] = rec4.round(2)
            df_gain["Δ(B4−B2) pts"] = (rec4 - rec2).round(2)

        # Tri par gain multimodal si dispo, sinon par gain vision
        sort_col = "Δ(B4−B2) pts" if "Δ(B4−B2) pts" in df_gain.columns else "Δ(B3−B2) pts"
        df_gain = df_gain.sort_values(sort_col, ascending=False, ignore_index=True)

        # Sélection top-K
        kmax = min(27, len(df_gain))
        topk = st.slider("Top améliorations à afficher", 5, kmax, min(12, kmax), key="gain_topk")
        st.markdown("**Tableau des gains par classe (en points de rappel)**")
        st.dataframe(df_gain.head(topk), use_container_width=True)

        # Barplot des gains (multimodal en priorité)
        if "Δ(B4−B2) pts" in df_gain.columns:
            sub = df_gain.head(topk)[["libellé","Δ(B4−B2) pts"]]
            figg = px.bar(sub, x="Δ(B4−B2) pts", y="libellé", orientation="h",
                        title="Top gains — Multimodal B4 vs Texte B2 (points de rappel)")
            figg.update_layout(height=520, margin=dict(t=60,b=20,l=0,r=0))
            st.plotly_chart(figg, use_container_width=True)

        if "Δ(B3−B2) pts" in df_gain.columns:
            with st.expander("Voir aussi : gains Vision B3 vs Texte B2"):
                sub = df_gain.sort_values("Δ(B3−B2) pts", ascending=False).head(topk)[["libellé","Δ(B3−B2) pts"]]
                figg2 = px.bar(sub, x="Δ(B3−B2) pts", y="libellé", orientation="h",
                            title="Top gains — Vision B3 vs Texte B2 (points de rappel)")
                figg2.update_layout(height=520, margin=dict(t=60,b=20,l=0,r=0))
                st.plotly_chart(figg2, use_container_width=True)

        # Option : classes en baisse
        with st.expander("Classes en baisse (Δ négatif)"):
            cols_neg = [c for c in ("Δ(B4−B2) pts","Δ(B3−B2) pts") if c in df_gain.columns]
            if cols_neg:
                out = df_gain.sort_values(cols_neg[0], ascending=True).head(topk)
                st.dataframe(out, use_container_width=True)
            else:
                st.caption("B3/B4 non disponibles → pas de Δ négatifs à afficher.")

    # 4) Résultats (F1 + matrice de confusion si dispo)
    preds_path = Path("results/preds_b2.csv")
    if preds_path.exists():
        try:
            dfp = pd.read_csv(preds_path)
            # on cherche y_true/y_pred
            y_true_col = next((c for c in dfp.columns if c.lower() in ("y_true","y","label","target","prdtypecode_true")), None)
            y_pred_col = next((c for c in dfp.columns if c.lower() in ("y_pred","pred","prediction","prdtypecode_pred")), None)
            if y_true_col and y_pred_col:
                f1 = f1_score(dfp[y_true_col], dfp[y_pred_col], average="weighted")
                st.metric("F1 pondéré (val)", f"{f1:.3f}")
                # == Matrice de confusion (pourcentages & labels) ==
                # 1) On détermine les classes présentes et on trie par fréquence (y_true)
                counts = dfp[y_true_col].value_counts().sort_values(ascending=False)
                max_k = int(min(40, len(counts)))  # sécurité
                k = st.slider("Top classes à afficher", 10, max_k, min(30, max_k), key="b2_cm_topk")

                sel_classes = counts.index[:k].tolist()

                # 2) Matrice + normalisation par ligne (en %)
                cm = confusion_matrix(dfp[y_true_col], dfp[y_pred_col], labels=sel_classes)
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

                # 3) Libellés lisibles via LABEL_MAP
                def _nice(lab):
                    # lab peut être int/str ; on essaie str puis int
                    return LABEL_MAP.get(str(lab), LABEL_MAP.get(int(lab) if str(lab).isdigit() else lab, str(lab)))

                tick_labels = [_nice(c) for c in sel_classes]

                # 4) Plot Plotly avec annotations
                figcm = px.imshow(
                    cm_pct,
                    text_auto=".2f",               # affiche 2 décimales
                    aspect="auto",
                    color_continuous_scale="Blues",
                    zmin=0, zmax=100,
                    title=f"Matrice de confusion — Pipeline Texte (top {len(sel_classes)} classes)",
                )

                # Axes avec libellés lisibles + rotation des X
                figcm.update_xaxes(
                    title_text="Predicted label",
                    tickmode="array",
                    tickvals=list(range(len(sel_classes))),
                    ticktext=tick_labels,
                    tickangle=-45,
                )
                figcm.update_yaxes(
                    title_text="True label",
                    tickmode="array",
                    tickvals=list(range(len(sel_classes))),
                    ticktext=tick_labels,
                )

                figcm.update_layout(height=900, margin=dict(t=60, b=160, l=0, r=0))
                plotly_auto(figcm)
            else:
                st.info("`results/preds_b2.csv` trouvé, mais colonnes y_true / y_pred manquantes.")
        except Exception as e:
            st.warning(f"Impossible de lire `results/preds_b2.csv` ({e}).")
    else:
        st.caption("Astuce : dépose `results/preds_b2.csv` pour calculer F1 et tracer la matrice de confusion automatiquement.")
# === Parcours animé B3 (image) ================================================
def show_b3_walkthrough():


    # 4) Résultats (F1 + matrice de confusion si dispo)
    preds_path = Path("results/preds_b3.csv")
    if preds_path.exists():
        try:
            dfp = pd.read_csv(preds_path)
            y_true_col = next((c for c in dfp.columns if c.lower() in ("y_true","y","label","target","prdtypecode_true")), None)
            y_pred_col = next((c for c in dfp.columns if c.lower() in ("y_pred","pred","prediction","prdtypecode_pred")), None)
            if y_true_col and y_pred_col:
                f1w = f1_score(dfp[y_true_col], dfp[y_pred_col], average="weighted")
                st.metric("F1 pondéré (val)", f"{f1w:.3f}")

                # --- Matrice de confusion normalisée (% par ligne) + labels lisibles ---
                counts = dfp[y_true_col].value_counts().sort_values(ascending=False)
                max_k = int(min(40, len(counts)))
                k = st.slider("Top classes à afficher", 10, max_k, min(30, max_k), key="b3_cm_topk")
                sel = counts.index[:k].tolist()

                cm = confusion_matrix(dfp[y_true_col], dfp[y_pred_col], labels=sel)
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0) * 100.0

                def _nice(lab):
                    return LABEL_MAP.get(str(lab), LABEL_MAP.get(int(lab), str(lab)))

                tick_labels = [_nice(x) for x in sel]

                figcm = px.imshow(
                    cm_pct, text_auto=".2f", aspect="auto",
                    color_continuous_scale="Blues", zmin=0, zmax=100,
                    title=f"Matrice de confusion — Pipeline Images (top {len(sel)} classes)",
                )
                figcm.update_xaxes(
                    title_text="Predicted label",
                    tickmode="array", tickvals=list(range(len(sel))), ticktext=tick_labels, tickangle=-45,
                )
                figcm.update_yaxes(
                    title_text="True label",
                    tickmode="array", tickvals=list(range(len(sel))), ticktext=tick_labels,
                )
                figcm.update_layout(height=900, margin=dict(t=60, b=160, l=0, r=0))
                st.plotly_chart(figcm, width='stretch')
            else:
                st.info("`results/preds_b3.csv` trouvé, mais colonnes y_true / y_pred manquantes.")
        except Exception as e:
            st.warning(f"Impossible de lire `results/preds_b3.csv` ({e}).")
    else:
        st.caption("Dépose `results/preds_b3.csv` pour calculer F1 et tracer la matrice de confusion automatiquement.")

with meth_tab:
    st.subheader("Méthode & pipeline (résumé)")
    st.divider()
    st.subheader("Diagramme du pipeline B4 (texte + image)")
    show_b2_walkthrough()
    show_b3_walkthrough()
    st.markdown(
        """
        - **Texte** : TF‑IDF → SVD/PCA → classif (LogReg / LinearSVC).
        - **Image** : CNN ResNet et/ou Vit pré‑entraîné (embeddings) → classif.
        - **Fusion** : concat features texte + image → classif final.
        - **CV** : K-fold stratifié, export des embeddings et diagnostics.
        """
    )


# ----------------------------
# Tab – Diagnostics modèle
# ----------------------------
with diag_tab:
    import os, re
    from pathlib import Path
    from PIL import Image
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.subheader("Diagnostics modèle")
    st.caption("Sorties générées (CSV/figures) et analyse des confusions.")

    # --------- chemins de base
    APP_DIR = Path(__file__).resolve().parent
    RAPPORT_DIR = (APP_DIR / "rapport").resolve()

    def get_repo_root():
        env_root = os.getenv("RAKUTEN_REPO_ROOT")
        if env_root and Path(env_root).expanduser().exists():
            return Path(env_root).expanduser().resolve()
        here = Path(__file__).resolve()
        candidate = here.parents[1] if len(here.parents) > 1 else here.parent
        return candidate if (candidate / "results").exists() else Path.cwd()

    REPO_ROOT = get_repo_root()
    FIG_DIR_DEFAULT = REPO_ROOT / "results" / "figures"
    REP_DIR_DEFAULT = REPO_ROOT / "results" / "reports"

    # --------- helpers
    def _percentify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit en pourcentages (x100, arrondi à l’unité) toutes les colonnes numériques
        SAUF les colonnes d'index/id/label.
        """
        out = df.copy()
        # colonnes à laisser telles quelles
        keep_patterns = re.compile(r"(unnamed|id|idx|index|classe|class|code|prdtype)", re.I)
        num_cols = [c for c in out.columns
                    if pd.api.types.is_numeric_dtype(out[c]) and not keep_patterns.search(str(c))]
        for c in num_cols:
            out[c] = (out[c] * 100.0).round(0).astype("Int64")
        return out

    def _df_to_percent_str(df: pd.DataFrame) -> pd.DataFrame:
        """
        Version 'jolie' pour affichage : ajoute le signe % aux colonnes numériques déjà x100.
        """
        out = df.copy()
        keep_patterns = re.compile(r"(unnamed|id|idx|index|classe|class|code|prdtype)", re.I)
        for c in out.columns:
            if pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_float_dtype(out[c]):
                if not keep_patterns.search(str(c)):
                    out[c] = out[c].apply(lambda v: f"{int(v)}%" if pd.notna(v) else "")
        return out

    # ----------------------------
    # B2 — Par classe (magnitude signée) → graphe unique empilé (sans tableaux)
    # ----------------------------
    st.divider()
    st.markdown("""
    **Comment lire ce graphique ?**  
    Pour chaque **classe** et chaque **bloc de variables** (TF-IDF mots, chars, etc.), on trace la
    **contribution moyenne signée** au **score linéaire** (magnitude de |x·w| avec le signe).
    - **> 0** : le bloc **pousse** vers la classe ; **< 0** : il **éloigne** de la classe.  
    L’échelle est en **unités de score** (sans dimension).  
    """)


    # 1) Chemins FIXES vers tes CSV
    POS_B2 = Path(r"C:\Users\colle\Dropbox\10 Learning\01 LES MINES DATASCIENTEST\Rakuten\rakuten-main\results\reports\block_importance_b2_per_class_pos_mag.csv")
    NEG_B2 = Path(r"C:\Users\colle\Dropbox\10 Learning\01 LES MINES DATASCIENTEST\Rakuten\rakuten-main\results\reports\block_importance_b2_per_class_neg_mag.csv")

    if not POS_B2.exists() or not NEG_B2.exists():
        st.error("CSV introuvables : vérifie les chemins POS_B2 / NEG_B2.")
    else:
        # 2) Lecture
        df_pos = pd.read_csv(POS_B2)
        df_neg = pd.read_csv(NEG_B2)

        # 3) Colonnes : première = id de classe ; le reste = blocs
        id_col = df_pos.columns[0]
        block_cols = df_pos.columns[1:].tolist()

        # 4) Passage en long + signe
        dfp = df_pos.melt(id_vars=[id_col], value_vars=block_cols,
                        var_name="bloc", value_name="val")
        dfn = df_neg.melt(id_vars=[id_col], value_vars=block_cols,
                        var_name="bloc", value_name="val")
        dfn["val"] = -dfn["val"]

        dfall = pd.concat([dfp, dfn], ignore_index=True)

        # 5) Libellés lisibles via LABEL_MAP
        def _nice_lab(v):
            s = str(v)
            return LABEL_MAP.get(s, LABEL_MAP.get(int(s), s)) if s.isdigit() or s in LABEL_MAP else s
        dfall["classe"] = dfall[id_col].apply(_nice_lab)

        # 6) Tri des classes par somme de |impact|
        order = (dfall.groupby("classe")["val"]
                        .apply(lambda x: x.abs().sum())
                        .sort_values(ascending=True)  # du plus faible au plus fort (axe Y ascendant)
                        .index.tolist())
        dfall["classe"] = pd.Categorical(dfall["classe"], categories=order, ordered=True)

        # 7) Graphe empilé signé (style "stacked bars")
        import plotly.graph_objects as go


        # Matrice "signée" en large : +pos  −neg  pour chaque bloc
        signed = df_pos.copy()
        for b in block_cols:
            signed[b] = df_pos[b] - df_neg[b]    # le CSV "neg" contient la magnitude; on la rend négative

        # Libellés lisibles + tri des classes par impact total |.| (ascendant pour une lecture naturelle)
        signed["classe"] = df_pos[id_col].apply(_nice_lab)
        signed["_abs_total"] = signed[block_cols].abs().sum(axis=1)
        signed = signed.sort_values("_abs_total", ascending=True)

        # Traces empilées : une par bloc, orientation horizontale
        fig = go.Figure()
        for b in block_cols:
            fig.add_trace(go.Bar(
                y=signed["classe"],
                x=signed[b],
                name=b,
                orientation="h"
            ))

        fig.update_layout(
            title="Importance par bloc — B2 (par classe, impact signé, magnitude)",
            height=900,
            margin=dict(t=60, b=20, l=0, r=0),
            legend_title_text="Bloc de variables",
            barmode="relative",       # ← empilement avec signes (droite = +, gauche = −)
            bargap=0.15,
            bargroupgap=0
        )
        fig.update_xaxes(
            title_text="Contribution moyenne au score linéaire (|x·w|)",
            zeroline=True,
            zerolinewidth=1
        )
        fig.update_yaxes(title_text="Classe")

        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Matrices de confusion — B2 & B4 (depuis *top_confusions* FR)
    # ----------------------------
    st.divider()
    st.header("Matrices de confusion — B2 (texte) & B4 (multimodal)")

    # Chemins des top_confusions (priorité rapport/)
    b2_top_path = RAPPORT_DIR / "b2_top_confusions.csv"
    b4_top_path = RAPPORT_DIR / "b4_top_confusions.csv"
    if not b2_top_path.exists():
        b2_top_path = REP_DIR_DEFAULT / "b2_top_confusions.csv"
    if not b4_top_path.exists():
        b4_top_path = REP_DIR_DEFAULT / "b4_top_confusions.csv"

    def _select_top_confused(cm, labels, k=8):
        """
        Sélectionne les k classes avec le plus de 'confusion' totale,
        mesurée comme la somme des hors-diagonales sur la ligne ET la colonne.
        Retourne (cm_reduite, labels_reduits) dans l'ordre décroissant de confusion.
        """
        import numpy as np
        cm = np.asarray(cm, dtype=float)
        off = cm.copy()
        np.fill_diagonal(off, 0.0)
        score = off.sum(axis=1) + off.sum(axis=0)  # masse d'erreur reçue + envoyée
        order = np.argsort(-score)[:k]            # indices des k plus “confusés”
        # pas de tri supplémentaire → on garde l'ordre par intensité de confusion
        cm_k = cm[np.ix_(order, order)]
        labels_k = [labels[i] for i in order]
        return cm_k, labels_k

    def load_top_confusions_df(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = None
        for sep in (",", ";", "\t"):
            try:
                tmp = pd.read_csv(path, sep=sep)
                if not tmp.empty:
                    df = tmp
                    break
            except Exception:
                continue
        if df is None or df.empty:
            return None

        cols = {c.lower(): c for c in df.columns}
        t_id  = next((cols[k] for k in cols if k in ("classe_vraie_id","true","y_true","target","prdtypecode_true","vrai")), None)
        t_nom = next((cols[k] for k in cols if k in ("classe_vraie_nom","true_name","vrai_nom")), None)
        p_id  = next((cols[k] for k in cols if k in ("classe_pred_id","pred","y_pred","prediction","prdtypecode_pred","pré","pre","predicted")), None)
        p_nom = next((cols[k] for k in cols if k in ("classe_pred_nom","pred_name","pré_nom","pre_nom")), None)
        taux  = next((cols[k] for k in cols if k in ("taux","pct","percent","percentage","pourcentage")), None)
        ncol  = next((cols[k] for k in cols if k in ("comptes","count","n","nb","freq","value","val")), None)

        if not (t_id and p_id and taux):
            st.warning(f"Format non reconnu pour {path.name}. Colonnes lues : {list(df.columns)}")
            st.dataframe(df.head(), use_container_width=True)
            return None

        out = pd.DataFrame()
        out["true_id"]  = df[t_id].astype(str)
        out["pred_id"]  = df[p_id].astype(str)
        out["true_nom"] = df[t_nom].astype(str) if t_nom else out["true_id"]
        out["pred_nom"] = df[p_nom].astype(str) if p_nom else out["pred_id"]

        tx = df[taux].astype(str).str.replace("%","", regex=False).str.replace(",",".", regex=False)
        tx = pd.to_numeric(tx, errors="coerce")
        if tx.max() is not None and tx.max() <= 1.0:
            tx = tx * 100.0
        out["taux_pct"] = tx.fillna(0.0)

        if ncol:
            out["comptes"] = pd.to_numeric(df[ncol], errors="coerce").fillna(0).astype(int)
        else:
            out["comptes"] = 0

        out["Taux (%)]"] = out["taux_pct"].round(0).astype(int)
        return out

    def cm_from_top_df(df_std: pd.DataFrame):
        labels = sorted(set(df_std["true_id"]).union(df_std["pred_id"]),
                        key=lambda x: (x.isdigit(), x))
        mat = df_std.pivot_table(index="true_id", columns="pred_id",
                                 values="taux_pct", aggfunc="sum", fill_value=0.0)
        mat = mat.reindex(index=labels, columns=labels, fill_value=0.0)
        return mat.to_numpy(dtype=float), labels

    def _plot_cm_percent(cm, labels, title, key):
        fig = px.imshow(
            cm, text_auto=".0f", aspect="auto",
            color_continuous_scale="Blues", zmin=0, zmax=100,
            title=f"{title} — % par ligne"
        )
        fig.update_xaxes(title_text="Prédit",
                         tickmode="array",
                         tickvals=list(range(len(labels))),
                         ticktext=[_nice_label(x) for x in labels],
                         tickangle=-45)
        fig.update_yaxes(title_text="Vrai",
                         tickmode="array",
                         tickvals=list(range(len(labels))),
                         ticktext=[_nice_label(x) for x in labels])
        try:
            st.plotly_chart(fig, width='stretch', key=key)
        except TypeError:
            st.plotly_chart(fig, use_container_width=True, key=key)

    def _warn_if_mode_collapse(cm, labels):
        col_tot = cm.sum(axis=0)
        tot = col_tot.sum()
        if tot > 0:
            heavy = (col_tot / tot >= 0.01).sum()
            if heavy <= 1:
                st.warning(
                    "Les prédictions se concentrent presque exclusivement sur **une seule classe** "
                    f"(dominante: `{labels[int(col_tot.argmax())]}`) — vérifie dump/mapping."
                )

    # Commentaires analytiques
    st.markdown("- Plusieurs erreurs semblent **structurelles** : **frontières de catalogue** peu nettes (produits pouvant relever de deux catégories).")
    st.markdown("- **Libellés vendeurs hétérogènes** (titres vagues, marketing) → le modèle **B2** est plus exposé.")
    st.markdown("- **Produits multi-usages / kits** (activité manuelle vs jouet éducatif) ⇒ hésitations.")
    st.markdown("- Le **multimodal B4** réduit certaines ambiguïtés via l’image, mais reste sensible si la photo n’explicite pas l’usage.")
    st.markdown(
        "> **Exemple : 1280 ↔ 1281** — **1280** (Jouets enfants & bébés, ex. chien à tirer) vs **1281** (Jeux & loisirs enfants, ex. puzzle). "
        "Textes/visuels se recouvrent (**\"jeu\", \"jouet\", \"éveil\"**), frontière **intrinsèquement floue**."
    )

    # Lectures + affichages
    col_b2, col_b4 = st.columns(2)

    with col_b2:
        df_b2 = load_top_confusions_df(b2_top_path)
        if df_b2 is None:
            st.warning(f"Impossible de lire {b2_top_path.name}.")
        else:
            st.markdown("**Top confusions — B2 (texte)**")
            st.dataframe(
                df_b2[["true_id","true_nom","pred_id","pred_nom","taux_pct","comptes"]]
                      .rename(columns={
                          "true_id":"classe_vraie_id","true_nom":"classe_vraie_nom",
                          "pred_id":"classe_pred_id","pred_nom":"classe_pred_nom",
                          "taux_pct":"Taux (%)"
                      })
                      .assign(**{"Taux (%)": lambda d: d["Taux (%)"].round(0).astype(int)}),
                use_container_width=True
            )
            cm_b2, labels_b2 = cm_from_top_df(df_b2)
            # Limiter aux Top classes les plus confusées (8 par défaut)
            topk_b2 = st.slider("Top classes confusées (B2)", 4, min(20, len(labels_b2)), 8, key="b2_top_conf")
            cm_b2k, labels_b2k = _select_top_confused(cm_b2, labels_b2, k=topk_b2)
            _plot_cm_percent(cm_b2k, labels_b2k, "Matrice de confusion — **B2 (texte)**", key="cm_b2")
            _warn_if_mode_collapse(cm_b2k, labels_b2k)


    with col_b4:
        df_b4 = load_top_confusions_df(b4_top_path)
        if df_b4 is None:
            st.warning(f"Impossible de lire {b4_top_path.name}.")
        else:
            st.markdown("**Top confusions — B4 (multimodal)**")
            st.dataframe(
                df_b4[["true_id","true_nom","pred_id","pred_nom","taux_pct","comptes"]]
                      .rename(columns={
                          "true_id":"classe_vraie_id","true_nom":"classe_vraie_nom",
                          "pred_id":"classe_pred_id","pred_nom":"classe_pred_nom",
                          "taux_pct":"Taux (%)"
                      })
                      .assign(**{"Taux (%)": lambda d: d["Taux (%)"].round(0).astype(int)}),
                use_container_width=True
            )
            cm_b4, labels_b4 = cm_from_top_df(df_b4)
            topk_b4 = st.slider("Top classes confusées (B4)", 4, min(20, len(labels_b4)), 8, key="b4_top_conf")
            cm_b4k, labels_b4k = _select_top_confused(cm_b4, labels_b4, k=topk_b4)
            _plot_cm_percent(cm_b4k, labels_b4k, "Matrice de confusion — **B4 (multimodal)**", key="cm_b4")
            _warn_if_mode_collapse(cm_b4k, labels_b4k)


    # Notes d'interprétation
    st.markdown("""
- **Limites de la classification**  
  1) **Flou taxonomique** (ex. *jouet d’éveil* vs *jeu éducatif*).  
  2) **Qualité du signal** (titres marketing génériques, descriptions manquantes).  
  3) **Visuels trompeurs** (boîte/packshot).  
  4) **Biais de fréquence** (classes majoritaires mieux apprises).

- **Pistes d’amélioration**  
  1) Enrichir descriptions (règles/LLM).  
  2) Ajouter des **métadonnées** (âge conseillé, nb de pièces, matériau).  
  3) **Règles métier** pour 1280/1281 (puzzle/jeu société → 1281 ; doudou/chien à tirer → 1280). 
        Nous avons créé une dizaine de règles de métier comme celles-ci où notre modèle définit si pour chaque produit il répond à une règle métier.
        
        BABY  = ("bebe","doudou","peluche","poupee","siegeauto","biberon","poussette","couches", "tetine")
        BOOK  = ("tome","volume","broche","poche","edition","edition","manga","isbn") 
  4) **Active learning** sur les paires les plus confondues + **revue humaine** ciblée.
""")

    # ----------------------------
    # Exemples illustratifs 1280 ↔ 1281
    # ----------------------------
    st.subheader("Exemples illustratifs 1280 ↔ 1281")
    IMAGES_BASE_DIR = (APP_DIR / "demo_images").resolve()
    ex1_base = "image_879232910_product_124649496.jpg"  # 1280
    ex2_base = "image_1285148307_product_4066272585.jpg"  # 1281

    def resolve_image_path(p: str, base_dir: Path) -> str | None:
        if not p:
            return None
        if isinstance(p, str) and (p.startswith("http://") or p.startswith("https://")):
            return p
        p0 = Path(p)
        if p0.exists():
            return str(p0)
        p1 = base_dir / p0.name
        return str(p1) if p1.exists() else None

    ex1 = resolve_image_path(ex1_base, IMAGES_BASE_DIR)
    ex2 = resolve_image_path(ex2_base, IMAGES_BASE_DIR)

    c1, c2 = st.columns(2)
    with c1:
        if ex1:
            try:
                img1 = Image.open(ex1).convert("RGB")
                st.image(img1, caption="Attendu: 1280 – Jouets enfants & bébés", width="content")
            except Exception as e:
                st.warning(f"Impossible d'ouvrir {ex1_base} ({e})")
        else:
            st.warning(f"Fichier introuvable: {ex1_base}")
    with c2:
        if ex2:
            try:
                img2 = Image.open(ex2).convert("RGB")
                st.image(img2, caption="Attendu: 1281 – Jeux et loisirs enfants", width="content")
            except Exception as e:
                st.warning(f"Impossible d'ouvrir {ex2_base} ({e})")
        else:
            st.warning(f"Fichier introuvable: {ex2_base}")
            
# ----------------------------
# Tab 3 – Prédiction (modèle hébergé)
# ----------------------------
# --- Modèle démo (XGB image-only) ---
# DEFAULT_MODEL_URL = "artifacts/b4_inference_lite.joblib"

# ==== Lazy downloader pour artifacts & data (à coller en haut de config.py) ====
from pathlib import Path
import os, hashlib
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1] if len(APP_DIR.parents) > 1 else APP_DIR

ART_DIR = (REPO_ROOT / "artifacts"); ART_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = (REPO_ROOT / "data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

# Renseigne ici tes liens (GitHub Release / HF / Dropbox ?dl=1)
# NB: tu peux aussi les mettre dans .streamlit/secrets.toml (voir plus bas)
DOWNLOADS = {
    # artifacts
    ART_DIR / "text_preproc.joblib": "https://www.dropbox.com/scl/fi/zpftggop63g87uiev9nzt/text_preproc.joblib?rlkey=ogzkino3681ttoa26fc04a282&dl=1",
    ART_DIR / "final_estimator.joblib": "https://www.dropbox.com/scl/fi/kdmd7njht3anvctd2zjbv/final_estimator.joblib?rlkey=vks5rl4cxrnvlbsojkmvcezwm&dl=1",
}

def _download(url, dest: Path, min_bytes=1024):
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
        size = tmp.stat().st_size
        if size < min_bytes:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Téléchargement trop petit ({size}o) pour {dest.name}")
        tmp.replace(dest)

def ensure_files():
    missing = []
    for dest, url in DOWNLOADS.items():
        if dest.exists():
            continue
        if not url or url.startswith("PUT_URL_HERE"):
            missing.append(dest.name)
            continue
        try:
            with st.spinner(f"Téléchargement de {dest.name}…"):
                _download(url, dest)
        except Exception as e:
            st.warning(f"Échec download {dest.name}: {e}")
            missing.append(dest.name)
    if missing:
        st.error("Fichiers manquants : " + ", ".join(missing))
    return len(missing) == 0
def _s(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

DOWNLOADS = {
        ART_DIR / "text_preproc.joblib": _s("https://www.dropbox.com/scl/fi/zpftggop63g87uiev9nzt/text_preproc.joblib?rlkey=ogzkino3681ttoa26fc04a282&dl=1",""),
        ART_DIR / "final_estimator.joblib": _s("https://www.dropbox.com/scl/fi/kdmd7njht3anvctd2zjbv/final_estimator.joblib?rlkey=vks5rl4cxrnvlbsojkmvcezwm&dl=1",""),
}

@st.cache_resource
def load_demo_xgb_image_only():
    import json, joblib, numpy as np
    from pathlib import Path
    if not Path(DEMO_EMB_NPZ).exists() or not Path(DEMO_IDX_JSON).exists() or not Path(DEMO_XGB_JOBLIB).exists():
        st.info("Mode démo XGB : fichiers manquants. Lance d’abord les scripts de pré-calcul/entraînement.")
        return None, None, None
    X = np.load(DEMO_EMB_NPZ)["X"].astype("float32", copy=False)
    idx = json.loads(Path(DEMO_IDX_JSON).read_text(encoding="utf-8"))["paths"]
    pack = joblib.load(DEMO_XGB_JOBLIB)
    return X, idx, pack

def predict_demo_xgb_image_only(image_abs_path: str):
    X, idx, pack = load_demo_xgb_image_only()
    if pack is None:
        st.warning("Mode démo XGB indisponible (artefacts manquants).")
        return None, {}
    f = _resnet50_embed_one(image_abs_path)  # (1, 2048)
    model = pack["model"]
    classes = np.array(pack["label_encoder_classes_"])
    proba = model.predict_proba(f)[0]
    y = int(proba.argmax())
    return classes[y], {classes[i]: float(p) for i, p in enumerate(proba)}

# --- Simulation (texte + image avec estimateur allégé) ---
with sim_tab:
    st.subheader("Démo  (texte + image de la galerie)")
    try:
        est, txt_ct, Ximg, idx, labels, order = _load_inference_stack()
    except Exception as e:
        st.error(f"Artefacts manquants ou illisibles : {e}")
        st.stop()

    # Sélecteur d'image (par basename affiché)
    from pathlib import Path as _P
    options = [(_P(p).name, p) for p in idx]
    disp2path = {disp: p for disp, p in options}
    sel_disp = st.selectbox(
        "Image",
        options=[d for d, _ in options],
        index=0 if options else 0,
        key="demo_img_sel",
    )
    image_path = disp2path.get(sel_disp, None)

    # Entrées texte
    colL, colR = st.columns([2, 1])
    with colL:
        designation = st.text_input("Désignation", value="Console rétro portable 400 jeux")
        description = st.text_area("Description (optionnel)", value="")
        run = st.button("Prédire", type="primary")
    with colR:
        if image_path:
            try:
                if image_path.startswith("http"):
                    st.image(image_path, caption=sel_disp, use_container_width=True)
                else:
                    st.image(Image.open(image_path).convert("RGB"), caption=sel_disp, use_container_width=True)
            except Exception:
                st.caption("Image non affichable.")

    if run:
        # 1) Prédiction
        y, proba = predict_estimator({"designation": designation, "description": description}, image_path)

        # 2) Récup des classes (ordre interne du modèle)
        # 'labels' vient de meta['classes_'] si dispo → c'est ce qu'on veut
        classes = labels if (labels is not None and len(labels)) else getattr(est, "classes_", None)
        if classes is None:
            classes = list(range(len(proba)))

        # 3) id interne (index dans classes), prdtypecode et libellé lisible
        try:
            cls_id = list(classes).index(int(y) if str(y).isdigit() else y)
        except ValueError:
            cls_id = None

        def _nice_label(c):
            s = str(c)
            return LABEL_MAP.get(s, LABEL_MAP.get(int(s) if s.isdigit() else c, s))

        if cls_id is None:
            st.success(f"Classe prédite : {y} — {_nice_label(y)}")
        else:
            st.success(f"Classe prédite : id={cls_id} | prdtypecode={y} — {_nice_label(y)}")

        # 4) Top-k — prdtypecode, libellé, proba (sans colonne 'id')
        def _prob_for(c):
            return float(
                proba.get(c, proba.get(str(c), proba.get(int(c) if str(c).isdigit() else c, 0.0)))
        )

        rows = [
            {
                "prdtypecode": int(c) if str(c).isdigit() else c,
                "libellé": _nice_label(c),
                "proba": _prob_for(c),
            }
            for c in classes
        ]
        dd = pd.DataFrame(rows).sort_values("proba", ascending=False).head(15)
        _show_table(dd, hide_index=True)

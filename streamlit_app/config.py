# -*- coding: utf-8 -*-
"""
Streamlit – Rakuten Multimodal Dashboard (corrigé)
- Supprime toute dépendance à un labels_map JSON (mapping en dur)
- Force l'utilisation de streamlit_app/demo_images pour l'affichage des images
- Corrige les erreurs de DuplicateElementId et remplace use_container_width par width
"""

from __future__ import annotations

import os
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

# ----------------------------
# Constantes démo
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
IMAGES_BASE_DIR = (APP_DIR / "demo_images").resolve()  # dossier d'images démo
DEMO_CSV = IMAGES_BASE_DIR / "demo_images.csv"         # CSV de démo

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

# ----------------------------
# Sidebar – Data & Config
# ----------------------------
st.set_page_config(page_title="Rakuten Multimodal Dashboard", layout="wide")
st.title("Rakuten – Dashboard Multimodal")

st.sidebar.header("Données")
up = st.sidebar.file_uploader("CSV des données (optionnel)", type=["csv"], accept_multiple_files=False)
path_hint = st.sidebar.text_input("…ou chemin vers un CSV local", value="")

df = load_csv(up if up is not None else (path_hint if path_hint else None))
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
expl_tab, meth_tab, sim_tab = st.tabs(["Exploration", "Méthode", "Simulation"])

# ----------------------------
# Tab 1 – Exploration
# ----------------------------
with expl_tab:
    if not df.empty:
        st.subheader("Aperçu du dataset")
        st.dataframe(df.head(n_show), width='stretch')

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
                st.plotly_chart(figb, width='stretch')

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
                st.plotly_chart(figt, width='stretch')

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
    import math
    from sklearn.metrics import f1_score, confusion_matrix
    import plotly.express as px

    # 1) Définition des étapes (tu peux ajuster les bullets)
    steps = [
        {
            "title": "Création du pipeline textuel",
            "bullets": [
                "Structure scikit-learn mémoire/cache",
                "Entrées brutes : designation + description"
            ],
        },
        {
            "title": "Nettoyage & normalisation (TextCleaner)",
            "bullets": [
                "Minuscules, accents/HTML, emojis",
                "Traductions via dictionnaire (FR/EN/DE)",
                "Stopwords + stemming léger"
            ],
        },
        {
            "title": "Vectorisation TF-IDF (mots)",
            "bullets": [
                "n-grammes (1–2), sublinear_tf",
                "min_df / max_df réglés pour le bruit"
            ],
        },
        {
            "title": "Pipeline caractères (option)",
            "bullets": [
                "TF-IDF char (2–6) pour fautes/variantes",
                "Robuste aux typos et concaténations"
            ],
        },
        {
            "title": "Petites features texte",
            "bullets": [
                "has_desc, title_len, text_stats, language",
                "Signal complémentaire simple"
            ],
        },
        {
            "title": "Fusion & pondérations",
            "bullets": [
                "FeatureUnion: word + char + feats",
                "Poids ex: word=1.0, char=0.4, feats=0.2"
            ],
        },
        {
            "title": "Standardisation & classifieur",
            "bullets": [
                "StandardScaler(with_mean=False)",
                "LogisticRegression(saga) ou LinearSVC (OvR)"
            ],
        },
        {
            "title": "Évaluation",
            "bullets": [
                "Score cible : F1 pondéré",
                "Matrice de confusion pour les erreurs"
            ],
        },
    ]

    # 2) État (index d'étape)
    if "b2_step" not in st.session_state:
        st.session_state.b2_step = 0
    i = st.session_state.b2_step
    n = len(steps)
    fill = (i + 1) / n  # niveau du cylindre (0..1)

    # 3) Mise en page
    c1, c2 = st.columns([1, 2])

    # 3a) Cylindre (CSS + HTML)
    cyl_css = f"""
    <style>
    .cyl-wrap {{
        height: 360px; width: 130px; position: relative;
        margin: 10px auto; 
    }}
    .cyl {{
        position: absolute; bottom: 0; left: 0; right: 0;
        height: 100%; width: 100%;
        border-radius: 65px;
        border: 3px solid #94a3b8;              /* slate-400 */
        background: linear-gradient(180deg,#f8fafc 0%, #e2e8f0 100%); /* slate-50→200 */
        box-shadow: inset 0 8px 12px rgba(0,0,0,0.06);
        overflow: hidden;
    }}
    .cyl-fill {{
        position: absolute; bottom: 0; left: 0; right: 0;
        height: {int(fill*100)}%;
        background: linear-gradient(180deg,#60a5fa 0%, #2563eb 100%); /* blue-400→600 */
        transition: height 300ms ease;
    }}
    .cyl-gloss {{
        position: absolute; top: 10px; left: 18px; right: 18px; height: 20px;
        border-radius: 10px;
        background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0));
        filter: blur(0.2px);
    }}
    </style>
    """
    with c1:
        st.markdown(cyl_css, unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="cyl-wrap">
              <div class="cyl">
                <div class="cyl-fill"></div>
                <div class="cyl-gloss"></div>
              </div>
            </div>
            <div style="text-align:center; color:#334155;">Étape {i+1}/{n}</div>
            """,
            unsafe_allow_html=True,
        )

    # 3b) Explication
    with c2:
        st.markdown(f"### {steps[i]['title']}")
        for b in steps[i]["bullets"]:
            st.markdown(f"- {b}")

        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if st.button("⟵ Précédent", disabled=(i==0)):
                st.session_state.b2_step = max(0, i-1)
                st.rerun()
        with colB:
            if st.button("Recommencer"):
                st.session_state.b2_step = 0
                st.rerun()
        with colC:
            if st.button("Suivant ⟶", type="primary"):
                st.session_state.b2_step = min(n-1, i+1)
                st.rerun()

    st.divider()

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
                import plotly.express as px
                figcm = px.imshow(
                    cm_pct,
                    text_auto=".2f",               # affiche 2 décimales
                    aspect="auto",
                    color_continuous_scale="Blues",
                    zmin=0, zmax=100,
                    title=f"Matrice de confusion — Baseline B2 (top {len(sel_classes)} classes)",
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
                st.plotly_chart(figcm, width='stretch')
            else:
                st.info("`results/preds_b2.csv` trouvé, mais colonnes y_true / y_pred manquantes.")
        except Exception as e:
            st.warning(f"Impossible de lire `results/preds_b2.csv` ({e}).")
    else:
        st.caption("Astuce : dépose `results/preds_b2.csv` pour calculer F1 et tracer la matrice de confusion automatiquement.")
# === Parcours animé B3 (image) ================================================
def show_b3_walkthrough():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, confusion_matrix
    import plotly.express as px
    import streamlit as st

    # 1) Étapes (pixels OU CNN selon la config de ton entraînement)
    steps = [
        {
            "title": "Chargement des images",
            "bullets": [
                "ImageLoader : lecture RGB + resize",
                "Chemins reconstruits depuis imageid/productid"
            ],
        },
        {
            "title": "Aplatissement des pixels",
            "bullets": [
                "(H, W, 3) → vecteur 1D",
                "Conversion en float32"
            ],
        },
        {
            "title": "Réduction de dimension (option)",
            "bullets": [
                "PCA (dense, whiten) ou TruncatedSVD",
                "Stabilise & compresse avant la régression"
            ],
        },
        {
            "title": "Embeddings CNN (option)",
            "bullets": [
                "ResNet18/50/101 pré-entraîné ImageNet",
                "L2-normalisation des embeddings"
            ],
        },
        {
            "title": "Standardisation & classifieur",
            "bullets": [
                "StandardScaler(with_mean=False)",
                "LogisticRegression(saga) ou LinearSVC (OvR)"
            ],
        },
        {
            "title": "Évaluation",
            "bullets": [
                "Score cible : F1 pondéré",
                "Matrice de confusion (top classes)"
            ],
        },
    ]

    # 2) État (index d’étape)
    if "b3_step" not in st.session_state:
        st.session_state.b3_step = 0
    i = st.session_state.b3_step
    n = len(steps)
    fill = (i + 1) / n  # niveau du cylindre (0..1)

    # 3) Mise en page
    c1, c2 = st.columns([1, 2])

    # 3a) Cylindre (CSS + HTML)
    cyl_css = f"""
    <style>
    .cyl-wrap {{
        height: 360px; width: 130px; position: relative;
        margin: 10px auto;
    }}
    .cyl {{
        position: absolute; bottom: 0; left: 0; right: 0;
        height: 100%; width: 100%;
        border-radius: 65px;
        border: 3px solid #94a3b8;
        background: linear-gradient(180deg,#f8fafc 0%, #e2e8f0 100%);
        box-shadow: inset 0 8px 12px rgba(0,0,0,0.06);
        overflow: hidden;
    }}
    .cyl-fill {{
        position: absolute; bottom: 0; left: 0; right: 0;
        height: {int(fill*100)}%;
        background: linear-gradient(180deg,#60a5fa 0%, #2563eb 100%);
        transition: height 300ms ease;
    }}
    .cyl-gloss {{
        position: absolute; top: 10px; left: 18px; right: 18px; height: 20px;
        border-radius: 10px;
        background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0));
        filter: blur(0.2px);
    }}
    </style>
    """
    with c1:
        st.markdown(cyl_css, unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="cyl-wrap">
              <div class="cyl">
                <div class="cyl-fill"></div>
                <div class="cyl-gloss"></div>
              </div>
            </div>
            <div style="text-align:center; color:#334155;">Étape {i+1}/{n}</div>
            """,
            unsafe_allow_html=True,
        )

    # 3b) Explication + boutons
    def _rerun():
        try:
            st.rerun()
        except Exception:
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

    with c2:
        st.markdown(f"### {steps[i]['title']}")
        for b in steps[i]["bullets"]:
            st.markdown(f"- {b}")

        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if st.button("⟵ Précédent", key="b3_prev", disabled=(i==0)):
                st.session_state.b3_step = max(0, i-1); _rerun()
        with colB:
            if st.button("Recommencer", key="b3_reset"):
                st.session_state.b3_step = 0; _rerun()
        with colC:
            if st.button("Suivant ⟶", key="b3_next", type="primary"):
                st.session_state.b3_step = min(n-1, i+1); _rerun()

    st.divider()

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
                    title=f"Matrice de confusion — Baseline B3 (top {len(sel)} classes)",
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
    st.subheader("Parcours B2 (animé)")
    show_b2_walkthrough()
    st.markdown(
        """
        - **Texte** : TF‑IDF → SVD/PCA → classif (LogReg / LinearSVC).
        - **Image** : CNN ResNet pré‑entraîné (embeddings) → classif.
        - **Fusion** : concat features texte + image → classif final.
        - **CV** : K-fold stratifié, export des embeddings et diagnostics.
        """
    )
    st.divider()
    st.subheader("Parcours B3 (animé)")
    show_b3_walkthrough()
    # Schéma (optionnel)
    dot = """
    digraph G {
      rankdir=LR; node [shape=box];
      TXT [label="Texte\nTF-IDF → SVD"]; IMG [label="Image\nResNet (embeddings)"];
      FUS [label="Fusion\nConcat features"]; CLS [label="Classifier"];
      TXT -> FUS; IMG -> FUS; FUS -> CLS;
    }
    """
    st.graphviz_chart(dot, width='stretch')

# ----------------------------
# Tab 3 – Simulation (démo sans modèle)
# ----------------------------
with sim_tab:
    st.subheader("Simulation (démo)")
    if df.empty:
        st.info("Charge un CSV (ou utilise la démo) pour simuler.")
    else:
        idx = st.number_input("Index de ligne", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
        row = df.iloc[int(idx)]

        # Texte
        if col_text != "(aucune)" and col_text in df.columns:
            st.markdown("**Texte**")
            st.write(str(row[col_text])[:2000])

        # Image
        if col_img != "(aucune)" and col_img in df.columns:
            rp = resolve_image_path(str(row[col_img]), IMAGES_BASE_DIR)
            if rp:
                st.markdown("**Image**")
                try:
                    if rp.startswith("http"):
                        st.image(rp, caption=os.path.basename(rp), width='content')
                    else:
                        input_img = Image.open(rp).convert("RGB")
                        st.image(input_img, caption=os.path.basename(rp), width='content')
                except Exception:
                    st.warning("Impossible d'ouvrir l'image.")

        # Placeholder prédiction
        st.info("Intégrez ici votre modèle (joblib) pour afficher une prédiction et des probabilités.")

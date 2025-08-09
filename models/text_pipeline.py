# Pipeline textuelle pour nettoyer et vectoriser les textes 

# Text_pipeline définit un pipeline de transformation complet 
# (nettoyage + vectorisation) et sera utilisé pour entraîner 
# le modèle ou être combiné à d'autres pipelines (image, cascade).

# Pipeline est un objet sklearn qui permet d’enchaîner plusieurs 
# transformations dans l’ordre. 
# 


# TextCleaner est la classe créée pour nettoyer et combiner 
# les colonnes designation + description.


#TextTfidfVectorizer : encapsule TfidfVectorizer, 
# transforme les textes en vecteurs numériques.

# Pipeline textuelle pour nettoyer et vectoriser les textes 

# -----------------------------------------------------------------------------
# Branche texte pour le pipeline multimodal :
# - Nettoyage + TF-IDF via TextCleaner (stemming only) + TextTfidfVectorizer
# - Ajout d'une feature binaire has_description (0/1)
# - Chargement automatique d'un translate_map (EN/DE -> FR) si présent
# -----------------------------------------------------------------------------

import os
import json
from typing import Optional, Dict

from sklearn.pipeline import make_pipeline, FeatureUnion

from features.text_cleaner import TextCleaner, HasDescriptionFlag, DesignationLength 
# stemming only
from features.text_vectorizer import TextTfidfVectorizer          
# TF-IDF propre


def _load_translate_map(path: Optional[str]) -> Dict[str, str]:
    """
    Charge un mapping token->traduction depuis un fichier JSON s'il existe.
    Formats acceptés :
      - liste d'objets [{"token": "...", "translation": "..."}]
      - dict {"token": "translation", ...}
    Retourne {} si aucun fichier n'est présent ou si parsing impossible.
    """
    candidates = []
    if path:
        candidates.append(path)

    # Fallbacks dans config/
    candidates += [
        os.path.join("config", "translate_map.json"),
        os.path.join("config", "translate_map_starter_from_cleaned.json"),
        os.path.join("config", "translate_map_starter.json"),
    ]

    for p in candidates:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return {d["token"]: d["translation"] for d in data if "token" in d and "translation" in d}
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception as e:
                print(f"[WARN] Impossible de charger le translate_map depuis {p}: {e}")
                continue

    return {}


def create_text_pipeline(
    max_features: int = 5000,
    translate_map_path: Optional[str] = None,
    use_stem: bool = True
):
    """
    Construit la branche texte à brancher dans le FeatureUnion global.

    - max_features : nombre max de features TF-IDF (par défaut 5000)
    - translate_map_path : chemin explicite vers un JSON (sinon auto dans config/)
    - use_stem : active le stemming Snowball (recommandé)
    """
    translate_map = _load_translate_map(translate_map_path)
    if translate_map:
        print(f">> translate_map chargé ({len(translate_map)} entrées).")
    else:
        print(">> Aucun translate_map trouvé : TextCleaner fonctionnera sans traduction.")

    text_tfidf = make_pipeline(
        TextCleaner(
            remove_html=True,
            translate_map=translate_map,  # peut être vide
            use_stem=use_stem
        ),
        TextTfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # unigrams + bigrams (e-commerce)
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
            strip_accents="unicode",
            lowercase=False,            # déjà fait dans TextCleaner
            token_pattern=r"(?u)\b\w+\b",
            dtype="float32"
        )
    )

    # Union : vecteurs TF-IDF + indicateur binaire has_description
    text_branch = FeatureUnion([
        ("tfidf", text_tfidf),
        ("has_desc", HasDescriptionFlag()),
        ("title_len", DesignationLength())
    ])

    return text_branch

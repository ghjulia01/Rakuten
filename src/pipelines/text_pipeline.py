"""
Pipeline de traitement textuel pour la classification Rakuten
----------------------------------------------------------
Ce module gère l'ensemble du traitement textuel :
1. Nettoyage et normalisation du texte
2. Vectorisation TF-IDF (mots et caractères)
3. Features additionnelles (statistiques, langue, etc.)
4. Réduction dimensionnelle SVD (optionnelle)
5. Combinaison pondérée des différentes features
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple


import numpy as np
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from src.features.text.cleaner import TextCleaner, HasDescriptionFlag, DesignationLength
from src.features.text.vectorizer import TextTfidfVectorizer
from src.features.text.stats import ( TextStatistics, TextStatisticsPro, LanguageDetector, Chi2LexiconFeatures
)
from src.utils.profiling import profile_func

# Configuration du logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utilitaires de configuration
# -----------------------------------------------------------------------------
@profile_func
def _load_translate_map(path: Optional[str]) -> Dict[str, str]:
    """
    Charge le dictionnaire de traduction depuis un fichier JSON.
    
    Args:
        path: Chemin vers le fichier JSON de traduction
        
    Returns:
        Dictionnaire {token: traduction} ou {} si erreur
    """
    candidates: list[str] = []
    if path:
        candidates.append(path)

    # Chemins par défaut
    candidates += [
        os.path.join("config", "translate_map.json"),
        os.path.join("config", "translate_map_starter_from_cleaned.json"),
    ]

    for p in candidates:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return {
                        d["token"]: d["translation"]
                        for d in data
                        if isinstance(d, dict) and "token" in d and "translation" in d
                    }
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Échec chargement translate_map depuis {p}: {e}")
    return {}
@profile_func
def _coerce_df_param(val: Any, *, is_max: bool = False) -> float | int:
    """
    Normalise les paramètres min_df/max_df pour sklearn.
    
    Args:
        val: Valeur à normaliser
        is_max: True si max_df, False si min_df
        
    Returns:
        Valeur normalisée (int >= 1 ou float ∈ [0,1])
    """
    if isinstance(val, int):
        if val == 0:
            return 0.0
        if is_max and val == 1:
            return 1.0
        return val
    try:
        v = float(val)
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.0 if not is_max else 1.0

# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
@profile_func
def create_text_pipeline(
    *,
    # 1) Cleaner
    translate_map_path: Optional[str] = None,
    use_stem: bool = True,
    clean_special: bool = True,
    handle_emojis: bool = True,
    remove_numbers: bool = False,
    use_lemmatization: bool = False,   # exposé pour usage ultérieur si besoin

    # 2) TF-IDF (branche word)
    max_features: int = 100_000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int | float = 2,
    max_df: int | float = 0.95,
    sublinear_tf: bool = True,
    norm: str = "l2",
    strip_accents: str | None = "unicode",  # None | "ascii" | "unicode"

    # 3) Features additionnelles
    use_language_detection: bool = True,
    use_text_stats: bool = True,
    use_text_stats_pro: bool = False,      
    use_lexicon: bool = False,             
    lexicon_top_k: int = 20,               
    lexicon_min_df: int | float = 3,
    lexicon_max_df: int | float = 0.95,
) -> FeatureUnion:
    """
    Crée le pipeline complet de traitement textuel.
    
    Le pipeline combine:
    1. TF-IDF sur les mots (avec nettoyage)
    2. Indicateurs de présence/longueur
    3. Statistiques textuelles
    4. Détection de langue
    
    Note: Les stopwords sont gérés dans TextCleaner via STOPWORDS_ALL
    """
    logger.info("Création du pipeline textuel...")
    
    # 1. Chargement du dictionnaire de traduction
    translate_map = _load_translate_map(translate_map_path)
    if translate_map:
        logger.info(f"Dictionnaire de traduction chargé ({len(translate_map)} entrées)")
    
    # 2. Pipeline TF-IDF principal
    text_tfidf = make_pipeline(
        TextCleaner(
            remove_html=True,
            translate_map_path=translate_map_path,
            use_stem=use_stem,
            clean_special=clean_special,
            handle_emojis=handle_emojis,
            remove_numbers=remove_numbers,
        ),
        TextTfidfVectorizer(
            analyzer="word",
            max_features=int(max_features),
            ngram_range=(int(ngram_min), int(ngram_max)),
            min_df=_coerce_df_param(min_df, is_max=False),
            max_df=_coerce_df_param(max_df, is_max=True),
            sublinear_tf=bool(sublinear_tf),
            norm=str(norm),
            strip_accents=strip_accents,
            lowercase=False,  # Déjà fait par TextCleaner
            dtype=np.float32,
            stop_words=None  
        ),
    )

    # 3. Construction du FeatureUnion
    transformers = [
        ("tfidf", text_tfidf),
        ("has_desc", HasDescriptionFlag()),
        ("title_len", DesignationLength()),
    ]

    if use_text_stats:
        transformers.append(("text_stats", TextStatistics()))

    if use_text_stats_pro:
        transformers.append(("text_stats_pro", TextStatisticsPro()))
    
    if use_language_detection:
        transformers.append(("language", LanguageDetector()))

    if use_lexicon:
        transformers.append((
            "lexicon",
            Chi2LexiconFeatures(top_k=lexicon_top_k, binary=True, min_df=lexicon_min_df, max_df=lexicon_max_df)
        ))

    return FeatureUnion(transformers)


@profile_func
def create_text_pipeline_from_cfg(cfg_text: Dict[str, Any]) -> Pipeline:
    """
    Construit le pipeline texte depuis la config avec SVD optionnelle.
    
    Ajoute automatiquement la réduction SVD si activée dans config.
    """
    tmap_path = cfg_text.get("translate_map_path", None)
    n_map = len(_load_translate_map(tmap_path)) if tmap_path else 0
    logger.info(
        "Construction du pipeline depuis la configuration… translate_map_path=%s (n=%d)",
        tmap_path, n_map
    )

    # 1) Construire la branche "word" complète (avec ses sous-transformeurs)
    word_branch = create_text_pipeline(
        # Cleaner
        translate_map_path=cfg_text.get("translate_map_path", None),
        use_stem=cfg_text.get("use_stem", True),
        clean_special=cfg_text.get("clean_special", True),
        handle_emojis=cfg_text.get("handle_emojis", True),
        remove_numbers=cfg_text.get("remove_numbers", False),
        use_lemmatization=cfg_text.get("use_lemmatization", False),

        # TF-IDF (word)
        max_features=cfg_text.get("max_features", 100_000),
        ngram_min=cfg_text.get("ngram_min", 1),
        ngram_max=cfg_text.get("ngram_max", 2),
        min_df=cfg_text.get("min_df", 2),
        max_df=cfg_text.get("max_df", 0.95),
        sublinear_tf=cfg_text.get("sublinear_tf", True),
        norm=cfg_text.get("norm", "l2"),
        strip_accents=cfg_text.get("strip_accents", "unicode"),

        # Features additionnelles
        use_language_detection=cfg_text.get("use_language_detection", True),
        use_text_stats=cfg_text.get("use_text_stats", True),
        use_text_stats_pro=cfg_text.get("use_text_stats_pro", False),
        use_lexicon=cfg_text.get("lexicon", {}).get("enabled", False),
        lexicon_top_k=cfg_text.get("lexicon", {}).get("top_k", 20),
        lexicon_min_df=cfg_text.get("lexicon", {}).get("min_df", 3),
        lexicon_max_df=cfg_text.get("lexicon", {}).get("max_df", 0.95)
    )

    # 2) Optionnel : ajouter la branche caractères
    transformers = [("tfidf_word", word_branch)]
    char_enabled = bool(cfg_text.get("char", {}).get("enabled", False))
    if char_enabled:
        char_cfg = cfg_text["char"]
        logger.info("Ajout du pipeline de caractères...")
        char_pipeline = make_pipeline(
            TextCleaner(
                use_stem=False,
                translate_map_path=cfg_text.get("translate_map_path", None),
                clean_special=cfg_text.get("clean_special", True),
                handle_emojis=cfg_text.get("handle_emojis", True),
                remove_numbers=cfg_text.get("remove_numbers", False),
            ),
            TextTfidfVectorizer(
                analyzer=str(char_cfg.get("analyzer", "char_wb")),  # ← lit le TOML
                ngram_range=(int(char_cfg.get("ngram_min", 2)), int(char_cfg.get("ngram_max", 6))),
                min_df=char_cfg.get("min_df", 2),
                max_df=char_cfg.get("max_df", 0.95),
                sublinear_tf=bool(char_cfg.get("sublinear_tf", True)),
                strip_accents=char_cfg.get("strip_accents", None),  # souvent None côté char
                dtype=np.float32,
            ),
        )
        transformers.append(("tfidf_char", char_pipeline))

    # 3) Répartition et application des poids (robuste aux branches désactivées)
    raw_weights = dict(cfg_text.get("weights", {}) or {})
    if raw_weights:
        logger.info("Application des poids (config): %s", raw_weights)

    # Branches effectivement présentes
    # - internes (dans la sous-union 'word_branch')
    inner_present = {name for name, _ in word_branch.transformer_list}
    # - top-level (dans la grande union texte)
    top_present   = {name for name, _ in transformers}

    # Filtrer les poids selon la présence réelle
    inner_weights = {k: float(v) for k, v in raw_weights.items() if k in inner_present}
    top_weights   = {k: float(v) for k, v in raw_weights.items() if k in top_present}

    # Poids ignorés (ex. language si use_language_detection=false, tfidf_char si char désactivé, etc.)
    dropped = sorted(set(raw_weights) - (set(inner_weights) | set(top_weights)))
    if dropped:
        logger.warning("Weights ignorés (pas présents à ce niveau): %s", dropped)

    # Appliquer les poids internes à la branche word
    if inner_weights:
        word_branch.set_params(transformer_weights=inner_weights)

    # Construire le FeatureUnion top-level avec les poids filtrés
    feature_union = FeatureUnion(transformers, transformer_weights=(top_weights or None))
    
    # ========================================
    # 4) Appliquer SVD si activée
    # ========================================
    svd_cfg = cfg_text.get("svd", {})
    svd_enabled = bool(svd_cfg.get("enabled", False))
    
    # DEBUG: Afficher la config SVD
    logger.info(f"[DEBUG] Config SVD reçue: {svd_cfg}")
    logger.info(f"[DEBUG] SVD enabled: {svd_enabled}")
    
    if svd_enabled:
        n_components = int(svd_cfg.get("n_components", 500))
        random_state = int(svd_cfg.get("random_state", 42))
        l2norm = bool(svd_cfg.get("l2norm", True))
        
        logger.info(f" Ajout de la réduction SVD (n_components={n_components})")
        
        # Créer le pipeline avec SVD
        steps = [
            ("features", feature_union),
            ("svd", TruncatedSVD(n_components=n_components, random_state=random_state))
        ]
        
        if l2norm:
            steps.append(("l2norm", Normalizer(copy=False)))
            logger.info(f"   + Normalisation L2")
        
        return Pipeline(steps)
    else:
        # Pas de SVD, retourner juste le FeatureUnion dans un Pipeline
        logger.info(" SVD désactivée - Pipeline sans réduction dimensionnelle")
        return Pipeline([("features", feature_union)])
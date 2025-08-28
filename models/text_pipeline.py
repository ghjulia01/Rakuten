"""
Branche TEXTE du pipeline Rakuten — version très commentée (FR)

But :
- Construire une branche texte modulaire (nettoyer → vectoriser) à intégrer dans le FeatureUnion.
- Lire autant que possible les hyperparamètres depuis le fichier TOML (via train_model.py).
- Offrir deux modes de construction :
  1) create_text_pipeline(...)
     → recevoir des paramètres explicites (appel direct et test unitaire facilités).
  2) create_text_pipeline_from_cfg(cfg_text)
     → lire une sous-config TOML (cfg["text"]) et appeler create_text_pipeline.

Choix d'implémentation :
- Utiliser TextCleaner (normaliser, retirer HTML, traduire via translate_map, stemmer) ;
- Utiliser TextTfidfVectorizer (wrapper TF-IDF cohérent avec sklearn) ;
- Ajouter des features simples (HasDescriptionFlag, DesignationLength) via FeatureUnion ;
- Forcer dtype=np.float64 pour éviter le warning sklearn (float32 → float64) ;
- Ne pas baisser en lowercase dans le vectorizer (déjà fait dans TextCleaner).

Remarque :
- Laisser au script principal (train_model.py) la responsabilité d'ouvrir le TOML
  et de passer cfg["text"]. Ici, proposer un helper pour plus de confort.
"""

from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any

import numpy as np
from sklearn.pipeline import make_pipeline, FeatureUnion

from features.text_cleaner import TextCleaner, HasDescriptionFlag, DesignationLength
from features.text_vectorizer import TextTfidfVectorizer


# -----------------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------------

def _load_translate_map(path: Optional[str]) -> Dict[str, str]:
    """Charger un dictionnaire token→traduction depuis JSON.

    Accepter :
      - liste d'objets {"token": "...", "translation": "..."}
      - dictionnaire {token: translation}
    Retourner {} si absent/invalide.

    Procéder :
      - essayer le chemin explicite ;
      - fallback dans ./config/ pour quelques noms usuels ;
      - sécuriser le parsing JSON.
    """
    candidates: list[str] = []
    if path:
        candidates.append(path)

    # Définir des fallbacks usuels (adapter si besoin)
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
                    return {
                        d["token"]: d["translation"]
                        for d in data
                        if isinstance(d, dict) and "token" in d and "translation" in d
                    }
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception as e:  # garder robuste en dev
                print(f"[WARN] Échec de chargement translate_map depuis {p}: {e}")
                continue
    return {}


def _coerce_df_param(val: Any, *, is_max: bool = False) -> float | int:
    """Normaliser min_df/max_df au format accepté par sklearn.

    - Accepter int >= 1 (compte absolu de documents) ;
    - Accepter float ∈ [0.0, 1.0] (proportion) ;
    - Convertir 0 → 0.0 et 1 (pour max_df) → 1.0.
    """
    if isinstance(val, int):
        if val == 0:
            return 0.0
        if is_max and val == 1:
            return 1.0
        return val
    try:
        v = float(val)
    except Exception:
        v = 0.0 if not is_max else 1.0
    return v


# -----------------------------------------------------------------------------
# Fabriques de pipeline
# -----------------------------------------------------------------------------

def create_text_pipeline(
    *,
    # --- Nettoyage / normalisation ---
    translate_map_path: Optional[str] = None,
    use_stem: bool = True,
    # --- TF-IDF ---
    max_features: int = 50_000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: float | int = 2,
    max_df: float | int = 0.95,
    sublinear_tf: bool = True,
    norm: str = "l2",
    strip_accents: Optional[str] = "unicode",
    stop_words: Optional[str | list[str]] = None,
) -> FeatureUnion:
    """Construire la branche texte : nettoyer → TF‑IDF → features simples.

    Paramètres principaux (à lire depuis TOML dans train_model.py ou via helper) :
      - translate_map_path : définir le chemin JSON du dictionnaire de traduction.
      - use_stem          : activer le stemming (réduire la variance lexicale).
      - max_features      : limiter la dimension du vocabulaire (contrôler mémoire/temps).
      - ngram_min/max     : définir la fenêtre d'ngrams (1–2 classique en e‑commerce).
      - min_df / max_df   : filtrer mots trop rares / trop fréquents.
      - sublinear_tf      : utiliser 1 + log(tf) (souvent bénéfique pour LR/SVM).
      - norm              : définir la normalisation ("l2" par défaut).
      - strip_accents     : homogénéiser les accents ("unicode" recommandé).
      - stop_words        : définir des stopwords ("french", "english" ou liste) si besoin.

    Retourner : FeatureUnion combinant (TF‑IDF) + (HasDescriptionFlag) + (DesignationLength).
    """
    # 1) Charger le translate_map (optionnel)
    translate_map = _load_translate_map(translate_map_path)
    if translate_map:
        print(f">> translate_map chargé ({len(translate_map)} entrées).")
    else:
        print(">> Aucun translate_map trouvé : exécuter TextCleaner sans traduction.")

    # 2) Normaliser min_df/max_df au format sklearn
    min_df = _coerce_df_param(min_df, is_max=False)
    max_df = _coerce_df_param(max_df, is_max=True)

    # 3) Construire le sous-pipeline TF‑IDF (cleaner → vectorizer)
    text_tfidf = make_pipeline(
        # a) Nettoyer les textes (déjà lower/strip/punct selon ton TextCleaner)
        TextCleaner(
            remove_html=True,
            translate_map=translate_map,
            use_stem=use_stem,
        ),
        # b) Vectoriser en TF‑IDF ; forcer dtype=np.float64 pour éviter le warning
        TextTfidfVectorizer(
            max_features=int(max_features),
            ngram_range=(int(ngram_min), int(ngram_max)),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=bool(sublinear_tf),
            norm=str(norm),
            strip_accents=strip_accents,
            lowercase=False,              # déjà géré par TextCleaner
            token_pattern=r"(?u)\b\w+\b",
            stop_words=stop_words,
            dtype=np.float64,
        ),
    )

    # 4) Ajouter des features dérivées simples via FeatureUnion
    text_branch = FeatureUnion([
        ("tfidf", text_tfidf),
        ("has_desc", HasDescriptionFlag()),
        ("title_len", DesignationLength()),
    ])

    return text_branch


def create_text_pipeline_from_cfg(cfg_text: Dict[str, Any]) -> FeatureUnion:
    """Construire la branche texte en lisant une sous-config TOML (cfg["text"]).

    Exemple d'usage (dans train_model.py) :
        text_branch = create_text_pipeline_from_cfg(cfg.get("text", {}))

    Champs attendus (tous optionnels) :
        max_features, ngram_min, ngram_max, min_df, max_df,
        sublinear_tf, use_stem, translate_map_path,
        norm, strip_accents, stop_words.
    """
    # a) Lire les valeurs avec défauts robustes
    max_features = cfg_text.get("max_features", 50_000)
    ngram_min = cfg_text.get("ngram_min", 1)
    ngram_max = cfg_text.get("ngram_max", 2)
    min_df = cfg_text.get("min_df", 2)
    max_df = cfg_text.get("max_df", 0.95)
    sublinear_tf = cfg_text.get("sublinear_tf", True)
    use_stem = bool(cfg_text.get("use_stem", True))
    translate_map_path = cfg_text.get("translate_map_path", None)
    norm = cfg_text.get("norm", "l2")
    strip_accents = cfg_text.get("strip_accents", "unicode")
    stop_words = cfg_text.get("stop_words", None)

    # b) Appeler la fabrique principale
    return create_text_pipeline(
        translate_map_path=translate_map_path,
        use_stem=use_stem,
        max_features=max_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm=norm,
        strip_accents=strip_accents,
        stop_words=stop_words,
    )

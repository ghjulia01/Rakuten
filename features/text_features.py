# features/text_features.py
"""
Features additionnelles pour l'analyse de texte.
Fournir des statistiques textuelles et une détection de langue robustes
aux différents formats d'entrée (DataFrame, Series, liste).
"""

from __future__ import annotations
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from langdetect import detect

logger = logging.getLogger(__name__)

# --- utilitaire ---------------------------------------------------------------
def _to_text_series(X: object) -> pd.Series:
    """Normaliser X en une Series de textes, quelle que soit sa forme d'entrée.
    - Si X est un DataFrame, combiner 'designation' + 'description' quand dispo.
    - Si X est une Series/itérable, convertir en str.
    - Toujours retourner une Series de longueur n_samples.
    """
    # Si DataFrame: combiner proprement
    if isinstance(X, pd.DataFrame):
        cols = [c for c in ("designation", "description") if c in X.columns]
        if len(cols) == 0:
            # Prendre la première colonne s'il n'y a pas les colonnes attendues
            if X.shape[1] == 0:
                return pd.Series([], index=X.index, dtype=str)
            return X.iloc[:, 0].fillna("").astype(str)
        if len(cols) == 1:
            return X[cols[0]].fillna("").astype(str)
        # 2 colonnes
        return (X[cols[0]].fillna("") + " " + X[cols[1]].fillna("")).astype(str)

    # Si Series déjà
    if isinstance(X, pd.Series):
        return X.fillna("").astype(str)

    # Sinon: essayer comme itérable
    try:
        return pd.Series([("" if v is None else str(v)) for v in X])
    except Exception:
        # Dernier recours: un seul élément
        return pd.Series([("" if X is None else str(X))])

# --- transformeurs ------------------------------------------------------------
class TextStatistics(BaseEstimator, TransformerMixin):
    """Extraire des statistiques simples sur le texte.
    Sortie: ndarray (n_samples, 4) = [n_mots, len_moyenne, diversité_lexicale, ratio_majuscules]
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        s = _to_text_series(X)  # normaliser l'entrée
        features = []
        for text in s:
            # gérer les valeurs non-string
            if not isinstance(text, str) or text == "":
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            words = text.split()
            n_words = float(len(words))
            n_chars = float(len(text))
            avg_word_len = (n_chars / n_words) if n_words > 0 else 0.0
            lex_div = (len(set(words)) / n_words) if n_words > 0 else 0.0
            caps_ratio = (sum(c.isupper() for c in text) / n_chars) if n_chars > 0 else 0.0

            features.append([n_words, avg_word_len, lex_div, caps_ratio])

        return np.asarray(features, dtype=np.float32)

class LanguageDetector(BaseEstimator, TransformerMixin):
    """Détecter la langue et renvoyer un one-hot (fr, en, de par défaut).
    Sortie: ndarray (n_samples, n_langues)
    """
    def __init__(self, languages: Optional[List[str]] = None, min_length: int = 10):
        self.languages = languages or ["fr", "en", "de"]
        self.min_length = int(min_length)

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        s = _to_text_series(X)  # normaliser l'entrée
        out = np.zeros((len(s), len(self.languages)), dtype=np.float32)
        for i, text in enumerate(s):
            if not isinstance(text, str) or len(text) < self.min_length:
                continue
            try:
                lang = detect(text)
                if lang in self.languages:
                    out[i, self.languages.index(lang)] = 1.0
            except Exception as e:
                logger.debug(f"LangDetect erreur sur l'élément {i}: {e}")
        return out
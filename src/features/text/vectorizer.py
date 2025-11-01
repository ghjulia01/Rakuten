# -----------------------------------------------------------------------------
# Vectorisation TF-IDF pour textes déjà nettoyés par TextCleaner.
# - Pas de prétraitement lourd ici : le nettoyage est fait en amont.
# - Paramètres utiles exposés (ngrammes, min_df, max_df, strip accents, etc.)
# -----------------------------------------------------------------------------

# features/text_vectorizer.py
# -----------------------------------------------------------------------------
# Vectorisation TF-IDF pour textes déjà nettoyés (TextCleaner).
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Union, Tuple, List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.profiling import profile_func

StopWords = Optional[Union[str, List[str]]]

class TextTfidfVectorizer(BaseEstimator, TransformerMixin):
    @profile_func
    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.95,
        sublinear_tf: bool = True,
        norm: str = "l2",
        strip_accents: Optional[str] = "unicode",
        lowercase: bool = False,                # déjà en minuscules via TextCleaner
        token_pattern: str = r"(?u)\b(?=\w*[A-Za-z])\w+\b",
        dtype: Union[str, np.dtype] = "float64",
        stop_words: StopWords = None,
        analyzer: str = "word",   # "word" | "char" | "char_wb"       
    ):
        # stocker explicitement chaque paramètre (clone-friendly)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.dtype = dtype
        self.stop_words = stop_words
        self.analyzer = analyzer           

        self.vectorizer: Optional[TfidfVectorizer] = None

    @profile_func
    def _make_vec(self) -> TfidfVectorizer:
        # Forcer float64 pour éviter le warning sklearn (float32→float64)
        dtype = np.float64 if str(self.dtype) not in ("float32", "np.float32") else np.float64
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            norm=self.norm,
            strip_accents=self.strip_accents,
            lowercase=self.lowercase,
            analyzer=self.analyzer,
            token_pattern=(None if self.analyzer != "word" else self.token_pattern),
            dtype=dtype,
            stop_words=self.stop_words,         
        )

    @profile_func
    def fit(self, X, y=None):
        self.vectorizer = self._make_vec()
        self.vectorizer.fit(X)
        return self

    @profile_func
    def transform(self, X):
        if self.vectorizer is None:
            raise RuntimeError("TextTfidfVectorizer non fitted : appeler fit() avant transform().")
        return self.vectorizer.transform(X)

    @profile_func
    def get_feature_names_out(self):
        if self.vectorizer is None:
            raise RuntimeError("TextTfidfVectorizer non fitted.")
        return self.vectorizer.get_feature_names_out()

    # Laisser sklearn régler/inspecter les hyperparamètres
    @profile_func
    def get_params(self, deep=True):
        return {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "norm": self.norm,
            "strip_accents": self.strip_accents,
            "lowercase": self.lowercase,
            "token_pattern": self.token_pattern,
            "dtype": self.dtype,
            "stop_words": self.stop_words,
            "analyzer": self.analyzer,
        }

    @profile_func
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        # recréer le vectorizer avec les nouveaux params au prochain fit()
        self.vectorizer = None
        return self
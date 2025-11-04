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
import unicodedata
from langdetect import DetectorFactory
DetectorFactory.seed = 0

from src.utils.profiling import profile_func

logger = logging.getLogger(__name__)

@profile_func
def _fold(s: str) -> str:
    # lower + suppression des accents
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


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

    @profile_func
    def fit(self, X, y=None):
        return self

    @profile_func
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
            avg_word_len = (sum(len(w) for w in words) / n_words) if n_words > 0 else 0.0
            lex_div = (len(set(words)) / n_words) if n_words > 0 else 0.0
            caps_ratio = (sum(c.isupper() for c in text) / n_chars) if n_chars > 0 else 0.0

            features.append([n_words, n_chars, avg_word_len, lex_div, caps_ratio])

        return np.asarray(features, dtype=np.float32)
    
    @profile_func
    def get_feature_names_out(self, input_features=None):
        return np.array(["n_words", "n_chars","avg_word_len","lex_div","caps_ratio"], dtype=object)

class LanguageDetector(BaseEstimator, TransformerMixin):
    """Détecter la langue et renvoyer un one-hot (fr, en, de par défaut).
    Sortie: ndarray (n_samples, n_langues)
    """
    @profile_func
    def __init__(self, languages: Optional[List[str]] = None, min_length: int = 10, max_chars: int = 500):
        self.languages = languages or ["fr", "en", "de"]
        self.min_length = int(min_length)
        self.max_chars = int(max_chars)

    @profile_func
    def fit(self, X, y=None):
        return self

    @profile_func
    def transform(self, X) -> np.ndarray:
        s = _to_text_series(X)  # normaliser l'entrée
        out = np.zeros((len(s), len(self.languages)), dtype=np.float32)
        for i, text in enumerate(s):
            if not isinstance(text, str) or len(text) < self.min_length:
                continue
            try:
                text = text[: self.max_chars]
                lang = detect(text)
                if lang in self.languages:
                    out[i, self.languages.index(lang)] = 1.0
            except Exception as e:
                logger.debug(f"LangDetect erreur sur l'élément {i}: {e}")
        return out
    
    def get_feature_names_out(self, input_features=None):
        return np.array([f"lang_{c}" for c in self.languages], dtype=object)
    
# --- Statistiques textuelles avancées ------------------------------------------
import re
class TextStatisticsPro(BaseEstimator, TransformerMixin):
    """
    Statistiques enrichies pour titres/descriptions.
    Sortie: ndarray (n_samples, K) avec K 25 features.
    """
    _re_year   = re.compile(r"\b(19|20)\d{2}\b")
    _re_dim    = re.compile(r"\b\d+\s*[xX×]\s*\d+\b")
    _re_isbn   = re.compile(r"\b(?:97[89][- ]?)?\d{9}[\dxX]\b")
    _re_digits = re.compile(r"\d")
    _re_upper_token = re.compile(r"^[A-Z]{2,}$")

    UNITS = ("cm","mm","kg","g","l","ml","gb","mah","cl")
    SIZES = ("xs","s","m","l","xl","xxl")
    PLAT  = ("ps5","ps4","ps3","playstation","xbox","switch","pc")
    BOOK  = ("tome","volume","broché","poche","édition","edition","manga","isbn")
    BABY  = ("bébé","doudou","peluche","poupee","siegeauto","biberon","poussette","siège","couches", "tétine", "tetine")
    TOY   = ("figurine","wargame","warhammer","lego","playmobil")
    GARD  = ("jardin","tondeuse","sécateur","outil")
    FOOD  = ("bio","goût","saveur")
    LOT   = ("lot", "pack", "bundle")
    GAMING= ("gamer", "gaming", "gamers", "ps","xbox", "switch",   "nintendo", 
             "nintendo switch", "xbox", "xbox one", "xbox series", "ps4", "ps5", "playstation", "playstation 4", "playstation 5") 
    STREAMING = ("streaming", "stream", "digital", "code", "origin", "uplay", "steam", "epic", "gog", "battle.net", "clé" )
    PUZZLE = ("puzzle", "casse-tête", "casse tete", "énigme", "enigme")

    def fit(self, X, y=None):
        self._UNITS_F = {_fold(t) for t in self.UNITS}
        self._SIZES_F = {_fold(t) for t in self.SIZES}
        self._PLAT_F  = {_fold(t) for t in self.PLAT}
        self._BOOK_F  = {_fold(t) for t in self.BOOK}
        self._BABY_F  = {_fold(t) for t in self.BABY}
        self._TOY_F   = {_fold(t) for t in self.TOY}
        self._GARD_F  = {_fold(t) for t in self.GARD}
        self._FOOD_F  = {_fold(t) for t in self.FOOD}
        self._LOT_F   = {_fold(t) for t in self.LOT}
        self._GAMING_F = {_fold(t) for t in self.GAMING}
        self._STREAMING_F = {_fold(t) for t in self.STREAMING}
        self._PUZZLE_F = {_fold(t) for t in self.PUZZLE}
        return self

    def transform(self, X) -> np.ndarray:
        s = _to_text_series(X)
        out = []
        for text in s:
            if not isinstance(text, str): text = ""
            t = text.strip()
            n_chars = float(len(t))
            words = t.split()
            n_words = float(len(words))
            uniq = len(set(words))
            avg_wlen = (sum(len(w) for w in words) / n_words) if n_words>0 else 0.0
            lex_div = (uniq / n_words) if n_words>0 else 0.0

            # ratios
            digits = len(self._re_digits.findall(t))
            digit_ratio = digits / n_chars if n_chars>0 else 0.0
            punct = sum(c in ".,;:!?()[]{}-/+&" for c in t)
            punct_ratio = punct / n_chars if n_chars>0 else 0.0
            caps_ratio = (sum(c.isupper() for c in t) / n_chars) if n_chars>0 else 0.0
            upper_tok_ratio = (sum(1 for w in words if self._re_upper_token.match(w)) / n_words) if n_words>0 else 0.0
            non_ascii_ratio = (sum(ord(c)>127 for c in t) / n_chars) if n_chars>0 else 0.0

            # patterns/bools
            has_year = 1.0 if self._re_year.search(t) else 0.0
            has_dim  = 1.0 if self._re_dim.search(t.lower()) else 0.0
            has_isbn = 1.0 if self._re_isbn.search(t.replace(" ", "").lower()) else 0.0
            has_euro = 1.0 if "€" in t or "eur" in t.lower() else 0.0
            has_percent = 1.0 if "%" in t else 0.0
            age_flag = 1.0 if re.search(r"\b\d+\s*(ans|\+)\b", t.lower()) else 0.0

            low = _fold(t) 
            def any_in_folded(fset): return 1.0 if any(tok in low for tok in fset) else 0.0
            UNITS_F = getattr(self, "_UNITS_F", {_fold(z) for z in self.UNITS})
            SIZES_F = getattr(self, "_SIZES_F", {_fold(z) for z in self.SIZES})
            PLAT_F  = getattr(self, "_PLAT_F",  {_fold(z) for z in self.PLAT})
            BOOK_F  = getattr(self, "_BOOK_F",  {_fold(z) for z in self.BOOK})
            BABY_F  = getattr(self, "_BABY_F",  {_fold(z) for z in self.BABY})
            TOY_F   = getattr(self, "_TOY_F",   {_fold(z) for z in self.TOY})
            GARD_F  = getattr(self, "_GARD_F",  {_fold(z) for z in self.GARD})
            FOOD_F  = getattr(self, "_FOOD_F",  {_fold(z) for z in self.FOOD})
            LOT_F   = getattr(self, "_LOT_F",   {_fold(z) for z in self.LOT})
            GAMING_F = getattr(self, "_GAMING_F", {_fold(z) for z in self.GAMING})
            STREAMING_F = getattr(self, "_STREAMING_F", {_fold(z) for z in self.STREAMING})
            PUZZLE_F = getattr(self, "_PUZZLE_F", {_fold(z) for z in self.PUZZLE})

            u_units    = any_in_folded(UNITS_F)
            u_sizes    = any_in_folded(SIZES_F)
            f_plat     = any_in_folded(PLAT_F)
            f_book     = any_in_folded(BOOK_F)
            f_baby     = any_in_folded(BABY_F)
            f_toy      = any_in_folded(TOY_F)
            f_gard     = any_in_folded(GARD_F)
            f_food     = any_in_folded(FOOD_F)
            f_lot      = any_in_folded(LOT_F)
            f_gaming   = any_in_folded(GAMING_F)
            f_streaming= any_in_folded(STREAMING_F)
            f_puzzle   = any_in_folded(PUZZLE_F)

            out.append([
                n_words, avg_wlen, lex_div, caps_ratio,
                digit_ratio, punct_ratio, upper_tok_ratio, non_ascii_ratio,
                has_year, has_dim, has_isbn, has_euro, has_percent, age_flag,
                u_units, u_sizes, f_plat, f_book, f_baby, f_toy, f_gard, f_food,
                f_lot, f_gaming, f_streaming, f_puzzle
            ])
        return np.asarray(out, dtype=np.float32)
    

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "n_words","avg_wlen","lex_div","caps_ratio",
            "digit_ratio","punct_ratio","upper_tok_ratio","non_ascii_ratio",
            "has_year","has_dim","has_isbn","has_euro","has_percent","age_flag",
            "has_units","has_sizes","plat_flag","book_flag","baby_flag","toy_flag","garden_flag","food_flag",
            "lot_flag","gaming_flag","streaming_flag","puzzle_flag"
        ], dtype=object)
    
# --- Lexiques par classe (chi2) ----------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

class Chi2LexiconFeatures(BaseEstimator, TransformerMixin):
    """
    Construit automatiquement un mini-lexique (top_k tokens) par classe via chi²,
    puis retourne, pour chaque échantillon, un score par classe = nombre de hits
    de mots du lexique (ou 0/1 si binary=True).

    Sortie: ndarray (n_samples, n_classes)
    """
    def __init__(
        self,
        top_k: int = 20,
        min_df: int | float = 3,
        max_df: int | float = 0.98,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 120_000,
        binary: bool = False,
    ):
        self.top_k = int(top_k)
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = tuple(ngram_range)
        self.max_features = int(max_features)
        self.binary = bool(binary)

        self.vectorizer_: TfidfVectorizer | None = None
        self.classes_: np.ndarray | None = None
        self.lexicons_: dict[int, set[str]] = {}

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Chi2LexiconFeatures.fit exige y (les labels).")
        s = _to_text_series(X)
        self.vectorizer_ = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",  # déjà nettoyé par TextCleaner en amont
            norm=None,        # chi² n'a pas besoin du norm
        )
        Xt = self.vectorizer_.fit_transform(s)
        self._analyzer_ = self.vectorizer_.build_analyzer()
        vocab = np.array(list(self.vectorizer_.vocabulary_.keys()))
        # remettre vocab dans l'ordre des colonnes
        inv_vocab = np.empty(len(self.vectorizer_.vocabulary_), dtype=object)
        for tok, j in self.vectorizer_.vocabulary_.items():
            inv_vocab[j] = tok
        vocab = inv_vocab

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.lexicons_.clear()

        for c in self.classes_:
            y_bin = (y == c).astype(int)
            scores, _ = chi2(Xt, y_bin)
            scores = np.nan_to_num(scores, nan=0.0)
            top_idx = np.argsort(scores)[::-1][: self.top_k]
            self.lexicons_[int(c)] = set(str(t) for t in vocab[top_idx] if isinstance(t, str))

        return self

    def transform(self, X) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Transformer non fit().")
        s = _to_text_series(X)
        
        # CORRECTION: Tokenisation manuelle pour éviter FutureWarning sklearn
        # On ne dépend plus de self._analyzer_ qui utilise le vectorizer interne
        docs_tokens = [set(txt.lower().split()) for txt in s]
        
        out = np.zeros((len(s), len(self.classes_)), dtype=np.float32)
        for j, c in enumerate(self.classes_):
            lex = self.lexicons_.get(int(c), set())
            if not lex:
                continue
            for i, toks in enumerate(docs_tokens):
                hits = len(lex.intersection(toks))
                out[i, j] = 1.0 if (self.binary and hits > 0) else float(hits)
        return out

    def get_feature_names_out(self, input_features=None):
        if self.classes_ is None:
            return np.array([], dtype=object)
        return np.array([f"lexicon_cls_{int(c)}" for c in self.classes_], dtype=object)
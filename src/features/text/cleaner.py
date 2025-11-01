"""
Pre-processing textuel et feature engineering pour classification Rakuten
---------------------------------------------------------------------

Ce module fournit plusieurs transformateurs pour le traitement du texte :
1. TextCleaner : Nettoyage et normalisation du texte
   - Fusion designation + description
   - Nettoyage HTML et caractères spéciaux
   - Gestion des emojis
   - Stemming multilingue (FR/EN/DE)
   - Traduction via dictionnaire
2. Features additionnelles :
   - HasDescriptionFlag : Indicateur présence/absence de description
   - DesignationLength : Longueur du titre
   - LanguageFeaturizer : Détection de langue
"""

# === Imports ===============================================================
from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
import json
import pandas as pd
import unicodedata
import emoji
from langdetect import detect
import logging
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from typing import Dict, List, Optional, Set, Tuple

from src.utils.profiling import profile_func, list_debug_add

# === Configuration =======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_LANGUAGES = ("french", "english", "german")

# Stopwords multilingues
STOPWORDS: Dict[str, Set[str]] = {
    lang: set(stopwords.words(lang)) 
    for lang in SUPPORTED_LANGUAGES
}
STOPWORDS_ALL = set.union(*STOPWORDS.values())

# -------------------------------------------------------------------
# Mots vagues à supprimer (faible pouvoir discriminant pour la catégorie)
# -------------------------------------------------------------------
mots_vagues = {
    "vie", "magic", "set", "produit", "produits", "article",
    "pièce", "pièces", "new", "die", "life","boite", "boîte", 
    "format", "modèle", "kit", "assortiment", "item", "tome", "import",
    "accessoire", "accessoires", "ensemble", "petit", "petite", "grand", "grande",
    "gros", "grosse", "mini", "maxi", "super", "ultra", "pcs", "pcs.", "pc",
    "piece", "pieces", "der", "dernier", "dernière", "nouveau", "nouvelle",
    "ancien", "ancienne", "original", "originale",
    "noir", "noire", "blanc", "blanche", "rouge", "bleu", "jaune", "vert", "rose",
    "orange", "gris", "grise", "marron", "violet", "violette", "turquoise", "argent",
    "doré", "or", "cuivre", "beige", "ivoire", "auucne", "aucune", "aucun", "aucuns",
    "aucunes", "aucunement", "und", "magideal", "allemand", "allemande", "deutsch",
    "deutsche", "german", "germane", "germans", "japonais", "japonaise", "japonaises",
    "français", "française", "francais", "francaises", "francophone", "francophones",
    "anglais", "anglaise", "english", "englishes", "complet", "complete", "completes",
    "jap", "japon", "sans", "intégré", "intégrée", "intégrés", "intégrées", "rare", 
    "commun", "communes",
    "neuf", "neuve", "neuves", "neufs", "occasion", "occasions", "occasionnel",
    "occasionnelle", "occasionnels", "occasionnelles", "occasionnellement",
    "générique", "génériques", "anti", "tout", "toute", "tous", "toutes", "design", 
    "home", "style", "mode", "fashion", "vol",
    "année", "années", "voir", "largeur", "longueur", "hauteur", "largeure", "microns",
    "comment","cet", "plus", "moins", "très", "peu", "peut", "facile",
    "facilement", "difficile", "difficilement", "simple", "simplement", "complexe",
    "complexes", "complexité", "complexité", "complexités", "léger", "légère", "légers", "différents",
    "différente", "différentes",
}

# === Classes ===========================================================
class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Nettoie et normalise le texte pour la classification.
    
    Paramètres:
        combine_cols: Tuple[str, str]  colonnes à fusionner (designation, description)
        remove_html: bool              supprimer les balises HTML
        translate_map_path: Optional[str]  JSON de traduction (token -> token)
        use_stem: bool                 activer stemming Snowball
        stem_langs: Tuple[str, ...]    langues ("french","english","german", …)
        clean_special: bool            nettoyer caractères spéciaux/ctrl
        handle_emojis: bool            convertir les emojis en :emoji:
        remove_numbers: bool           retirer les chiffres avant tokenisation
    """
    @profile_func
    def __init__(
        self,
        combine_cols: Tuple[str, str] = ("designation", "description"),
        remove_html: bool = True,
        translate_map_path: Optional[str] = None,
        use_stem: bool = True,
        stem_langs: Tuple[str, ...] = SUPPORTED_LANGUAGES,
        clean_special: bool = True,
        handle_emojis: bool = True,
        remove_numbers: bool = False,
    ):
        # Attributs utilisateur
        self.combine_cols = combine_cols
        self.remove_html = remove_html
        self.translate_map_path = translate_map_path
        self.use_stem = use_stem
        self.stem_langs = stem_langs
        self.clean_special = clean_special
        self.handle_emojis = handle_emojis
        self.remove_numbers = remove_numbers

        # Internes initialisés ici (évite AttributeError en fit/transform)
        self._stemmers = {
            lang: SnowballStemmer(lang) for lang in stem_langs
            if lang in SnowballStemmer.languages
        }
        self._translate_map: Dict[str, str] = {}

    # ------------------------- Utils de nettoyage -------------------------
    @profile_func
    def _clean_special_chars(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        if self.handle_emojis:
            text = emoji.demojize(text)
        if self.clean_special:
            text = unicodedata.normalize("NFKD", text)
            text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        return text

    def _strip_html(self, text: str) -> str:
        text = re.sub(r"<(br|hr|p)[^>]*/?>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&(nbsp|gt|lt|amp|quot|apos);", " ", text)
        return text

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------------ API sklearn ---------------------------
    @profile_func
    def fit(self, X, y=None):
        # (ré)initialiser le translate_map
        self._translate_map = {}
        if self.translate_map_path:
            try:
                with open(self.translate_map_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._translate_map = data
                elif isinstance(data, list):
                    self._translate_map = {
                        d["token"]: d["translation"]
                        for d in data
                        if isinstance(d, dict) and "token" in d and "translation" in d
                    }
            except Exception as e:
                logger.warning("Erreur chargement translate_map (%s): %s",
                               self.translate_map_path, e)

        logger.info(
            "TextCleaner prêt | translate_map=%d, stem=%s, langs=%s, "
            "clean_special=%s, emojis=%s, remove_numbers=%s",
            len(self._translate_map), self.use_stem, self.stem_langs,
            self.clean_special, self.handle_emojis, self.remove_numbers
        )
        return self

    @profile_func
    def clean_text(self, text: str) -> str:
        try:
            if pd.isnull(text):
                return ""
            # 1) spéciaux/emoji
            text = self._clean_special_chars(text)
            # 2) HTML
            if self.remove_html:
                text = self._strip_html(text)
            # 3) normalisation
            text = self._normalize(text)
            # 4) tokens
            tokens = text.split()
            # 5) traduction
            if self._translate_map:
                tokens = [self._translate_map.get(t, t) for t in tokens]
            # 6) stopwords + mots vagues
            tokens = [t for t in tokens if t not in STOPWORDS_ALL and t not in mots_vagues]
            # 7) stemming
            if self.use_stem and self._stemmers:
                stemmed = []
                for tok in tokens:
                    forms = [stemmer.stem(tok) for stemmer in self._stemmers.values()]
                    stemmed.append(min(forms, key=len))
                tokens = stemmed
            return " ".join(tokens) if tokens else "__empty__"
        except Exception as e:
            logger.error("Erreur nettoyage texte: %s", e)
            return "__error__"

    @profile_func
    def transform(self, X: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "_translate_map"):
            # sécurité si fit non appelé
            self.fit(X)
        # colonnes nécessaires
        missing = [c for c in self.combine_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        combined = (
            X[self.combine_cols[0]].fillna("").astype(str) + " " +
            X[self.combine_cols[1]].fillna("").astype(str)
        )
        return combined.apply(self.clean_text)

class HasDescriptionFlag(BaseEstimator, TransformerMixin):
    """Feature binaire indiquant la présence d'une description."""
    
    @profile_func
    def __init__(self, col_name: str = "description", 
                 out_name: str = "has_description"):
        self.col_name = col_name
        self.out_name = out_name

    @profile_func
    def fit(self, X, y=None):
        return self

    @profile_func
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.col_name not in X.columns:
            raise ValueError(f"Colonne '{self.col_name}' absente de X")
        series = X[self.col_name].notna().astype(int)
        return pd.DataFrame({self.out_name: series}, index=X.index)

class DesignationLength(BaseEstimator, TransformerMixin):
    """Feature numérique donnant la longueur de la désignation."""
    
    def __init__(self, col_name: str = "designation"):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.col_name not in X.columns:
            raise ValueError(f"Colonne '{self.col_name}' absente de X")
        lengths = X[self.col_name].fillna("").astype(str).str.len()
        return pd.DataFrame({"designation_length": lengths}, index=X.index)

class LanguageFeaturizer(BaseEstimator, TransformerMixin):
    """Features one-hot de détection de langue."""
    
    @profile_func
    def __init__(self, min_length: int = 5):
        self.min_length = min_length
        
    @profile_func
    def fit(self, X, y=None):
        return self
        
    @profile_func
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            languages = X.apply(
                lambda x: detect(x) if pd.notna(x) and len(str(x)) > self.min_length 
                else 'unknown'
            )
            return pd.get_dummies(languages, prefix='lang')
        except Exception as e:
            logger.warning(f"Erreur dans la détection de langue: {e}")
            return pd.DataFrame(index=X.index)
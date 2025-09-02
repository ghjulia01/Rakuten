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
    "lot", "vie", "magic", "set", "produit", "produits", "article",
    "pièce", "pièces", "new", "die", "life","boite", "boîte", "pack",
    "format", "modèle", "kit", "assortiment", "item", "tome", "import",
    "accessoire", "accessoires", "ensemble", "collection", "gamme", "série",
    "version", "volumes", "volume", "édition", "edition", "édition spéciale",
    "édition limitée", "série limitée", "petit", "petite", "grand", "grande",
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
    "jap", "japon", "sans", "intégré", "intégrée", "intégrés", "intégrées",
    "pvc", "plastique", "acier", "aluminium", "rare", "commun", "communes",
    "neuf", "neuve", "neuves", "neufs", "occasion", "occasions", "occasionnel",
    "occasionnelle", "occasionnels", "occasionnelles", "occasionnellement",
    "générique", "génériques", "anti", "tout", "toute", "tous", "toutes",
    "stream", "design", "home", "style", "mode", "fashion", "vol",
    "année", "années", "voir", "largeur", "longueur", "hauteur", "largeure", "microns",
    "comment", "extension", "extensions", "cet", "x", "plus", "moins", "très", "peu", "peut", "facile",
    "facilement", "difficile", "difficilement", "simple", "simplement", "complexe",
    "complexes", "complexité", "complexité", "complexités", "léger", "légère", "légers", "différents",
    "différente", "différentes",
}

# === Classes ===========================================================
class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Nettoie et normalise le texte pour la classification.
    
    Paramètres:
        combine_cols: Tuple[str, str]
            Colonnes à fusionner (designation, description)
        remove_html: bool
            Si True, supprime les balises HTML
        translate_map_path: Optional[str]
            Chemin vers le fichier JSON de traduction
        use_stem: bool
            Si True, applique le stemming
        stem_langs: Tuple[str, ...]
            Langues pour le stemming
        clean_special: bool
            Si True, nettoie les caractères spéciaux
        handle_emojis: bool
            Si True, convertit les emojis en texte
    """
    def __init__(
        self,
        combine_cols: Tuple[str, str] = ("designation", "description"),
        remove_html: bool = True,
        translate_map_path: Optional[str] = None,
        use_stem: bool = True,
        stem_langs: Tuple[str, ...] = SUPPORTED_LANGUAGES,
        clean_special: bool = True,
        handle_emojis: bool = True
    ):
        # Validation des langues supportées
        if not all(lang in SnowballStemmer.languages for lang in stem_langs):
            invalid = set(stem_langs) - set(SnowballStemmer.languages)
            raise ValueError(f"Langues non supportées: {invalid}")
        
        # Attribution des paramètres
        self.combine_cols = combine_cols
        self.remove_html = remove_html
        self.translate_map_path = translate_map_path
        self.use_stem = use_stem
        self.stem_langs = stem_langs
        self.clean_special = clean_special
        self.handle_emojis = handle_emojis
        
        # Initialisation des stemmers
        self._stemmers = {
            lang: SnowballStemmer(lang)
            for lang in stem_langs
            if lang in SnowballStemmer.languages
        }

    def fit(self, X, y=None):
        """
        Initialise les composants du nettoyage.
        
        Args:
            X: DataFrame d'entrée (non utilisé)
            y: Labels (non utilisé)
            
        Returns:
            self: Retourne l'instance pour le chainage
        """
        # Log des paramètres
        logger.info(f"Fitting TextCleaner with parameters: stem={self.use_stem}, "
                   f"clean_special={self.clean_special}, langs={self.stem_langs}")
        
        # Initialisation du dictionnaire de traduction
        self._translate_map = {}
        if self.translate_map_path:
            try:
                with open(self.translate_map_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._translate_map = data
                    elif isinstance(data, list):
                        self._translate_map = {
                            d["token"]: d["translation"]
                            for d in data
                            if isinstance(d, dict) 
                            and "token" in d 
                            and "translation" in d
                        }
                    logger.info(f"Dictionnaire de traduction chargé: {len(self._translate_map)} entrées")
            except Exception as e:
                logger.warning(f"Erreur chargement translate_map: {e}")
        
        return self

    def _clean_special_chars(self, text: str) -> str:
        """Nettoie les caractères spéciaux et emojis."""
        if not isinstance(text, str):
            return ""
            
        if self.handle_emojis:
            text = emoji.demojize(text)
            
        if self.clean_special:
            text = unicodedata.normalize('NFKD', text)
            text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        return text

    def _strip_html(self, text: str) -> str:
        """Supprime le HTML en préservant le contenu."""
        # Remplacer balises bloc par espace
        text = re.sub(r'<(br|hr|p)[^>]*/?>', ' ', text, flags=re.I)
        # Supprimer autres balises
        text = re.sub(r'<[^>]+>', '', text)
        # Nettoyer entités HTML
        text = re.sub(r'&(nbsp|gt|lt|amp|quot|apos);', ' ', text)
        return text

    def _normalize(self, text: str) -> str:
        """Normalisation basique du texte."""
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def clean_text(self, text: str) -> str:
        """Pipeline complet de nettoyage."""
        try:
            if pd.isnull(text):
                return ""

            # 1. Nettoyage caractères spéciaux
            text = self._clean_special_chars(text)

            # 2. Suppression HTML si activé
            if self.remove_html:
                text = self._strip_html(text)

            # 3. Normalisation basique
            text = self._normalize(text)

            # 4. Tokenization
            tokens = text.split()

            # 5. Traduction si dictionnaire chargé
            if self._translate_map:
                tokens = [self._translate_map.get(t, t) for t in tokens]

            # 6. Filtrage stopwords et mots vagues
            tokens = [
                t for t in tokens 
                if t not in STOPWORDS_ALL and t not in mots_vagues
            ]

            # 7. Stemming si activé
            if self.use_stem and self._stemmers:
                stemmed = []
                for token in tokens:
                    forms = [stemmer.stem(token) 
                            for stemmer in self._stemmers.values()]
                    stemmed.append(min(forms, key=len))
                tokens = stemmed

            return " ".join(tokens) if tokens else "__empty__"

        except Exception as e:
            logger.error(f"Erreur nettoyage texte: {e}")
            return "__error__"

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Combine et nettoie les colonnes textuelles."""
        if not hasattr(self, '_translate_map'):
            logger.warning("TextCleaner not fitted. Call fit first.")
            self.fit(X)
            
        try:
            # Vérifier colonnes requises
            missing = [col for col in self.combine_cols if col not in X.columns]
            if missing:
                raise ValueError(f"Colonnes manquantes: {missing}")

            # Combiner les colonnes
            combined = (
                X[self.combine_cols[0]].fillna("") + 
                " " + 
                X[self.combine_cols[1]].fillna("")
            )

            # Nettoyage avec gestion erreurs
            return combined.apply(self.clean_text)

        except Exception as e:
            logger.error(f"Erreur transformation: {e}")
            raise

class HasDescriptionFlag(BaseEstimator, TransformerMixin):
    """Feature binaire indiquant la présence d'une description."""
    
    def __init__(self, col_name: str = "description", 
                 out_name: str = "has_description"):
        self.col_name = col_name
        self.out_name = out_name

    def fit(self, X, y=None):
        return self

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
    
    def __init__(self, min_length: int = 5):
        self.min_length = min_length
        
    def fit(self, X, y=None):
        return self
        
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
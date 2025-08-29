# ---------------------------------------------------------------------------------
# Pre-processing textuel et feature engineering (version "stemming only")
# ---------------------------------------------------------------------------------
# Ce module fournit :
# 1) TextCleaner : nettoie et fusionne (designation + description) en un texte unique
#    - Suppression HTML
#    - Normalisation (minuscule, ponctuation, espaces)
#    - Stopwords FR/EN/DE + mots vagues spécifiques au retail
#    - Option de traduction simple via dictionnaire (ex. "black" -> "noir", "schwarz" -> "noir")
#    - **Stemming uniquement** (pas de lemmatisation) via Snowball (FR/EN/DE)
# 2) HasDescriptionFlag : ajoute une feature binaire 0/1 indiquant la présence d'une description
# ---------------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
import pandas as pd

# NLTK stopwords
from nltk.corpus import stopwords

# Stemming (rapide, dispo FR/EN/DE)
from nltk.stem.snowball import SnowballStemmer

# -------------------------------------------------------------------
# Stopwords multilingues
# -------------------------------------------------------------------
stop_fr = set(stopwords.words('french'))
stop_en = set(stopwords.words('english'))
stop_de = set(stopwords.words('german'))
stop_all = stop_fr.union(stop_en).union(stop_de)

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

# -------------------------------------------------------------------
# Transformateur : Texte nettoyé (fusion designation + description)
# -------------------------------------------------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self,
                 combine_cols=("designation", "description"),
                 remove_html: bool = True,
                 translate_map: dict | None = None,
                 use_stem: bool = True,
                 stem_langs: tuple = ("french", "english", "german")):
        """
        Parameters
        ----------
        combine_cols : tuple
            Colonnes à fusionner (dans l'ordre).
        remove_html : bool
            Supprime les balises HTML (<...>) avant tout traitement.
        translate_map : dict | None
            Mapping facultatif mot->traduction (ex: {'black': 'noir', 'schwarz': 'noir'}).
            NB : pas de détection automatique de langue ; les clés doivent être en minuscules.
        use_stem : bool
            Si True, applique un stemming (Snowball) pour normaliser les formes.
        stem_langs : tuple
            Langues à essayer pour le stemming (ordre d'application).
        """
        self.combine_cols = combine_cols
        self.remove_html = remove_html
        self.translate_map = translate_map or {}
        self.use_stem = use_stem
        self.stem_langs = stem_langs

        # Prépare les stemmers Snowball disponibles
        self._stemmers = {}
        for lang in stem_langs:
            try:
                self._stemmers[lang] = SnowballStemmer(lang)
            except Exception:
                pass  # ignore langues non supportées

    def _strip_html(self, text: str) -> str:
        # Supprimer toutes les balises <...>
        return re.sub(r"<[^>]*>", " ", text)

    def _normalize(self, text: str) -> str:
        # Minuscule, suppression ponctuation, espaces multiples
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _translate_tokens(self, tokens):
        if not self.translate_map:
            return tokens
        return [self.translate_map.get(t, t) for t in tokens]

    def _remove_stop_and_vague(self, tokens):
        return [t for t in tokens if t not in stop_all and t not in mots_vagues]

    def _stem_token(self, token: str) -> str:
        # Applique en cascade les stemmers disponibles et garde la forme la plus courte
        forms = [token]
        for _, stemmer in self._stemmers.items():
            try:
                forms.append(stemmer.stem(token))
            except Exception:
                continue
        return min(forms, key=len)

    def clean_text(self, text: str) -> str:
        if pd.isnull(text):
            return ""

        if self.remove_html:
            text = self._strip_html(text)

        text = self._normalize(text)
        tokens = text.split()

        # Traduction simple avant filtrage stopwords
        tokens = self._translate_tokens(tokens)

        # Stopwords + mots vagues
        tokens = self._remove_stop_and_vague(tokens)

        # Stemming (si activé)
        if self.use_stem and self._stemmers:
            tokens = [self._stem_token(t) for t in tokens]

        # Si aucun token après nettoyage, jeton de secours si tout est vide
        if not tokens:
            return "__empty__"

        return " ".join(tokens)

    # Fit vide pour compat sklearn
    def fit(self, X, y=None):
        return self

    # Transform : fusionne les colonnes et nettoie
    def transform(self, X: pd.DataFrame):
        assert all(col in X.columns for col in self.combine_cols), \
            f"Colonnes attendues manquantes: {self.combine_cols}"
        X_combined = X[self.combine_cols[0]].fillna("") + " " + X[self.combine_cols[1]].fillna("")
        return X_combined.apply(self.clean_text)


# -------------------------------------------------------------------
# Transformateur : Colonne binaire has_description (0/1)
# -------------------------------------------------------------------
class HasDescriptionFlag(BaseEstimator, TransformerMixin):
    """
    Produit une DataFrame (n,1) nommée 'has_description' valant 1 si la description n'est pas NaN,
    0 sinon. Peut être ajoutée via FeatureUnion à côté du TF-IDF.
    """
    def __init__(self, col_name: str = "description", out_name: str = "has_description"):
        self.col_name = col_name
        self.out_name = out_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if self.col_name not in X.columns:
            raise ValueError(f"Colonne '{self.col_name}' absente de X")
        series = X[self.col_name].notna().astype(int)
        return pd.DataFrame({self.out_name: series}, index=X.index)
class DesignationLength(BaseEstimator, TransformerMixin):
    """
    Transformateur qui calcule la longueur de la désignation en caractères.
    Retourne un DataFrame (n, 1) avec le nom 'designation_length'.
    """
    def __init__(self, col_name: str = "designation"):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if self.col_name not in X.columns:
            raise ValueError(f"Colonne '{self.col_name}' absente de X")
        lengths = X[self.col_name].fillna("").astype(str).str.len()
        return pd.DataFrame({"designation_length": lengths}, index=X.index)

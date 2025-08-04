# Pre-processing textuelle et feature engineering
# BaseEstimator et TransformerMixin sont des classes de scikit-learn.
# BaseEstimator permet d’hériter du comportement standard des objets sklearn 
# (représentation, compatibilité GridSearchCV).
# TransformerMixin permet de définir les méthodes fit et transform.

# Pre-processing textuelle: Nettoyage de la base de données

# Nettoyage des désignations pour l'analyse textuelle

from sklearn.base import BaseEstimator, TransformerMixin
import re, string, pandas as pd
from nltk.corpus import stopwords


# Stopwords multilingues
stop_fr = set(stopwords.words('french'))
stop_en = set(stopwords.words('english'))
stop_de = set(stopwords.words('german'))
stop_all = stop_fr.union(stop_en).union(stop_de)

# Mots vagues à supprimer (déjà fournis pendant l'EDA se référé plus haut)
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
    "comment", "extension", "extensions"
}


# Standardisation des textes manquants comme des chaînes vides
# afin de ne pas avoir de NaN dans les colonnes de texte
# qui peuvent bloquer le rest du pipeline comme avec text.lower() ou du re.sub

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, combine_cols=("designation", "description")):
        self.combine_cols = combine_cols

    def clean_text(self, text):
        if pd.isnull(text):
            return ""
         # Minuscule
        text = text.lower()
        # Supprimer ponctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        # Supprimer les espaces multiples
        text = re.sub(r"\s+", " ", text).strip()
        # Tokenisation simple par split
        tokens = text.split()
        # Suppression des stopwords + mots vagues
        tokens = [t for t in tokens if t not in stop_all and t not in mots_vagues]
        return " ".join(tokens)

    # Fit est nécessaire même vide, pour permettre l’intégration dans un pipeline
    # car les pipelines sklearn attendent un fit et un transform
    def fit(self, X, y=None):
        return self

    # Transformation pour combiner les colonnes et nettoyer le texte
    # Vérification que les colonnes demandées (designation, description)
    # existent bien dans X
    # Combinaison des colonnes designation et description
    # en une seule colonne de texte nettoyé
    # en remplaçant les NaN par "" et en les séparant par un espace
    # Cela crée un texte unique par produit, combinant les deux sources.
    # Application de la fonction clean_text à chaque ligne de ce texte fusionné.
    def transform(self, X):
        assert all(col in X.columns for col in self.combine_cols)
        X_combined = X[self.combine_cols[0]].fillna("") + " " + X[self.combine_cols[1]].fillna("")
        return X_combined.apply(self.clean_text)
    # Retourne une série de texte nettoyé

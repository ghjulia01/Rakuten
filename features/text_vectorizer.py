# **Pre-processing et feature engineering - TextTfidfVectorizer**

# Dans cette étape, text_vectorizer.py contient la classe TextTfidfVectorizer, 
# un wrapper autour de TfidfVectorizer.
# Il sert à apprendre un vocabulaire sur les textes nettoyés et transformer 
# ces textes en vecteurs numériques utilisables par un modèle.

#Le TfidfVectorizer est utilisé pour transformer le texte en vecteurs de 
# caractéristiques numérique, en tenant compte de la fréquence 
# des mots dans le corpus.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class TextTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)

# Apprentissage du vocabulaire et des poids TF-IDF
    # Fit est nécessaire même vide, pour permettre l’intégration dans un pipeline
    # X est une série de textes propres (ex. désignation + description nettoyées).
    # Elle ne retourne pas de données, mais elle met en mémoire le vocabulaire.
    # On retourne self pour permettre le chaînage (pipeline.fit().transform()).
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self
# Utilise le vocabulaire appris pour transformer chaque texte en vecteur de scores TF-IDF.
# Résultat : une matrice sparse de dimensions (n_texts, n_features)
# Cette méthode ne modifie pas le vocabulaire : elle applique celui appris au fit.
    def transform(self, X):
        return self.vectorizer.transform(X)
    
# Comme TextTfidfVectorizer ne peut pas gérer plusieurs colonnes,
# on crée un transformateur qui combine les colonnes de texte.
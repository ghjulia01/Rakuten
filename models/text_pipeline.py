# Pipeline textuelle pour nettoyer et vectoriser les textes 

# Text_pipeline définit un pipeline de transformation complet 
# (nettoyage + vectorisation) et sera utilisé pour entraîner 
# le modèle ou être combiné à d'autres pipelines (image, cascade).

# Pipeline est un objet sklearn qui permet d’enchaîner plusieurs 
# transformations dans l’ordre. 
# 


# TextCleaner est la classe créée pour nettoyer et combiner 
# les colonnes designation + description.


#TextTfidfVectorizer : encapsule TfidfVectorizer, 
# transforme les textes en vecteurs numériques.

# Pipeline textuelle pour nettoyer et vectoriser les textes 

import sys
import os
sys.path.append(os.path.abspath(".."))

from sklearn.pipeline import Pipeline
from features.text_cleaner import TextCleaner
from features.text_vectorizer import TextTfidfVectorizer

def create_text_pipeline():
    #  Retourne un pipeline prêt à l’emploi
    return Pipeline([
        ("cleaner", TextCleaner(combine_cols=("designation", "description"))),
        # L'étape 1, Combine les deux colonnes texte, nettoie : 
        # minuscules, stopwords, ponctuation, mots vagues
        # et produit une série pd.Series de texte propre (une ligne par produit)
        ("vectorizer", TextTfidfVectorizer(max_features=5000))
        # L'étape 2, utilise TfidfVectorizer pour ppliquer la vectorisation TF-IDF 
        # sur les textes nettoyés et produit une matrice sparse de dimensions 
        # (n_samples, n_features)
    ])
    
# -----------------------------------------------------------------------------
# Vectorisation TF-IDF pour textes déjà nettoyés par TextCleaner.
# - Pas de prétraitement lourd ici : le nettoyage est fait en amont.
# - Paramètres utiles exposés (ngrammes, min_df, max_df, strip accents, etc.)
# - dtype=float32 pour alléger la mémoire.
# -----------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class TextTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 max_features: int = 5000,
                 ngram_range=(1, 2),
                 min_df=2,
                 max_df=0.95,
                 sublinear_tf=True,
                 norm="l2",
                 strip_accents="unicode",
                 lowercase=False,
                 token_pattern=r"(?u)\b\w+\b",
                 dtype="float32"):
        """
        Paramètres principaux
        ---------------------
        max_features : nb max de features conservées (par fréquence TF-IDF).
        ngram_range : (1,2) => unigrams + bigrams (bien pour le e-commerce).
        min_df : ignorer les termes très rares (ex. 2 documents mini).
        max_df : ignorer les termes trop fréquents (ex. présents dans >95% des docs).
        sublinear_tf : log-scaling des TF pour limiter l’influence des répétitions.
        norm : normalisation des vecteurs ('l2' par défaut).
        strip_accents : 'unicode' => supprime les accents pour harmoniser.
        lowercase : False car TextCleaner passe déjà en minuscules.
        token_pattern : pattern simple (mots alphanumériques) – le texte est déjà propre.
        dtype : float32 pour réduire la mémoire sans perte notable.
        """
        self.params = dict(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
            strip_accents=strip_accents,
            lowercase=lowercase,
            token_pattern=token_pattern,
            dtype=dtype
        )
        self.vectorizer = TfidfVectorizer(**self.params)

    def fit(self, X, y=None):
        # X : série de textes déjà nettoyés (TextCleaner)
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        # Retourne une matrice sparse (n_docs, n_features)
        return self.vectorizer.transform(X)

    # Utiles si tu veux sérialiser / recharger proprement
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()

    def get_params(self, deep=True):
        return {"max_features": self.params["max_features"], **self.params}

    def set_params(self, **kwargs):
        # Permet d’ajuster les hyperparams via GridSearchCV si besoin
        self.params.update(kwargs)
        self.vectorizer = TfidfVectorizer(**self.params)
        return self
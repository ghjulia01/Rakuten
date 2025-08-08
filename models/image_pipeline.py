
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix
from features.image_loader import ImageLoader

def create_image_pipeline(image_dir, image_size=(64, 64), memory=None):
    """
    Attend un DataFrame avec une colonne 'productid'.
    - sélectionne productid
    - charge + redimensionne + normalise (ImageLoader)
    - aplatit (n,h,w,3) -> (n, h*w*3)*
    - convertit en sparse pour s’aligner avec la branche TF-IDF
    """
    # 1) Sélection de la colonne productid -> array[str]
    select_pid = FunctionTransformer(
        lambda df: df["productid"].astype(str).values,
        validate=False
    )

    # 2) Aplatissement en 2D si nécessaire
    def _flatten(X):
        X = np.asarray(X)
        return X.reshape((X.shape[0], -1)) if X.ndim == 4 else X
    flatten = FunctionTransformer(_flatten, validate=False)

    # 3) Conversion en sparse CSR (pour matcher TF-IDF)
    """ Cela renvoie une matrice sparse veut dire que l’objet 
    qu’on retourne n’est pas un gros tableau NumPy classique 
    rempli de zéros, mais une structure optimisée qui ne stocke 
    que les valeurs non nulles et leurs positions.
    """
    to_sparse = FunctionTransformer(lambda X: csr_matrix(X), 
                                    accept_sparse=True, 
                                    validate=False)

    return Pipeline(steps=[
        ("select_pid", select_pid),
        ("loader", ImageLoader(image_dir=image_dir, image_size=image_size)),
        ("flatten", flatten),
        ("to_sparse", to_sparse),
    ], memory=memory)

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from features.image_loader import ImageLoader

def create_image_pipeline(image_dir, image_size=(64, 64)):
    """
    Attend un DataFrame avec une colonne 'productid'.
    - sélectionne productid
    - charge + redimensionne + normalise (ImageLoader)
    - aplatit (n,h,w,3) -> (n, h*w*3)
    """
    select_pid = FunctionTransformer(lambda df: df["productid"].astype(str).values, 
                                     validate=False)

    def _flatten(X):
        X = np.asarray(X)
        return X.reshape((X.shape[0], -1)) if X.ndim == 4 else X

    flatten = FunctionTransformer(_flatten, validate=False)

    return Pipeline([
        ("select_pid", select_pid),
        ("loader", ImageLoader(image_dir=image_dir, image_size=image_size)),
        ("flatten", flatten),
    ])
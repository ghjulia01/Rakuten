
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix
from features.image_loader import ImageLoader
from sklearn.decomposition import TruncatedSVD, PCA

def create_image_pipeline(image_dir, image_size=(64, 64), dim_reduction=None, memory=None):
    """
    Attend un DataFrame avec une colonne 'productid'.
    - sélectionne productid
    - charge + redimensionne + normalise (ImageLoader)
    - aplatit (n,h,w,3) -> (n, h*w*3)
    - optionnel: réduction de dimension (SVD/PCA) configurable
    - pour SVD: conversion en sparse CSR (compatible avec TF-IDF)
    """
    # 1) Sélection de la colonne productid -> array[str]
    select_pid = FunctionTransformer(
        lambda df: df["productid"].astype(str).values, validate=False
    )

    # 2) Aplatissement en 2D si nécessaire
    def _flatten(X):
        X = np.asarray(X)
        return X.reshape((X.shape[0], -1)) if X.ndim == 4 else X
    flatten = FunctionTransformer(_flatten, validate=False)

    steps = [
        ("select_pid", select_pid),
        ("loader", ImageLoader(image_dir=image_dir, image_size=image_size)),
        ("flatten", flatten),
    ]

    # 3) Dimensionality reduction (optional)
    if dim_reduction and dim_reduction.get("enabled", False):
        method = (dim_reduction.get("method", "svd") or "svd").lower()
        n_comp = int(dim_reduction.get("n_components", 200))
        rs = dim_reduction.get("random_state", 42)

        if method == "svd":
            # SVD sur sparse (recommandé)
            steps += [
                ("to_sparse", FunctionTransformer(lambda X: csr_matrix(X), accept_sparse=True, validate=False)),
                ("svd", TruncatedSVD(n_components=n_comp, random_state=rs)),
            ]
        elif method == "pca":
            # PCA dense (mémoire ↑) : standardisation avec moyenne nécessaire
            steps += [
                ("scaler", StandardScaler(with_mean=True)),
                ("pca", PCA(n_components=n_comp, random_state=rs)),
            ]
        else:
            # fallback: juste sparse
            steps += [("to_sparse", FunctionTransformer(lambda X: csr_matrix(X), accept_sparse=True, validate=False))]
    else:
        # Pas de réduction: on reste en sparse comme avant
        steps += [("to_sparse", FunctionTransformer(lambda X: csr_matrix(X), accept_sparse=True, validate=False))]

    return Pipeline(steps=steps, memory=memory)
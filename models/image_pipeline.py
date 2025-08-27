
# models/image_pipeline.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import csr_matrix
from features.image_loader import ImageLoader

# ---------- helpers picklables  ----------
def _select_productid(df):
    # df est un DataFrame; on retourne un 1D array de productid (str)
    return df["productid"].astype(str).to_numpy()

def _flatten_images(X):
    X = np.asarray(X)
    # (n, H, W, C) -> (n, H*W*C)
    return X.reshape((X.shape[0], -1)) if X.ndim == 4 else X

def _to_sparse_csr(X):
    return csr_matrix(X)


def create_image_pipeline(image_dir, image_size=(64, 64), dim_reduction=None, memory=None):
    """
    Attend un DataFrame avec une colonne 'productid'.
    Étapes :
      - sélection 'productid'
      - chargement + redimensionnement + normalisation (ImageLoader)
      - flatten
      - réduction de dimension optionnelle (PCA dense ou SVD sparse)

    Remarque :
      - Pour PCA (dense), on ne convertit PAS en sparse.
      - Pour SVD, on convertit en CSR avant TruncatedSVD.
    """
    steps = [
        ("select_pid", FunctionTransformer(_select_productid, validate=False)),
        ("loader", ImageLoader(image_dir=image_dir, image_size=image_size)),
        ("flatten", FunctionTransformer(_flatten_images, validate=False)),
    ]

    if dim_reduction and dim_reduction.get("enabled", False):
        method = (dim_reduction.get("method", "pca") or "pca").lower()
        n_comp = int(dim_reduction.get("n_components", 200))
        rs = dim_reduction.get("random_state", 42)

        if method == "pca":
            steps += [
                ("scaler", StandardScaler(with_mean=True)),
                ("pca", PCA(n_components=n_comp, random_state=rs)),
            ]
        elif method == "svd":
            steps += [
                ("to_sparse", FunctionTransformer(_to_sparse_csr, accept_sparse=True, validate=False)),
                ("svd", TruncatedSVD(n_components=n_comp, random_state=rs)),
            ]
        else:
            # méthode inconnue -> pas de réduction
            pass

    # Si pas de réduction, on reste en DENSE (meilleur pour images + PCA ultérieure).
    return Pipeline(steps=steps, memory=memory)
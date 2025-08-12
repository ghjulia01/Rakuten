
import numpy as np
from scipy.sparse import csr_matrix, issparse
from models.image_pipeline import create_image_pipeline

def test_image_pipeline_sparse_output(img_dir, mini_df):
    pipe = create_image_pipeline(image_dir=img_dir, image_size=(64,64))
    Xt = pipe.fit_transform(mini_df)  # needs DataFrame with productid
    assert issparse(Xt)
    assert isinstance(Xt, csr_matrix)
    # n_samples x (H*W*3)
    assert Xt.shape[0] == 4
    assert Xt.shape[1] == 64*64*3
    # Finite
    arr = Xt.toarray()
    assert np.isfinite(arr).all()
    # Missing image row is zeros
    assert (arr[3] == 0.0).all()

def test_image_pipeline_order_invariance(img_dir, mini_df):
    pipe = create_image_pipeline(image_dir=img_dir, image_size=(32,32))
    Xt1 = pipe.fit_transform(mini_df)
    # Shuffle rows
    X2 = mini_df.sample(frac=1.0, random_state=42)
    Xt2 = pipe.transform(X2)
    # Shape depends only on n and size
    assert Xt1.shape[1] == Xt2.shape[1] == 32*32*3
    # Same product ids should map to same vectors regardless of order
    # We compare the set of norms as a proxy (exact order may differ)
    n1 = np.linalg.norm(Xt1.toarray(), axis=1)
    n2 = np.linalg.norm(Xt2.toarray(), axis=1)
    assert set(np.round(n1,6)) == set(np.round(n2,6))

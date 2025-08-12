
import numpy as np
import pytest
from features.image_loader import ImageLoader

def test_image_loader_shapes_and_norm(img_dir, mini_df):
    loader = ImageLoader(image_dir=img_dir, image_size=(64,64))
    # Fit builds the index from filenames
    loader.fit(mini_df["productid"].astype(str).values)

    arr = loader.transform(mini_df["productid"].astype(str).values)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 64, 64, 3)
    assert arr.dtype == np.float32

    # Values are normalized in [0,1]
    assert np.isfinite(arr).all()
    assert arr.min() >= 0.0 and arr.max() <= 1.0

    # Missing image (productid=4) should be zeros
    assert np.all(arr[3] == 0.0)

def test_image_loader_respects_resize(img_dir, mini_df):
    loader = ImageLoader(image_dir=img_dir, image_size=(32,48))  # HxW
    loader.fit(mini_df["productid"].astype(str).values)
    arr = loader.transform(mini_df["productid"].astype(str).values)
    assert arr.shape == (4, 32, 48, 3)

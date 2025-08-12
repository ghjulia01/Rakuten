import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



import os
import random
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PIL import Image


# Ensure NLTK stopwords are available for tests that rely on them
@pytest.fixture(scope="session", autouse=True)
def _ensure_nltk_stopwords():
    try:
        from nltk.corpus import stopwords  # noqa: F401
    except Exception:
        import nltk
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pytest.skip("NLTK stopwords unavailable; skipping text-cleaning tests.", allow_module_level=True)

# Determinism where possible
@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(42)
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"

@pytest.fixture
def mini_df_text():
    return pd.DataFrame({
        "designation": ["Robe noire coton", "Veste en cuir", "Câble USB type-C"],
        "description": ["Belle robe <i>noire</i>", None, "Câble rapide pour téléphone"],
        "productid": [1, 2, 3],
        "imageid": [100, 200, 300],
    }, index=[101, 102, 103])

@pytest.fixture
def img_dir(tmp_path):
    """Create a temporary directory with a few test images following Rakuten naming."""
    d = tmp_path / "imgs"
    d.mkdir()

    # Helper to create an RGB image (WxH) filled with a color
    def solid(name, size, color):
        im = Image.new("RGB", size, color)
        im.save(d / name, format="JPEG")

    # Helper to create an image with a mid-gray rectangle to trigger stats
    def with_rect(name, size, bg=(255,255,255), rect=(30,30,70,70), rect_color=(120,120,120)):
        im = Image.new("RGB", size, bg)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(im)
        draw.rectangle(rect, fill=rect_color)
        im.save(d / name, format="JPEG")

    # Create images with the Rakuten pattern: image_{imageid}_product_{productid}.jpg
    #  - One white image (should give occupancy=0)
    solid("image_100_product_1.jpg", (80, 60), (255,255,255))
    #  - One black image (should give occupancy=0 as well due to threshold)
    solid("image_200_product_2.jpg", (120, 90), (0,0,0))
    #  - One with a mid-gray rectangle detectable by stats
    with_rect("image_300_product_3.jpg", (100, 100), rect=(25,25,75,75), rect_color=(120,120,120))

    return str(d)

@pytest.fixture
def mini_df_image():
    """Small DataFrame with productid/imageid that match the created files + one missing."""
    import pandas as pd
    return pd.DataFrame({
        "productid": [1, 2, 3, 4],     # 4 is missing on purpose
        "imageid":   [100,200,300,400]
    }, index=[10,11,12,13])



import os
import random
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
def mini_df():
    return pd.DataFrame({
        "designation": ["Robe noire coton", "Veste en cuir", "Câble USB type-C"],
        "description": ["Belle robe <i>noire</i>", None, "Câble rapide pour téléphone"],
        "productid": [1, 2, 3],
        "imageid": [100, 200, 300],
    }, index=[101, 102, 103])

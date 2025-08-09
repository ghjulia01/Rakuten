
import numpy as np
import pandas as pd
from features.text_vectorizer import TextTfidfVectorizer

def test_text_tfidf_vectorizer_shapes_and_dtype():
    texts = pd.Series(["robe coton", "veste coton noir", "noir coton veste"], name="clean_text")
    vec = TextTfidfVectorizer(max_features=5, ngram_range=(1, 2), min_df=1, max_df=1.0, dtype="float32")
    X = vec.fit_transform(texts)
    assert X.shape[0] == 3
    assert X.dtype == np.float32
    assert len(vec.get_feature_names_out()) <= 5

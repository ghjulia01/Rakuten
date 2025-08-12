
import pandas as pd
from models.text_pipeline import create_text_pipeline

def test_text_pipeline_empty_inputs():
    pipe = create_text_pipeline(max_features=10)
    X = pd.DataFrame({
        "designation": [None, ""],
        "description": [None, ""],
        "productid": [1, 2],
        "imageid": [10, 20]
    })
    Xt = pipe.fit_transform(X)
    assert Xt.shape[0] == 2

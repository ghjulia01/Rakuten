
import joblib
import numpy as np
from models.text_pipeline import create_text_pipeline

def test_text_pipeline_serialization_roundtrip(tmp_path, mini_df):
    pipe = create_text_pipeline(max_features=50)
    Xt1 = pipe.fit_transform(mini_df)

    file = tmp_path / "text_pipe.joblib"
    joblib.dump(pipe, file)
    pipe2 = joblib.load(file)

    Xt2 = pipe2.transform(mini_df)
    assert np.allclose(Xt1.toarray(), Xt2.toarray(), atol=1e-6)

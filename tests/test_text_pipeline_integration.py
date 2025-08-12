
import numpy as np
from models.text_pipeline import create_text_pipeline

def test_text_pipeline_feature_union_train_test_consistency(mini_df):
    pipe = create_text_pipeline(max_features=20, translate_map_path=None, use_stem=True)

    X_train = mini_df
    X_test = mini_df.sample(frac=1.0, random_state=42)  # same schema

    Xt_train = pipe.fit_transform(X_train)
    Xt_test = pipe.transform(X_test)

    # Same number of columns between train and test (fixed vocab after fit)
    assert Xt_train.shape[1] == Xt_test.shape[1]
    # Same number of rows as input
    assert Xt_train.shape[0] == X_train.shape[0]
    # No NaN present
    assert not np.isnan(Xt_train.toarray()).any()

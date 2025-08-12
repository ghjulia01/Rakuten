
import numpy as np
import pandas as pd
from features.image_stats import ImageStatsFeaturizer

def test_image_stats_basic(img_dir, mini_df):
    stats = ImageStatsFeaturizer(
        image_dir=img_dir,
        imgid_col="imageid",
        pid_col="productid",
        white_threshold=230,
        black_threshold=25,
        min_area=16,
        out_prefix="feat_"
    )
    out = stats.fit_transform(mini_df)

    # Output frame with expected columns and index
    assert list(out.columns) == ["feat_width","feat_height","feat_occ"]
    assert list(out.index) == [10,11,12,13]
    assert (out.dtypes == np.float32).all()

    # For product 1 (white), occupancy should be ~0
    assert out.loc[10,"feat_occ"] == 0.0
    # For product 2 (black), occupancy ~0 also (below black_threshold)
    assert out.loc[11,"feat_occ"] == 0.0
    # For product 3 (rectangle), occupancy > 0 and width/height > 0
    assert out.loc[12,"feat_occ"] > 0.0
    assert out.loc[12,"feat_width"]  > 0.0
    assert out.loc[12,"feat_height"] > 0.0
    # Missing image (product 4) => zeros
    assert (out.loc[13] == 0.0).all()

def test_image_stats_switch_dir(tmp_path, img_dir, mini_df):
    # Move directory and ensure set_image_dir takes effect
    from shutil import copytree
    new_dir = tmp_path / "other"
    copytree(img_dir, new_dir)

    stats = ImageStatsFeaturizer(image_dir=img_dir, out_prefix="x_")
    _ = stats.fit(mini_df)
    stats.set_image_dir(str(new_dir))
    out2 = stats.transform(mini_df)
    assert list(out2.columns) == ["x_width","x_height","x_occ"]
    # Values should be identical after switching directory
    assert (out2.values >= 0).all()

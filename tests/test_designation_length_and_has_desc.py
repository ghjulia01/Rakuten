
import pandas as pd
import pytest

# Adjust the import path to your project layout if needed
from features.text_cleaner import DesignationLength, HasDescriptionFlag

def test_designation_length_basic():
    X = pd.DataFrame({"designation": ["abc", "défghi", None]}, index=[10, 11, 12])
    out = DesignationLength().fit_transform(X)
    assert list(out.columns) == ["designation_length"]
    assert list(out.index) == [10, 11, 12]
    assert out.loc[10, "designation_length"] == 3
    assert out.loc[11, "designation_length"] == 6
    assert out.loc[12, "designation_length"] == 0

def test_has_description_flag():
    X = pd.DataFrame({"description": ["ok", None, ""]}, index=[1, 2, 3])
    out = HasDescriptionFlag().fit_transform(X)
    assert list(out.columns) == ["has_description"]
    assert out.loc[1, "has_description"] == 1
    assert out.loc[2, "has_description"] == 0
    assert out.loc[3, "has_description"] == 1  # empty string is not NaN

def test_missing_columns_raise():
    with pytest.raises(ValueError):
        DesignationLength().transform(pd.DataFrame({"wrong": ["x"]}))
    with pytest.raises(ValueError):
        HasDescriptionFlag().transform(pd.DataFrame({"wrong": ["x"]}))

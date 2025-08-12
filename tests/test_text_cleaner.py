
import pandas as pd
from features.text_cleaner import TextCleaner

def test_text_cleaner_end_to_end():
    tc = TextCleaner(
        translate_map={"black": "noir"},
        use_stem=True
    )
    X = pd.DataFrame({
        "designation": ["<b>Robe</b> black", None],
        "description": ["nouveau modèle", "Un texte EN with and the"]
    })
    out = tc.fit_transform(X)  # Series of cleaned strings
    assert isinstance(out.iloc[0], str)
    assert "<b>" not in out.iloc[0]               # HTML removed
    assert "black" not in out.iloc[0]             # translated if present
    assert "nouveau" not in out.iloc[0]           # vague words removed
    assert out.iloc[1] != ""                      # not everything removed

def test_text_cleaner_handles_nan():
    tc = TextCleaner()
    X = pd.DataFrame({"designation": [None], "description": [None]})
    out = tc.fit_transform(X)
    assert isinstance(out.iloc[0], str)
    assert out.iloc[0] in ("", "__empty__")

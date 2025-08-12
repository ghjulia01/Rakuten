
import json
from models.text_pipeline import _load_translate_map

def test_load_translate_map_file(tmp_path):
    p = tmp_path / "map.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"black": "noir"}, f)
    mp = _load_translate_map(str(p))
    assert mp.get("black") == "noir"

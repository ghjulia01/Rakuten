import types
import numpy as np
import pytest
import pandas as pd
# pytest -q --tb=short tests/test_pipelines.py


def _dummy_y(n=60):
    # classes 10 (major), 40, 50
    base = [10, 40, 50] * (n // 3)
    if len(base) < n:
        base += [10] * (n - len(base))
    return pd.Series(base[:n])

# --- import résilient du module ---
def _import_train_model():
    try:
        from main import train_model as tm
        return tm
    except Exception:
        import importlib
        tm = importlib.import_module("train_model")
        return tm


# --- featurizer CNN bidon pour éviter les I/O/torch ---
class _DummyCNNFeaturizer:
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Retourne un embedding constant (n, 2048)
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.zeros((n, 2048), dtype=np.float32)


def _minimal_cfg(model_name: str) -> dict:
    """Config minuscule mais suffisante pour créer les pipelines sans I/O lourdes."""
    return {
        "model": {"name": model_name},
        "random": {"seed": 42},
        "cache": {"enabled": False},
        "text": {
            "svd": {"enabled": True, "n_components": 8, "l2norm": True},
            # le détail des sous-pipelines texte n'est pas utilisé tant qu'on ne fit pas
        },
        "images": {
            "train_dir": "dummy_train_dir",   # non utilisé par le featurizer bidon
            "test_dir": "dummy_test_dir",
            "cnn": {
                "enabled": True,
                "arch": "resnet50",
                "batch_size": 8,
                "device": "cpu",
                "use_imagenet_norm": True,
                "dim_reduction": {"enabled": True, "n_components": 16, "random_state": 42},
            },
            "stats_combined": {"enabled": False},
        },
        "fusion": {
            # On prune les pixels pour avoir uniquement texte + cnn
            "weights": {"text": 1.0, "image_cnn": 1.0, "image_pixels": 0}
        },
        "sampling": {"major_class": 10, "major_cap": 100, "tail_min": 5},
    }


@pytest.fixture(autouse=True)
def monkeypatch_cnn_featurizer(monkeypatch):
    """Remplace CNNFeaturizer par une version bidon pour tous les tests."""
    tm = _import_train_model()
    monkeypatch.setattr(tm, "CNNFeaturizer", _DummyCNNFeaturizer, raising=True)


def _has_l2norm_in_cnn_branch_cmb(pipe) -> bool:
    """Inspecte la pipeline COMBINED (B4) : step 'features' -> FeatureUnion -> 'image_cnn'."""
    features = pipe.named_steps["features"]  # FeatureUnion
    trfs = dict(features.transformer_list)
    assert "image_cnn" in trfs, "Branche 'image_cnn' absente de la FeatureUnion (features)."
    cnn_branch = trfs["image_cnn"]          # sklearn.Pipeline
    return "l2norm" in getattr(cnn_branch, "named_steps", {})


def _has_l2norm_in_cnn_branch_b3(pipe) -> bool:
    """Inspecte la pipeline B3 : step 'img' -> FeatureUnion([('img', <pipeline>)]) -> pipeline CNN/pixels."""
    img_union = pipe.named_steps["img"]     # FeatureUnion
    trfs = dict(img_union.transformer_list)
    assert "img" in trfs, "Sous-branche 'img' absente de la FeatureUnion (img)."
    img_branch = trfs["img"]                # sklearn.Pipeline (CNN ou pixels)
    return "l2norm" in getattr(img_branch, "named_steps", {})


# -------- Tests ---------

@pytest.mark.parametrize("tree_model", ["xgb", "lgbm"])
def test_combined_cnn_l2_off_for_tree_models(tree_model):
    tm = _import_train_model()
    cfg = _minimal_cfg(tree_model)
    pipe = tm.create_combined_pipeline(cfg, under_strategy={}, over_strategy={}, seed=42)
    assert _has_l2norm_in_cnn_branch_cmb(pipe) is False, "L2 devrait être OFF pour les modèles arbres."


def test_combined_cnn_l2_on_for_linear_models():
    tm = _import_train_model()
    cfg = _minimal_cfg("lr")
    pipe = tm.create_combined_pipeline(cfg, under_strategy={}, over_strategy={}, seed=42)
    assert _has_l2norm_in_cnn_branch_cmb(pipe) is True, "L2 devrait être ON pour les modèles linéaires."


@pytest.mark.parametrize("tree_model", ["xgb", "lgbm"])
def test_b3_cnn_l2_off_for_tree_models(tree_model):
    """Ce test échouera si B3 n'appelle pas create_cnn_branch_from_cfg(..., apply_l2=not is_tree_model)."""
    tm = _import_train_model()
    cfg = _minimal_cfg(tree_model)
    pipe, need_cols = tm.build_baseline_pipeline("b3", cfg, seed=42)
    # On n'entraîne pas ; on inspecte juste la structure
    assert _has_l2norm_in_cnn_branch_b3(pipe) is False, "B3 devrait couper L2 pour les modèles arbres."


@pytest.mark.parametrize("kind, step_name", [("b2", "clf"), ("b3", "clf"), ("b4", "model")])
@pytest.mark.parametrize("tree_model", ["xgb", "lgbm"])
def test_label_wrapper_used_for_tree_models(kind, step_name, tree_model):
    tm = _import_train_model()
    cfg = _minimal_cfg(tree_model)
    if kind == "b4":
        y = _dummy_y(60)
        pipe, _ = tm.build_baseline_pipeline(kind, cfg, seed=42, y_train=y)
    else:
        pipe, _ = tm.build_baseline_pipeline(kind, cfg, seed=42)
    assert isinstance(pipe.named_steps[step_name], tm.LabelEncodingClassifier), \
        f"{kind}: le classifieur devrait être enveloppé par LabelEncodingClassifier pour {tree_model}."


@pytest.mark.parametrize("kind, step_name", [("b2", "clf"), ("b3", "clf"), ("b4", "model")])
def test_label_wrapper_not_used_for_lr(kind, step_name):
    tm = _import_train_model()
    cfg = _minimal_cfg("lr")
    if kind == "b4":
        y = _dummy_y(60)
        pipe, _ = tm.build_baseline_pipeline(kind, cfg, seed=42, y_train=y)
    else:
        pipe, _ = tm.build_baseline_pipeline(kind, cfg, seed=42)
    assert not isinstance(pipe.named_steps[step_name], tm.LabelEncodingClassifier), \
        f"{kind}: pas de wrapper attendu pour LR."
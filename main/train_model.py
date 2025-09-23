# -*- coding: utf-8 -*-
"""
Classification Rakuten — pipeline consolidée (b2/b3/b4)
- CV avec early stopping (dans l’espace de features)
- Entraînement final + re-fit full data au best_iter+1
- Sauvegardes .joblib b2/b3/b4
- Exports analytiques: ACP/SVD, SHAP, BLOCKs
- Re-pointage images vers test_dir
- Préparation HuggingFace pour ViT (cache local + fallback offline)

Dépendances : numpy, pandas, scikit-learn, imbalanced-learn, xgboost ou lightgbm (selon config),
matplotlib, shap (optionnel), transformers (pour ViT via CNNFeaturizer).
"""

import os, re, time, json, random, argparse, logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

# sklearn / imblearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline as SkPipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler

from joblib import Memory

# --- modules projet
import tomllib  # Python 3.11+
from features.image_loader import ImageLoader
from features.image_stats import ImageStatsCombinedFeaturizer
from models.text_pipeline import create_text_pipeline_from_cfg
from models.image_pipeline import create_image_pipeline_from_cfg
from models.cnn_features import CNNFeaturizer  # gère ResNet/ViT selon cfg


DEFAULT_CFG = Path(__file__).resolve().parents[1] / "features" / "config.toml"

# --- logging helper (compat tools/peek_features.py) ---
def setup_logging(level=logging.INFO,
                  fmt="%(asctime)s | %(levelname)s | %(message)s",
                  name="rakuten",
                  log_dir: str | None = None,
                  filename: str | None = None,
                  *, force: bool = False):
    """
    Initialise le logging console + (optionnel) fichier.
    - log_dir: dossier où écrire le .log (nom par défaut: <name>.log)
    - filename: chemin complet du fichier de log (prend le pas sur log_dir)
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename or log_dir:
        if not filename:
            os.makedirs(log_dir, exist_ok=True)
            filename = os.path.join(log_dir, f"{name}.log")
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        handlers.append(logging.FileHandler(filename, mode="a", encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=force)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

logger = setup_logging()


# -------------------- Config & Seed --------------------
def load_config(config_path: str | Path | None = None) -> dict:
    cfg_path = Path(config_path) if config_path else DEFAULT_CFG
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg

def init_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_cache(cfg: dict) -> Optional[Memory]:
    use_cache = bool(cfg.get("compute", {}).get("use_cache", False))
    if not use_cache:
        return None
    cache_dir = os.path.join(cfg["outputs"]["log_dir"], "skcache")
    os.makedirs(cache_dir, exist_ok=True)
    return Memory(cache_dir)



# -------------------- HuggingFace (ViT) --------------------
def setup_hf_env(cfg: dict) -> None:
    """Prépare un cache HF local et désactive la télémétrie (robuste en entreprise/proxy)."""
    base = cfg.get("outputs", {}).get("log_dir", "logs")
    hf_cache = os.path.join(base, "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

def ensure_vit_available(model_name: str = "google/vit-base-patch16-224", revision: str = "main") -> None:
    """Tente de télécharger le processor + modèle ViT (une fois), sinon offline."""
    try:
        import transformers as tfm
        cfg = tfm.AutoConfig.from_pretrained(model_name, revision=revision, local_files_only=False)
        if getattr(cfg, "model_type", None) == "vit":
            cfg.add_pooling_layer = False
        _ = tfm.AutoImageProcessor.from_pretrained(
            model_name, revision=revision, use_fast=True, local_files_only=False
        )
        _ = tfm.AutoModel.from_pretrained(model_name, revision=revision, config=cfg, local_files_only=False)
        logger.info("[HF] ViT '%s' disponible (réseau).", model_name)
    except Exception as e:
        logger.warning("[HF] Réseau indisponible (%s). Tentative offline…", e)
        import transformers as tfm
        cfg = tfm.AutoConfig.from_pretrained(model_name, revision=revision, local_files_only=True)
        if getattr(cfg, "model_type", None) == "vit":
            cfg.add_pooling_layer = False
        _ = tfm.AutoImageProcessor.from_pretrained(
            model_name, revision=revision, use_fast=True, local_files_only=True
        )
        _ = tfm.AutoModel.from_pretrained(model_name, revision=revision, config=cfg, local_files_only=True)
        logger.info("[HF] ViT '%s' trouvé en cache local.", model_name)

# -------------------- Sampling --------------------
def make_sampling_strategies(y_train: pd.Series,
                             major_class: int,
                             major_cap: int,
                             tail_min: int) -> Tuple[dict, dict]:
    vc = y_train.value_counts()
    under = {
        int(cls): (min(int(cnt), int(major_cap)) if int(cls) == int(major_class) else int(cnt))
        for cls, cnt in vc.items()
    }
    over = {int(cls): int(tail_min) for cls, cnt in vc.items() if int(cnt) < int(tail_min)}
    return under, over

class AdaptiveUnderSampler(BaseSampler):
    _parameter_constraints = {"cap_dict": [dict, None], "random_state": [None, int], "sampling_strategy": [str, dict, float, callable, None]}
    _sampling_type = "under-sampling"
    def __init__(self, cap_dict=None, random_state=None, sampling_strategy="auto"):
        self.cap_dict = cap_dict or {}
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
    def _fit_resample(self, X, y):
        y_arr = np.asarray(y).ravel()
        cnt = {k: int(v) for k, v in zip(*np.unique(y_arr, return_counts=True))}
        sampling_strategy = {cls: min(n, self.cap_dict.get(cls, n)) for cls, n in cnt.items()}
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
        return rus.fit_resample(X, y_arr)
    def _more_tags(self):
        return {"allow_nan": False, "X_types": ["2darray", "sparse"]}

# -------------------- Utils --------------------
class ToFloat32:
    def fit(self, X, y=None): return self
    def transform(self, X):
        try:
            return X.astype("float32") if sparse.issparse(X) else X.astype(np.float32)
        except Exception:
            return X
# -------------------- Summary helpers --------------------
def _append_summary_row(baseline: str, cv_splits: int, f1_macro: float, f1_weighted: float,
                        train_infer_time_sec: float, notes: str, out_csv: str = "baseline_results_summary.csv"):
    """Append one line to the summary CSV, create header if missing."""
    import os, csv
    header = ["baseline","cv_splits","f1_macro","f1_weighted","train_infer_time_sec","notes"]
    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        # notes may contain commas -> keep it as the last field; csv.writer will quote if needed
        w.writerow([baseline, cv_splits, f1_macro, f1_weighted, f"{train_infer_time_sec:.3f}", notes])

def _make_notes(pipe, cfg: dict, *, y_train=None, y_val=None, best_iter=None) -> str:
    """Faire une synthèse textuelle de la config + pipeline."""
    # text/image switches
    text_on  = bool(cfg.get("text", {}))
    image_on = bool(cfg.get("images", {}))
    dummy_on = False

    # CNN/ViT details
    img_cfg  = cfg.get("images", {}) or {}
    cnn_cfg  = img_cfg.get("cnn", {}) or {}
    vit_cfg  = img_cfg.get("cnn_vit", {}) or {}
    cnn_arch = cnn_cfg.get("arch", None)
    cnn_svd  = (cnn_cfg.get("dim_reduction", {}) or {}).get("enabled", None)
    cnn_svd_k= (cnn_cfg.get("dim_reduction", {}) or {}).get("n_components", None)
    vit_on   = bool(vit_cfg.get("enabled", False))

    # active branches + weights (si dispo)
    active_branches, weights = None, None
    try:
        feats = pipe.named_steps.get("features", None)
        if feats is not None:
            # FeatureUnion
            union = feats.named_steps["union"] if "union" in feats.named_steps else feats
            active_branches = [name for name, _ in getattr(union, "transformer_list", [])]
            weights = getattr(union, "transformer_weights", None)
    except Exception:
        pass

    # sizes
    n_train = (len(y_train) if y_train is not None else None)
    n_val   = (len(y_val) if y_val is not None else None)

    # classifier name + eval_metric from ctor if available
    clf = pipe.named_steps.get("model", None)
    clf_name = clf.__class__.__name__ if clf is not None else "NA"
    eval_metric_ctor = getattr(getattr(clf, "est_", clf), "eval_metric", None)

    parts = [
        f"text_only={text_on and not image_on}",
        f"image_only={image_on and not text_on}",
        f"dummy={dummy_on}",
        f"CNN arch={cnn_arch} svd={cnn_svd}/{cnn_svd_k}",
        f"ViT={'on' if vit_on else 'off'}",
        f"branches={active_branches}" if active_branches else None,
        f"weights={weights}" if weights else None,
        f"train={n_train} val={n_val}" if n_train is not None else None,
        f"clf={clf_name}",
        f"best_iter={best_iter if best_iter is not None else 'NA'}",
        f"eval_metric={eval_metric_ctor}",
    ]
    return " | ".join(p for p in parts if p)

# -------------------- LabelEncoding wrapper --------------------
class LabelEncodingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator
    def get_params(self, deep=True):
        params = {"base_estimator": self.base_estimator}
        if deep and hasattr(self.base_estimator, "get_params"):
            for k, v in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{k}"] = v
        return params
    def set_params(self, **params):
        if "base_estimator" in params:
            self.base_estimator = params.pop("base_estimator")
        base_params = {k.split("__",1)[1]: v for k, v in params.items() if k.startswith("base_estimator__")}
        if base_params and hasattr(self.base_estimator, "set_params"):
            self.base_estimator.set_params(**base_params)
        return self
    def fit(self, X, y, **fit_params):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        if "eval_set" in fit_params and isinstance(fit_params["eval_set"], list):
            new_eval = []
            for (Xv, yv) in fit_params["eval_set"]:
                new_eval.append((Xv, self.le_.transform(np.asarray(yv))))
            fit_params["eval_set"] = new_eval
        self.est_ = clone(self.base_estimator)
        try:
            K = len(self.le_.classes_)
            if K > 2 and hasattr(self.est_, "get_params"):
                params = self.est_.get_params()
                if "num_class" in params:
                    self.est_.set_params(num_class=K)
                if self.est_.__class__.__name__ == "XGBClassifier":
                    if params.get("objective", None) not in ("multi:softprob", "multi:softmax"):
                        self.est_.set_params(objective="multi:softprob")
        except Exception:
            pass
        self.est_.fit(X, y_enc, **fit_params)
        return self
    def predict(self, X):
        y_enc = self.est_.predict(X)
        return self.le_.inverse_transform(np.asarray(y_enc))
    def predict_proba(self, X):
        if hasattr(self.est_, "predict_proba"):
            return self.est_.predict_proba(X)
        raise AttributeError("Le modèle sous-jacent ne supporte pas predict_proba.")

# -------------------- Classifier builder --------------------
def build_classifier(cfg: dict, seed: int):
    model_cfg = cfg.get("model", {}) or {}
    name = str(model_cfg.get("name", "lr")).lower()
    n_jobs = int(cfg.get("compute", {}).get("n_jobs", -1))

    if name in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=int(model_cfg.get("xgb", {}).get("n_estimators", model_cfg.get("n_estimators", 2000))),
            learning_rate=float(model_cfg.get("xgb", {}).get("learning_rate", model_cfg.get("learning_rate", 0.05))),
            max_depth=int(model_cfg.get("xgb", {}).get("max_depth", model_cfg.get("max_depth", 8))),
            subsample=float(model_cfg.get("xgb", {}).get("subsample", model_cfg.get("subsample", 0.8))),
            colsample_bytree=float(model_cfg.get("xgb", {}).get("colsample_bytree", model_cfg.get("colsample_bytree", 0.8))),
            reg_alpha=float(model_cfg.get("xgb", {}).get("reg_alpha", model_cfg.get("reg_alpha", 0.0))),
            reg_lambda=float(model_cfg.get("xgb", {}).get("reg_lambda", model_cfg.get("reg_lambda", 1.0))),
            tree_method=str(model_cfg.get("xgb", {}).get("tree_method", model_cfg.get("tree_method", "hist"))),
            n_jobs=n_jobs, random_state=seed, objective="multi:softprob", eval_metric="mlogloss",
        )

    if name in ("lgbm", "lightgbm"):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=int(model_cfg.get("lgbm", {}).get("n_estimators", model_cfg.get("n_estimators", 1200))),
            learning_rate=float(model_cfg.get("lgbm", {}).get("learning_rate", model_cfg.get("learning_rate", 0.05))),
            num_leaves=int(model_cfg.get("lgbm", {}).get("num_leaves", model_cfg.get("num_leaves", 255))),
            max_depth=int(model_cfg.get("lgbm", {}).get("max_depth", model_cfg.get("max_depth", -1))),
            feature_fraction=float(model_cfg.get("lgbm", {}).get("feature_fraction", model_cfg.get("feature_fraction", 0.8))),
            bagging_fraction=float(model_cfg.get("lgbm", {}).get("bagging_fraction", model_cfg.get("bagging_fraction", 0.8))),
            bagging_freq=int(model_cfg.get("lgbm", {}).get("bagging_freq", model_cfg.get("bagging_freq", 1))),
            min_child_samples=int(model_cfg.get("lgbm", {}).get("min_child_samples", model_cfg.get("min_child_samples", 40))),
            reg_alpha=float(model_cfg.get("lgbm", {}).get("reg_alpha", model_cfg.get("reg_alpha", 0.0))),
            reg_lambda=float(model_cfg.get("lgbm", {}).get("reg_lambda", model_cfg.get("reg_lambda", 1.0))),
            n_jobs=n_jobs, random_state=seed, objective="multiclass",
        )

    if name == "svc":
        cw = "balanced" if model_cfg.get("use_class_weight", False) else None
        return LinearSVC(
            C=float(model_cfg.get("svc", {}).get("C", model_cfg.get("C", 1.0))),
            tol=float(model_cfg.get("svc", {}).get("tol", model_cfg.get("tol", 5e-4))),
            max_iter=int(model_cfg.get("svc", {}).get("max_iter", model_cfg.get("max_iter", 4000))),
            class_weight=cw, loss="squared_hinge", dual=False, penalty="l2",
        )

    # défaut : LR
    use_cw = bool(model_cfg.get("use_class_weight", False))
    class_weight = "balanced" if use_cw else None
    return LogisticRegression(
        solver=str(model_cfg.get("lr", {}).get("solver", model_cfg.get("solver", "saga"))),
        penalty=str(model_cfg.get("lr", {}).get("penalty", model_cfg.get("penalty", "l2"))),
        C=float(model_cfg.get("lr", {}).get("C", model_cfg.get("C", 1.0))),
        max_iter=int(model_cfg.get("lr", {}).get("max_iter", model_cfg.get("max_iter", 4000))),
        tol=float(model_cfg.get("lr", {}).get("tol", model_cfg.get("tol", 1e-3))),
        verbose=int(model_cfg.get("lr", {}).get("verbose", model_cfg.get("verbose", 0))),
        class_weight=class_weight,
        fit_intercept=True,
        n_jobs=int(cfg.get("compute", {}).get("n_jobs", -1)),
    )

# -------------------- CNN / ViT branches --------------------
def create_cnn_branch_from_cfg(images_cfg: dict, section: str = "cnn", apply_l2: bool = True) -> SkPipeline:
    cnn_cfg = images_cfg.get(section, {}) or {}
    if not bool(cnn_cfg.get("enabled", False)):
        raise ValueError(f"[images.{section}.enabled] = false")

    # --- lecture des augs & label smoothing depuis TOML ---
    aug_cfg = cnn_cfg.get("aug", {}) or {}
    rrc_cfg = aug_cfg.get("random_resized_crop", {}) or {}

    aug_hflip_p      = float(aug_cfg.get("hflip_p", aug_cfg.get("aug_hflip_p", 0.2)))
    aug_color_jitter = float(aug_cfg.get("color_jitter", 0.05))
    mixup_alpha      = float(aug_cfg.get("mixup_alpha", 0.1))
    cutmix_alpha     = float(aug_cfg.get("cutmix_alpha", 0.0))
    rrc_scale        = tuple(rrc_cfg.get("scale", (0.9, 1.0)))
    rrc_ratio        = tuple(rrc_cfg.get("ratio", (0.95, 1.05)))

    label_smoothing  = float(cnn_cfg.get("label_smoothing", 0.0))

    # >>> LOG DE CONTROLE
    logger.info(
        "[AUG] %s | hflip_p=%.2f | color_jitter=%.2f | mixup=%.2f | cutmix=%.2f | "
        "rrc_scale=%s | rrc_ratio=%s | label_smoothing=%.3f",
        section, aug_hflip_p, aug_color_jitter, mixup_alpha, cutmix_alpha,
        rrc_scale, rrc_ratio, label_smoothing
    )

    # garde-fou: avertir si hors plages usuelles
    if not (0.0 <= aug_hflip_p <= 1.0):
        logger.warning("[AUG] hflip_p hors plage: %.3f", aug_hflip_p)
    if not (0.0 <= mixup_alpha <= 1.0):
        logger.warning("[AUG] mixup_alpha suspect: %.3f", mixup_alpha)
    if not (0.0 <= cutmix_alpha <= 1.0):
        logger.warning("[AUG] cutmix_alpha suspect: %.3f", cutmix_alpha)
    if not (0.0 <= label_smoothing < 0.2):
        logger.warning("[AUG] label_smoothing inhabituel: %.3f", label_smoothing)

    featurizer = CNNFeaturizer(
        image_dir=images_cfg["train_dir"],
        arch=str(cnn_cfg.get("arch", "resnet50")),
        batch_size=int(cnn_cfg.get("batch_size", 12)),
        device=str(cnn_cfg.get("device", "auto")),
        use_imagenet_norm=bool(cnn_cfg.get("use_imagenet_norm", True)),
        trainable_last_layers=int(cnn_cfg.get("trainable_last_layers", 1)),
        fallback_zero=bool(cnn_cfg.get("fallback_zero", True)),
        dtype=str(cnn_cfg.get("dtype", "float32")),
        num_workers=int(cnn_cfg.get("num_workers", 0)),
        finetune_epochs=int(cnn_cfg.get("finetune_epochs", 0)),
        finetune_lr=float(cnn_cfg.get("finetune_lr", 3e-4)),
        finetune_weight_decay=float(cnn_cfg.get("finetune_weight_decay", 0.01)),
        finetune_max_n=int(cnn_cfg.get("finetune_max_n", 8000)),
        # ViT/HF
        hf_model_name=cnn_cfg.get("hf_model_name", None),
        hf_revision=cnn_cfg.get("hf_revision", "main"),
        hf_feature_dim=cnn_cfg.get("hf_feature_dim", 768),
        hf_use_fast=bool(cnn_cfg.get("use_fast", True)),
        ft_patience=int(cnn_cfg.get("ft_patience", 3)),
        aug_hflip_p=aug_hflip_p,
        aug_color_jitter=aug_color_jitter,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        random_resized_crop_scale=rrc_scale,
        random_resized_crop_ratio=rrc_ratio,
        label_smoothing=label_smoothing,
    )
    steps = [("cnn", featurizer)]
    dr = cnn_cfg.get("dim_reduction", {}) or {}
    if bool(dr.get("enabled", False)):
        steps += [("to32_pre", ToFloat32()),
                  ("svd", TruncatedSVD(n_components=int(dr.get("n_components", 256)),
                                       random_state=int(dr.get("random_state", 42))))]
        if apply_l2:
            steps.append(("l2norm", Normalizer(copy=False)))
        steps.append(("to32_post", ToFloat32()))
    else:
        if apply_l2:
            steps.append(("l2norm", Normalizer(copy=False)))
    return SkPipeline(steps)

# -------------------- Combined pipeline (b4) --------------------
def create_combined_pipeline(cfg: dict, under_strategy: dict, over_strategy: dict, seed: int) -> ImbPipeline:
    text_branch = create_text_pipeline_from_cfg(cfg.get("text", {}))

    model_name = str(cfg.get("model", {}).get("name", "lr")).lower()
    is_tree_model = model_name in ("xgb", "xgboost", "lgbm", "lightgbm")

    svd_cfg = (cfg.get("text", {}).get("svd", {}) or {})
    if bool(svd_cfg.get("enabled", True)):
        text_branch = SkPipeline([
            ("text", text_branch),
            ("to32_pre", ToFloat32()),
            ("svd", TruncatedSVD(n_components=int(svd_cfg.get("n_components", 700)),
                                 random_state=int(svd_cfg.get("random_state", seed)))),
            *([("l2", Normalizer(copy=False))] if (svd_cfg.get("l2norm", True) and not is_tree_model) else []),
            ("to32_post", ToFloat32()),
        ])

    transformers = [("text", text_branch)]
    images_cfg = cfg.get("images", {}) or {}
    fusion_w = (cfg.get("fusion", {}) or {}).get("weights", {}) or {}

    want_pixels = not (fusion_w.get("image_pixels", None) == 0)
    if want_pixels:
        img_pixels = create_image_pipeline_from_cfg(images_cfg, use_test_dir=False)
        transformers.append(("image_pixels", img_pixels))

    stats_c = images_cfg.get("stats_combined", {}) or {}
    if bool(stats_c.get("enabled", False)) and not (fusion_w.get("image_stats_combined", None) == 0):
        transformers.append(("image_stats_combined", ImageStatsCombinedFeaturizer(
            image_dir=images_cfg["train_dir"],
            imgid_col="imageid", pid_col="productid",
            white_threshold=int(stats_c.get("white_threshold", 230)),
            black_threshold=int(stats_c.get("black_threshold", 25)),
            min_area=int(stats_c.get("min_area", 16)),
            prefix_basic=str(stats_c.get("prefix_basic", "img_")),
            prefix_pro=str(stats_c.get("prefix_pro", "pro_")),
        )))

    # CNN (ResNet)
    if bool(images_cfg.get("cnn", {}).get("enabled", False)) and not (fusion_w.get("image_cnn", None) == 0):
        transformers.append(("image_cnn", create_cnn_branch_from_cfg(images_cfg, "cnn", apply_l2=not is_tree_model)))

    # ViT (HF) — préparer le cache et s’assurer que le modèle est dispo
    if bool(images_cfg.get("cnn_vit", {}).get("enabled", False)) and not (fusion_w.get("image_cnn_vit", None) == 0):
        setup_hf_env(cfg)
        try:
            ensure_vit_available(images_cfg["cnn_vit"].get("hf_model_name", "google/vit-base-patch16-224"),
                                 images_cfg["cnn_vit"].get("hf_revision", "main"))
        except Exception as e:
            logger.warning("[ViT] Préchargement échoué (%s). Le featurizer tentera quand même le chargement.", e)
        transformers.append(("image_cnn_vit", create_cnn_branch_from_cfg(images_cfg, "cnn_vit", apply_l2=not is_tree_model)))

    # Ne garder que les poids des transformeurs effectivement présents
    present = [name for name, _ in transformers]
    tw = {}
    for k, v in (fusion_w or {}).items():
        if k in present:
            tw[k] = float(v)
        else:
            # log informatif si un poids est fourni pour une branche absente
            logger.warning("[fusion] Poids fourni pour '%s' mais branche absente → ignoré.", k)

    union = FeatureUnion(transformer_list=transformers, transformer_weights=(tw or None))
    features = SkPipeline([("union", union)])

    under = AdaptiveUnderSampler(cap_dict=under_strategy, random_state=seed)
    over  = RandomOverSampler(sampling_strategy=over_strategy, random_state=seed)

    base_clf = build_classifier(cfg, seed)
    if is_tree_model:
        clf = LabelEncodingClassifier(base_clf)
        scaler_step = []  # arbres : pas de scaler
    else:
        clf = base_clf
        scaler_step = [("scaler", StandardScaler(with_mean=False))]

    cache = get_cache(cfg)
    steps = [("features", features), ("under", under), ("over", over), *scaler_step, ("model", clf)]
    pipe = ImbPipeline(steps, memory=cache)
    return pipe

# -------------------- Baselines builder --------------------
def build_baseline_pipeline(kind: str, cfg: dict, seed: int, y_train: Optional[pd.Series]=None):
    if kind == "b0":  # trivial
        return DummyClassifier(strategy="most_frequent"), ["designation"]
    if kind == "b1":
        return DummyClassifier(strategy="stratified", random_state=seed), ["designation"]
    if kind == "b2":  # texte only
        text_branch = create_text_pipeline_from_cfg(cfg.get("text", {}))
        model_name = str(cfg.get("model", {}).get("name", "lr")).lower()
        clf = build_classifier(cfg, seed)
        if model_name in ("xgb","xgboost","lgbm","lightgbm"):
            clf = LabelEncodingClassifier(clf)
        pipe = SkPipeline([("features", text_branch), ("model", clf)], memory=get_cache(cfg))
        return pipe, ["designation","description"]
    if kind == "b3":  # image only (pixels/CNN/ViT/… selon cfg)
        images_cfg = cfg.get("images", {})
        # optionnel : prépare HF si ViT activé
        if bool(images_cfg.get("cnn_vit", {}).get("enabled", False)):
            setup_hf_env(cfg)
            try:
                ensure_vit_available(images_cfg["cnn_vit"].get("hf_model_name", "google/vit-base-patch16-224"),
                                     images_cfg["cnn_vit"].get("hf_revision", "main"))
            except Exception as e:
                logger.warning("[ViT] Préchargement échoué (%s).", e)
        img_branch = create_image_pipeline_from_cfg(images_cfg, use_test_dir=False)
        model_name = str(cfg.get("model", {}).get("name", "lr")).lower()
        clf = build_classifier(cfg, seed)
        if model_name in ("xgb","xgboost","lgbm","lightgbm"):
            clf = LabelEncodingClassifier(clf)
        pipe = SkPipeline([("features", img_branch), ("model", clf)], memory=get_cache(cfg))
        return pipe, ["productid","imageid"]
    if kind == "b4":
        if y_train is None:
            raise ValueError("b4 nécessite y_train pour construire les stratégies de sampling.")
        under, over = make_sampling_strategies(
            y_train,
            major_class=cfg["sampling"]["major_class"],
            major_cap=cfg["sampling"]["major_cap"],
            tail_min=cfg["sampling"]["tail_min"],
        )
        pipe = create_combined_pipeline(cfg, under, over, seed)
        return pipe, ["designation","description","productid","imageid"]
    raise ValueError(f"Baseline inconnue: {kind}")

# -------------------- CV (features-only + ES) --------------------
def run_baseline_and_report(kind: str, X_train: pd.DataFrame, y_train: pd.Series, cfg: dict, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    seed = int(cfg.get("random", {}).get("seed", 42))
    splits = int(cfg.get("cv", {}).get("splits", 3))
    shuffle = bool(cfg.get("cv", {}).get("shuffle", True))
    cv_seed = int(cfg.get("cv", {}).get("random_state", seed))

    pipe, need_cols = build_baseline_pipeline(kind, cfg, seed, y_train=y_train if kind=="b4" else None)

    # --- Safe CV for long-tail datasets ---
    min_count = int(y_train.value_counts().min())
    if min_count >= 2:
        new_splits = min(splits, min_count)
        if new_splits != splits:
            logger.warning("[CV] n_splits=%d > min class count=%d → réduction à %d",
                       splits, min_count, new_splits)
        cv = StratifiedKFold(n_splits=new_splits, shuffle=shuffle, random_state=cv_seed)
    else:
        logger.warning("[CV] Au moins une classe n'a qu'un seul exemple (min=%d). "
                    "Fallback vers une CV non stratifiée.", min_count)
        from sklearn.model_selection import ShuffleSplit
        cv = ShuffleSplit(n_splits=max(3, splits), test_size=0.1, random_state=cv_seed)
    y_pred_cv = np.empty_like(y_train.values)
    if min_count < splits:
        rare = y_train.value_counts().sort_values().head(10)
        logger.info("[CV] Comptes par classe (10 plus rares) :\n%s", rare.to_string())

    t0 = time.time()
    n_splits_used = cv.get_n_splits()
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
        logger.info("[CV] Fold %d/%d", fold, n_splits_used)

        X_tr, y_tr = X_train.iloc[tr][need_cols], y_train.iloc[tr]
        X_va, y_va = X_train.iloc[va][need_cols], y_train.iloc[va]

        # 1) features-only
        pre_feat = SkPipeline([("features", pipe.named_steps["features"])])
        pre_feat.fit(X_tr, y_tr)
        Z_tr = pre_feat.transform(X_tr); Z_va = pre_feat.transform(X_va)

        # 2) sampling TRAIN only (si présents)
        Z_tr_rs, y_tr_rs = Z_tr, y_tr
        if "under" in pipe.named_steps and "over" in pipe.named_steps:
            Z_tr_rs, y_tr_rs = pipe.named_steps["under"].fit_resample(Z_tr, y_tr)
            Z_tr_rs, y_tr_rs = pipe.named_steps["over"].fit_resample(Z_tr_rs, y_tr_rs)

        # 3) scaler (si présent)
        Z_va_rs = Z_va
        if "scaler" in pipe.named_steps:
            scaler = pipe.named_steps["scaler"]
            scaler.fit(Z_tr_rs)
            Z_tr_rs = scaler.transform(Z_tr_rs)
            Z_va_rs = scaler.transform(Z_va)

        # 4) Classifier (avec ES si possible) 
        clf = pipe.named_steps["model"]
        if hasattr(clf, "fit"):
            # XGB/LGBM via LabelEncodingClassifier -> eval_set re-encodé en interne
            try:
                try:
                    # XGBoost <= 1.x : OK d’avoir eval_metric + early_stopping_rounds
                    clf.fit(
                        Z_tr_rs, y_tr_rs,
                        eval_set=[(Z_tr_rs, y_tr_rs), (Z_va_rs, y_va)],
                        eval_metric=["mlogloss", "merror"],
                        early_stopping_rounds=50,
                        verbose=False,
                    )
                except TypeError:
                    # XGBoost >= 2.0 : eval_metric ne doit PAS être passé à fit()
                    import xgboost as xgb
                    cb = [xgb.callback.EarlyStopping(rounds=50, save_best=True, maximize=False)]
                    clf.fit(
                        Z_tr_rs, y_tr_rs,
                        eval_set=[(Z_tr_rs, y_tr_rs), (Z_va_rs, y_va)],
                        callbacks=cb,
                        verbose=False,
                    )       
            except TypeError:
                # modèles sans ES -> fit standard
                clf.fit(Z_tr_rs, y_tr_rs)
        y_pred_cv[va] = clf.predict(Z_va_rs)

    # OOF
    dt = time.time() - t0
    f1m = f1_score(y_train, y_pred_cv, average="macro")
    f1w = f1_score(y_train, y_pred_cv, average="weighted")
    logger.info("[CV] F1-macro=%.4f | F1-weighted=%.4f (%.1fs)", f1m, f1w, dt)

    
    out_oof = Path(cfg["outputs"].get("oof_out", f"{outdir}/preds_oof_{kind}.csv"))
    pd.DataFrame({"y_true": y_train.astype(str).values, "y_pred": y_pred_cv.astype(str)}, index=y_train.index).to_csv(out_oof, index=True)
    logger.info("[CV] OOF écrit: %s", out_oof)
    
    report_dir = Path("results")
    report_dir.mkdir(parents=True, exist_ok=True)

    # rapports bruts
    txt_path = report_dir / f"report_{kind}_cv.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== Baseline {kind.upper()} — CV (n_splits={n_splits_used}) ===\n")
        f.write(f"F1-macro     : {f1m:.4f}\nF1-weighted  : {f1w:.4f}\n")
        f.write(f"Time (sec)   : {dt:.2f}\n\n")
        f.write(classification_report(y_train, y_pred_cv))

    # version lisible (si tu as un mapping id->nom, sinon garde le brut)
    readable_txt = report_dir / f"report_{kind}_cv_readable.txt"
    with open(readable_txt, "w", encoding="utf-8") as f:
        f.write(f"=== Baseline {kind.upper()} — CV (n_splits={n_splits_used}) ===\n")
        f.write(f"F1-macro     : {f1m:.4f}\nF1-weighted  : {f1w:.4f}\n\n")
        f.write(classification_report(y_train, y_pred_cv))  # remplacer par version “target_names” si tu as le mapping

    # résumé Markdown
    md_path = report_dir / f"report_{kind}_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Baseline {kind.upper()} — CV ({n_splits_used} folds) — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- **F1-macro**: {f1m:.4f}\n- **F1-weighted**: {f1w:.4f}\n")

    # per-class CSV (brut)
    import pandas as pd
    prec, rec, f1s, sup = precision_recall_fscore_support(y_train, y_pred_cv, labels=np.unique(y_train))
    df_rep = pd.DataFrame({
        "class": np.unique(y_train),
        "precision": prec, "recall": rec, "f1": f1s, "support": sup
    })
    csv_path = report_dir / f"report_{kind}_per_class_readable.csv"
    df_rep.to_csv(csv_path, index=False)

    # résumé CV dans baseline_results_summary.csv
    try:
        notes = _make_notes(pipe, cfg, y_train=y_train, y_val=None, best_iter=None)
        splits_used = n_splits_used
        _append_summary_row(kind.upper(), splits_used, f1m, f1w, dt, notes,
                            out_csv=cfg.get("outputs", {}).get("summary_csv", "baseline_results_summary.csv"))
        logger.info("[CV] Résumé append → baseline_results_summary.csv")
    except Exception as e:
        logger.warning("[CV] Append résumé échoué: %s", e)

    return f1m, y_pred_cv
# -------------------- Courbes & repointage --------------------
def _save_curve(xs, ys, title, ylabel, out_png):
    plt.figure(figsize=(7.5, 4))
    plt.plot(xs, label="train")
    plt.plot(ys, label="val")
    plt.xlabel("iteration"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"[INFO] Courbe: {out_png}")

def _plot_training_curves_from_est(est):
    try:
        if est.__class__.__name__ == "XGBClassifier":
            hist = est.evals_result()
            for metric in hist["validation_0"].keys():
                _save_curve(hist["validation_0"][metric], hist["validation_1"][metric],
                            f"Final fit — XGB ({metric})", metric,
                            f"results/training_curve_final_xgb_{metric}.png")
        elif "LGBM" in est.__class__.__name__ and hasattr(est, "evals_result_"):
            hist = est.evals_result_
            train_key = "training" if "training" in hist else next(iter(hist.keys()))
            valid_key = "valid_1" if "valid_1" in hist else "valid_0"
            for metric in hist[train_key].keys():
                _save_curve(hist[train_key][metric], hist[valid_key][metric],
                            f"Final fit — LGBM ({metric})", metric,
                            f"results/training_curve_final_lgbm_{metric}.png")
    except Exception as e:
        print(f"[WARN] Courbes non générées: {e}")

def _repoint_images_to_test(pipe: Union[SkPipeline, ImbPipeline], cfg: dict):
    features_step = pipe.named_steps["features"]
    feat_union = features_step.named_steps["union"] if (hasattr(features_step, "named_steps") and "union" in features_step.named_steps) else features_step
    image_test_dir = cfg["images"]["test_dir"]
    new_list = []
    for name, sub in getattr(feat_union, "transformer_list", []):
        if name == "image_pixels":
            if "loader" in getattr(sub, "named_steps", {}):
                loader = sub.named_steps["loader"]
                if hasattr(loader, "set_image_dir"): loader.set_image_dir(image_test_dir)
                elif hasattr(loader, "image_dir"): loader.image_dir = image_test_dir
        elif name == "image_stats_combined":
            if hasattr(sub, "set_image_dir"): sub.set_image_dir(image_test_dir)
            else: sub.image_dir = image_test_dir
        elif name in ("image_cnn", "image_cnn_vit"):
            if "cnn" in getattr(sub, "named_steps", {}):
                cnn = sub.named_steps["cnn"]
                if hasattr(cnn, "set_image_dir"): cnn.set_image_dir(image_test_dir)
                else: cnn.image_dir = image_test_dir
        new_list.append((name, sub))
    feat_union.transformer_list = new_list

# -------------------- Exports analytiques (ACP / SHAP / BLOCKs) --------------------
def _features_only_transform(pipe_or_model, X, y=None):
    if isinstance(pipe_or_model, (ImbPipeline, SkPipeline)) and "features" in pipe_or_model.named_steps:
        features_step = pipe_or_model.named_steps["features"]
        clf = pipe_or_model.named_steps.get("model") or pipe_or_model.named_steps.get("clf")
        has_scaler = "scaler" in pipe_or_model.named_steps
    else:
        features_step = pipe_or_model.named_steps.get("text") or pipe_or_model.named_steps.get("img") or pipe_or_model.named_steps.get("features")
        clf = pipe_or_model.named_steps.get("model") or pipe_or_model.named_steps.get("clf")
        has_scaler = "scaler" in pipe_or_model.named_steps

    pre_feat = SkPipeline([("features", features_step)])
    pre_feat.fit(X, y)
    Z = pre_feat.transform(X)

    feat_union = features_step.named_steps["union"] if (hasattr(features_step, "named_steps") and "union" in features_step.named_steps) else features_step
    slices = {}
    start = 0
    for name, trans in getattr(feat_union, "transformer_list", []):
        try:
            Zi = SkPipeline([(name, trans)]).fit(X, y).transform(X)
            end = start + Zi.shape[1]
            slices[name] = (start, end)
            start = end
        except Exception:
            pass

    scaler = pipe_or_model.named_steps["scaler"] if has_scaler else None
    if scaler is not None:
        Z = scaler.fit_transform(Z)
    return Z, slices, has_scaler, scaler, clf, pre_feat

def export_blocks_importance(pipe, X, y, outdir="results", tag="bX"):
    Z, slices, _, _, clf, _ = _features_only_transform(pipe, X, y)
    is_tree = hasattr(clf, "est_") and clf.est_.__class__.__name__ in {"XGBClassifier","LGBMClassifier"}
    importances = {}
    if not is_tree and hasattr(clf, "coef_"):
        W = clf.coef_
        w_mag = np.mean(np.abs(W), axis=0) if W.ndim > 1 else np.abs(W)
        for name, (a, b) in slices.items():
            importances[name] = float(np.mean(w_mag[a:b]))
    else:
        try:
            import shap
            base = clf.est_ if hasattr(clf, "est_") else clf
            expl = shap.TreeExplainer(base, data=Z, feature_perturbation="interventional", model_output="probability")
            SH = expl.shap_values(Z)
            SH_abs = np.mean(np.abs(SH), axis=0) if isinstance(SH, list) else np.mean(np.abs(SH), axis=0)
            for name, (a, b) in slices.items():
                importances[name] = float(np.mean(SH_abs[a:b]))
        except Exception as e:
            print(f"[WARN] SHAP indisponible pour BLOCKs: {e}")
            return
    df = pd.DataFrame({"block": list(importances.keys()), "importance": list(importances.values())}).sort_values("importance", ascending=False)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{outdir}/blocks_importance_{tag}.csv", index=False)
    plt.figure(figsize=(8,6)); plt.barh(df["block"], df["importance"]); plt.gca().invert_yaxis()
    plt.title(f"Importance par bloc — {tag}"); plt.xlabel("importance moyenne"); plt.tight_layout()
    plt.savefig(f"{outdir}/blocks_importance_{tag}.png", dpi=150); plt.close()

def export_pca_preview(pipe, X, y, outdir="results", tag="bX", n_comp=100):
    Z, _, _, _, _, _ = _features_only_transform(pipe, X, y)
    svd = TruncatedSVD(n_components=min(n_comp, Z.shape[1]-1), random_state=42)
    Zs = svd.fit_transform(Z)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(Zs[:, :10]).to_csv(f"{outdir}/features_{tag}_svd10_preview.csv", index=False)
    pd.DataFrame({"component": np.arange(len(svd.explained_variance_ratio_)),
                  "var_ratio": svd.explained_variance_ratio_}).to_csv(f"{outdir}/pca_{tag}_explained_variance.csv", index=False)

def export_shap(pipe, X, y, outdir="results", tag="bX", max_samples=3000):
    Z, _, _, _, clf, _ = _features_only_transform(pipe, X, y)
    if Z.shape[0] > max_samples:
        idx = np.random.RandomState(42).choice(Z.shape[0], size=max_samples, replace=False)
        Z = Z[idx]; y = y.iloc[idx] if isinstance(y, pd.Series) else y[idx]
    try:
        import shap
        base = clf.est_ if hasattr(clf, "est_") else clf
        if base.__class__.__name__ in {"XGBClassifier","LGBMClassifier"}:
            expl = shap.TreeExplainer(base, data=Z, feature_perturbation="interventional", model_output="probability")
            SH = expl.shap_values(Z)
        else:
            expl = shap.LinearExplainer(base, Z, feature_dependence="independent")
            SH = expl.shap_values(Z)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        shap.summary_plot(SH, Z, plot_type="bar", show=False, max_display=20)
        plt.tight_layout(); plt.savefig(f"{outdir}/shap_{tag}_bar.png", dpi=150); plt.close()
        shap.summary_plot(SH, Z, show=False, max_display=20)
        plt.tight_layout(); plt.savefig(f"{outdir}/shap_{tag}_beeswarm.png", dpi=150); plt.close()
        print(f"[INFO] SHAP: figures sous {outdir}")
    except Exception as e:
        print(f"[WARN] SHAP non généré ({e}).")

# -------------------- Final training with ES + refit --------------------
def train_and_predict_on_test(pipe: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, cfg: dict):
    logger.info(">> Entraînement final avec early stopping (split 90/10)")
    need_cols = ["designation","description","productid","imageid"]

    X_tr, X_va, y_tr, y_va = train_test_split(X_train[need_cols], y_train, test_size=0.10, random_state=42, stratify=y_train)

    # 1) features-only
    pre_feat = SkPipeline([("features", pipe.named_steps["features"])])
    pre_feat.fit(X_tr, y_tr)
    Z_tr = pre_feat.transform(X_tr); Z_va = pre_feat.transform(X_va)

    # 2) sampling TRAIN
    Z_tr_rs, y_tr_rs = pipe.named_steps["under"].fit_resample(Z_tr, y_tr)
    Z_tr_rs, y_tr_rs = pipe.named_steps["over"].fit_resample(Z_tr_rs, y_tr_rs)

    # 3) scaler
    has_scaler = "scaler" in pipe.named_steps
    Z_va_rs = Z_va
    if has_scaler:
        scaler = pipe.named_steps["scaler"]
        scaler.fit(Z_tr_rs); Z_tr_rs = scaler.transform(Z_tr_rs); Z_va_rs = scaler.transform(Z_va)

    # 4) fit classifieur avec ES
    clf = pipe.named_steps["model"]
    try:
        # XGBoost <= 1.x
        clf.fit(
            Z_tr_rs, y_tr_rs,
            eval_set=[(Z_tr_rs, y_tr_rs), (Z_va_rs, y_va)],
            eval_metric=["mlogloss", "merror"],
            early_stopping_rounds=50,
            verbose=False,
        )
    except TypeError:
        # XGBoost >= 2.0 : utiliser les callbacks
        import xgboost as xgb
        cb = [xgb.callback.EarlyStopping(rounds=50, save_best=True, maximize=False)]
        clf.fit(
            Z_tr_rs, y_tr_rs,
            eval_set=[(Z_tr_rs, y_tr_rs), (Z_va_rs, y_va)],
            callbacks=cb,
            verbose=False,
        )
    _plot_training_curves_from_est(clf.est_)

    # -- après clf.fit(...) et avant refit full data --
    best_iter = None
    best_score = None

    # XGBoost >= 1.3 scikit wrapper expose souvent best_iteration_ / best_score
    if hasattr(clf, "best_iteration_"):
        best_iter = clf.best_iteration_
    if hasattr(clf, "best_score"):
        best_score = clf.best_score

    # fallback via booster (robuste XGB 1.x / 2.x)
    try:
        booster = clf.get_booster()
        if best_iter is None and hasattr(booster, "best_iteration"):
            best_iter = booster.best_iteration
        if best_score is None and hasattr(booster, "best_score"):
            best_score = booster.best_score
    except Exception:
        pass

    logger.info(f"[XGB] best_iter={best_iter} | best_score={best_score}")

    # Refitting au meilleur nombre d’arbres si dispo
    if best_iter is not None:
        n_estimators_refit = int(best_iter) + 1
        logger.info(f"[XGB] Refit full data at n_estimators={n_estimators_refit}")


    # 5) re-fit FULL DATA — toujours SANS under/over
    t_train = time.time()

    # pipeline minimal: features (+ scaler si présent) + model
    steps = [("features", pre_feat.named_steps["features"])]
    if has_scaler:
        steps.append(("scaler", scaler))
    final_clf = clf  # on réutilise le classif entraîné (ses hyperparams + best_iter)

    # si best_iter connu, on ajuste n_estimators via le wrapper
    if best_iter is not None:
        n_estimators_refit = int(best_iter) + 1
        logger.info("[XGB] Refit full data avec n_estimators=%d", n_estimators_refit)
        final_clf = clone(clf).set_params(**{"base_estimator__n_estimators": n_estimators_refit})

    final_pipe = SkPipeline(steps + [("model", final_clf)], memory=getattr(pipe, "memory", None))
    final_pipe.fit(X_train[need_cols], y_train)
    train_dt = time.time() - t_train

    # remplace l'ancien pipe par le final pour la suite (exports + préd test)
    pipe = final_pipe
  

    # 6) repointer images → test
    _repoint_images_to_test(pipe, cfg)

    # 7) exports analytiques b4 (optionnels)
    try:
        export_blocks_importance(pipe, X_train[need_cols], y_train, outdir="results", tag="b4")
        export_pca_preview(pipe, X_train[need_cols], y_train, outdir="results", tag="b4")
        export_shap(pipe, X_train[need_cols], y_train, outdir="results", tag="b4")
    except Exception as e:
        print(f"[WARN] Exports analytiques b4 échoués: {e}")

    # 8) prédire test
    y_pred = pipe.predict(X_test[need_cols])
    return pipe, y_pred

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CFG), help="Chemin du fichier TOML.")
    parser.add_argument("--baseline", type=str, default="b4", choices=["b0","b1","b2","b3","b4"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_seeds(int(cfg.get("random", {}).get("seed", 42)))

    # CSV
    X_train = pd.read_csv(cfg["paths"]["x_train_csv"])
    y_train = pd.read_csv(cfg["paths"]["y_train_csv"]).iloc[:, 0]
    X_test  = pd.read_csv(cfg["paths"]["x_test_csv"])



    # --- Sanity checks & auto-fix target / stray columns ---
    TARGET_COL = (cfg.get("data", {}) or {}).get("target", "prdtypecode")

    # 1) Si la cible est manifestement un index (ex: 'Unnamed: 0') ou unique par ligne,
    #    on essaie de la récupérer proprement.
    if (getattr(y_train, "name", None) is None
        or str(y_train.name).lower().startswith("unnamed")
        or y_train.nunique(dropna=False) == len(y_train)):

        # a) Si la cible est dans X_train, on la sort de X -> y
        if TARGET_COL in X_train.columns:
            y_train = X_train.pop(TARGET_COL).astype(int)
            logger.warning("[data] Cible récupérée depuis X_train['%s'] .", TARGET_COL)
        else:
            # b) Sinon on relit le CSV y_train en entier et on prend la bonne colonne si elle existe
            y_df = pd.read_csv(cfg["paths"]["y_train_csv"])
            if TARGET_COL in y_df.columns:
                y_train = y_df[TARGET_COL].astype(int)
                logger.warning("[data] Cible depuis y_train.csv['%s'] .", TARGET_COL)
            else:
                raise ValueError(f"[data] Target column '{TARGET_COL}' introuvable dans X_train et y_train.csv")

    # 2) On supprime d'éventuelles colonnes d'index oubliées dans X (ex: 'Unnamed: 0')
    drop_unnamed = [c for c in X_train.columns if c.lower().startswith("unnamed")]
    if drop_unnamed:
        X_train = X_train.drop(columns=drop_unnamed)
        logger.info("[data]  X_train: %s", drop_unnamed)

    # --- Sanity: colonnes attendues par les baselines ---
    required_b2 = ["designation","description"]
    required_b3 = ["productid","imageid"]
    required_b4 = required_b2 + required_b3

    need = required_b4 if args.baseline=="b4" else (required_b3 if args.baseline=="b3" else required_b2)
    missing = [c for c in need if c not in X_train.columns]
    if missing:
        raise ValueError(f"[data] Colonnes manquantes pour {args.baseline}: {missing}")

    # --- Sanity: répertoires image ---
    imgs = cfg.get("images", {}) or {}
    for key in ["train_dir","test_dir"]:
        if key in imgs:
            p = Path(imgs[key])
            if not p.exists():
                raise FileNotFoundError(f"[images] {key} inexistant: {p}")

    # 3) Petit diagnostic pour confirmer
    vc = y_train.value_counts()
    logger.info("[data] Target '%s' → %d classes | min_count=%d | max_count=%d",
                y_train.name, vc.size, int(vc.min()), int(vc.max()))
    
    # --- Limite globale via env: RAKUTEN_MAX_N (APRES réparation de y_train) ---
    try:
        max_n_env = int(os.getenv("RAKUTEN_MAX_N", "0") or "0")
    except ValueError:
        max_n_env = 0

   
    if max_n_env > 0 and len(X_train) > max_n_env:
        n_before = len(X_train)
        seed = int(cfg.get("random", {}).get("seed", 42))
        from sklearn.model_selection import StratifiedShuffleSplit
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=max_n_env, random_state=seed)
            tr_idx, _ = next(sss.split(X_train, y_train))
            reason = "stratified"
        except ValueError as e:
            logger.warning("[data] StratifiedShuffleSplit impossible (%s) → fallback non stratifié.", e)
            rng = np.random.RandomState(seed)
            tr_idx = rng.choice(len(X_train), size=max_n_env, replace=False)
            reason = "random"

        X_train = X_train.iloc[tr_idx].reset_index(drop=True)
        y_train = y_train.iloc[tr_idx].reset_index(drop=True)
        logger.warning("[data] Sous-échantillonnage global activé (%s): %d → %d via RAKUTEN_MAX_N",
                    reason, n_before, len(y_train))
 

    # CV (score + OOF)
    try:
        f1m, _ = run_baseline_and_report(args.baseline, X_train, y_train, cfg, outdir="results")
    except Exception as e:
        logger.exception("[FATAL] run_baseline_and_report(%s) a échoué", args.baseline)
        raise

    seed = int(cfg.get("random", {}).get("seed", 42))

    # Entraînement final + sauvegardes selon baseline
    if args.baseline in {"b2","b3"}:
        pipe, need_cols = build_baseline_pipeline(args.baseline, cfg, seed)
        pipe.fit(X_train[need_cols], y_train)

        # Préd test & joblib
        Path(cfg["outputs"]["pred_out"]).parent.mkdir(parents=True, exist_ok=True)
        y_pred = pipe.predict(X_test[need_cols])
        pd.DataFrame({"id": X_test.index, "y_pred": y_pred}).to_csv(
            cfg["outputs"]["pred_out"].replace(".csv", f"_{args.baseline}.csv"), index=False)

        Path(cfg["outputs"]["model_out"]).parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(pipe, cfg["outputs"]["model_out"].replace(".joblib", f"_{args.baseline}.joblib"))
        logger.info("Pipeline %s sauvegardée.", args.baseline.upper())

        try:
            t0f = time.time()
            # On mesure le temps de prédiction pour avoir train+infer approximatif si tu veux
            _ = pipe.predict(X_train[need_cols].iloc[:100])  # petit ping pour éviter coût froid
            train_infer_dt = time.time() - t0f
            notes_train = _make_notes(pipe, cfg, y_train=y_train, y_val=None, best_iter=None)
            _append_summary_row(args.baseline.upper()+"-FINAL", 1, float("nan"), float("nan"),
                                train_infer_dt, notes_train,
                                out_csv=cfg.get("outputs", {}).get("summary_csv", "baseline_results_summary.csv"))
            logger.info("[TrainFinal %s] Résumé append → baseline_results_summary.csv", args.baseline.upper())
        except Exception as e:
            logger.warning("[TrainFinal %s] Append résumé échoué: %s", args.baseline.upper(), e)

        # Exports analytiques
        try:
            export_blocks_importance(pipe, X_train[need_cols], y_train, outdir="results", tag=args.baseline)
            export_pca_preview(pipe, X_train[need_cols], y_train, outdir="results", tag=args.baseline)
            export_shap(pipe, X_train[need_cols], y_train, outdir="results", tag=args.baseline)
        except Exception as e:
            print(f"[WARN] Exports analytiques {args.baseline} échoués: {e}")

    elif args.baseline == "b4":
        under, over = make_sampling_strategies(y_train, cfg["sampling"]["major_class"], cfg["sampling"]["major_cap"], cfg["sampling"]["tail_min"])
        pipe = create_combined_pipeline(cfg, under, over, seed)
        pipe, y_pred = train_and_predict_on_test(pipe, X_train, y_train, X_test, cfg)

        # Sauvegardes
        Path(cfg["outputs"]["pred_out"]).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": X_test.index, "y_pred": y_pred}).to_csv(cfg["outputs"]["pred_out"], index=False)
        import joblib
        Path(cfg["outputs"]["model_out"]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, cfg["outputs"]["model_out"])
        logger.info("Pipeline b4 sauvegardée: %s", cfg["outputs"]["model_out"])

if __name__ == "__main__":
    main()
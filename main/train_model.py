"""
Classification Rakuten – Baselines & Pipeline (version commentée)

Objectif général :
- Construire une pipeline multimodale (texte + image) avec rééchantillonnage pour traiter le déséquilibre des classes.
- Comparer rapidement deux classifieurs (LogisticRegression et LinearSVC) en validation croisée.
- Entraîner le modèle complet sur X_train/Y_train puis prédire sur X_test.

Baselines :
- B0 — Naïf (majoritaire) : DummyClassifier(strategy="most_frequent").
- B1 — Naïf (aléatoire stratifié) : DummyClassifier(strategy="stratified").
- B2 — Texte seul (TF-IDF → LR) : branche texte sans rééchantillonnage (référence).
- B3 — Image seule (ImageLoader → flatten → PCA → LR) : sans texte.
- B4 — Multimodal complet : Texte (TF-IDF) + Images (pixels + stats optionnelles).
      Rééchantillonnage : under-sampling adaptatif par fold + over-sampling ciblé.
      (Under-sampling “capé” par classe, recalculé à chaque fold → CV-safe.)

Fonctionnalités CLI :
- --model {lr,svc} : définir le classifieur final (défaut : lr).
- --compare : comparer LR et LinearSVC via validation croisée stratifiée (F1-macro)
              sur X_train/Y_train, en conservant le split Rakuten d’origine.
- --config : lire un fichier TOML pour centraliser chemins & hyperparams.

Détails pipeline :
- Fusionner texte+image avec FeatureUnion.
- Branche images “pixels” : charger RGB, redimensionner, aplatir, (option) PCA/SVD.
- Branche images “stats” : ImageStatsFeaturizer (width, height, occupancy, white_ratio, black_ratio).
  Les seuils venir du TOML : [images.stats] white_threshold, black_threshold, min_area.
  Si out_prefix="auto", faire refléter ces seuils dans les noms de colonnes (ex. img_w230_b25_*).
- Rééchantillonnage :
  * Under-sampling : AdaptiveUnderSampler(cap_dict) → pour chaque fold, fixer la cible à min(plafond, effectif_du_fold).
  * Over-sampling : RandomOverSampler(sampling_strategy=...).
- Placer StandardScaler(with_mean=False) APRÈS under/over pour scaler la distribution réellement vue par le modèle.
- Sérialiser avec Joblib, sauvegarder les prédictions X_test (index produit), journaliser les étapes.

Exécution :
# Baselines
python -m main.train_model --config features/config.toml --baseline b0
python -m main.train_model --config features/config.toml --baseline b1
python -m main.train_model --config features/config.toml --baseline b2
python -m main.train_model --config features/config.toml --baseline b3

# Modèle multimodal “B4” (entraînement + prédictions)
python -m main.train_model --config features/config.toml

# Comparaison LR vs SVC (CV stratifiée F1-macro, avec under adaptatif)
python -m main.train_model --config features/config.toml --compare
"""

# === Importer les bibliothèques standard et ML ==================================
# === Importer les bibliothèques standard et ML ==================================
import os
import re
import time
import argparse
import joblib
import json
import random
from collections import Counter
from pathlib import Path
import logging
import toml
from typing import Dict, Any, Union, Optional, Tuple, List
from tqdm.auto import tqdm
from datetime import datetime


import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline as SkPipeline, FeatureUnion, make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler

# Importer nos modules personnalisés
from features.image_loader import ImageLoader
from models.text_pipeline import create_text_pipeline_from_cfg
from models.image_pipeline import create_image_pipeline
from models.image_pipeline import create_image_pipeline_from_cfg 
from features.image_stats import ImageStatsCombinedFeaturizer
from models.image_pipeline import diagnostic_reduction
from models.cnn_features import CNNFeaturizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from logging.handlers import RotatingFileHandler


logger = logging.getLogger(__name__)

class TqdmLoggingHandler(logging.StreamHandler):
    """Écrit les logs via tqdm.write pour ne pas casser la barre de progression."""
    def emit(self, record):
        try:
            from tqdm.auto import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            super().emit(record)

def setup_logging(log_dir: str = "results/logs", level=logging.INFO):
    log_dir = os.path.expanduser(os.path.expandvars(log_dir))
    os.makedirs(log_dir, exist_ok=True)

    # Format commun
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Handlers fichiers
    train_fh = RotatingFileHandler(
        os.path.join(log_dir, "training.log"),
        encoding="utf-8", maxBytes=5_000_000, backupCount=3, delay=True
    )
    train_fh.setLevel(level)
    train_fh.setFormatter(fmt)

    img_fh = RotatingFileHandler(
        os.path.join(log_dir, "image_processing.log"),
        encoding="utf-8", maxBytes=5_000_000, backupCount=3, delay=True
    )
    img_fh.setLevel(level)
    img_fh.setFormatter(fmt)

    # Handler console compatible tqdm
    console_h = TqdmLoggingHandler()
    console_h.setLevel(level)
    console_h.setFormatter(fmt)

    # (Re)configurer radicalement la racine
    logging.basicConfig(level=level, handlers=[train_fh, console_h], force=True)

    # Attacher un fichier dédié aux loggers image
    for name in ["models.image_pipeline", "features.image_loader", "features.image_stats"]:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.addHandler(img_fh)     # écrira dans image_processing.log
        lg.propagate = True       # et remontera aussi vers la racine

    # Exemple : logger principal
    root = logging.getLogger(__name__)
    root.info("Logging initialisé — logs dans %s", os.path.abspath(log_dir))

# Importer ImageLoader depuis le module features

def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Valide la configuration en vérifiant la présence des sections requises.
    
    Args:
        cfg: Dictionnaire de configuration chargé depuis le fichier TOML
        
    Raises:
        ValueError: Si une section requise est manquante
    """
    required = ['paths', 'model', 'compute']
    for key in required:
        if key not in cfg:
            raise ValueError(f"Configuration manquante: {key}")
            
def train_model(config_path: str = "features/config.toml"):
    """Entraîne le modèle avec la configuration spécifiée."""
    logger.info("Chargement de la configuration...")
    
    try:
        with open(config_path) as f:
            config = toml.load(f)
        validate_config(config)  # Validation après chargement
        logger.debug("Configuration validée avec succès")
    except Exception as e:
        logger.error(f"Erreur de configuration: {e}")
        raise


# === Définir le chemin par défaut du TOML ======================================
# Définir un chemin par défaut vers features/config.toml (relatif à ce fichier)
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "features" / "config.toml"

# === Charger la configuration TOML =============================================
# Utiliser tomllib (Py>=3.11) sinon rétroporter sur tomli
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # fallback si environnement < 3.11
    import tomli as tomllib


def load_config(config_path: str | Path | None = None) -> dict:
    """Charger le fichier TOML et retourner un dict Python."""
    cfg_path = Path(config_path) if config_path else DEFAULT_CFG
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg


# === Initialiser les graines de hasard (reproductibilité) =======================
def init_seeds(seed: int) -> None:
    """Fixer les graines pour numpy, random et Python hash."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# === Construire les stratégies d'échantillonnage ================================
def make_sampling_strategies(y_train: pd.Series,
                             major_class: int = 2583,
                             major_cap: int = 6000,
                             tail_min: int = 1500):
    """
    Construire deux stratégies pour imblearn :
      - under : plafonner uniquement la classe major_class à major_cap
      - over  : remonter toutes les classes d'effectif < tail_min à tail_min
    Retourner (under_dict, over_dict).
    """
    vc = y_train.value_counts()
    under = {
        int(cls): (min(int(cnt), int(major_cap)) if int(cls) == int(major_class) else int(cnt))
        for cls, cnt in vc.items()
    }
    over = {int(cls): int(tail_min) for cls, cnt in vc.items() if int(cnt) < int(tail_min)}
    return under, over


# === Définir un under-sampler adaptatif (CV-safe) ===============================

class AdaptiveUnderSampler(BaseSampler):
    """
    Under-sampler adaptatif : à chaque fold de CV, borner l'effectif de chaque
    classe à min(plafond_configuré, effectif_du_fold).
    cap_dict : {classe: plafond_max} (ex. {2583: 6000})
    """
    # imblearn/sklearn validations
    _parameter_constraints = {
        "cap_dict": [dict, None],
        "random_state": [None, int],
        "sampling_strategy": [str, dict, float, callable, None],  # ex. 'auto'
    }
    _sampling_type = "under-sampling"  # requis par BaseSampler

    def __init__(self, cap_dict=None, random_state=None, sampling_strategy="auto"):
        self.cap_dict = cap_dict or {}
        self.random_state = random_state
        # requis par BaseSampler.fit_resample (même si non utilisé ensuite)
        self.sampling_strategy = sampling_strategy

    def _fit_resample(self, X, y):
        # sécuriser y en 1D
        y_arr = np.asarray(y).ravel()

        # effectifs observés dans CE fold
        cnt = Counter(y_arr)

        # stratégie bornée par cap_dict : min(effectif_fold, plafond_configuré)
        sampling_strategy = {
            cls: min(n, self.cap_dict.get(cls, n))
            for cls, n in cnt.items()
        }

        # déléguer à RandomUnderSampler avec cette stratégie
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_res, y_res = rus.fit_resample(X, y_arr)
        return X_res, y_res

    def _more_tags(self):
        return {"allow_nan": False, "X_types": ["2darray", "sparse"]}

# --- utils mémoire : caster en float32 pour réduire l'empreinte ---
from scipy import sparse
class ToFloat32:
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        try:
            return X.astype("float32") if sparse.issparse(X) else X.astype(np.float32)
        except Exception:
            return X  # garde-fou si un bloc ne supporte pas astype 
          
# === Fabriquer le classifieur à partir de la config =============================
def build_classifier(cfg: dict, seed: int):
    """
    Construire le classifieur final depuis la section [model] du TOML.

    Support :
      - name="lr"  -> LogisticRegression
      - name="svc" -> LinearSVC (utile pour TF-IDF clairsemé)

    Clés TOML reconnues (section [model]) :
      name              = "lr" | "svc"
      solver            = "saga" | "lbfgs" | "liblinear" | "newton-cg" | "sag"
      penalty           = "l2" | "l1" | "elasticnet" | "none"
      C                 = float
      max_iter          = int
      tol               = float
      verbose           = int (0/1)
      use_class_weight  = true/false   # -> "balanced" si true
      fit_intercept     = true/false
      l1_ratio          = float (0..1) # utilisé seulement si penalty="elasticnet"
      multi_class       = "auto" | "ovr" | "multinomial"  # si omis: on laisse sklearn décider
      ovr               = true/false   # envelopper en OneVsRestClassifier
    """
    model = cfg.get("model", {}) or {}
    name  = str(model.get("name", "lr")).lower()

    # lire le nombre de jobs global
    n_jobs = int(cfg.get("compute", {}).get("n_jobs", -1))

    # === LinearSVC (option texte) =============================================
    if name == "svc":
        cw = "balanced" if model.get("use_class_weight", False) else None
        loss = str(model.get("loss", "squared_hinge"))   # "hinge" ou "squared_hinge"
        dual = bool(model.get("dual", True))             # voir note ci-dessous
        penalty = str(model.get("penalty", "l2"))        # "l2" (par défaut) ou "l1" (avec dual=False)
        return LinearSVC(
            C=float(model.get("C", 1.0)),
            tol=float(model.get("tol", 1e-3)),
            max_iter=int(model.get("max_iter", 2000)),
            class_weight=cw,
            loss=loss,
            dual=dual,
            penalty=penalty,

        )

    # === LogisticRegression ====================================================
    use_cw = bool(model.get("use_class_weight", False))
    class_weight = "balanced" if use_cw else None

    solver        = str(model.get("solver", "saga"))
    penalty       = str(model.get("penalty", "l2"))
    C             = float(model.get("C", 1.0))
    max_iter      = int(model.get("max_iter", 3000))
    tol           = float(model.get("tol", 1e-3))
    verbose       = int(model.get("verbose", 0))
    fit_intercept = bool(model.get("fit_intercept", True))

    # multi_class : ne PAS forcer si absent (évite FutureWarning)
    multi = model.get("multi_class", None)
    if multi is not None:
        multi = str(multi)

    # l1_ratio : n’utiliser QUE si elasticnet (sinon éviter le warning)
    l1_ratio = model.get("l1_ratio", None)
    if penalty != "elasticnet":
        l1_ratio = None
    else:
        l1_ratio = 0.0 if l1_ratio is None else float(l1_ratio)

    # gardes-fous solver/penalty
    if penalty == "l1" and solver not in {"liblinear", "saga"}:
        raise ValueError("penalty='l1' nécessite solver 'liblinear' ou 'saga'.")
    if penalty == "elasticnet" and solver != "saga":
        raise ValueError("penalty='elasticnet' nécessite solver 'saga'.")

    # construire les paramètres sans passer les clés inutiles
    params = dict(
        solver=solver,
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=n_jobs,             # ignoré par certains solvers (OK)
        fit_intercept=fit_intercept,
    )
    if l1_ratio is not None:
        params["l1_ratio"] = l1_ratio
    if multi is not None and multi.lower() != "auto":
        params["multi_class"] = multi

    base_lr = LogisticRegression(**params)

    # Option : envelopper en One-vs-Rest explicite
    if bool(model.get("ovr", False)):
        from sklearn.multiclass import OneVsRestClassifier
        return OneVsRestClassifier(base_lr, n_jobs=n_jobs)

    return base_lr

# === Construire les pipelines baselines sans rééchantillonnage ==================
from typing import Optional
from joblib import Memory

def get_cache(cfg: dict):
    """Retourne un objet Memory si use_cache=true dans [compute], sinon None."""
    use_cache = bool(cfg.get("compute", {}).get("use_cache", False))
    if not use_cache:
        return None
    cache_dir = os.path.join(cfg["outputs"]["log_dir"], "skcache")
    os.makedirs(cache_dir, exist_ok=True)
    return Memory(cache_dir)

def build_baseline_pipeline(
    kind: str,
    cfg: dict,
    seed: int,
    y_train: Optional[pd.Series] = None,   # <- nouveau
):
    """
    Construire une pipeline baseline.
    kind in {'b0','b1','b2','b3','b4'}
    """
    need_cols = ["designation", "description", "productid", "imageid"]

    if kind == "b0":
        pipe = DummyClassifier(strategy="most_frequent")
        return pipe, ["designation"]

    if kind == "b1":
        pipe = DummyClassifier(strategy="stratified", random_state=seed)
        return pipe, ["designation"]

    if kind == "b2":
        text_branch = create_text_pipeline_from_cfg(cfg.get("text", {}))
        clf = build_classifier(cfg, seed)

        cache = get_cache(cfg)
        if cache is not None and hasattr(cache, "location"):
            logger.info(f"Cache sklearn activé: {cache.location}")

        pipe = SkPipeline(
            [("text", text_branch), ("clf", clf)],
            memory=cache
        )
        return pipe, ["designation", "description"]

    if kind == "b3":
    # Si [images.cnn.enabled]=true → utiliser CNN, sinon pixels
        use_cnn = bool(cfg.get("images", {}).get("cnn", {}).get("enabled", False))
        if use_cnn:
            img_branch = create_cnn_branch_from_cfg(cfg["images"])
        else:
            img_branch = create_image_pipeline_from_cfg(cfg["images"], use_test_dir=False)

        img_union = FeatureUnion([("img", img_branch)])

        stats_c = cfg.get("images", {}).get("stats_combined", {})
        if bool(stats_c.get("enabled", False)):
            img_union.transformer_list.append((
                "image_stats_combined",
                ImageStatsCombinedFeaturizer(
                    image_dir=cfg["images"]["train_dir"],
                    imgid_col="imageid", pid_col="productid",
                    white_threshold=int(stats_c.get("white_threshold", 230)),
                    black_threshold=int(stats_c.get("black_threshold", 25)),
                    min_area=int(stats_c.get("min_area", 16)),
                    prefix_basic=str(stats_c.get("prefix_basic", "img_")),
                    prefix_pro=str(stats_c.get("prefix_pro", "pro_")),
                )
            ))

        clf = build_classifier(cfg, seed)
        cache = get_cache(cfg)
        pipe = SkPipeline([("img", img_union), ("clf", clf)], memory=cache)
        return pipe, ["productid", "imageid"]
    
    # ---------- B4 : multimodal (texte + image) avec sampling ----------
    if kind == "b4":
        if y_train is None:
            raise ValueError("b4 a besoin de y_train pour calculer les stratégies d'échantillonnage.")
        under, over = make_sampling_strategies(
            y_train,
            major_class=cfg["sampling"]["major_class"],
            major_cap=cfg["sampling"]["major_cap"],
            tail_min=cfg["sampling"]["tail_min"],
        )
        seed = int(cfg.get("random", {}).get("seed", 42))
        pipe = create_combined_pipeline(cfg, under, over, seed)
        return pipe, ["designation", "description", "productid", "imageid"]

    # ---------- fin ----------
    raise ValueError(f"Baseline inconnue: {kind}")


# === Exécuter une baseline et écrire un rapport CV ==============================
def run_baseline_and_report(kind: str, X_train, y_train, cfg: dict, outdir="results"):
    """Exécute la baseline avec validation croisée et diagnostics."""
    logger.info(f"Évaluation baseline {kind}...")
    """
    Exécuter la baseline 'kind' en CV, puis écrire un résumé clair :
      - results/baseline_results_summary.csv (append)
      - results/report_{kind}_cv.txt        (rapport complet par classe)
    """
    os.makedirs(outdir, exist_ok=True)

    seed = int(cfg.get("random", {}).get("seed", 42))
    splits = int(cfg.get("cv", {}).get("splits", 3))
    shuffle = bool(cfg.get("cv", {}).get("shuffle", True))
    cv_seed = int(cfg.get("cv", {}).get("random_state", seed))

    pipe, need_cols = build_baseline_pipeline(kind, cfg, seed, y_train=y_train if kind=="b4" else None)

    # Construire une CV stratifiée
    cv = StratifiedKFold(n_splits=splits, shuffle=shuffle, random_state=cv_seed)

    # ===== Export features pour ACP/SHAP (OOF, même ordre que y_pred_cv) =====
    export_features = True  # False pour couper
    feat_blocks = []        # liste de blocs (sparse/dense) par fold (validation uniquement)
    feat_indices = []       # index du jeu de validation, pour réordonner ensuite
    use_svd_for_preview = True   # compressions 2D/100D pour ACP rapide (optionnel)
    svd_components_preview = 100 # 100 pour ACP exploratoire, 2 pour projeter directement
    EXPORT_K = 256   # (128/256/512 au choix)

    # ==== Prédictions OOF avec barre de progression ====
    from sklearn.base import clone
    import numpy as np

    splits_iter = list(cv.split(X_train, y_train))
    y_pred_cv = np.empty_like(y_train.values)


    pbar = tqdm(
    total=len(splits_iter),
    desc=f"{kind.upper()} CV",
    unit="fold",
    dynamic_ncols=True,  # s’adapte à la largeur du terminal
    leave=True
)
    t0_total = time.time()
    try:
        for fold_idx, (tr, va) in enumerate(splits_iter, 1):
            logger.info(f"Fold {fold_idx}/{cv.get_n_splits()} | train={len(tr)} val={len(va)}")
            logger.info("Classes train: %s", pd.Series(y_train.iloc[tr]).value_counts().to_dict())
            logger.info("Classes val  : %s", pd.Series(y_train.iloc[va]).value_counts().to_dict())
            t0 = time.time()
            model = clone(pipe)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This Pipeline instance is not fitted yet",
                    category=FutureWarning,
                    module=r"sklearn\.pipeline",
                )
                model.fit(X_train.iloc[tr][need_cols], y_train.iloc[tr])
                y_pred_cv[va] = model.predict(X_train.iloc[va][need_cols])

            # ----- Export des features transformées du split de validation -----
            if export_features:
                try:
                    Xtr = X_train.iloc[tr][need_cols]
                    Xv  = X_train.iloc[va][need_cols]

                    if kind == "b4":
                        Ztr = model.named_steps["features"].transform(Xtr)
                        if "scaler" in model.named_steps:
                            Ztr = model.named_steps["scaler"].transform(Ztr)
                        Zv = model.named_steps["features"].transform(Xv)
                        if "scaler" in model.named_steps:
                            Zv = model.named_steps["scaler"].transform(Zv)

                    elif kind == "b2":
                        Ztr = model.named_steps["text"].transform(Xtr)
                        Zv  = model.named_steps["text"].transform(Xv)

                    elif kind == "b3":
                        Ztr = model.named_steps["img"].transform(Xtr)
                        Zv  = model.named_steps["img"].transform(Xv)

                    else:
                        Ztr = Zv = None

                    if Ztr is not None and Zv is not None:
                        # SVD "local au fold" appris sur TRAIN → projette VAL vers une dimension fixe
                        k = min(EXPORT_K, (Ztr.shape[1] - 1)) if hasattr(Ztr, "shape") else EXPORT_K
                        if k < 2:  # garde-fou si très peu de colonnes
                            k = 2
                        svd_fold = TruncatedSVD(n_components=k, random_state=cv_seed)
                        Ztr_red = svd_fold.fit_transform(Ztr)
                        var_ratio = getattr(svd_fold, "explained_variance_ratio_", None)
                        if var_ratio is not None:
                            logger.debug("Fold %d SVD var_explained=%.2f%%", fold_idx, 100*var_ratio.sum())
                        Zv_red  = svd_fold.transform(Zv)

                        feat_blocks.append(Zv_red)            # (n_val, k) — dimension fixe
                        feat_indices.append(X_train.index[va])
                except Exception as e:
                    logger.warning("Export features impossible sur fold %d: %s", fold_idx, e)
            pbar.set_postfix(time=f"{time.time()-t0:.1f}s")
            pbar.update(1)
    finally:
        pbar.close()

    dt = time.time() - t0_total

    preds_df = pd.DataFrame({
        "y_true": y_train,
        "y_pred": y_pred_cv
    }, index=y_train.index)

    preds_path = os.path.join(outdir, f"preds_{kind}.csv")
    preds_df.to_csv(preds_path, index=True)
    print(f"> Prédictions sauvegardées: {preds_path}")

    # ===== Sauvegarde finale des features OOF =====
    if export_features and len(feat_blocks) == len(splits_iter):
        try:  
            from scipy import sparse        
            # Empilement en respectant l'ordre OOF
            # On réordonne avec l'index global de X_train
            all_idx = np.concatenate([np.asarray(idx) for idx in feat_indices])   # index pandas des lignes
            # Concaténer les blocs (sparse compatible)
            if sparse.issparse(feat_blocks[0]):
                Z_all = sparse.vstack(feat_blocks, format="csr")
            else:
                Z_all = np.vstack(feat_blocks)

            # Réordonner Z_all selon l'ordre de y_train (OOF dans l'ordre de l'index)
            # astuce : on crée une position pour chaque index
            pos = pd.Series(np.arange(len(all_idx)), index=all_idx)
            order = pos.loc[y_train.index].values   # positions correspondant à l'ordre de y_train
            if sparse.issparse(Z_all):
                Z_all = Z_all[order]
            else:
                Z_all = Z_all[order, :]

            # Sauvegarde sparse + une version "preview" compressée pour ACP
            os.makedirs(outdir, exist_ok=True)
            base = os.path.join(outdir, f"features_{kind}")

            # 3.1) Sauvegarde sparse (si applicable)
            try:
                from scipy.sparse import save_npz
                if sparse.issparse(Z_all):
                    save_npz(base + "_oof.npz", Z_all)
                    logger.info("Features OOF sauvegardées (sparse): %s_oof.npz", base)
                else:
                    np.save(base + "_oof.npy", Z_all)
                    logger.info("Features OOF sauvegardées (npy): %s_oof.npy", base)
            except Exception as e:
                logger.warning("Impossible de sauver en sparse/NPY: %s", e)

            # 3.2) Aperçu SVD 100D pour ACP rapide (optionnel)
            if use_svd_for_preview:
                try:
                    # Si déjà scalé (with_mean=False), on peut appliquer directement SVD
                    svd = TruncatedSVD(n_components=svd_components_preview, random_state=cv_seed)
                    Z_svd = svd.fit_transform(Z_all)
                    # Sauvegarder un CSV léger (avec y_true/y_pred pour colorer en ACP)
                    svd_df = pd.DataFrame(Z_svd, index=y_train.index,
                                      columns=[f"svd_{i+1}" for i in range(svd_components_preview)])
                    svd_df["y_true"] = y_train.astype(str).values
                    svd_df["y_pred"] = pd.Series(y_pred_cv, index=y_train.index).astype(str).values
                    svd_csv = base + f"_svd{svd_components_preview}_preview.csv"
                    svd_df.to_csv(svd_csv, index=True)
                    logger.info("Aperçu SVD %dD sauvegardé: %s", svd_components_preview, svd_csv)
                except Exception as e:
                    logger.warning("SVD preview impossible: %s", e)

        except Exception as e:
            logger.warning("Consolidation features OOF échouée: %s", e)

    # ==== Harmoniser les types pour éviter toute surprise int/str ====
    yt = y_train.astype(str)
    yp = pd.Series(y_pred_cv, index=y_train.index).astype(str)

    # ==== Métriques & rapport ====
    from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support

    f1_macro = f1_score(yt, yp, average="macro")
    f1_weighted = f1_score(yt, yp, average="weighted")

    # --- mapping lisible ---
    labels_map_path = os.path.join("features", "labels_map.json")
    lblmap = None
    if os.path.exists(labels_map_path):
        import json
        with open(labels_map_path, "r", encoding="utf-8") as f:
            lblmap = {str(k): v for k, v in json.load(f).items()}

    # Construire un tableau de labels (dans l'ordre trié) et leurs noms lisibles
    labels_vec = np.array(sorted(np.unique(yt)), dtype=str)
    target_names = [lblmap.get(l, l) for l in labels_vec] if lblmap else None

    # Rapports sklearn (id + lisible)
    report = classification_report(yt, yp, labels=labels_vec, digits=4, zero_division=0)

    report_readable = None
    if target_names is not None:
        report_readable = classification_report(
            yt, yp, labels=labels_vec, target_names=target_names, digits=4, zero_division=0
        )

    # === Rapport par classe (CSV propre) ===
    prec, rec, f1, supp = precision_recall_fscore_support(yt, yp, labels=labels_vec, zero_division=0)
    per_class = pd.DataFrame({
        "class_id": labels_vec,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": supp
    }).sort_values("support", ascending=False)

    # Sauvegarder le résumé


    use_cnn = bool(cfg.get("images", {}).get("cnn", {}).get("enabled", False))
    notes = []
    notes.append(f"dummy={kind in ['b0','b1']}")
    notes.append(f"text_only={kind=='b2'}")
    notes.append(f"image_only={kind=='b3'}")
    notes.append(f"multimodal={kind=='b4'}")
    if kind in ['b3','b4']:
        notes.append(f"cnn={'true' if use_cnn else 'false'}")
    notes_str = " | ".join(notes)

    summary_csv = os.path.join(outdir, "baseline_results_summary.csv")
    row = pd.DataFrame([{
        "baseline": kind.upper(),
        "cv_splits": splits,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_infer_time_sec": round(dt, 3),
        "notes": notes_str,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }])
    row.to_csv(summary_csv, index=False, mode="a", header=not os.path.exists(summary_csv))


    # Sauvegarder le rapport détaillé
    with open(os.path.join(outdir, f"report_{kind}_cv.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== Baseline {kind.upper()} — CV (n_splits={splits}) ===\n")
        f.write(f"F1-macro     : {f1_macro:.4f}\n")
        f.write(f"F1-weighted  : {f1_weighted:.4f}\n")
        f.write(f"Time (sec)   : {dt:.2f}\n\n")
        f.write(report)

    # Afficher un résumé console
    print(f"[{kind.upper()}] F1-macro={f1_macro:.4f} | F1-weighted={f1_weighted:.4f} | time={dt:.1f}s")
    print(f"> Résumé: {summary_csv}")
    print(f"> Rapport: {os.path.join(outdir, f'report_{kind}_cv.txt')}")
    print(f"[CV] terminé ({splits} folds)")
    if report_readable is not None:
        with open(os.path.join(outdir, f"report_{kind}_cv_readable.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Baseline {kind.upper()} — CV (n_splits={splits}) ===\n")
            f.write(f"F1-macro     : {f1_macro:.4f}\n")
            f.write(f"F1-weighted  : {f1_weighted:.4f}\n\n")
            f.write(report_readable)


    # === Version lisible (mapping) ===
    if lblmap is not None:
        per_class_readable = per_class.assign(
            class_name=per_class["class_id"].map(lblmap).fillna(per_class["class_id"])
        )
        cols = ["class_id", "class_name", "support", "precision", "recall", "f1"]
        per_class_readable[cols].to_csv(os.path.join(outdir, f"report_{kind}_per_class_readable.csv"), index=False)

    # === Petit résumé Markdown (lisible) ===
    md_lines = [
        f"# Baseline {kind.upper()} — CV ({splits} folds) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **F1-macro**: {f1_macro:.4f}",
        f"- **F1-weighted**: {f1_weighted:.4f}",
        "",
        "## Top 20 classes (par support)",
    ]
    top20 = per_class.head(20).copy()
    if os.path.exists(labels_map_path):
        top20["name"] = top20["class_id"].map(lblmap).fillna(top20["class_id"])
        md_lines.append("| id | nom | support | precision | recall | f1 |")
        md_lines.append("|---:|:-----|-------:|----------:|-------:|----:|")
        for _, r in top20.iterrows():
            md_lines.append(f"| {r['class_id']} | {top20.loc[_,'name']} | {int(r['support'])} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    else:
        md_lines.append("| id | support | precision | recall | f1 |")
        md_lines.append("|---:|--------:|----------:|-------:|----:|")
        for _, r in top20.iterrows():
            md_lines.append(f"| {r['class_id']} | {int(r['support'])} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")

    with open(os.path.join(outdir, f"report_{kind}_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # ---- Diagnostics (pour baselines images) ----
    if kind in ("b3", "b4"):
        logger.info("Exécution diagnostics baseline %s...", kind.upper())
        sample_size = min(1000, len(X_train))
        sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[sample_idx]
        y_sample = y_train.iloc[sample_idx]

    # Entraîner un modèle sur l'échantillon pour diagnostiquer
        pipe.fit(X_sample[need_cols], y_sample)

    # 1) Diagnostic réduction (PCA / SVD)
        info = diagnostic_reduction(pipe)
        logger.info("Diagnostic réduction (%s): %s", kind.upper(), info)

    # 2) Autres diagnostics (classes prédites, proba, F1 sur l'échantillon)
        diags = diagnostic_baseline(pipe, X_sample[need_cols], y_sample)
        diags["reduction_info"] = info

    # 2bis) Diagnostic CNN (si présent)
        try:
            features_step = pipe.named_steps.get("features") if "features" in pipe.named_steps else None
            if features_step is not None:
                if hasattr(features_step, "named_steps") and "union" in features_step.named_steps:
                    feat_union = features_step.named_steps["union"]
                else:
                    feat_union = features_step
                for name, sub in getattr(feat_union, "transformer_list", []):
                    if name == "image_cnn" and hasattr(sub, "named_steps") and "cnn" in sub.named_steps:
                        cnn = sub.named_steps["cnn"]
                        if hasattr(cnn, "get_diagnostics"):
                            diags["cnn_info"] = cnn.get_diagnostics()
                            logger.info("Diagnostic CNN: %s", diags["cnn_info"])
                    else:
                        logger.debug("Pas de CNN dans le transformeur '%s' — ignoré pour le diagnostic.", name)
        except Exception as e:
            logger.warning("Impossible d'extraire le diagnostic CNN: %s", e)

    # 3) Sauvegarder
        diag_path = os.path.join(outdir, f"diagnostics_{kind}.json")
        with open(diag_path, "w") as f:
            json.dump(diags, f, indent=2)
        logger.info("Diagnostics sauvegardés: %s", diag_path)

def diagnostic_baseline(
    pipe: Union[SkPipeline, ImbPipeline],
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    outdir: str = "results"
) -> Dict[str, Any]:
    """
    Effectue des diagnostics sur le pipeline.
    
    Args:
        pipe: Pipeline entraîné (sklearn ou imblearn)
        X_sample: Échantillon de données
        y_sample: Labels correspondants
        outdir: Dossier de sortie des résultats
        
    Returns:
        Dict contenant les métriques de diagnostic
    """
    logger.info("Exécution diagnostics pipeline...")
    
    diagnostics = {}
    
    try:
        # 1. Vérification réduction dimension
        if hasattr(pipe, "named_steps"):
            for step_name, step in pipe.named_steps.items():
                if hasattr(step, "explained_variance_ratio_"):
                    var_ratio = step.explained_variance_ratio_.sum()
                    diagnostics[f"{step_name}_variance_explained"] = var_ratio
                    logger.info(f"Variance expliquée {step_name}: {var_ratio:.2%}")

        # 1. Vérification réduction dimension SVD
        if 'svd' in pipe.named_steps:
            svd = pipe.named_steps['svd']
            var_ratio = svd.explained_variance_ratio_.sum()
            diagnostics["variance_explained"] = float(var_ratio)
            logger.info(f"Variance expliquée SVD: {var_ratio:.2%}")
        
        # 2. Test prédictions
        y_pred = pipe.predict(X_sample)
        unique_classes = np.unique(y_pred)
        diagnostics["n_predicted_classes"] = len(unique_classes)
        diagnostics["predicted_classes"] = unique_classes.tolist()
        logger.info(f"Classes uniques prédites: {unique_classes}")
        
        # 3. Vérification probabilités
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_sample)
            max_prob = float(probs.max())
            mean_prob = float(probs.mean())
            diagnostics["max_probability"] = max_prob
            diagnostics["mean_probability"] = mean_prob
            logger.info(f"Probabilité max: {max_prob:.3f}, moyenne: {mean_prob:.3f}")
            
        # 4. Calcul F1-score sur échantillon
        if y_sample is not None:
            f1_macro = f1_score(y_sample, y_pred, average='macro')
            diagnostics["sample_f1_macro"] = float(f1_macro)
            logger.info(f"F1-macro sur échantillon: {f1_macro:.4f}")
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Erreur durant diagnostic: {e}")
        return {"error": str(e)}

# === Construire la branche CNN depuis la config =================================
# (avec post-réduction optionnelle)    
def create_cnn_branch_from_cfg(images_cfg: dict) -> SkPipeline:
    """
    Construire la branche CNN (embedding ResNet) depuis [images.cnn] du TOML.
    Support : post-réduction optionnelle (TruncatedSVD) + normalisation L2.
    """
    cnn_cfg = images_cfg.get("cnn", {}) or {}
    if not bool(cnn_cfg.get("enabled", False)):
        raise ValueError("CNN demandée mais [images.cnn.enabled] est false dans le TOML.")

    image_dir = images_cfg["train_dir"]
    featurizer = CNNFeaturizer(
        image_dir=image_dir,
        arch=str(cnn_cfg.get("arch", "resnet50")),
        batch_size=int(cnn_cfg.get("batch_size", 16)),
        device=str(cnn_cfg.get("device", "auto")),
        use_imagenet_norm=bool(cnn_cfg.get("use_imagenet_norm", True)),
        fallback_zero=bool(cnn_cfg.get("fallback_zero", True)),
        dtype=str(cnn_cfg.get("dtype", "float32")),
    )

    steps = [("cnn", featurizer)]

    dr = cnn_cfg.get("dim_reduction", {}) or {}
    if bool(dr.get("enabled", False)):
        n_comp = int(dr.get("n_components", 256))
        rs = int(dr.get("random_state", 42))
        steps += [
            ("svd", TruncatedSVD(n_components=n_comp, random_state=rs)),
            ("l2norm", Normalizer(copy=False)),
        ]
    else:
        steps += [("l2norm", Normalizer(copy=False))]

    return SkPipeline(steps)


# === Construire la pipeline multimodale complète ================================
def create_combined_pipeline(cfg: dict, under_strategy: dict, over_strategy: dict, seed: int):
    """Construire la pipeline texte+image, rééchantillonner, scaler, et ajouter le modèle final."""
    # Utiliser la fabrique qui lit directement toutes les options depuis [text]
    text_branch = create_text_pipeline_from_cfg(cfg.get("text", {}))

    # --- NEW: SVD global sur la branche texte (optionnel via [text.svd]) ---
    svd_cfg = (cfg.get("text", {}).get("svd", {}) or {})
    if bool(svd_cfg.get("enabled", True)):
        n_comp = int(svd_cfg.get("n_components", 600))   # 400–600 typiquement
        rs     = int(svd_cfg.get("random_state", seed))
        l2norm = bool(svd_cfg.get("l2norm", True))
        from sklearn.pipeline import Pipeline as _SkPipe
        steps_txt = [("to32", ToFloat32()),
                     ("svd", TruncatedSVD(n_components=n_comp, random_state=rs))]
        if l2norm:
            steps_txt.append(("l2", Normalizer(copy=False)))
        text_branch = _SkPipe([("text", text_branch), *steps_txt])

    # Construire la branche IMAGES (pixels) depuis le TOML
    image_pixels = create_image_pipeline_from_cfg(cfg["images"], use_test_dir=False)

    # On garde image_train_dir pour la branche "stats" (si activée)
    image_train_dir = cfg["images"]["train_dir"]

    # Ajouter éventuellement la branche IMAGES (stats)
    transformers = [("text", text_branch), ("image_pixels", image_pixels)]
    # Ajouter éventuellement la branche CNN
    try:
        if bool(cfg.get("images", {}).get("cnn", {}).get("enabled", False)):
            image_cnn = create_cnn_branch_from_cfg(cfg["images"])
            transformers.append(("image_cnn", image_cnn))
    except Exception as e:
        logger.warning("CNN désactivée (raison: %s)", e)
        
    # Ajouter la branche IMAGES (stats combinées)
    stats_c = cfg.get("images", {}).get("stats_combined", {})
    if bool(stats_c.get("enabled", False)):
        transformers.append(("image_stats_combined", ImageStatsCombinedFeaturizer(
            image_dir=image_train_dir,
            imgid_col="imageid", pid_col="productid",
            white_threshold=int(stats_c.get("white_threshold", 230)),
            black_threshold=int(stats_c.get("black_threshold", 25)),
            min_area=int(stats_c.get("min_area", 16)),
            prefix_basic=str(stats_c.get("prefix_basic", "img_")),
            prefix_pro=str(stats_c.get("prefix_pro", "pro_")),
        )))

    # Fusionner les branches
    fusion_cfg = (cfg.get("fusion", {}) or {}).get("weights", None)
    present = {name for name, _ in transformers}
    fusion_weights = {k: v for k, v in (fusion_cfg or {}).items() if k in present} or None

    # 1) union SANS memory
    union = FeatureUnion(
        transformer_list=transformers,
        transformer_weights=fusion_weights,
        n_jobs=1                          # <— clé pour la stabilité mémoire pendant fit
    )
    # Construire le classifieur final
    model = build_classifier(cfg, seed)

    # 2) enveloppe AVEC memory
    cache = get_cache(cfg)
    if cache is not None and hasattr(cache, "location"):
        logger.info(f"Cache sklearn activé: {cache.location}")


    # Construire la pipeline Imbalanced-Learn
    pipe = ImbPipeline(steps=[
        ("features", union),
        ("under", AdaptiveUnderSampler(cap_dict=under_strategy, random_state=seed)),
        ("over", RandomOverSampler(sampling_strategy=over_strategy, random_state=seed)),
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model),
    ], memory=cache)
    return pipe


# === Entraîner sur le train, prédire sur le test ================================
def train_and_predict_on_test(X_train, y_train, X_test, cfg: dict):
    """Entraîner la pipeline complète sur X_train/Y_train puis prédire y_test (labels) pour X_test."""
    # Lire les seeds
    seed = int(cfg.get("random", {}).get("seed", 42))

    # Construire les stratégies d'échantillonnage à partir de y_train
    under, over = make_sampling_strategies(
        y_train,
        major_class=cfg["sampling"]["major_class"],
        major_cap=cfg["sampling"]["major_cap"],
        tail_min=cfg["sampling"]["tail_min"],
    )

    # Construire la pipeline complète
    seed = int(cfg.get("random", {}).get("seed", 42))
    pipe = create_combined_pipeline(cfg, under, over, seed)

    # Entraîner la pipeline
    print(">> Entraîner la pipeline complète…")
    pipe.fit(X_train, y_train)

    # Re-pointer les dossiers images vers TEST pour l'inférence
    print(">> Re-pointer les lecteurs d'images vers le dossier TEST…")
    features_step = pipe.named_steps["features"]  # peut être une FeatureUnion OU un Pipeline
    # Si c’est un Pipeline avec la union dedans, aller chercher la union
    if hasattr(features_step, "named_steps") and "union" in features_step.named_steps:
        feat_union = features_step.named_steps["union"]
    else:
        feat_union = features_step  # c’est la FeatureUnion directement

    image_test_dir = cfg["images"]["test_dir"]

    # Parcourir les transformeurs et mettre à jour les répertoires d'images
    new_list = []
    for name, sub in feat_union.transformer_list:
        if name == "image_pixels":
            # sub est une sklearn.Pipeline (loader → transformations)
            if "loader" in sub.named_steps:
                loader = sub.named_steps["loader"]
                if hasattr(loader, "set_image_dir"):
                    loader.set_image_dir(image_test_dir)
                elif hasattr(loader, "image_dir"):
                    loader.image_dir = image_test_dir
                else:
                    raise RuntimeError("ImageLoader n'avoir ni 'set_image_dir' ni attribut 'image_dir'.")
            new_list.append((name, sub))

        elif name == "image_stats_combined":   # NEW
            if hasattr(sub, "set_image_dir"):
                sub.set_image_dir(image_test_dir)
            else:
                sub.image_dir = image_test_dir
            new_list.append((name, sub))
    
        elif name == "image_cnn":
            # sub est une sklearn.Pipeline ; le featurizer est 'cnn'
            if "cnn" in sub.named_steps:
                cnn = sub.named_steps["cnn"]
                if hasattr(cnn, "set_image_dir"):
                    cnn.set_image_dir(image_test_dir)
                else:
                    cnn.image_dir = image_test_dir
            new_list.append((name, sub))
        else:
            new_list.append((name, sub))
    feat_union.transformer_list = new_list

    # Prédire sur X_test
    print(">> Prédire sur X_test…")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This Pipeline instance is not fitted yet",
            category=FutureWarning,
            module=r"sklearn\.pipeline",
        )
        y_pred = pipe.predict(X_test)

    return pipe, y_pred


# === Comparer LR et SVC en CV stratifiée =======================================
def compare_models_cv(X_train, y_train, cfg: dict):
    """Comparer LR et LinearSVC avec la même pipeline (sauf le classifieur), en CV F1-macro."""
    seed = int(cfg.get("random", {}).get("seed", 42))
    n_jobs = int(cfg.get("compute", {}).get("n_jobs", 1))
    cv_splits = int(cfg.get("cv", {}).get("splits", 3))
    shuffle = bool(cfg.get("cv", {}).get("shuffle", True))
    cv_seed = int(cfg.get("cv", {}).get("random_state", seed))

    # Construire les stratégies under/over à partir de y_train
    under, over = make_sampling_strategies(
        y_train,
        major_class=cfg["sampling"]["major_class"],
        major_cap=cfg["sampling"]["major_cap"],
        tail_min=cfg["sampling"]["tail_min"],
    )

    # Définir la CV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=shuffle, random_state=cv_seed)

    # Évaluer les deux modèles
    rows = []
    for name in ["lr", "svc"]:
        cfg_local = {**cfg, "model": {**cfg.get("model", {}), "name": name}}
        pipe = create_combined_pipeline(cfg_local, under, over, seed)
        scores = cross_val_score(
            pipe, X_train, y_train,
            scoring="f1_macro", cv=cv, n_jobs=n_jobs
        )
        print(f"   - {name.upper()} | F1-macro = {scores.mean():.4f} ± {scores.std():.4f} | {scores}")
        rows.append({
            "model": name.upper(),
            "cv_mean_f1_macro": scores.mean(),
            "cv_std": scores.std(),
            "cv_scores": scores.tolist(),
        })

    return pd.DataFrame(rows)


# === Parser les arguments CLI ===================================================
def parse_args():
    """Définir les arguments CLI et retourner l'objet Namespace."""
    p = argparse.ArgumentParser(description="Entraîner la pipeline texte+image avec rééchantillonnage ; comparer LR vs SVC en option")
    p.add_argument("--config", default=str(DEFAULT_CFG), help="Chemin vers le TOML (défaut: features/config.toml)")
    p.add_argument("--compare", action="store_true", help="Comparer LR vs SVC via CV sur X_train (F1-macro)")
    p.add_argument("--baseline", choices=["b0", "b1", "b2", "b3", "b4"], help="Exécuter une baseline simple et sortir")
    p.add_argument("--model", choices=["lr", "svc"], default=None, help="Forcer le modèle (écraser [model].name dans le TOML)")
    p.add_argument("--compare-all", action="store_true", 
                  help="Compare tous les modèles (B0-B4) et génère des graphiques")
    return p.parse_args()


# === Fonction principale ========================================================
def main():
    # Lire les arguments et la configuration
    args = parse_args()
    cfg = load_config(args.config) # Charger le TOML
    validate_config(cfg)
    # Si --model est fourni, écraser le nom du modèle de la config
    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    if args.compare_all:
        from tools.compare_models import compare_all_models
        compare_all_models()
        return
    setup_logging(log_dir=cfg.get("outputs", {}).get("log_dir", "results/logs"))
    logger.info("Configuration chargée.")
    # Initialiser les graines pour la reproductibilité
    seed = int(cfg.get("random", {}).get("seed", 42))
    init_seeds(seed)

    # Charger les chemins de données depuis le TOML
    print(">> Charger les données…")
    X_train = pd.read_csv(cfg["paths"]["x_train_csv"], index_col=0)
    y_train = pd.read_csv(cfg["paths"]["y_train_csv"], index_col=0).squeeze()
    X_test = pd.read_csv(cfg["paths"]["x_test_csv"], index_col=0)

    # Option de downsampling via variable d'environnement
    max_n = int(os.environ.get("RAKUTEN_MAX_N", "0"))
    if max_n > 0:
        X_train = X_train.iloc[:max_n].copy()
        y_train = y_train.iloc[:max_n].copy()
        logger.info(f"Downsampling activé: RAKUTEN_MAX_N={max_n} -> X_train={len(X_train)} y_train={len(y_train)}")
         
    # Vérifier les colonnes nécessaires
    needed = ["designation", "description", "productid", "imageid"]
    for col in needed:
        if col not in X_train.columns:
            raise ValueError(f"Colonne manquante dans X_train : '{col}'")
        if col not in X_test.columns:
            raise ValueError(f"Colonne manquante dans X_test : '{col}'")

    # Exécuter une baseline et sortir si demandé
    if args.baseline:
        run_baseline_and_report(args.baseline, X_train, y_train, cfg, outdir="results")
        return

    # Comparer LR vs SVC en CV si demandé
    if args.compare:
        print(">> Comparer LR vs SVC (CV F1-macro)…")
        df_cmp = compare_models_cv(X_train[needed], y_train, cfg)
        out_cmp = cfg["outputs"]["compare_out"]
        os.makedirs(os.path.dirname(out_cmp), exist_ok=True)
        df_cmp.to_csv(out_cmp, index=False)
        print(f">> Sauvegarder les résultats de comparaison : {out_cmp}")
        return

    # Entraîner la pipeline complète et prédire sur X_test
    pipe, y_pred = train_and_predict_on_test(X_train[needed], y_train, X_test[needed], cfg)

    # Sauvegarder le modèle et les prédictions
    os.makedirs(os.path.dirname(cfg["outputs"]["model_out"]), exist_ok=True)
    joblib.dump(pipe, cfg["outputs"]["model_out"])
    print(f">> Sauvegarder le modèle : {cfg['outputs']['model_out']}")

    pred_df = pd.DataFrame(y_pred, index=X_test.index, columns=["predicted_label"])
    pred_df.to_csv(cfg["outputs"]["pred_out"])
    print(f">> Sauvegarder les prédictions : {cfg['outputs']['pred_out']}")


# === Entrée script ==============================================================
if __name__ == "__main__":
    main()

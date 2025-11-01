
# models/image_pipeline.py
# =======================================================
# Pipeline images : charger -> aplatir -> (réduire) -> normaliser
# =======================================================
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, Union, Iterable
import numpy as np
import logging

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, Normalizer
from sklearn.decomposition import PCA, TruncatedSVD
from src.features.image.loader import ImageLoader

from src.utils.profiling import profile_func

logger = logging.getLogger(__name__)


# ---------- Helpers picklables (nécessaires aux FunctionTransformer) ----------
@profile_func
def _to_float32(X: np.ndarray) -> np.ndarray:
    """Convertir en float32 (éviter les copies inutiles)."""
    return X.astype(np.float32, copy=False)

@profile_func
def _flatten_images(X: np.ndarray) -> np.ndarray:
    """Aplatir (n, H, W, C) -> (n, H*W*C). Lever une erreur si dimensions inattendues."""
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"Attendre un tenseur 4D (n,H,W,C) ; reçu shape={X.shape}")
    return X.reshape((X.shape[0], -1))


# ---------- Recherche récursive d’un réducteur (PCA / TruncatedSVD) ----------
@profile_func
def _iter_pipeline_steps(obj: Any) -> Iterable[tuple[str, Any]]:
    """Itérer sur les (name, step) d’un objet type Pipeline/FeatureUnion (récursif)."""
    # Pipeline sklearn
    if hasattr(obj, "steps"):
        for name, step in obj.steps:
            yield name, step
            yield from _iter_pipeline_steps(step)
    # FeatureUnion / ColumnTransformer
    if hasattr(obj, "transformer_list"):
        for name, step in obj.transformer_list:
            yield name, step
            yield from _iter_pipeline_steps(step)
    # Composite (OneVsRest, etc.) -> essayer d’accéder à l’estimateur interne
    for attr in ("estimator", "classifier", "regressor", "pipeline", "base_estimator"):
        if hasattr(obj, attr):
            inner = getattr(obj, attr)
            if inner is not obj:
                yield from _iter_pipeline_steps(inner)

@profile_func
def _find_reducer(obj: Any) -> Optional[tuple[str, Union[PCA, TruncatedSVD]]]:
    """Trouver la première étape PCA/TruncatedSVD rencontrée (récursif)."""
    for name, step in _iter_pipeline_steps(obj):
        if isinstance(step, (PCA, TruncatedSVD)):
            return name, step
    return None


# ---------- Fabriques de pipelines ----------
@profile_func
def create_image_pipeline(
    image_dir: str,
    image_size: Tuple[int, int] = (128, 128),
    dim_reduction: Optional[Dict[str, Any]] = None,
    memory: Optional[str] = None,
) -> SkPipeline:
    """
    Construire une pipeline image *générique* prête pour un classifieur linéaire.

    Étapes :
      1) Charger les images depuis `image_dir` (le DataFrame d’entrée doit contenir
         **productid** ET **imageid** ; ne pas filtrer en amont).
      2) Convertir en float32 (si besoin).
      3) Aplatir en vecteurs.
      4) Réduire la dimension si demandé (PCA dense ou TruncatedSVD).
      5) Normaliser / standardiser pour stabiliser l’entraînement.

    Args:
        image_dir: Dossier racine des images.
        image_size: Taille cible (H, W). (utiliser `create_image_pipeline_from_cfg`
                    pour lire cette valeur directement depuis le TOML)
        dim_reduction: Dictionnaire optionnel :
            {
              "enabled": true/false,
              "method": "pca" | "truncated_svd",
              "n_components": int,
              "random_state": 42
            }
        memory: Chemin de cache joblib (optionnel).

    Returns:
        Pipeline sklearn.
    """
    cfg = dim_reduction or {}
    enabled = bool(cfg.get("enabled", False))
    method = str(cfg.get("method", "pca")).lower()
    n_comp = int(cfg.get("n_components", 150))
    rs = int(cfg.get("random_state", 42))

    steps = [
        # Ne pas “sélectionner productid uniquement” : l’ImageLoader
        #  a besoin de productid ET imageid pour résoudre les chemins fichiers.
        ("loader", ImageLoader(image_dir=image_dir, image_size=image_size)),
        ("to_float", FunctionTransformer(_to_float32, accept_sparse=False)),
        ("flatten", FunctionTransformer(_flatten_images, accept_sparse=False)),
    ]

    if enabled:
        if method in ("svd", "truncated_svd"):
            # -> Réduction SVD (accepter dense/sparse) + normalisation L2
            steps += [
                ("svd", TruncatedSVD(n_components=n_comp, random_state=rs)),
                ("l2norm", Normalizer(copy=False)),
            ]
            logger.info("Réduction: TruncatedSVD (n_components=%d, rs=%d) + L2 norm", n_comp, rs)

        elif method == "pca":
            # -> PCA dense : centrer/standardiser + whitener
            steps += [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pca", PCA(
                    n_components=n_comp,
                    svd_solver="randomized",
                    whiten=True,
                    random_state=rs,
                )),
            ]
            logger.info("Réduction: PCA (n_components=%d, whiten=True, rs=%d)", n_comp, rs)

        else:
            # Méthode inconnue -> pas de réduction, normaliser légèrement
            steps += [("scaler", StandardScaler(with_mean=False))]
            logger.warning("Méthode de réduction inconnue '%s' -> pas de réduction.", method)
    else:
        # Pas de réduction -> standardiser (sans centrage pour limiter le coût)
        steps += [("scaler", StandardScaler(with_mean=False))]
        logger.info("Réduction désactivée -> standardiser (with_mean=False).")

    return SkPipeline(steps=steps, memory=memory)

@profile_func
def create_image_pipeline_from_cfg(
    images_cfg: Dict[str, Any],
    *,
    use_test_dir: bool = False,
    memory: Optional[str] = None
) -> SkPipeline:
    """
    Construire une pipeline image **à partir de la section [images] du TOML**.

    Le TOML attendu ressemble à :
        [images]
        size = [128, 128]
        train_dir = "data/images/images/image_train"
        test_dir  = "data/images/images/image_test"
        [images.dim_reduction]
        enabled = true
        method = "pca"            # ou "truncated_svd"
        n_components = 150
        random_state = 42

    Args:
        images_cfg: Dictionnaire `cfg["images"]`.
        use_test_dir: Utiliser `test_dir` au lieu de `train_dir` (prédiction).
        memory: Chemin de cache joblib (optionnel).

    Returns:
        Pipeline sklearn.
    """
    size = images_cfg.get("size", [128, 128])
    img_dir_key = "test_dir" if use_test_dir else "train_dir"
    img_dir = images_cfg.get(img_dir_key)
    if not img_dir:
        raise ValueError(f"Clé '{img_dir_key}' absente de [images] dans le TOML.")

    dr_cfg = images_cfg.get("dim_reduction", {}) or {}
    pipe = create_image_pipeline(
        image_dir=img_dir,
        image_size=size,
        dim_reduction=dr_cfg,
        memory=memory,
    )
    logger.info("Pipeline image créée depuis TOML (%s) avec size=%s", img_dir_key, size)
    return pipe


# ---------- Diagnostic (réduction / compression) ----------
@profile_func
def diagnostic_reduction(pipe: SkPipeline) -> Dict[str, Any]:
    """
    Calculer des métriques de réduction **après fit** :
      - type de réducteur (PCA/SVD)
      - nombre de composantes retenues
      - somme des ratios de variance expliquée (si disponible)
      - niveau dans la pipeline où se trouve le réducteur

    Args:
        pipe: Pipeline (déjà entraînée).

    Returns:
        dict avec clés : reducer_type, reducer_name, n_components, explained_variance_ratio_sum (optionnel).
    """
    out: Dict[str, Any] = {}
    found = _find_reducer(pipe)
    if not found:
        out["reducer_type"] = None
        out["message"] = "Aucun réducteur trouvé dans la pipeline."
        return out

    name, reducer = found
    out["reducer_name"] = name
    out["reducer_type"] = type(reducer).__name__
    # n_components_: attribut PCA et TruncatedSVD (une fois fit)
    n_comp = getattr(reducer, "n_components_", None) or getattr(reducer, "n_components", None)
    out["n_components"] = int(n_comp) if n_comp is not None else None

    # explained_variance_ratio_ : dispo pour PCA et TruncatedSVD
    evr = getattr(reducer, "explained_variance_ratio_", None)
    if evr is not None:
        try:
            out["explained_variance_ratio_sum"] = float(np.sum(evr))
        except Exception:
            pass

    return out
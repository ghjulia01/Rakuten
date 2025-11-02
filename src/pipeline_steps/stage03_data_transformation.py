"""
Etape 3 : Transformation des donnees (Data Transformation).
==========================================================

Cette etape applique le reechantillonnage et construit les features.

Responsabilites :
- Reechantillonnage (under/over sampling)
- Construction des pipelines de features (texte + image + CNN)
- Fusion ponderee des branches
- Transformation des donnees train et test

Utilisation:
    from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
    
    pipeline = DataTransformationPipeline(config)
    X_train_t, y_train_t, X_test_t, feature_pipeline = pipeline.run(
        X_train, y_train, X_test
    )

"""
from __future__ import annotations

import logging
from typing import Tuple, Any

import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion

from src.data.sampling import apply_sampling
from src.pipelines.text_pipeline import create_text_pipeline_from_cfg
from src.pipelines.image_pipeline import create_image_pipeline_from_cfg
from src.features.cnn_features import CNNFeaturizer
from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class DataTransformationPipeline:
    """
    Pipeline de transformation des donnees.
    
    Applique le reechantillonnage et construit les features texte/image/CNN.
    
    Attributes:
        config: Configuration complete du projet
        feature_pipeline: Pipeline sklearn de features (apres fit)
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> pipeline = DataTransformationPipeline(config)
        >>> X_train_t, y_train_t, X_test_t, pipe = pipeline.run(
        ...     X_train, y_train, X_test
        ... )
    """
    
    def __init__(self, config):
        """
        Initialise le pipeline de transformation.
        
        Args:
            config: Objet Config contenant tous les parametres
        """
        self.config = config
        self.feature_pipeline = None
        
        logger.info("=" * 70)
        logger.info("ETAPE 3 : TRANSFORMATION DES DONNEES")
        logger.info("=" * 70)
    
    def apply_resampling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applique le reechantillonnage pour equilibrer les classes.
        
        Args:
            X_train: Features d'entrainement
            y_train: Labels d'entrainement
            
        Returns:
            Tuple (X_train_resampled, y_train_resampled)
        """
        logger.info("\n--- Reechantillonnage ---")
        
        # Parametres depuis la config
        sampling_config = self.config.sampling
        
        major_class = sampling_config.get("major_class", 2583)
        major_cap = sampling_config.get("major_cap", 2500)
        tail_min = sampling_config.get("tail_min", 1500)
        random_state = self.config.random_seed
        
        logger.info(f"Parametres:")
        logger.info(f"  - Classe majoritaire: {major_class}")
        logger.info(f"  - Cap majorite: {major_cap}")
        logger.info(f"  - Min minorite: {tail_min}")
        
        # Appliquer le sampling
        X_resampled, y_resampled = apply_sampling(
            X=X_train,
            y=y_train,
            major_class=major_class,
            major_cap=major_cap,
            tail_min=tail_min,
            random_state=random_state
        )
        
        logger.info(f"[OK] Reechantillonnage termine: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def build_feature_pipeline(self) -> FeatureUnion:
        """
        Construit le pipeline de features (texte + image + CNN).
        
        Returns:
            FeatureUnion sklearn combinant toutes les branches activees
        """
        logger.info("\n--- Construction du pipeline de features ---")
        
        transformers = []
        weights = {}
        
        # ========================================
        # Branche Texte
        # ========================================
        logger.info("Construction de la branche texte...")
        text_pipeline = create_text_pipeline_from_cfg(self.config.text)
        transformers.append(("text", text_pipeline))
        
        # Poids de la branche texte
        text_weight = self.config.get("fusion.weights.text", 1.0)
        weights["text"] = float(text_weight)
        logger.info(f"  Poids texte: {text_weight}")
        
        # ========================================
        # Branche Image (pixels, stats, etc.)
        # ========================================
        # Verifier si les images traditionnelles sont activees
        image_pixels_enabled = self.config.get("features.image.pixels.enabled", False)
        image_stats_enabled = self.config.get("features.image.stats.enabled", False)
        
        if image_pixels_enabled or image_stats_enabled:
            logger.info("Construction de la branche image (pixels/stats)...")
            image_pipeline = create_image_pipeline_from_cfg(
                self.config.images,
                use_test_dir=False  # Train d'abord
            )
            transformers.append(("image_pixels", image_pipeline))
            
            # Poids
            image_pixels_weight = self.config.get("fusion.weights.image_pixels", 0.5)
            weights["image_pixels"] = float(image_pixels_weight)
            logger.info(f"  Poids image_pixels: {image_pixels_weight}")
        
        # ========================================
        # Branche CNN ResNet/ViT
        # ========================================
        cnn_enabled = self.config.get("features.image.cnn.enabled", False)
        
        if cnn_enabled:
            logger.info("Construction de la branche CNN...")
            
            # Importer CNNFeaturizer
            try:
                from src.features.cnn_features import CNNFeaturizer
            except ImportError as e:
                logger.error(f"Impossible d'importer CNNFeaturizer: {e}")
                logger.warning("CNN desactive - continuons sans CNN")
                cnn_enabled = False
            
            if cnn_enabled:
                # Parametres CNN depuis config
                cnn_config = self.config.get("features.image.cnn", {})
                
                # Determiner le chemin des images
                image_dir = self.config.get("paths.image_train_dir", 
                                           self.config.get("images.train_dir", 
                                                          "data/images/images/image_train"))
                
                logger.info(f"  Architecture: {cnn_config.get('arch', 'resnet50')}")
                logger.info(f"  Batch size: {cnn_config.get('batch_size', 16)}")
                logger.info(f"  Device: {cnn_config.get('device', 'auto')}")
                logger.info(f"  Image dir: {image_dir}")
                
                # Creer le CNNFeaturizer
                cnn_transformer = CNNFeaturizer(
                    image_dir=image_dir,
                    arch=cnn_config.get("arch", "resnet50"),
                    batch_size=cnn_config.get("batch_size", 16),
                    device=cnn_config.get("device", "auto"),
                    use_imagenet_norm=cnn_config.get("use_imagenet_norm", True),
                    fallback_zero=cnn_config.get("fallback_zero", True),
                    dtype=cnn_config.get("dtype", "float32"),
                    num_workers=cnn_config.get("num_workers", 0),
                    
                    # Fine-tuning
                    finetune_epochs=cnn_config.get("finetune_epochs", 0),
                    finetune_lr=cnn_config.get("finetune_lr", 0.0003),
                    finetune_weight_decay=cnn_config.get("finetune_weight_decay", 0.01),
                    finetune_max_n=cnn_config.get("finetune_max_n", 8000),
                    trainable_last_n=cnn_config.get("trainable_last_n", 0),
                    ft_patience=cnn_config.get("ft_patience", 3),
                    label_smoothing=cnn_config.get("label_smoothing", 0.0),
                    
                    # Augmentation
                    aug_hflip_p=cnn_config.get("aug_hflip_p", 0.2),
                    aug_color_jitter=cnn_config.get("aug_color_jitter", 0.0),
                    random_resized_crop_scale=cnn_config.get("random_resized_crop_scale", [0.9, 1.0]),
                    random_resized_crop_ratio=cnn_config.get("random_resized_crop_ratio", [0.95, 1.05]),
                    mixup_alpha=cnn_config.get("mixup_alpha", 0.0),
                    cutmix_alpha=cnn_config.get("cutmix_alpha", 0.0),
                    
                    # Grad-CAM
                    save_head_path=cnn_config.get("save_head_path", "artifacts/head_ft.pt"),
                    save_head_normalize=cnn_config.get("save_head_normalize", True),
                    
                    # Optimisation
                    foreach=cnn_config.get("foreach", True),
                )
                
                # Ajouter au pipeline
                transformers.append(("image_cnn", cnn_transformer))
                
                # Poids
                cnn_weight = self.config.get("fusion.weights.image_cnn", 1.0)
                weights["image_cnn"] = float(cnn_weight)
                logger.info(f"  Poids image_cnn: {cnn_weight}")
                
                # SVD post-CNN (optionnel)
                svd_enabled = self.config.get("features.image.cnn.svd.enabled", False)
                if svd_enabled:
                    logger.info("  SVD post-CNN active")
        
        # ========================================
        # Branche ViT (optionnel)
        # ========================================
        vit_enabled = self.config.get("features.image.vit.enabled", False)
        
        if vit_enabled:
            logger.info("Construction de la branche ViT...")
            logger.warning("ViT non encore implemente dans ce fichier")
            # TODO: Ajouter support ViT si necessaire
        
        # ========================================
        # Creer le FeatureUnion
        # ========================================
        if not transformers:
            raise ValueError("Aucune branche de features activee ! "
                           "Verifier la configuration.")
        
        feature_pipeline = FeatureUnion(
            transformers,
            transformer_weights=weights if weights else None
        )
        
        logger.info(f"[OK] Pipeline cree avec {len(transformers)} branche(s)")
        logger.info(f"  Branches: {[name for name, _ in transformers]}")
        
        return feature_pipeline
    
    def transform_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        feature_pipeline: FeatureUnion
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforme les donnees avec le pipeline de features.
        
        Args:
            X_train: Features d'entrainement (brutes)
            y_train: Labels d'entrainement  
            X_test: Features de test (brutes)
            feature_pipeline: Pipeline sklearn a appliquer
            
        Returns:
            Tuple (X_train_transformed, X_test_transformed)
        """
        logger.info("\n--- Transformation des donnees ---")
        logger.info(f"[INFO] Train: {len(X_train)} echantillons a transformer")
        logger.info(f"[INFO] Test: {len(X_test)} echantillons a transformer")
        
        # ========================================
        # Fit + Transform sur train
        # ========================================
        logger.info("\n[1/2] Fit + Transform sur TRAIN...")
        with Timer("Fit + Transform sur train"):
            X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
        
        logger.info(f"[OK] Train transforme: {X_train_transformed.shape}")
        logger.info(f"  Type: {type(X_train_transformed)}")
        logger.info(f"  Progression: {len(X_train)}/{len(X_train)} échantillons traités ✓")
        
        # Afficher les stats si c'est sparse
        if hasattr(X_train_transformed, 'nnz'):
            density = X_train_transformed.nnz / np.prod(X_train_transformed.shape)
            logger.info(f"  Densite: {density:.4f} ({density*100:.2f}%)")
        
        # ========================================
        # Transform sur test
        # ========================================
        logger.info("\n[2/2] Transform sur TEST...")
        with Timer("Transform sur test"):
            X_test_transformed = feature_pipeline.transform(X_test)
        
        logger.info(f"[OK] Test transforme: {X_test_transformed.shape}")
        
        return X_train_transformed, X_test_transformed
    
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Series, np.ndarray, FeatureUnion]:
        """
        Execute le pipeline de transformation complet.
        
        Args:
            X_train: Features d'entrainement (brutes)
            y_train: Labels d'entrainement
            X_test: Features de test (brutes)
            
        Returns:
            Tuple (X_train_transformed, y_train_resampled, 
                   X_test_transformed, feature_pipeline)
        """
        with Timer("Transformation des donnees"):
            
            # ========================================
            # 1. Reechantillonnage
            # ========================================
            X_train_resampled, y_train_resampled = self.apply_resampling(
                X_train, y_train
            )
            
            # ========================================
            # 2. Construction du pipeline
            # ========================================
            self.feature_pipeline = self.build_feature_pipeline()
            
            # ========================================
            # 3. Transformation
            # ========================================
            X_train_transformed, X_test_transformed = self.transform_data(
                X_train_resampled,
                y_train_resampled,
                X_test,
                self.feature_pipeline
            )
            
            # ========================================
            # 4. Resume final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RESUME DE LA TRANSFORMATION")
            logger.info("=" * 70)
            logger.info(f"[INFO] X_train : {X_train.shape} -> {X_train_transformed.shape}")
            logger.info(f"[INFO] y_train : {y_train.shape} -> {y_train_resampled.shape}")
            logger.info(f"[INFO] X_test  : {X_test.shape} -> {X_test_transformed.shape}")
            logger.info(f"[INFO] Pipeline sauvegarde : {self.feature_pipeline is not None}")
            logger.info("=" * 70 + "\n")
            
            return (
                X_train_transformed,
                y_train_resampled,
                X_test_transformed,
                self.feature_pipeline
            )


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de DataTransformationPipeline")
    print("="*70 + "\n")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Stage 1: Ingestion
        stage1 = DataIngestionPipeline(config)
        X_train, y_train, X_test = stage1.run()
        
        # Stage 2: Validation
        stage2 = DataValidationPipeline(config)
        validation_ok = stage2.run(X_train, y_train, X_test)
        
        if not validation_ok:
            print("\n[ERROR] Validation echouee - arret du pipeline")
        else:
            # Stage 3: Transformation
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, pipeline = stage3.run(
                X_train, y_train, X_test
            )
            
            print("\n[OK] Transformation terminee avec succes!")
            print(f"  X_train transforme: {X_train_t.shape}")
            print(f"  y_train reechantillonne: {y_train_t.shape}")
            print(f"  X_test transforme: {X_test_t.shape}")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Erreur: {e}")
        print("Verification que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n[ERROR] Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
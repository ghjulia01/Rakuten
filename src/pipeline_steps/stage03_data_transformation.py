"""
Étape 3 : Transformation des données (Data Transformation).
==========================================================

Cette étape applique le rééchantillonnage et construit les features.

Responsabilités :
- Rééchantillonnage (under/over sampling)
- Construction des pipelines de features (texte + image)
- Fusion pondérée des branches
- Transformation des données train et test

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
from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class DataTransformationPipeline:
    """
    Pipeline de transformation des données.
    
    Applique le rééchantillonnage et construit les features texte/image.
    
    Attributes:
        config: Configuration complète du projet
        feature_pipeline: Pipeline sklearn de features (après fit)
        
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
            config: Objet Config contenant tous les paramètres
        """
        self.config = config
        self.feature_pipeline = None
        
        logger.info("=" * 70)
        logger.info("ÉTAPE 3 : TRANSFORMATION DES DONNÉES")
        logger.info("=" * 70)
    
    def apply_resampling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applique le rééchantillonnage pour équilibrer les classes.
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            
        Returns:
            Tuple (X_train_resampled, y_train_resampled)
        """
        logger.info("\n--- Rééchantillonnage ---")
        
        # Paramètres depuis la config
        sampling_config = self.config.sampling
        
        major_class = sampling_config.get("major_class", 2583)
        major_cap = sampling_config.get("major_cap", 2500)
        tail_min = sampling_config.get("tail_min", 1500)
        random_state = self.config.random_seed
        
        logger.info(f"Paramètres:")
        logger.info(f"  - Classe majoritaire: {major_class}")
        logger.info(f"  - Cap majorité: {major_cap}")
        logger.info(f"  - Min minorité: {tail_min}")
        
        # Appliquer le sampling
        X_resampled, y_resampled = apply_sampling(
            X=X_train,
            y=y_train,
            major_class=major_class,
            major_cap=major_cap,
            tail_min=tail_min,
            random_state=random_state
        )
        
        logger.info(f" Rééchantillonnage terminé: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def build_feature_pipeline(self) -> FeatureUnion:
        """
        Construit le pipeline de features (texte + image).
        
        Returns:
            FeatureUnion sklearn combinant les branches texte et image
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
        # Branche Image (si activée)
        # ========================================
        image_enabled = (
            self.config.get("images.cnn.enabled", False) or
            self.config.get("images.cnn_vit.enabled", False) or
            self.config.get("images.dim_reduction.enabled", False)
        )
        
        if image_enabled:
            logger.info("Construction de la branche image...")
            image_pipeline = create_image_pipeline_from_cfg(
                self.config.images,
                use_test_dir=False  # Train d'abord
            )
            transformers.append(("image", image_pipeline))
            
            # Poids de la branche image
            if self.config.get("images.cnn_vit.enabled", False):
                image_weight = self.config.get("fusion.weights.image_cnn_vit", 1.0)
            elif self.config.get("images.cnn.enabled", False):
                image_weight = self.config.get("fusion.weights.image_cnn", 1.0)
            else:
                image_weight = self.config.get("fusion.weights.image_pixels", 0.5)
            
            weights["image"] = float(image_weight)
            logger.info(f"  Poids image: {image_weight}")
        else:
            logger.info("Branche image désactivée (config)")
        
        # ========================================
        # Créer le FeatureUnion
        # ========================================
        feature_pipeline = FeatureUnion(
            transformers,
            transformer_weights=weights if weights else None
        )
        
        logger.info(f" Pipeline créé avec {len(transformers)} branche(s)")
        
        return feature_pipeline
    
    def transform_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        feature_pipeline: FeatureUnion
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforme les données avec le pipeline de features.
        
        Args:
            X_train: Features d'entraînement (brutes)
            y_train: Labels d'entraînement  
            X_test: Features de test (brutes)
            feature_pipeline: Pipeline sklearn à appliquer
            
        Returns:
            Tuple (X_train_transformed, X_test_transformed)
        """
        logger.info("\n--- Transformation des données ---")
        logger.info(f" Train: {len(X_train)} échantillons à transformer")
        logger.info(f" Test: {len(X_test)} échantillons à transformer")
        
        # ========================================
        # Fit + Transform sur train
        # ========================================
        logger.info("\n Étape 1/2 : Fit + Transform sur TRAIN...")
        with Timer("Fit + Transform sur train"):
            X_train_transformed = feature_pipeline.fit_transform(X_train, y_train)
        
        logger.info(f" Train transformé: {X_train_transformed.shape}")
        logger.info(f"  Type: {type(X_train_transformed)}")
        logger.info(f"  Progression: {len(X_train)}/{len(X_train)} échantillons traités ✓")
        
        # Afficher les stats si c'est sparse
        if hasattr(X_train_transformed, 'nnz'):
            density = X_train_transformed.nnz / np.prod(X_train_transformed.shape)
            logger.info(f"  Densité: {density:.4f} ({density*100:.2f}%)")
        
        # ========================================
        # Transform sur test
        # ========================================
        logger.info("\n Étape 2/2 : Transform sur TEST...")
        with Timer("Transform sur test"):
            X_test_transformed = feature_pipeline.transform(X_test)
        
        logger.info(f" Test transformé: {X_test_transformed.shape}")
        logger.info(f"  Progression: {len(X_test)}/{len(X_test)} échantillons traités ✓")
        
        return X_train_transformed, X_test_transformed
    
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Series, np.ndarray, FeatureUnion]:
        """
        Exécute le pipeline de transformation complet.
        
        Args:
            X_train: Features d'entraînement (brutes)
            y_train: Labels d'entraînement
            X_test: Features de test (brutes)
            
        Returns:
            Tuple (X_train_transformed, y_train_resampled, 
                   X_test_transformed, feature_pipeline)
        """
        with Timer("Transformation des données"):
            
            # ========================================
            # 1. Rééchantillonnage
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
                X_test,
                self.feature_pipeline
            )
            
            # ========================================
            # 4. Résumé final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RÉSUMÉ DE LA TRANSFORMATION")
            logger.info("=" * 70)
            logger.info(f" X_train : {X_train.shape} → {X_train_transformed.shape}")
            logger.info(f" y_train : {y_train.shape} → {y_train_resampled.shape}")
            logger.info(f" X_test  : {X_test.shape} → {X_test_transformed.shape}")
            logger.info(f" Pipeline sauvegardé : {self.feature_pipeline is not None}")
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
            print("\n Validation échouée - arrêt du pipeline")
        else:
            # Stage 3: Transformation
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, pipeline = stage3.run(
                X_train, y_train, X_test
            )
            
            print("\n Transformation terminée avec succès!")
            print(f"  X_train transformé: {X_train_t.shape}")
            print(f"  y_train rééchantillonné: {y_train_t.shape}")
            print(f"  X_test transformé: {X_test_t.shape}")
        
    except FileNotFoundError as e:
        print(f"\n Erreur: {e}")
        print("Vérification que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
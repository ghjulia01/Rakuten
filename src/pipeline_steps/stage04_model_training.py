"""
Étape 4 : Entraînement du modèle (Model Training).
==================================================

Cette étape entraîne le modèle sur les données transformées.
Elle correspond à la stage04 du projet wine_quality.

Responsabilités :
- Créer le modèle selon la configuration
- Entraîner sur les données transformées
- Sauvegarder le modèle et le pipeline de features

Utilisation:
    from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline
    
    pipeline = ModelTrainingPipeline(config)
    model = pipeline.run(X_train_transformed, y_train_resampled)

Auteur: Projet Rakuten
Date: 2024
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import joblib

from src.models.model_trainer import ModelTrainer
from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Pipeline d'entraînement du modèle.
    
    Entraîne le modèle final sur les données transformées.
    
    Attributes:
        config: Configuration complète du projet
        trainer: Instance de ModelTrainer
        model: Modèle entraîné (après run)
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> pipeline = ModelTrainingPipeline(config)
        >>> model = pipeline.run(X_train_transformed, y_train_resampled)
    """
    
    def __init__(self, config):
        """
        Initialise le pipeline d'entraînement.
        
        Args:
            config: Objet Config contenant tous les paramètres
        """
        self.config = config
        self.trainer = None
        self.model = None
        
        logger.info("=" * 70)
        logger.info("ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE")
        logger.info("=" * 70)
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """
        Entraîne le modèle sur les données.
        
        Args:
            X_train: Features transformées (n_samples, n_features)
            y_train: Labels (n_samples,)
            
        Returns:
            Modèle entraîné
        """
        logger.info("\n--- Entraînement du modèle ---")
        
        # Créer le trainer
        self.trainer = ModelTrainer(
            model_config=self.config.model,
            random_state=self.config.random_seed
        )
        
        # Informations sur les données
        logger.info(f"Données d'entraînement: {X_train.shape}")
        logger.info(f"Labels: {y_train.shape}")
        logger.info(f"Nombre de classes: {len(np.unique(y_train))}")
        
        # Type de matrice (sparse ou dense)
        if hasattr(X_train, 'nnz'):
            logger.info(f"Matrice sparse - nnz: {X_train.nnz}")
            density = X_train.nnz / np.prod(X_train.shape)
            logger.info(f"Densité: {density:.4f} ({density*100:.2f}%)")
        else:
            logger.info("Matrice dense")
        
        # Entraîner
        with Timer(f"Entraînement {self.config.model['name'].upper()}"):
            self.model = self.trainer.train(X_train, y_train)
        
        logger.info("✓ Entraînement terminé")
        
        return self.model
    
    def save_model(
        self,
        model: Any,
        output_path: str
    ) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            model: Modèle entraîné
            output_path: Chemin de sauvegarde
        """
        logger.info("\n--- Sauvegarde du modèle ---")
        
        # Créer le dossier parent si nécessaire
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        self.trainer.save_model(model, str(output_path))
        
        logger.info(f"✓ Modèle sauvegardé: {output_path}")
    
    def save_full_pipeline(
        self,
        model: Any,
        feature_pipeline: Any,
        output_path: str
    ) -> None:
        """
        Sauvegarde le pipeline complet (features + modèle).
        
        Utile pour la prédiction : on peut charger tout d'un coup.
        
        Args:
            model: Modèle entraîné
            feature_pipeline: Pipeline de features (sklearn)
            output_path: Chemin de sauvegarde
        """
        logger.info("\n--- Sauvegarde du pipeline complet ---")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Créer un dict avec tout
        full_pipeline = {
            "feature_pipeline": feature_pipeline,
            "model": model,
            "config": {
                "model_name": self.config.model["name"],
                "random_seed": self.config.random_seed,
            }
        }
        
        # Sauvegarder
        joblib.dump(full_pipeline, output_path)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Pipeline complet sauvegardé: {output_path}")
        logger.info(f"  Taille: {size_mb:.2f} MB")
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_pipeline: Any = None
    ) -> Any:
        """
        Exécute le pipeline d'entraînement complet.
        
        Args:
            X_train: Features transformées
            y_train: Labels
            feature_pipeline: Pipeline de features (optionnel, pour sauvegarde complète)
            
        Returns:
            Modèle entraîné
        """
        with Timer("Entraînement du modèle"):
            
            # ========================================
            # 1. Entraîner le modèle
            # ========================================
            self.model = self.train_model(X_train, y_train)
            
            # ========================================
            # 2. Sauvegarder le modèle seul
            # ========================================
            model_path = self.config.paths.get("model_out", "models/model.joblib")
            
            # Remplacer les placeholders dans le chemin
            model_name = self.config.model["name"]
            model_path = model_path.replace("{kind}", model_name)
            model_path = model_path.replace("{phase}", "final")
            
            self.save_model(self.model, model_path)
            
            # ========================================
            # 3. Sauvegarder le pipeline complet (si fourni)
            # ========================================
            if feature_pipeline is not None:
                full_pipeline_path = model_path.replace(".joblib", "_full_pipeline.joblib")
                self.save_full_pipeline(
                    self.model,
                    feature_pipeline,
                    full_pipeline_path
                )
            
            # ========================================
            # 4. Résumé final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RÉSUMÉ DE L'ENTRAÎNEMENT")
            logger.info("=" * 70)
            logger.info(f"✓ Modèle: {self.config.model['name'].upper()}")
            logger.info(f"✓ Données: {X_train.shape}")
            logger.info(f"✓ Sauvegardé: {model_path}")
            if feature_pipeline is not None:
                logger.info(f"✓ Pipeline complet: {full_pipeline_path}")
            logger.info("=" * 70 + "\n")
            
            return self.model


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de ModelTrainingPipeline")
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
            print("\n✗ Validation échouée - arrêt du pipeline")
        else:
            # Stage 3: Transformation
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, feature_pipeline = stage3.run(
                X_train, y_train, X_test
            )
            
            # Stage 4: Training
            stage4 = ModelTrainingPipeline(config)
            model = stage4.run(X_train_t, y_train_t, feature_pipeline)
            
            print("\n✓ Entraînement terminé avec succès!")
            print(f"  Modèle: {type(model).__name__}")
            print(f"  Sauvegardé dans: models/")
        
    except FileNotFoundError as e:
        print(f"\n✗ Erreur: {e}")
        print("Assurez-vous que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n✗ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()

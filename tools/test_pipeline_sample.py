#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test rapide du pipeline sur un échantillon réduit.
============================================================

Ce script permet de vérifier que tout fonctionne correctement
en lançant le pipeline complet sur un petit échantillon de données.

Utilisation:
    python tools/test_pipeline_sample.py --sample-size 2000  # Taille de l'échantillon
    python scripts/test_pipeline_sample.py # Utilise la taille par défaut (1000)
    python scripts/test_pipeline_sample.py --with-cv  # Active la validation croisée

"""
import sys
import logging
import argparse
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.config import load_config
from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline
from src.pipeline_steps.stage05_model_evaluation import ModelEvaluationPipeline
from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Test rapide du pipeline sur un échantillon"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Taille de l'échantillon (défaut: 1000)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.toml",
        help="Chemin vers le fichier de configuration"
    )
    
    parser.add_argument(
        "--with-cv",
        action="store_true",
        help="Activer la validation croisée"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Ignorer la validation des données"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbeux"
    )
    
    return parser.parse_args()


def main():
    """Fonction principale."""
    args = parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # En-tête
    logger.info("\n" + "=" * 70)
    logger.info(" TEST RAPIDE DU PIPELINE RAKUTEN")
    logger.info("=" * 70)
    logger.info(f"Échantillon: {args.sample_size} lignes")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 70 + "\n")
    
    try:
        # ========================================
        # Chargement de la configuration
        # ========================================
        config = load_config(args.config)
        logger.info(f" Configuration chargée")
        logger.info(f"  Modèle: {config.model['name'].upper()}")
        logger.info(f"  Random seed: {config.random_seed}")
        
        # Override CV si demandé
        if args.with_cv:
            config.cv['enabled'] = True
            logger.info(f"  Validation croisée: ACTIVÉE (--with-cv)")
        
        with Timer("Test pipeline complet"):
            
            # ========================================
            # ÉTAPE 1 : Data Ingestion
            # ========================================
            logger.info("\n" + "🔹" * 35)
            logger.info(" ÉTAPE 1/5 : INGESTION (échantillon)")
            
            stage1 = DataIngestionPipeline(config)
            X_train, y_train, X_test = stage1.run()
            
            #  ÉCHANTILLONNAGE
            logger.info(f"\n🔬 Échantillonnage à {args.sample_size} lignes...")
            sample_size_train = min(args.sample_size, len(X_train))
            sample_size_test = min(args.sample_size // 5, len(X_test))  # 20% pour test
            
            X_train = X_train.sample(n=sample_size_train, random_state=42)
            y_train = y_train.loc[X_train.index]
            X_test = X_test.sample(n=sample_size_test, random_state=42)
            
            logger.info(f" Échantillon train: {len(X_train)} lignes")
            logger.info(f" Échantillon test: {len(X_test)} lignes")
            
            # ========================================
            # ÉTAPE 2 : Data Validation
            # ========================================
            if not args.skip_validation:
                logger.info("\n" + "🔹" * 35)
                logger.info(" ÉTAPE 2/5 : VALIDATION")
                
                stage2 = DataValidationPipeline(config)
                validation_ok = stage2.run(X_train, y_train, X_test)
                
                if not validation_ok:
                    logger.error("\n✗ Validation échouée")
                    return 1
            else:
                logger.warning("\n⚠ Validation ignorée")
            
            # ========================================
            # ÉTAPE 3 : Data Transformation
            # ========================================
            logger.info("\n" + "🔹" * 35)
            logger.info(" ÉTAPE 3/5 : TRANSFORMATION")
            
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, feature_pipeline = stage3.run(
                X_train, y_train, X_test
            )
            
            logger.info(f"✓ Features: {X_train_t.shape[1]} colonnes")
            
            # ========================================
            # ÉTAPE 4 : Model Training
            # ========================================
            logger.info("\n" + "🔹" * 35)
            logger.info(" ÉTAPE 4/5 : ENTRAÎNEMENT")
            
            stage4 = ModelTrainingPipeline(config)
            model = stage4.run(X_train_t, y_train_t, feature_pipeline)
            
            # ========================================
            # ÉTAPE 5 : Model Evaluation
            # ========================================
            logger.info("\n" + "🔹" * 35)
            logger.info(" ÉTAPE 5/5 : ÉVALUATION")
            
            stage5 = ModelEvaluationPipeline(config)
            
            # Évaluation sur train
            logger.info(f"\n Évaluation sur train ({len(X_train_t)} échantillons)...")
            train_results = stage5.run(
                model, X_train_t, y_train_t,
                dataset_name="train_sample",
                trainer=stage4.trainer
            )
            
            # Prédictions sur test
            logger.info(f"\n Prédictions sur test ({len(X_test_t)} échantillons)...")
            test_results = stage5.run(
                model, X_test_t, y_true=None,
                dataset_name="test_sample",
                trainer=stage4.trainer
            )
            
            # ========================================
            # RÉSUMÉ FINAL
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info(" TEST TERMINÉ AVEC SUCCÈS !")
            logger.info("=" * 70)
            logger.info(f" Échantillon train: {len(X_train)} → {len(X_train_t)} lignes")
            logger.info(f" Échantillon test: {len(X_test)} → {len(X_test_t)} lignes")
            logger.info(f" Features: {X_train_t.shape[1]} colonnes")
            logger.info(f" Modèle: {config.model['name'].upper()}")
            
            if 'accuracy' in train_results:
                logger.info(f" Accuracy (train): {train_results['accuracy']:.4f}")
                logger.info(f" F1 (train): {train_results['f1_weighted']:.4f}")
            
            logger.info(f"✓ Prédictions: {len(test_results['predictions'])} générées")
            logger.info("=" * 70)
            
            logger.info("\nTOUS LES TESTS PASSENT !")
            logger.info("Le pipeline fonctionne correctement.")
            logger.info("Ok pour lancer sur les données complètes.\n")
            
            return 0
            
    except FileNotFoundError as e:
        logger.error(f"\n Erreur: Fichier non trouvé: {e}")
        logger.error("Vérification que les fichiers CSV existent dans data/raw/")
        return 1
        
    except Exception as e:
        logger.error(f"\n Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Pipeline d'entraînement complet - Orchestration des 5 étapes.
============================================================

Ce script orchestre les 5 étapes du pipeline d'entraînement :
1. Data Ingestion     - Chargement des données
2. Data Validation    - Validation de la qualité
3. Data Transformation - Features + sampling
4. Model Training     - Entraînement
5. Model Evaluation   - Évaluation

Utilisation:
    # Pipeline complet rapide
    python scripts/train_pipeline.py

    # Avec validation des données avec StatifiedKFold ou activation du mode CV dans le tomlib
    python scripts/train_pipeline.py --cv
    
    # Avec configuration personnalisée
    python scripts/train_pipeline.py --config config/config.toml
    
    # Mode verbeux
    python scripts/train_pipeline.py --verbose

"""
import argparse
import logging
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.config import load_config
from src.utils.profiling import Timer

from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline
from src.pipeline_steps.stage05_model_evaluation import ModelEvaluationPipeline

logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement complet Rakuten (5 étapes)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.toml",
        help="Chemin vers le fichier de configuration (défaut: config/config.toml)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Active les logs détaillés (niveau DEBUG)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Passe l'étape de validation (pas recommandé)"
    )
    
    parser.add_argument(
        "--evaluate-on-train",
        action="store_true",
        help="Évalue également sur le jeu d'entraînement"
    )
    
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Active la validation croisée (override config)"
    )
    
    return parser.parse_args()


def main():
    """Fonction principale du pipeline."""
    args = parse_args()
    
    # ========================================
    # Configuration du logging
    # ========================================
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE D'ENTRAÎNEMENT RAKUTEN")
    logger.info("=" * 70)
    logger.info("5 étapes : Ingestion → Validation → Transformation → Training → Evaluation")
    logger.info("=" * 70 + "\n")
    
    # ========================================
    # Chargement de la configuration
    # ========================================
    try:
        config = load_config(args.config)
        logger.info(f" Configuration chargée depuis: {args.config}")
        logger.info(f"  Modèle: {config.model['name'].upper()}")
        logger.info(f"  Random seed: {config.random_seed}")
        
        # Override CV si demandé
        if args.cv:
            config.cv['enabled'] = True
            logger.info(f"  Validation croisée: ACTIVÉE (--cv)")
            logger.info(f"    Folds: {config.cv.get('splits', 3)}")
        else:
            cv_status = "ACTIVÉE" if config.get("cv.enabled", False) else "DÉSACTIVÉE"
            logger.info(f"  Validation croisée: {cv_status}")
            
    except FileNotFoundError as e:
        logger.error(f" Fichier de configuration non trouvé: {e}")
        return 1
    except Exception as e:
        logger.error(f" Erreur lors du chargement de la configuration: {e}")
        return 1
    
    try:
        with Timer("Pipeline complet"):
            
            # ========================================
            # ÉTAPE 1 : Data Ingestion
            # ========================================
            logger.info("\n" + "*" * 35)
            logger.info(" ÉTAPE 1/5 : INGESTION DES DONNÉES")
            stage1 = DataIngestionPipeline(config)
            X_train, y_train, X_test = stage1.run()
            logger.info(f" Chargé : {len(X_train)} train + {len(X_test)} test")
            
            # ========================================
            # ÉTAPE 2 : Data Validation
            # ========================================
            if not args.skip_validation:
                logger.info("\n" + "*" * 35)
                logger.info(" ÉTAPE 2/5 : VALIDATION DES DONNÉES")
                stage2 = DataValidationPipeline(config)
                validation_ok = stage2.run(X_train, y_train, X_test)
                
                if not validation_ok:
                    logger.error("\n Validation échouée - Arrêt du pipeline")
                    logger.error(" erreurs présentes")
                    return 1
            else:
                logger.warning("\n Validation ignorée (--skip-validation)")
            
            # ========================================
            # ÉTAPE 3 : Data Transformation
            # ========================================
            logger.info("\n" + " * " * 35)
            logger.info(" ÉTAPE 3/5 : TRANSFORMATION DES DONNÉES")
            logger.info(f"   → Rééchantillonnage + Construction features")
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, feature_pipeline, feature_mapping = stage3.run(
                X_train, y_train, X_test
            )
            logger.info(f" Transformé : {X_train_t.shape[0]} train → {X_train_t.shape[1]} features")
            
            # ========================================
            # ÉTAPE 4 : Model Training
            # ========================================
            logger.info("\n" + "*" * 35)
            logger.info("ÉTAPE 4/5 : ENTRAÎNEMENT DU MODÈLE")
            logger.info(f"   → Modèle: {config.model['name'].upper()}")
            stage4 = ModelTrainingPipeline(config)
            model = stage4.run(X_train_t, y_train_t, feature_pipeline)
            logger.info(f" Modèle entraîné et sauvegardé")
            
            # ========================================
            # ÉTAPE 5 : Model Evaluation
            # ========================================
            logger.info("\n" + "*" * 35)
            logger.info("ÉTAPE 5/5 : ÉVALUATION DU MODÈLE")
            stage5 = ModelEvaluationPipeline(config)
            
            # Évaluation sur train (optionnel)
            if args.evaluate_on_train:
                logger.info(f"\nÉvaluation sur le jeu d'entraînement ({len(X_train_t)} échantillons)...")
                train_results = stage5.run(
                    model, X_train_t, y_train_t, 
                    dataset_name="train",
                    trainer=stage4.trainer,  # IMPORTANT : pour décodage
                    feature_mapping=feature_mapping, # passe le mapping
                    feature_pipeline=feature_pipeline  # pour SHAP
                )
            
            # Prédictions sur test (pas de labels)
            logger.info(f"\nGénération des prédictions sur le jeu de test ({len(X_test_t)} échantillons)...")
            test_results = stage5.run(
                model, X_test_t, y_true=None,
                dataset_name="test",
                trainer=stage4.trainer,  # pour décodage
                feature_mapping=feature_mapping, # passe le mapping
                feature_pipeline=feature_pipeline  # pour SHAP
            )
            logger.info(f" Prédictions générées: {len(test_results['predictions'])} échantillons")
        
            
            # Sauvegarder les prédictions test
            pred_output = config.paths.get("pred_out", "results/predictions/test_predictions.csv")
            pred_output = pred_output.replace("{kind}", config.model["name"])
            pred_output = pred_output.replace("{phase}", "final")
            
            import pandas as pd
            predictions_df = pd.DataFrame({
                "prediction": test_results["predictions"]
            }, index=X_test.index)
            
            pred_output_path = Path(pred_output)
            pred_output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(pred_output_path)
            
            logger.info(f"Prédictions test sauvegardées: {pred_output_path}")
        
        # ========================================
        # Résumé final
        # ========================================
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE TERMINÉ")
        logger.info("=" * 70)
        logger.info(f"\nRésultats:")
        logger.info(f"  • Modèle entraîné: {config.model['name'].upper()}")
        logger.info(f"  • Données d'entraînement: {X_train_t.shape}")
        logger.info(f"  • Données de test: {X_test_t.shape}")
        
        if args.evaluate_on_train and 'train_results' in locals():
            logger.info(f"\nPerformance sur train:")
            logger.info(f"  • Accuracy: {train_results['accuracy']:.4f}")
            logger.info(f"  • F1 (weighted): {train_results['f1_weighted']:.4f}")
            logger.info(f"  • F1 (macro): {train_results['f1_macro']:.4f}")
        
        logger.info(f"\nFichiers générés:")
        logger.info(f"  • Modèle: models/")
        logger.info(f"  • Prédictions: {pred_output_path}")
        logger.info(f"  • Métriques: results/metrics/")
        
        logger.info("\n" + "=" * 70 + "\n")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"\n Fichier non trouvé: {e}")
        logger.error("Vérification que tous les fichiers de données existent")
        return 1
    
    except Exception as e:
        logger.error(f"\n Erreur durant le pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
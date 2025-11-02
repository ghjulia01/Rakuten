#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test rapide du pipeline sur un échantillon réduit AVEC PROFILING DÉTAILLÉ.
====================================================================================

Ce script permet de vérifier que tout fonctionne correctement
en lançant le pipeline complet sur un petit échantillon de données,
et affiche un profiling détaillé de chaque étape de transformation.

Utilisation:
    python tools/test_pipeline_sample.py --sample-size 2000  # Taille de l'échantillon
    python tools/test_pipeline_sample.py # Utilise la taille par défaut (1000)
    python tools/test_pipeline_sample.py --with-cv  # Active la validation croisée
    python tools/test_pipeline_sample.py --profile-features  # Profiling détaillé des features
    # Profiling rapide (échantillon plus petit pour le profiling)
    python tools/test_pipeline_sample.py --sample-size 2000 --profile-features --profile-sample-size 300
    # Test complet avec tous les détails
    python tools/test_pipeline_sample.py --sample-size 2000 --profile-features --verbose

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

# Import du profiler de features
try:
    from src.utils.feature_profiler import profile_pipeline
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    logging.warning("  Feature profiler non disponible (src.utils.feature_profiler)")

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
        "--profile-features",
        action="store_true",
        help="Activer le profiling détaillé des transformateurs de features"
    )
    
    parser.add_argument(
        "--profile-sample-size",
        type=int,
        default=500,
        help="Taille de l'échantillon pour le profiling (défaut: 500, plus rapide)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbeux"
    )
    
    return parser.parse_args()


def profile_feature_pipeline(feature_pipeline, X_sample, y_sample, config):
    """
    Profile le pipeline de features en détail.
    
    Args:
        feature_pipeline: Pipeline de features à profiler
        X_sample: Échantillon de données X
        y_sample: Échantillon de labels y
        config: Configuration du projet
    """
    if not PROFILER_AVAILABLE:
        logger.warning("  Profiling ignoré : feature_profiler non disponible")
        return
    
    logger.info("\n" + "=" * 70)
    logger.info(" PROFILING DÉTAILLÉ DES FEATURES")
    logger.info("=" * 70)
    logger.info(f" Échantillon de profiling : {len(X_sample)} lignes")
    logger.info(f" Cela peut prendre quelques minutes...\n")
    
    try:
        # Profiler le pipeline complet
        results = profile_pipeline(
            pipeline=feature_pipeline,
            X=X_sample,
            y=y_sample,
            max_depth=3
        )
        
        # Afficher le résumé
        results.print_summary()
        
        # Identifier les goulots d'étranglement
        bottlenecks = results.get_bottlenecks(top_n=5)
        
        logger.info("\n" + "=" * 70)
        logger.info(" RECOMMANDATIONS D'OPTIMISATION")
        logger.info("=" * 70)
        
        total_transform = sum(p.transform_time for p in results.profiles)
        
        for i, bottleneck in enumerate(bottlenecks[:3], 1):
            pct = (bottleneck.transform_time / total_transform * 100) if total_transform > 0 else 0
            logger.info(f"\n{i}. {bottleneck.name}")
            logger.info(f"   ⏱  Temps: {bottleneck.transform_time:.2f}s ({pct:.1f}% du total)")
            logger.info(f"    Mémoire: {bottleneck.memory_mb:.1f} MB")
            logger.info(f"    Shape: {bottleneck.output_shape}")
            logger.info(f"    Débit: {bottleneck.samples_per_sec:.0f} samples/sec")
            
            # Suggestions d'optimisation
            if 'tfidf' in bottleneck.name.lower() or 'countvectorizer' in bottleneck.name.lower():
                logger.info("    Suggestion: Réduire max_features ou augmenter min_df")
            elif 'svd' in bottleneck.name.lower() or 'pca' in bottleneck.name.lower():
                logger.info("    Suggestion: Réduire n_components")
            elif bottleneck.memory_mb > 1000:
                logger.info("    Suggestion: Considérer une représentation sparse ou réduire la dimensionnalité")
        
        logger.info("\n" + "=" * 70 + "\n")
        
        # Sauvegarder les résultats détaillés
        import json
        from pathlib import Path
        
        output_dir = Path("artifacts/profiling")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "feature_profiling_results.json"
        with open(output_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        logger.info(f" Résultats détaillés sauvegardés : {output_file}")
        
    except Exception as e:
        logger.error(f" Erreur durant le profiling : {e}")
        import traceback
        traceback.print_exc()


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
    logger.info(f" Échantillon: {args.sample_size} lignes")
    logger.info(f"  Config: {args.config}")
    if args.profile_features:
        logger.info(f" Profiling: ACTIVÉ ({args.profile_sample_size} lignes)")
    logger.info("=" * 70 + "\n")
    
    try:
        # ========================================
        # Chargement de la configuration
        # ========================================
        config = load_config(args.config)
        logger.info(f" Configuration chargée")
        logger.info(f"   Modèle: {config.model['name'].upper()}")
        logger.info(f"   Random seed: {config.random_seed}")
        
        # Override CV si demandé
        if args.with_cv:
            config.cv['enabled'] = True
            logger.info(f"   Validation croisée: ACTIVÉE (--with-cv)")
        
        with Timer("Test pipeline complet"):
            
            # ========================================
            # ÉTAPE 1 : Data Ingestion
            # ========================================
            logger.info("\n" + "🔹" * 35)
            logger.info(" ÉTAPE 1/5 : INGESTION (échantillon)")
            
            stage1 = DataIngestionPipeline(config)
            X_train, y_train, X_test = stage1.run()
            
            # 🔬 ÉCHANTILLONNAGE
            logger.info(f"\n Échantillonnage à {args.sample_size} lignes...")
            sample_size_train = min(args.sample_size, len(X_train))
            sample_size_test = min(args.sample_size // 5, X_test.shape[0])  # 20% pour test
            
            X_train = X_train.sample(n=sample_size_train, random_state=42)
            y_train = y_train.loc[X_train.index]
            X_test = X_test.sample(n=sample_size_test, random_state=42)
            
            logger.info(f" Échantillon train: {len(X_train)} lignes")
            logger.info(f" Échantillon test: {X_test.shape[0]} lignes")
            
            # ========================================
            # ÉTAPE 2 : Data Validation
            # ========================================
            if not args.skip_validation:
                logger.info("\n" + "🔹" * 35)
                logger.info(" ÉTAPE 2/5 : VALIDATION")
                
                stage2 = DataValidationPipeline(config)
                validation_ok = stage2.run(X_train, y_train, X_test)
                
                if not validation_ok:
                    logger.error("\n Validation échouée")
                    return 1
            else:
                logger.warning("\n  Validation ignorée")
            
            # ========================================
            # ÉTAPE 3 : Data Transformation
            # ========================================
            logger.info("\n" + "🔹" * 35)
            logger.info(" ÉTAPE 3/5 : TRANSFORMATION")
            
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, feature_pipeline, feature_mapping = stage3.run(
                X_train, y_train, X_test
            )
            
            logger.info(f" Features: {X_train_t.shape[1]} colonnes")
            
            # ========================================
            # PROFILING DÉTAILLÉ (optionnel)
            # ========================================
            if args.profile_features:
                # Créer un sous-échantillon pour le profiling
                profile_size = min(args.profile_sample_size, len(X_train))
                X_profile = X_train.sample(n=profile_size, random_state=42)
                y_profile = y_train.loc[X_profile.index]
                
                logger.info(f"\n🔍 Lancement du profiling sur {profile_size} lignes...")
                profile_feature_pipeline(feature_pipeline, X_profile, y_profile, config)
            
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
            logger.info(f"\n Évaluation sur train ({X_train_t.shape[0]} échantillons)...")
            train_results = stage5.run(
                model, X_train_t, y_train_t,
                dataset_name="train_sample",
                trainer=stage4.trainer
            )
            
            # Prédictions sur test
            logger.info(f"\n Prédictions sur test ({X_test_t.shape[0]} échantillons)...")
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
            logger.info(f" Échantillon train: {len(X_train)} → {X_train_t.shape[0]} lignes")
            logger.info(f" Échantillon test: {X_test.shape[0]} → {X_test_t.shape[0]} lignes")
            logger.info(f" Features: {X_train_t.shape[1]} colonnes")
            logger.info(f" Modèle: {config.model['name'].upper()}")
            
            if 'accuracy' in train_results:
                logger.info(f" Accuracy (train): {train_results['accuracy']:.4f}")
                logger.info(f" F1 (train): {train_results['f1_weighted']:.4f}")
            
            logger.info(f" Prédictions: {len(test_results['predictions'])} générées")
            
            if args.profile_features:
                logger.info(f" Profiling: Résultats sauvegardés dans artifacts/profiling/")
            
            logger.info("=" * 70)
            
            logger.info("\n TOUS LES TESTS PASSENT !")
            logger.info(" Le pipeline fonctionne correctement.")
            logger.info(" Ok pour lancer sur les données complètes.\n")
            
            return 0
            
    except FileNotFoundError as e:
        logger.error(f"\n Erreur: Fichier non trouvé: {e}")
        logger.error(" Vérification que les fichiers CSV existent dans data/raw/")
        return 1
        
    except Exception as e:
        logger.error(f"\n Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
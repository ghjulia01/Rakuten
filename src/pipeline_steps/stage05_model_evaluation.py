"""
Étape 5 : Évaluation du modèle (Model Evaluation).
==================================================

Cette étape évalue les performances du modèle entraîné.

Responsabilités :
- Générer des prédictions
- Calculer les métriques (F1, accuracy, precision, recall)
- Créer la matrice de confusion
- Générer un rapport détaillé
- Sauvegarder les résultats

Utilisation:
    from src.pipeline_steps.stage05_model_evaluation import ModelEvaluationPipeline
    
    pipeline = ModelEvaluationPipeline(config)
    results = pipeline.run(model, X_test_transformed, y_test)

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class ModelEvaluationPipeline:
    """
    Pipeline d'évaluation du modèle.
    
    Évalue les performances et génère des rapports détaillés.
    
    Attributes:
        config: Configuration complète du projet
        results: Dictionnaire des résultats (après run)
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> pipeline = ModelEvaluationPipeline(config)
        >>> results = pipeline.run(model, X_val, y_val)
    """
    
    def __init__(self, config):
        """
        Initialise le pipeline d'évaluation.
        
        Args:
            config: Objet Config contenant tous les paramètres
        """
        self.config = config
        self.results = {}
        
        logger.info("=" * 70)
        logger.info("ÉTAPE 5 : ÉVALUATION DU MODÈLE")
        logger.info("=" * 70)
    
    def generate_predictions(
        self,
        model: Any,
        X: np.ndarray,
        dataset_name: str = "validation",
        trainer: Any = None
    ) -> np.ndarray:
        """
        Génère les prédictions du modèle.
        
        Args:
            model: Modèle entraîné
            X: Features (n_samples, n_features)
            dataset_name: Nom du dataset (pour les logs)
            trainer: ModelTrainer (optionnel, pour décodage automatique)
            
        Returns:
            Prédictions (n_samples,)
        """
        logger.info(f"\n--- Prédictions sur {dataset_name} ---")
        
        with Timer(f"Prédiction ({X.shape[0]} échantillons)"):
            # Si on a le trainer, utiliser sa méthode predict (avec décodage)
            if trainer is not None:
                y_pred = trainer.predict(X)
                logger.info("✓ Prédictions décodées vers classes originales")
            else:
                # Sinon, prédiction directe (attention : labels encodés!)
                y_pred = model.predict(X)
                logger.warning("⚠ Prédictions en labels encodés (0,1,2...) - pas de trainer fourni")
        
        logger.info(f"✓ {len(y_pred)} prédictions générées")
        
        return y_pred
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "validation"
    ) -> Dict[str, float]:
        """
        Calcule les métriques de classification.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            dataset_name: Nom du dataset
            
        Returns:
            Dictionnaire des métriques
        """
        logger.info(f"\n--- Calcul des métriques ({dataset_name}) ---")
        
        metrics = {}
        
        # Accuracy
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # F1 Score (weighted pour tenir compte du déséquilibre)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        logger.info(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        
        # F1 Score (macro pour voir la performance moyenne par classe)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        
        # Precision & Recall
        metrics["precision_weighted"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["recall_weighted"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        
        logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        
        return metrics
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Génère un rapport de classification détaillé.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            
        Returns:
            Rapport sous forme de string
        """
        logger.info("\n--- Rapport de classification ---")
        
        report = classification_report(y_true, y_pred, zero_division=0)
        
        # Afficher dans les logs (limité aux premières lignes)
        lines = report.split('\n')
        logger.info("Aperçu du rapport:")
        for line in lines[:10]:  # Premières 10 lignes
            if line.strip():
                logger.info(f"  {line}")
        
        if len(lines) > 10:
            logger.info(f"  ... ({len(lines) - 10} lignes supplémentaires)")
        
        return report
    
    def generate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Génère la matrice de confusion.
        
        Args:
            y_true: Labels réels
            y_pred: Labels prédits
            
        Returns:
            Matrice de confusion (n_classes, n_classes)
        """
        logger.info("\n--- Matrice de confusion ---")
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        logger.info(f"Dimensions: {conf_matrix.shape}")
        logger.info(f"Diagonal (prédictions correctes): {conf_matrix.diagonal().sum()}")
        logger.info(f"Total: {conf_matrix.sum()}")
        
        # Calculer l'accuracy depuis la matrice
        accuracy_from_matrix = conf_matrix.diagonal().sum() / conf_matrix.sum()
        logger.info(f"Accuracy (depuis matrice): {accuracy_from_matrix:.4f}")
        
        return conf_matrix
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str = "results/metrics"
    ) -> None:
        """
        Sauvegarde les résultats de l'évaluation.
        
        Args:
            results: Dictionnaire des résultats
            output_dir: Dossier de sortie
        """
        logger.info("\n--- Sauvegarde des résultats ---")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nom du modèle pour les fichiers
        model_name = self.config.model["name"]
        
        # 1. Sauvegarder les métriques en JSON
        metrics_file = output_dir / f"{model_name}_metrics.json"
        metrics_to_save = {
            k: v for k, v in results.items()
            if isinstance(v, (int, float, str, bool))
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        logger.info(f"✓ Métriques: {metrics_file}")
        
        # 2. Sauvegarder le rapport de classification
        if "classification_report" in results:
            report_file = output_dir / f"{model_name}_classification_report.txt"
            with open(report_file, 'w') as f:
                f.write(results["classification_report"])
            logger.info(f"✓ Rapport: {report_file}")
        
        # 3. Sauvegarder la matrice de confusion (CSV)
        if "confusion_matrix" in results:
            matrix_file = output_dir / f"{model_name}_confusion_matrix.csv"
            conf_matrix_df = pd.DataFrame(results["confusion_matrix"])
            conf_matrix_df.to_csv(matrix_file, index=False)
            logger.info(f"✓ Matrice: {matrix_file}")
    
    def run(
        self,
        model: Any,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        dataset_name: str = "validation",
        trainer: Any = None
    ) -> Dict[str, Any]:
        """
        Exécute le pipeline d'évaluation complet.
        
        Args:
            model: Modèle entraîné
            X: Features transformées
            y_true: Labels réels (optionnel pour test)
            dataset_name: Nom du dataset
            trainer: ModelTrainer (optionnel, pour décodage des prédictions)
            
        Returns:
            Dictionnaire des résultats
        """
        with Timer("Évaluation du modèle"):
            
            # ========================================
            # 1. Générer les prédictions
            # ========================================
            y_pred = self.generate_predictions(model, X, dataset_name, trainer)
            
            self.results["predictions"] = y_pred
            self.results["dataset_name"] = dataset_name
            
            # ========================================
            # 2. Si on a les vrais labels, calculer les métriques
            # ========================================
            if y_true is not None:
                # Métriques
                metrics = self.calculate_metrics(y_true, y_pred, dataset_name)
                self.results.update(metrics)
                
                # Rapport de classification
                report = self.generate_classification_report(y_true, y_pred)
                self.results["classification_report"] = report
                
                # Matrice de confusion
                conf_matrix = self.generate_confusion_matrix(y_true, y_pred)
                self.results["confusion_matrix"] = conf_matrix
                
                # Sauvegarder
                self.save_results(self.results)
            else:
                logger.info("\nPas de labels fournis - évaluation limitée aux prédictions")
            
            # ========================================
            # 3. Résumé final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RÉSUMÉ DE L'ÉVALUATION")
            logger.info("=" * 70)
            logger.info(f"✓ Dataset: {dataset_name}")
            logger.info(f"✓ Prédictions: {len(y_pred)}")
            
            if y_true is not None:
                logger.info(f"✓ Accuracy: {self.results['accuracy']:.4f}")
                logger.info(f"✓ F1 (weighted): {self.results['f1_weighted']:.4f}")
                logger.info(f"✓ F1 (macro): {self.results['f1_macro']:.4f}")
                logger.info(f"✓ Résultats sauvegardés dans: results/metrics/")
            
            logger.info("=" * 70 + "\n")
            
            return self.results


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
    from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de ModelEvaluationPipeline")
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
            
            # Stage 5: Evaluation (sur train pour cet exemple)
            stage5 = ModelEvaluationPipeline(config)
            results = stage5.run(model, X_train_t, y_train_t, dataset_name="train")
            
            print("\n✓ Évaluation terminée avec succès!")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1 (weighted): {results['f1_weighted']:.4f}")
            print(f"  F1 (macro): {results['f1_macro']:.4f}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Erreur: {e}")
        print("Assurez-vous que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n✗ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
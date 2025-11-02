"""
Étape 2 : Validation des données (Data Validation).
===================================================

Cette étape valide la qualité et la cohérence des données chargées.

Responsabilités :
- Vérifier les types de données
- Détecter les valeurs aberrantes
- Analyser la distribution des classes
- Générer un rapport de validation

Utilisation:
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    
    pipeline = DataValidationPipeline(config)
    validation_ok = pipeline.run(X_train, y_train, X_test)

"""
from __future__ import annotations

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class DataValidationPipeline:
    """
    Pipeline de validation des données.
    
    Vérifie la qualité des données avant la transformation.
    
    Attributes:
        config: Configuration complète du projet
        validation_report: Rapport de validation (dict)
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> pipeline = DataValidationPipeline(config)
        >>> is_valid = pipeline.run(X_train, y_train, X_test)
    """
    
    def __init__(self, config):
        """
        Initialise le pipeline de validation.
        
        Args:
            config: Objet Config contenant tous les paramètres
        """
        self.config = config
        self.validation_report = {}
        
        logger.info("=" * 70)
        logger.info("ÉTAPE 2 : VALIDATION DES DONNÉES")
        logger.info("=" * 70)
    
    def validate_schema(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> bool:
        """
        Valide que les schémas train et test sont cohérents.
        
        Args:
            X_train: DataFrame d'entraînement
            X_test: DataFrame de test
            
        Returns:
            True si les schémas sont valides, False sinon
        """
        logger.info("\n--- Validation du schéma ---")
        
        issues = []
        
        # 1. Vérifier les colonnes requises
        required_cols = ["designation", "description", "productid", "imageid"]
        
        for col in required_cols:
            if col not in X_train.columns:
                issues.append(f"Colonne manquante dans X_train: {col}")
            if col not in X_test.columns:
                issues.append(f"Colonne manquante dans X_test: {col}")
        
        # 2. Vérifier que train et test ont les mêmes colonnes
        if set(X_train.columns) != set(X_test.columns):
            issues.append("Les colonnes de train et test ne correspondent pas")
        
        # 3. Vérifier les types de données
        for col in X_train.columns:
            if col in X_test.columns:
                train_dtype = X_train[col].dtype
                test_dtype = X_test[col].dtype
                if train_dtype != test_dtype:
                    issues.append(
                        f"Type différent pour {col}: "
                        f"train={train_dtype}, test={test_dtype}"
                    )
        
        if issues:
            logger.error("✗ Problèmes de schéma détectés:")
            for issue in issues:
                logger.error(f"  - {issue}")
            self.validation_report["schema_valid"] = False
            self.validation_report["schema_issues"] = issues
            return False
        else:
            logger.info("✓ Schéma valide")
            self.validation_report["schema_valid"] = True
            return True
    
    def validate_data_quality(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> bool:
        """
        Valide la qualité des données.
        
        Args:
            X_train: DataFrame d'entraînement
            X_test: DataFrame de test
            
        Returns:
            True si la qualité est acceptable, False sinon
        """
        logger.info("\n--- Validation de la qualité ---")
        
        issues = []
        
        # 1. Vérifier les valeurs manquantes excessives
        for df, name in [(X_train, "train"), (X_test, "test")]:
            missing_pct = (df.isnull().sum() / len(df) * 100)
            critical_missing = missing_pct[missing_pct > 50]
            
            if len(critical_missing) > 0:
                issues.append(
                    f"{name}: colonnes avec >50% de valeurs manquantes: "
                    f"{list(critical_missing.index)}"
                )
        
        # 2. Vérifier les doublons d'ID
        if "productid" in X_train.columns:
            train_duplicates = X_train["productid"].duplicated().sum()
            test_duplicates = X_test["productid"].duplicated().sum()
            
            if train_duplicates > 0:
                issues.append(f"train: {train_duplicates} productid en doublon")
            if test_duplicates > 0:
                issues.append(f"test: {test_duplicates} productid en doublon")
        
        # 3. Vérifier que les textes ne sont pas tous vides
        if "designation" in X_train.columns:
            empty_designation = (
                X_train["designation"].fillna("").str.strip() == ""
            ).sum()
            pct_empty = empty_designation / len(X_train) * 100
            
            if pct_empty > 10:
                issues.append(
                    f"train: {pct_empty:.1f}% de désignations vides"
                )
        
        if issues:
            logger.warning("⚠ Problèmes de qualité détectés:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            self.validation_report["quality_valid"] = False
            self.validation_report["quality_issues"] = issues
            # On continue quand même (warning, pas error)
            return True
        else:
            logger.info("✓ Qualité des données acceptable")
            self.validation_report["quality_valid"] = True
            return True
    
    def validate_class_distribution(
        self,
        y_train: pd.Series
    ) -> bool:
        """
        Valide la distribution des classes.
        
        Args:
            y_train: Labels d'entraînement
            
        Returns:
            True si la distribution est acceptable, False sinon
        """
        logger.info("\n--- Validation de la distribution des classes ---")
        
        issues = []
        
        # 1. Vérifier le nombre de classes
        n_classes = y_train.nunique()
        expected_n_classes = self.config.get("data.n_classes", 27)
        
        if n_classes != expected_n_classes:
            issues.append(
                f"Nombre de classes incorrect: {n_classes} "
                f"(attendu: {expected_n_classes})"
            )
        
        logger.info(f"Nombre de classes: {n_classes}")
        
        # 2. Analyser la distribution
        class_counts = y_train.value_counts().sort_values()
        
        min_samples = class_counts.min()
        max_samples = class_counts.max()
        ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        logger.info(f"Classe la moins représentée: {min_samples} échantillons")
        logger.info(f"Classe la plus représentée: {max_samples} échantillons")
        logger.info(f"Ratio max/min: {ratio:.1f}x")
        
        # 3. Vérifier qu'il n'y a pas de classes trop rares
        very_rare = class_counts[class_counts < 10]
        if len(very_rare) > 0:
            issues.append(
                f"{len(very_rare)} classe(s) avec <10 échantillons: "
                f"{list(very_rare.index)}"
            )
        
        # 4. Vérifier le déséquilibre
        if ratio > 100:
            issues.append(
                f"Déséquilibre très important: ratio {ratio:.1f}x"
            )
        
        self.validation_report["n_classes"] = n_classes
        self.validation_report["min_samples_per_class"] = int(min_samples)
        self.validation_report["max_samples_per_class"] = int(max_samples)
        self.validation_report["imbalance_ratio"] = float(ratio)
        
        if issues:
            logger.warning("⚠ Problèmes de distribution détectés:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            self.validation_report["distribution_valid"] = False
            self.validation_report["distribution_issues"] = issues
            # On continue quand même (les warnings ne bloquent pas)
            return True
        else:
            logger.info("✓ Distribution des classes acceptable")
            self.validation_report["distribution_valid"] = True
            return True
    
    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> bool:
        """
        Exécute le pipeline de validation complet.
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            X_test: Features de test
            
        Returns:
            True si toutes les validations passent, False sinon
        """
        with Timer("Validation des données"):
            
            # ========================================
            # 1. Validation du schéma
            # ========================================
            schema_ok = self.validate_schema(X_train, X_test)
            
            if not schema_ok:
                logger.error("✗ Validation du schéma échouée - arrêt du pipeline")
                return False
            
            # ========================================
            # 2. Validation de la qualité
            # ========================================
            quality_ok = self.validate_data_quality(X_train, X_test)
            
            # ========================================
            # 3. Validation de la distribution
            # ========================================
            distribution_ok = self.validate_class_distribution(y_train)
            
            # ========================================
            # 4. Résumé final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RÉSUMÉ DE LA VALIDATION")
            logger.info("=" * 70)
            
            all_valid = schema_ok and quality_ok and distribution_ok
            
            logger.info(f"✓ Schéma: {'OK' if schema_ok else 'ERREUR'}")
            logger.info(f"✓ Qualité: {'OK' if quality_ok else 'ATTENTION'}")
            logger.info(f"✓ Distribution: {'OK' if distribution_ok else 'ATTENTION'}")
            
            if all_valid:
                logger.info("\n✓ VALIDATION RÉUSSIE - Pipeline peut continuer")
            else:
                logger.error("\n✗ VALIDATION ÉCHOUÉE - Corriger les erreurs")
            
            logger.info("=" * 70 + "\n")
            
            self.validation_report["overall_valid"] = all_valid
            
            return all_valid
    
    def get_report(self) -> Dict[str, Any]:
        """
        Retourne le rapport de validation complet.
        
        Returns:
            Dictionnaire contenant tous les résultats de validation
        """
        return self.validation_report


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de DataValidationPipeline")
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
        
        # Afficher le rapport
        report = stage2.get_report()
        print("\nRapport de validation:")
        for key, value in report.items():
            print(f"  {key}: {value}")
        
        if validation_ok:
            print("\n✓ Validation terminée avec succès!")
        else:
            print("\n✗ Validation échouée - voir les logs")
        
    except FileNotFoundError as e:
        print(f"\n✗ Erreur: {e}")
        print("S'assurer que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n✗ Erreur inattendue: {e}")

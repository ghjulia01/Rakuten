"""
Étape 1 : Ingestion des données (Data Ingestion).
=================================================
python 

Cette étape gère le chargement des fichiers CSV d'entraînement et de test.

Responsabilités :
- Charger X_train et y_train depuis les CSV
- Charger X_test depuis le CSV
- Valider la cohérence des données
- Afficher des statistiques de base

Utilisation:
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    
    pipeline = DataIngestionPipeline(config)
    X_train, y_train, X_test = pipeline.run()
"""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from src.data.load_data import (
    load_train_data,
    load_test_data,
    validate_dataframes,
    check_missing_values
)
from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Pipeline d'ingestion des données.
    
    Charge les données d'entraînement et de test depuis les fichiers CSV.
    
    Attributes:
        config: Configuration complète du projet
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> pipeline = DataIngestionPipeline(config)
        >>> X_train, y_train, X_test = pipeline.run()
    """
    
    def __init__(self, config):
        """
        Initialise le pipeline d'ingestion.
        
        Args:
            config: Objet Config contenant tous les paramètres
        """
        self.config = config
        logger.info("=" * 70)
        logger.info("ÉTAPE 1 : INGESTION DES DONNÉES")
        logger.info("=" * 70)
    
    def run(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Exécute le pipeline d'ingestion complet.
        
        Returns:
            Tuple (X_train, y_train, X_test)
            
        Raises:
            FileNotFoundError: Si les fichiers CSV n'existent pas
            ValueError: Si les données sont incohérentes
        """
        with Timer("Ingestion des données"):
            # ========================================
            # 1. Charger les données d'entraînement
            # ========================================
            logger.info("\n--- Chargement des données d'entraînement ---")
            X_train, y_train = load_train_data(
                x_train_path=self.config.paths["x_train_csv"],
                y_train_path=self.config.paths["y_train_csv"]
            )
            
            # ========================================
            # 2. Charger les données de test
            # ========================================
            logger.info("\n--- Chargement des données de test ---")
            X_test = load_test_data(
                x_test_path=self.config.paths["x_test_csv"]
            )
            
            # ========================================
            # 3. Valider la compatibilité
            # ========================================
            logger.info("\n--- Validation de la compatibilité ---")
            validate_dataframes(X_train, X_test)
            
            # ========================================
            # 4. Analyser les valeurs manquantes
            # ========================================
            logger.info("\n--- Analyse des valeurs manquantes ---")
            check_missing_values(X_train, "X_train")
            check_missing_values(X_test, "X_test")
            
            # ========================================
            # 5. Résumé final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RÉSUMÉ DE L'INGESTION")
            logger.info("=" * 70)
            logger.info(f"✓ X_train : {X_train.shape}")
            logger.info(f"✓ y_train : {y_train.shape}")
            logger.info(f"✓ X_test  : {X_test.shape}")
            logger.info(f"✓ Colonnes : {list(X_train.columns)}")
            logger.info(f"✓ Classes : {y_train.nunique()}")
            logger.info("=" * 70 + "\n")
            
            return X_train, y_train, X_test


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de DataIngestionPipeline")
    print("="*70 + "\n")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Créer et exécuter le pipeline
        pipeline = DataIngestionPipeline(config)
        X_train, y_train, X_test = pipeline.run()
        
        print("\n✓ Pipeline d'ingestion terminé avec succès!")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Erreur: {e}")
        print("S'assurer que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n✗ Erreur inattendue: {e}")

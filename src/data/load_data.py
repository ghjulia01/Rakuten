"""
Module de chargement des données pour le projet Rakuten.
======================================================

Ce module gère le chargement des fichiers CSV d'entraînement et de test,
ainsi que la validation de la cohérence des données.

Fonctions principales :
- load_train_data() : Charge X_train et y_train
- load_test_data() : Charge X_test
- validate_dataframes() : Vérifie la compatibilité train/test
- check_missing_values() : Analyse les valeurs manquantes

Utilisation typique :
    from src.data.load_data import load_train_data, load_test_data
    
    # Charger les données d'entraînement
    X_train, y_train = load_train_data(
        "data/raw/X_train_update.csv",
        "data/raw/Y_train_CVw08PX.csv"
    )
    
    # Charger les données de test
    X_test = load_test_data("data/raw/X_test_update.csv")

Auteur: Projet Rakuten
Date: 2024
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


def load_train_data(
    x_train_path: str,
    y_train_path: str,
    index_col: int = 0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Charge les données d'entraînement (features et labels).
    
    Cette fonction :
    1. Vérifie que les fichiers existent
    2. Charge les CSV avec pandas
    3. Extrait la colonne cible (y)
    4. Valide que X et y ont la même longueur
    5. Affiche des statistiques utiles
    
    Args:
        x_train_path: Chemin vers le fichier X_train.csv
                     Contient les features : designation, description, 
                     productid, imageid
        y_train_path: Chemin vers le fichier Y_train.csv
                     Contient les labels : prdtypecode (code de catégorie)
        index_col: Colonne à utiliser comme index (0 par défaut)
                  Généralement la première colonne est l'ID
        
    Returns:
        Tuple de (X_train, y_train)
        - X_train : DataFrame avec toutes les features
        - y_train : Series avec les codes de catégorie
        
    Raises:
        FileNotFoundError: Si un des fichiers n'existe pas
        ValueError: Si X_train et y_train n'ont pas la même longueur
        
    Exemple:
        >>> X_train, y_train = load_train_data(
        ...     "data/raw/X_train_update.csv",
        ...     "data/raw/Y_train_CVw08PX.csv"
        ... )
        >>> print(X_train.shape)
        (84916, 4)
        >>> print(y_train.shape)
        (84916,)
    """
    with Timer("Chargement des données d'entraînement"):
        # ========================================
        # Étape 1 : Vérifier que les fichiers existent
        # ========================================
        x_path = Path(x_train_path)
        y_path = Path(y_train_path)
        
        if not x_path.exists():
            raise FileNotFoundError(
                f"Fichier X_train non trouvé: {x_train_path}\n"
                f"Vérifiez que le fichier existe et que le chemin est correct."
            )
        if not y_path.exists():
            raise FileNotFoundError(
                f"Fichier Y_train non trouvé: {y_train_path}\n"
                f"Vérifiez que le fichier existe et que le chemin est correct."
            )
        
        # ========================================
        # Étape 2 : Charger les CSV
        # ========================================
        logger.info(f"Chargement de X_train depuis: {x_path}")
        X_train = pd.read_csv(x_path, index_col=index_col)
        
        logger.info(f"Chargement de y_train depuis: {y_path}")
        y_train = pd.read_csv(y_path, index_col=index_col)
        
        # ========================================
        # Étape 3 : Extraire la colonne cible
        # ========================================
        # Y_train peut être :
        # - Un DataFrame avec une colonne 'prdtypecode'
        # - Un DataFrame avec une seule colonne (nom quelconque)
        # On veut toujours une Series en sortie
        
        if 'prdtypecode' in y_train.columns:
            # Cas standard : colonne nommée 'prdtypecode'
            y_train = y_train['prdtypecode']
            logger.debug("Colonne 'prdtypecode' extraite")
        elif y_train.shape[1] == 1:
            # Cas : une seule colonne (peu importe le nom)
            y_train = y_train.iloc[:, 0]
            logger.debug(f"Colonne unique '{y_train.name}' extraite")
        else:
            # Cas problématique : plusieurs colonnes sans 'prdtypecode'
            logger.warning(
                f"Y_train a {y_train.shape[1]} colonnes. "
                f"Utilisation de la première : '{y_train.columns[0]}'"
            )
            y_train = y_train.iloc[:, 0]
        
        # ========================================
        # Étape 4 : Afficher les informations
        # ========================================
        logger.info(f"✓ X_train chargé: {X_train.shape}")
        logger.info(f"✓ y_train chargé: {y_train.shape}")
        
        # ========================================
        # Étape 5 : Valider l'alignement
        # ========================================
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Erreur: X_train ({len(X_train)} lignes) et y_train "
                f"({len(y_train)} lignes) n'ont pas la même taille !\n"
                f"Les données doivent être alignées."
            )
        
        # ========================================
        # Étape 6 : Afficher des statistiques utiles
        # ========================================
        logger.info(f"\nColonnes dans X_train: {list(X_train.columns)}")
        
        # Statistiques sur les classes
        n_classes = y_train.nunique()
        logger.info(f"Nombre de classes: {n_classes}")
        
        # Distribution des classes (top 5)
        class_counts = y_train.value_counts().sort_index()
        logger.info(f"\nDistribution des classes (top 5):")
        for cls, count in class_counts.head(5).items():
            pct = 100 * count / len(y_train)
            logger.info(f"  Classe {cls}: {count} échantillons ({pct:.1f}%)")
        
        if len(class_counts) > 5:
            logger.info(f"  ... et {len(class_counts) - 5} autres classes")
        
        return X_train, y_train


def load_test_data(
    x_test_path: str,
    index_col: int = 0
) -> pd.DataFrame:
    """
    Charge les données de test (features seulement, pas de labels).
    
    Args:
        x_test_path: Chemin vers le fichier X_test.csv
        index_col: Colonne à utiliser comme index (0 par défaut)
        
    Returns:
        DataFrame X_test avec toutes les features
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        
    Exemple:
        >>> X_test = load_test_data("data/raw/X_test_update.csv")
        >>> print(X_test.shape)
        (13812, 4)
    """
    with Timer("Chargement des données de test"):
        # Vérifier que le fichier existe
        x_path = Path(x_test_path)
        
        if not x_path.exists():
            raise FileNotFoundError(
                f"Fichier X_test non trouvé: {x_test_path}\n"
                f"Vérifiez que le fichier existe et que le chemin est correct."
            )
        
        # Charger le CSV
        logger.info(f"Chargement de X_test depuis: {x_path}")
        X_test = pd.read_csv(x_path, index_col=index_col)
        
        # Afficher les informations
        logger.info(f"✓ X_test chargé: {X_test.shape}")
        logger.info(f"Colonnes dans X_test: {list(X_test.columns)}")
        
        return X_test


def validate_dataframes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> None:
    """
    Valide que les DataFrames train et test ont des schémas compatibles.
    
    Vérifie que :
    - Les deux DataFrames ont les mêmes colonnes
    - Les colonnes sont dans le même ordre (important pour sklearn)
    
    Args:
        X_train: DataFrame d'entraînement
        X_test: DataFrame de test
        
    Raises:
        ValueError: Si les colonnes ne correspondent pas
        
    Exemple:
        >>> validate_dataframes(X_train, X_test)
        # Affiche: "✓ Train et test validés (schémas compatibles)"
    """
    logger.info("Validation de la compatibilité train/test...")
    
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    # Vérifier si les colonnes sont identiques
    if train_cols != test_cols:
        # Colonnes manquantes dans test
        missing_in_test = train_cols - test_cols
        # Colonnes en trop dans test
        extra_in_test = test_cols - train_cols
        
        # Construire un message d'erreur détaillé
        msg_parts = []
        if missing_in_test:
            msg_parts.append(
                f"Colonnes manquantes dans X_test: {missing_in_test}"
            )
        if extra_in_test:
            msg_parts.append(
                f"Colonnes en trop dans X_test: {extra_in_test}"
            )
        
        raise ValueError(
            "Les DataFrames train et test ont des colonnes différentes!\n"
            + "\n".join(msg_parts)
        )
    
    # Vérifier l'ordre des colonnes
    if list(X_train.columns) != list(X_test.columns):
        logger.warning(
            "Les colonnes sont les mêmes mais dans un ordre différent. "
            "Cela peut poser problème avec certains modèles."
        )
    
    logger.info("✓ Train et test validés (schémas compatibles)")


def check_missing_values(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Analyse et affiche les informations sur les valeurs manquantes.
    
    Utile pour comprendre la qualité des données et identifier
    les colonnes qui nécessitent un traitement spécial.
    
    Args:
        df: DataFrame à analyser
        name: Nom du DataFrame (pour l'affichage)
        
    Exemple:
        >>> check_missing_values(X_train, name="X_train")
        # Affiche:
        # Valeurs manquantes dans X_train:
        #   description: 2450 (2.88%)
        #   designation: 0 (0.00%)
    """
    logger.info(f"\nAnalyse des valeurs manquantes dans {name}...")
    
    # Compter les valeurs manquantes par colonne
    missing = df.isnull().sum()
    
    if missing.sum() > 0:
        # Il y a des valeurs manquantes
        logger.warning(f"Valeurs manquantes détectées dans {name}:")
        
        for col, count in missing[missing > 0].items():
            pct = 100 * count / len(df)
            logger.warning(f"  {col}: {count} valeurs ({pct:.2f}%)")
        
        total_missing = missing.sum()
        total_cells = len(df) * len(df.columns)
        total_pct = 100 * total_missing / total_cells
        
        logger.warning(
            f"\nTotal: {total_missing} valeurs manquantes sur "
            f"{total_cells} cellules ({total_pct:.2f}%)"
        )
    else:
        # Aucune valeur manquante
        logger.info(f"✓ Aucune valeur manquante dans {name}")


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    # Configuration du logging pour voir les messages
    from src.utils.logging_config import setup_logging
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Démonstration du chargement des données")
    print("="*70 + "\n")
    
    # Exemple fictif (adaptez les chemins à votre configuration)
    try:
        # Charger les données d'entraînement
        X_train, y_train = load_train_data(
            "data/raw/X_train_update.csv",
            "data/raw/Y_train_CVw08PX.csv"
        )
        
        # Charger les données de test
        X_test = load_test_data("data/raw/X_test_update.csv")
        
        # Valider la compatibilité
        validate_dataframes(X_train, X_test)
        
        # Analyser les valeurs manquantes
        check_missing_values(X_train, "X_train")
        check_missing_values(X_test, "X_test")
        
        print("\n" + "="*70)
        print("Chargement réussi!")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        print("\nCe script est une démonstration.")
        print("Adaptez les chemins aux fichiers de votre projet.")

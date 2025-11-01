"""
Module de rééchantillonnage pour gérer le déséquilibre des classes.
=================================================================

Le dataset Rakuten est déséquilibré : certaines catégories ont beaucoup
d'exemples (ex: classe 2583 avec ~5000 échantillons) tandis que d'autres
en ont très peu (ex: certaines classes avec <100 échantillons).

Ce module implémente une stratégie de rééchantillonnage adaptative :
1. **Undersampling** de la classe majoritaire (réduire à ~2500)
2. **Oversampling** des classes minoritaires (augmenter à ~1500)
3. Les classes intermédiaires restent inchangées

Avantages par rapport à class_weight :
- Meilleure convergence pour les modèles linéaires
- Réduit le temps d'entraînement (moins d'échantillons)
- Plus équilibré pour la validation croisée

Utilisation typique :
    from src.data.sampling import apply_sampling
    
    X_resampled, y_resampled = apply_sampling(
        X=X_train,
        y=y_train,
        major_class=2583,
        major_cap=2500,
        tail_min=1500,
        random_state=42
    )

Auteur: Projet Rakuten
Date: 2024
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import resample

from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


def apply_sampling(
    X: pd.DataFrame,
    y: pd.Series,
    major_class: int,
    major_cap: int,
    tail_min: int,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applique une stratégie de rééchantillonnage adaptative.
    
    Stratégie en 3 étapes :
    
    1. **Undersampling** (sous-échantillonnage) de la classe majoritaire
       - Si une classe a plus de `major_cap` échantillons
       - On tire aléatoirement `major_cap` échantillons
       - Exemple : classe 2583 avec 5000 échantillons → 2500
    
    2. **Oversampling** (sur-échantillonnage) des classes minoritaires
       - Si une classe a moins de `tail_min` échantillons
       - On tire avec remplacement pour atteindre `tail_min`
       - Exemple : classe avec 50 échantillons → 1500 (avec doublons)
    
    3. **Conservation** des classes intermédiaires
       - Les classes entre `tail_min` et `major_cap` restent inchangées
    
    Args:
        X: DataFrame des features (n_samples, n_features)
        y: Series des labels (n_samples,)
        major_class: Label de la classe majoritaire (ex: 2583)
        major_cap: Nombre max d'échantillons pour la classe majoritaire
        tail_min: Nombre min d'échantillons pour les petites classes
        random_state: Graine aléatoire pour la reproductibilité
        
    Returns:
        Tuple de (X_resampled, y_resampled)
        - Même structure que X et y en entrée
        - Nouvelle distribution plus équilibrée
        
    Raises:
        ValueError: Si X et y n'ont pas la même longueur
        
    Exemple:
        >>> X_train, y_train = load_train_data(...)
        >>> print(y_train.value_counts().describe())
        count    27.0
        mean     3145.0
        std      1876.0
        min      50.0
        max      5431.0
        
        >>> X_bal, y_bal = apply_sampling(
        ...     X_train, y_train,
        ...     major_class=2583,
        ...     major_cap=2500,
        ...     tail_min=1500
        ... )
        >>> print(y_bal.value_counts().describe())
        count    27.0
        mean     2000.0  # Plus équilibré !
        std      400.0
        min      1500.0
        max      2500.0
    """
    with Timer("Rééchantillonnage des données"):
        # ========================================
        # Validation des entrées
        # ========================================
        if len(X) != len(y):
            raise ValueError(
                f"X ({len(X)} lignes) et y ({len(y)} lignes) doivent "
                f"avoir la même longueur"
            )
        
        # ========================================
        # Afficher la distribution initiale
        # ========================================
        initial_dist = y.value_counts().sort_index()
        logger.info(f"\n{'='*60}")
        logger.info("Distribution AVANT rééchantillonnage:")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {len(y)} échantillons")
        logger.info(f"Nombre de classes: {y.nunique()}")
        logger.info(f"\nTop 5 classes:")
        for cls, count in initial_dist.head(5).items():
            logger.info(f"  Classe {cls}: {count} échantillons")
        
        # ========================================
        # Préparer la liste des indices à conserver
        # ========================================
        indices_to_keep = []
        
        # Traiter chaque classe individuellement
        for cls in y.unique():
            # Récupérer tous les indices de cette classe
            cls_indices = y[y == cls].index.tolist()
            n_samples = len(cls_indices)
            
            # ========================================
            # Décision de stratégie selon le nombre d'échantillons
            # ========================================
            
            if cls == major_class and n_samples > major_cap:
                # --------------------------------
                # CAS 1: Classe majoritaire → UNDERSAMPLING
                # --------------------------------
                sampled = resample(
                    cls_indices,
                    n_samples=major_cap,
                    random_state=random_state,
                    replace=False  # Sans remplacement (pas de doublons)
                )
                logger.info(
                    f"Classe {cls} (majoritaire): "
                    f"{n_samples} → {major_cap} (sous-échantillonnage)"
                )
                indices_to_keep.extend(sampled)
                
            elif n_samples < tail_min:
                # --------------------------------
                # CAS 2: Classe minoritaire → OVERSAMPLING
                # --------------------------------
                sampled = resample(
                    cls_indices,
                    n_samples=tail_min,
                    random_state=random_state,
                    replace=True  # AVEC remplacement (doublons possibles)
                )
                logger.info(
                    f"Classe {cls} (minoritaire): "
                    f"{n_samples} → {tail_min} (sur-échantillonnage)"
                )
                indices_to_keep.extend(sampled)
                
            else:
                # --------------------------------
                # CAS 3: Classe intermédiaire → INCHANGÉE
                # --------------------------------
                logger.info(
                    f"Classe {cls}: {n_samples} (inchangé)"
                )
                indices_to_keep.extend(cls_indices)
        
        # ========================================
        # Réindexer avec les échantillons sélectionnés
        # ========================================
        X_resampled = X.loc[indices_to_keep]
        y_resampled = y.loc[indices_to_keep]
        
        # ========================================
        # Mélanger aléatoirement (important !)
        # ========================================
        # Sans mélange, toutes les images d'une classe seraient consécutives
        # Ce qui peut poser problème pour la validation croisée
        shuffled_indices = np.random.RandomState(random_state).permutation(
            len(X_resampled)
        )
        X_resampled = X_resampled.iloc[shuffled_indices].reset_index(drop=True)
        y_resampled = y_resampled.iloc[shuffled_indices].reset_index(drop=True)
        
        # ========================================
        # Afficher la distribution finale
        # ========================================
        final_dist = y_resampled.value_counts().sort_index()
        logger.info(f"\n{'='*60}")
        logger.info("Distribution APRÈS rééchantillonnage:")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {len(y_resampled)} échantillons")
        logger.info(f"\nTop 5 classes:")
        for cls, count in final_dist.head(5).items():
            logger.info(f"  Classe {cls}: {count} échantillons")
        
        # Calcul du changement total
        diff = len(y_resampled) - len(y)
        sign = "+" if diff > 0 else ""
        logger.info(f"\nChangement total: {sign}{diff} échantillons")
        
        return X_resampled, y_resampled


def get_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calcule les poids de classe pour sklearn (format class_weight).
    
    Alternative au rééchantillonnage : au lieu de modifier le dataset,
    on peut dire au modèle de donner plus d'importance aux classes rares.
    
    Formule : weight[c] = n_samples / (n_classes * n_samples_class[c])
    
    Args:
        y: Series des labels
        
    Returns:
        Dictionnaire {classe: poids}
        - Classes rares → poids élevé (ex: 5.0)
        - Classes fréquentes → poids faible (ex: 0.2)
        
    Exemple:
        >>> weights = get_class_weights(y_train)
        >>> print(weights)
        {10: 3.2, 40: 1.5, 2583: 0.4, ...}
        
        # Utilisation avec sklearn
        >>> from sklearn.linear_model import LogisticRegression
        >>> model = LogisticRegression(class_weight=weights)
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculer les poids automatiquement
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    # Convertir en dictionnaire
    class_weights = dict(zip(classes, weights))
    
    logger.info("Poids de classe calculés:")
    # Afficher les 5 poids les plus élevés (classes les plus rares)
    sorted_weights = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 5 poids (classes rares):")
    for cls, weight in sorted_weights[:5]:
        logger.info(f"  Classe {cls}: {weight:.3f}")
    
    return class_weights


def analyze_class_distribution(
    y: pd.Series,
    name: str = "Dataset"
) -> pd.DataFrame:
    """
    Analyse et affiche des statistiques détaillées sur la distribution des classes.
    
    Utile pour comprendre le déséquilibre et décider de la stratégie de sampling.
    
    Args:
        y: Series des labels
        name: Nom du dataset (pour l'affichage)
        
    Returns:
        DataFrame avec les statistiques par classe :
        - class: Numéro de la classe
        - count: Nombre d'échantillons
        - percentage: Pourcentage du total
        
    Exemple:
        >>> stats = analyze_class_distribution(y_train, "Train")
        # Affiche:
        # Distribution des classes (Train):
        # ┌───────┬───────┬────────────┐
        # │ class │ count │ percentage │
        # ├───────┼───────┼────────────┤
        # │  2583 │  5431 │     6.4%   │
        # │  2705 │  4563 │     5.4%   │
        # │   ...
    """
    counts = y.value_counts().sort_index()
    percentages = 100 * counts / len(y)
    
    # Créer un DataFrame avec les statistiques
    stats = pd.DataFrame({
        'class': counts.index,
        'count': counts.values,
        'percentage': percentages.values
    })
    
    # Affichage formaté
    logger.info(f"\n{'='*60}")
    logger.info(f"Distribution des classes ({name})")
    logger.info(f"{'='*60}")
    logger.info(f"\n{stats.to_string(index=False)}")
    
    # Statistiques globales
    logger.info(f"\n{'='*60}")
    logger.info("Statistiques globales:")
    logger.info(f"{'='*60}")
    logger.info(f"Nombre total d'échantillons: {len(y)}")
    logger.info(f"Nombre de classes: {len(stats)}")
    logger.info(f"\nClasse la plus fréquente: {stats.iloc[0]['class']} "
                f"({int(stats.iloc[0]['count'])} échantillons)")
    logger.info(f"Classe la moins fréquente: {stats.iloc[-1]['class']} "
                f"({int(stats.iloc[-1]['count'])} échantillons)")
    logger.info(f"\nRatio max/min: {stats['count'].max() / stats['count'].min():.1f}x")
    
    return stats


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    # Configuration du logging
    from src.utils.logging_config import setup_logging
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Démonstration du rééchantillonnage")
    print("="*70 + "\n")
    
    # Créer des données exemple (simulées)
    np.random.seed(42)
    
    # Simuler un dataset déséquilibré
    # Classe 0: 5000 échantillons (majoritaire)
    # Classe 1: 2000 échantillons (intermédiaire)
    # Classe 2: 100 échantillons (minoritaire)
    
    y_example = pd.Series(
        [0] * 5000 + [1] * 2000 + [2] * 100,
        name='class'
    )
    
    X_example = pd.DataFrame({
        'feature1': np.random.randn(len(y_example)),
        'feature2': np.random.randn(len(y_example))
    })
    
    # Analyser la distribution initiale
    analyze_class_distribution(y_example, "Avant sampling")
    
    # Appliquer le rééchantillonnage
    X_bal, y_bal = apply_sampling(
        X=X_example,
        y=y_example,
        major_class=0,
        major_cap=2500,
        tail_min=1500,
        random_state=42
    )
    
    # Analyser la distribution finale
    analyze_class_distribution(y_bal, "Après sampling")
    
    print("\n" + "="*70)
    print("Démonstration terminée!")
    print("="*70 + "\n")

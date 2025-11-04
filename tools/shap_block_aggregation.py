"""
Script pour agréger les valeurs SHAP par blocs de features.

Au lieu d'avoir 102,058 importances individuelles, on regroupe par composante:
- text_tfidf: 1 importance globale
- text_has_desc: 1 importance
- text_stats: 1 importance (ou garder les 5 sous-features)
- image_cnn: 1 importance
etc.

Usage:
    python shap_block_aggregation.py --shap_values results/shap/shap_values.npy \
                                      --mapping artifacts/feature_mapping.json \
                                      --output results/shap/shap_blocks.png
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, Tuple

def load_feature_mapping(mapping_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Charge le mapping des features depuis le fichier JSON.
    
    Args:
        mapping_path: Chemin vers feature_mapping.json
        
    Returns:
        Dict {block_name: (start_idx, end_idx)}
    """
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    return mapping


def aggregate_shap_by_block(
    shap_values: np.ndarray,
    mapping: Dict[str, Tuple[int, int]],
    aggregation: str = "mean_abs"
) -> Dict[str, np.ndarray]:
    """
    Agrège les valeurs SHAP par bloc de features.
    
    Args:
        shap_values: Array (n_samples, n_features) des valeurs SHAP
        mapping: Dict {block_name: (start_idx, end_idx)}
        aggregation: Méthode d'agrégation
            - "mean_abs": Moyenne des valeurs absolues (importance)
            - "sum_abs": Somme des valeurs absolues
            - "mean": Moyenne signée (peut être négative)
            - "max_abs": Maximum des valeurs absolues
            
    Returns:
        Dict {block_name: array (n_samples,)} valeurs agrégées par échantillon
    """
    aggregated = {}
    
    for block_name, (start, end) in mapping.items():
        # Extraire les SHAP values du bloc
        block_shap = shap_values[:, start:end]
        
        # Agréger selon la méthode choisie
        if aggregation == "mean_abs":
            # Moyenne des valeurs absolues (importance moyenne)
            agg_values = np.mean(np.abs(block_shap), axis=1)
        elif aggregation == "sum_abs":
            # Somme des valeurs absolues (importance totale)
            agg_values = np.sum(np.abs(block_shap), axis=1)
        elif aggregation == "mean":
            # Moyenne signée (direction de l'effet)
            agg_values = np.mean(block_shap, axis=1)
        elif aggregation == "max_abs":
            # Feature la plus importante du bloc
            agg_values = np.max(np.abs(block_shap), axis=1)
        else:
            raise ValueError(f"Méthode d'agrégation inconnue: {aggregation}")
        
        aggregated[block_name] = agg_values
    
    return aggregated


def compute_global_importance(
    aggregated_shap: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Calcule l'importance globale de chaque bloc (moyenne sur tous les échantillons).
    
    Args:
        aggregated_shap: Dict {block_name: array (n_samples,)}
        
    Returns:
        Dict {block_name: importance_globale}
    """
    return {
        name: float(np.mean(values))
        for name, values in aggregated_shap.items()
    }


def plot_block_importance(
    block_importance: Dict[str, float],
    output_path: str = None,
    title: str = "Importance SHAP par bloc de features",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Visualise l'importance par bloc de features.
    
    Args:
        block_importance: Dict {block_name: importance}
        output_path: Chemin de sauvegarde (optionnel)
        title: Titre du graphique
        figsize: Taille de la figure
    """
    # Trier par importance décroissante
    sorted_blocks = sorted(block_importance.items(), key=lambda x: x[1], reverse=True)
    blocks, importances = zip(*sorted_blocks)
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=figsize)
    
    # Barplot horizontal
    colors = sns.color_palette("viridis", len(blocks))
    bars = ax.barh(range(len(blocks)), importances, color=colors)
    
    # Labels
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(blocks)
    ax.set_xlabel("Importance SHAP moyenne (|SHAP|)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Ajouter les valeurs sur les barres
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=10)
    
    # Grille
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Sauvegarder
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {output_path}")
    
    plt.show()


def plot_block_importance_per_class(
    shap_values: np.ndarray,
    y_pred: np.ndarray,
    mapping: Dict[str, Tuple[int, int]],
    output_path: str = None,
    top_classes: int = 10
):
    """
    Visualise l'importance par bloc pour chaque classe (top N classes).
    
    Args:
        shap_values: Array (n_samples, n_features)
        y_pred: Array (n_samples,) prédictions
        mapping: Dict des blocs
        output_path: Chemin de sauvegarde
        top_classes: Nombre de classes à afficher
    """
    from collections import Counter
    
    # Identifier les classes les plus fréquentes
    class_counts = Counter(y_pred)
    top_class_ids = [c for c, _ in class_counts.most_common(top_classes)]
    
    # Calculer importance par classe
    results = []
    
    for class_id in top_class_ids:
        # Sélectionner les échantillons de cette classe
        mask = y_pred == class_id
        shap_class = shap_values[mask]
        
        # Agréger par bloc
        agg_class = aggregate_shap_by_block(shap_class, mapping, aggregation="mean_abs")
        block_imp = compute_global_importance(agg_class)
        
        # Stocker
        for block_name, importance in block_imp.items():
            results.append({
                "class": class_id,
                "block": block_name,
                "importance": importance
            })
    
    # Créer un DataFrame pour heatmap
    import pandas as pd
    df = pd.DataFrame(results)
    pivot = df.pivot(index="block", columns="class", values="importance")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Importance SHAP"})
    ax.set_title("Importance SHAP par bloc et par classe", fontsize=14, fontweight='bold')
    ax.set_xlabel("Classe prédite", fontsize=12)
    ax.set_ylabel("Bloc de features", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap sauvegardée: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Agrégation SHAP par blocs de features")
    parser.add_argument("--shap_values", required=True, help="Chemin vers shap_values.npy")
    parser.add_argument("--mapping", required=True, help="Chemin vers feature_mapping.json")
    parser.add_argument("--predictions", help="Chemin vers y_pred.npy (optionnel, pour analyse par classe)")
    parser.add_argument("--output", default="results/shap/shap_blocks.png", help="Chemin de sortie")
    parser.add_argument("--per_class", action="store_true", help="Générer aussi l'analyse par classe")
    parser.add_argument("--aggregation", default="mean_abs", choices=["mean_abs", "sum_abs", "mean", "max_abs"],
                        help="Méthode d'agrégation")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AGRÉGATION SHAP PAR BLOCS DE FEATURES")
    print("=" * 70)
    
    # Charger les données
    print(f"\n[1/4] Chargement des SHAP values: {args.shap_values}")
    shap_values = np.load(args.shap_values)
    print(f"  Shape: {shap_values.shape}")
    
    print(f"\n[2/4] Chargement du mapping: {args.mapping}")
    mapping = load_feature_mapping(args.mapping)
    print(f"  Nombre de blocs: {len(mapping)}")
    for block_name, (start, end) in mapping.items():
        print(f"    - {block_name:25s}: colonnes {start:6d} - {end:6d} ({end-start:6d} features)")
    
    # Agréger
    print(f"\n[3/4] Agrégation par blocs (méthode: {args.aggregation})")
    aggregated = aggregate_shap_by_block(shap_values, mapping, aggregation=args.aggregation)
    block_importance = compute_global_importance(aggregated)
    
    print("\nImportance globale par bloc:")
    for block_name, importance in sorted(block_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {block_name:25s}: {importance:.6f}")
    
    # Visualiser
    print(f"\n[4/4] Génération du graphique: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_block_importance(block_importance, str(output_path))
    
    # Analyse par classe (optionnel)
    if args.per_class and args.predictions:
        print(f"\n[BONUS] Analyse par classe...")
        y_pred = np.load(args.predictions)
        output_per_class = output_path.parent / (output_path.stem + "_per_class" + output_path.suffix)
        plot_block_importance_per_class(shap_values, y_pred, mapping, str(output_per_class))
    
    print("\n" + "=" * 70)
    print("✓ Agrégation terminée avec succès !")
    print("=" * 70)


if __name__ == "__main__":
    main()
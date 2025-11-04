#!/usr/bin/env python3
"""
Script de diagnostic : Vérifier quelles branches du pipeline sont actives.
==========================================================================

Utilisation:
    python tools/check_pipeline_branches.py
    python tools/check_pipeline_branches.py --config config/config.toml
"""
import sys
import argparse
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline


def main():
    parser = argparse.ArgumentParser(description="Diagnostic du pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.toml",
        help="Chemin vers la config"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC DU PIPELINE")
    print("="*70)
    print(f"Config: {args.config}\n")
    
    # Charger config
    config = load_config(args.config)
    
    # ========================================
    # 1. Vérifier la config
    # ========================================
    print(" Configuration Détectée:")
    print("-" * 70)
    
    # Texte
    text_enabled = True  # Toujours activé
    print(f"✓ Texte: ACTIVÉ")
    print(f"  - max_features: {config.get('features.text.max_features', 'N/A')}")
    print(f"  - use_stem: {config.get('features.text.use_stem', 'N/A')}")
    
    # Images pixels
    pixels_enabled = config.get("features.image.pixels.enabled", False)
    print(f"\n{'✓' if pixels_enabled else '✗'} Images Pixels: {'ACTIVÉ' if pixels_enabled else 'DÉSACTIVÉ'}")
    if pixels_enabled:
        print(f"  - size: {config.get('features.image.pixels.size', 'N/A')}")
        print(f"  - PCA: {config.get('features.image.pixels.pca.enabled', False)}")
        if config.get('features.image.pixels.pca.enabled'):
            print(f"  - n_components: {config.get('features.image.pixels.pca.n_components', 'N/A')}")
    
    # Images stats
    stats_enabled = config.get("features.image.stats.enabled", False)
    print(f"\n{'✓' if stats_enabled else '✗'} Images Stats: {'ACTIVÉ' if stats_enabled else 'DÉSACTIVÉ'}")
    if stats_enabled:
        print(f"  - fast: {config.get('features.image.stats.fast', 'N/A')}")
    
    # CNN
    cnn_enabled = config.get("features.image.cnn.enabled", False)
    print(f"\n{'✓' if cnn_enabled else '✗'} CNN ResNet: {'ACTIVÉ' if cnn_enabled else 'DÉSACTIVÉ'}")
    if cnn_enabled:
        print(f"  - arch: {config.get('features.image.cnn.arch', 'N/A')}")
        print(f"  - batch_size: {config.get('features.image.cnn.batch_size', 'N/A')}")
    
    # ========================================
    # 2. Vérifier les poids de fusion
    # ========================================
    print("\n  Poids de Fusion:")
    print("-" * 70)
    
    weights = {
        "text": config.get("fusion.weights.text", 1.0),
        "image_pixels": config.get("fusion.weights.image_pixels", 0.0),
        "image_stats": config.get("fusion.weights.image_stats", 0.0),
        "image_cnn": config.get("fusion.weights.image_cnn", 0.0),
    }
    
    for name, weight in weights.items():
        status = "✓" if weight > 0 else "✗"
        print(f"{status} {name}: {weight}")
    
    # ========================================
    # 3. Construire le pipeline
    # ========================================
    print("\n🔧 Construction du Pipeline:")
    print("-" * 70)
    
    try:
        pipeline = DataTransformationPipeline(config)
        feature_pipeline = pipeline.build_feature_pipeline()
        
        print(f"✓ Pipeline construit avec succès")
        print(f"  Nombre de branches: {len(feature_pipeline.transformer_list)}")
        
        print("\n Branches Actives:")
        for name, transformer in feature_pipeline.transformer_list:
            print(f"  ✓ {name}")
            print(f"    Type: {type(transformer).__name__}")
        
        # Poids
        if hasattr(feature_pipeline, 'transformer_weights') and feature_pipeline.transformer_weights:
            print("\n  Poids Appliqués:")
            for name, weight in feature_pipeline.transformer_weights.items():
                print(f"  {name}: {weight}")
        
    except Exception as e:
        print(f"✗ Erreur lors de la construction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================
    # 4. Résumé et recommandations
    # ========================================
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    
    # Compter branches actives
    n_branches = len(feature_pipeline.transformer_list)
    
    if n_branches == 1:
        print("  UNE SEULE BRANCHE ACTIVE (texte uniquement)")
        print("\nRecommandations:")
        print("  1. Activer les images dans config.toml:")
        print("     [features.image.pixels]")
        print("     enabled = true")
        print("  2. Donner du poids aux images:")
        print("     [fusion.weights]")
        print("     image_pixels = 0.5")
    
    elif n_branches == 2:
        print("✓ DEUX BRANCHES ACTIVES (texte + images)")
        print("\nGain attendu: +2-5% F1-score")
        
        # Vérifier que les poids ne sont pas à 0
        non_zero_weights = sum(1 for w in weights.values() if w > 0)
        if non_zero_weights < 2:
            print("\n  ATTENTION: Certaines branches ont un poids de 0 !")
            print("   Les images ne contribueront pas au modèle.")
    
    else:
        print(f"✓ {n_branches} BRANCHES ACTIVES")
        print("\nGain attendu: +5-10% F1-score")
    
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
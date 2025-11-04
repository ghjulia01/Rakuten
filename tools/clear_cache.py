#!/usr/bin/env python3
"""
Nettoie le cache des features pour forcer le recalcul.
======================================================

Utilisation:
    python tools/clear_cache.py
    python tools/clear_cache.py --all  # Nettoie aussi les modèles
"""
import argparse
import shutil
from pathlib import Path


def clear_feature_cache():
    """Nettoie le cache des features."""
    cache_dir = Path("artifacts/cache/features")
    
    if cache_dir.exists():
        num_files = len(list(cache_dir.glob("*.joblib")))
        shutil.rmtree(cache_dir)
        print(f" Cache des features supprimé: {cache_dir}")
        print(f"   {num_files} fichier(s) supprimé(s)")
    else:
        print("ℹ  Pas de cache features à supprimer")


def clear_model_cache():
    """Nettoie le cache des modèles."""
    artifacts_dir = Path("artifacts")
    
    if artifacts_dir.exists():
        # Supprimer les modèles intermédiaires
        patterns = ["model_*.joblib", "*_full_pipeline.joblib"]
        total_deleted = 0
        
        for pattern in patterns:
            for file in artifacts_dir.glob(f"**/{pattern}"):
                file.unlink()
                total_deleted += 1
        
        if total_deleted > 0:
            print(f" {total_deleted} modèle(s) intermédiaire(s) supprimé(s)")
        else:
            print("ℹ  Pas de modèles intermédiaires à supprimer")
    else:
        print("ℹ  Dossier artifacts/ inexistant")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Nettoie le cache du pipeline"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Nettoie également les modèles intermédiaires"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("NETTOYAGE DU CACHE")
    print("="*70 + "\n")
    
    # Nettoyer le cache des features
    clear_feature_cache()
    
    # Nettoyer les modèles si demandé
    if args.all:
        clear_model_cache()
    
    print("\n" + "="*70)
    print(" Nettoyage terminé")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

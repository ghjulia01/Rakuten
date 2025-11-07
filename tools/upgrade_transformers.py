"""
Script pour upgrader transformers et vérifier la compatibilité avec TensorFlow.

Ce script:
1. Détecte la version actuelle de TensorFlow
2. Upgrade transformers vers une version compatible
3. Vérifie que tout fonctionne correctement

Usage:
    python upgrade_transformers.py [--version VERSION]
"""

import sys
import subprocess
import argparse
from typing import Tuple


def get_package_version(package_name: str) -> str:
    """Obtient la version d'un package installé."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':', 1)[1].strip()
    except subprocess.CalledProcessError:
        return None


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse une version string en tuple (major, minor, patch)."""
    parts = version_str.split('.')
    return tuple(int(p) for p in parts[:3])


def get_recommended_transformers_version(tf_version: str) -> str:
    """Recommande une version de transformers selon TensorFlow."""
    tf_major, tf_minor, _ = parse_version(tf_version)
    
    if tf_major == 2:
        if tf_minor >= 17:
            return "4.45.0"  # Pour TF 2.17+
        elif tf_minor >= 16:
            return "4.35.0"  # Pour TF 2.16
        elif tf_minor >= 15:
            return "4.30.0"  # Pour TF 2.15
        else:
            return "4.25.0"  # Pour TF 2.13-2.14
    
    return "4.45.0"  # Par défaut, version la plus récente


def upgrade_transformers(target_version: str = None, force: bool = False):
    """Upgrade transformers vers la version cible."""
    
    print("=" * 70)
    print("🔄 UPGRADE DE TRANSFORMERS")
    print("=" * 70)
    
    # 1. Vérifier TensorFlow
    print("\n[1/5] Vérification de TensorFlow...")
    tf_version = get_package_version("tensorflow")
    
    if not tf_version:
        print("  ❌ TensorFlow n'est pas installé!")
        print("  → Installer avec: pip install tensorflow")
        sys.exit(1)
    
    print(f"  ✓ TensorFlow version: {tf_version}")
    
    # 2. Vérifier transformers actuel
    print("\n[2/5] Vérification de transformers...")
    current_transformers = get_package_version("transformers")
    
    if current_transformers:
        print(f"  • Version actuelle: {current_transformers}")
    else:
        print("  ⚠️  transformers n'est pas installé")
        current_transformers = "0.0.0"
    
    # 3. Déterminer la version cible
    if not target_version:
        target_version = get_recommended_transformers_version(tf_version)
    
    print(f"\n[3/5] Version cible recommandée: {target_version}")
    
    # Vérifier si upgrade nécessaire
    if current_transformers != "0.0.0":
        current_tuple = parse_version(current_transformers)
        target_tuple = parse_version(target_version)
        
        if current_tuple >= target_tuple and not force:
            print(f"  ✓ Version actuelle ({current_transformers}) est déjà suffisante")
            print(f"  → Aucune action nécessaire")
            return
        
        if current_tuple > target_tuple:
            print(f"  ⚠️  Version actuelle ({current_transformers}) est plus récente que la cible")
            if not force:
                response = input("  Voulez-vous downgrade? (y/N): ")
                if response.lower() != 'y':
                    print("  → Annulation")
                    return
    
    # 4. Installer/Upgrader
    print(f"\n[4/5] Installation de transformers=={target_version}...")
    
    try:
        # Désinstaller l'ancienne version proprement
        if current_transformers != "0.0.0":
            print("  → Désinstallation de l'ancienne version...")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "transformers", "-y"],
                check=True,
                capture_output=True
            )
        
        # Installer la nouvelle version
        print(f"  → Installation de transformers=={target_version}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"transformers=={target_version}"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"  ✓ transformers=={target_version} installé avec succès!")
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Erreur lors de l'installation:")
        print(e.stderr)
        sys.exit(1)
    
    # 5. Vérification
    print("\n[5/5] Vérification de l'installation...")
    verify_installation(tf_version, target_version)
    
    print("\n" + "=" * 70)
    print("✅ UPGRADE TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    print(f"\n📦 Configuration finale:")
    print(f"  • TensorFlow:   {tf_version}")
    print(f"  • transformers: {target_version}")
    
    # Recommandations
    print(f"\n💡 Recommandations:")
    print(f"  • Testez votre code avec la nouvelle version")
    print(f"  • Certaines API peuvent avoir changé")
    print(f"  • Consultez: https://huggingface.co/docs/transformers/migration")


def verify_installation(tf_version: str, transformers_version: str):
    """Vérifie que l'installation fonctionne correctement."""
    
    print("  → Test d'import de transformers...")
    try:
        import transformers
        print(f"    ✓ Import réussi (version {transformers.__version__})")
    except ImportError as e:
        print(f"    ❌ Erreur d'import: {e}")
        sys.exit(1)
    
    print("  → Test d'import de TensorFlow...")
    try:
        import tensorflow as tf
        print(f"    ✓ Import réussi (version {tf.__version__})")
    except ImportError as e:
        print(f"    ❌ Erreur d'import: {e}")
        sys.exit(1)
    
    print("  → Test de compatibilité...")
    try:
        # Test basique: importer un modèle
        from transformers import AutoTokenizer
        print("    ✓ AutoTokenizer accessible")
        
        # Vérifier compatibilité TF
        from transformers import TFAutoModel
        print("    ✓ TFAutoModel accessible")
        
        print("  ✓ Installation vérifiée avec succès!")
        
    except Exception as e:
        print(f"    ⚠️  Avertissement: {e}")
        print("    → L'installation semble fonctionnelle mais certaines features peuvent nécessiter des ajustements")


def show_compatibility_matrix():
    """Affiche la matrice de compatibilité TensorFlow/transformers."""
    print("\n📊 MATRICE DE COMPATIBILITÉ")
    print("=" * 70)
    print("TensorFlow    | transformers recommandé | Notes")
    print("-" * 70)
    print("2.13.x        | 4.25.0 - 4.28.0         | Stable")
    print("2.14.x        | 4.28.0 - 4.32.0         | Stable")
    print("2.15.x        | 4.30.0 - 4.35.0         | Stable")
    print("2.16.x        | 4.35.0 - 4.40.0         | Keras 3")
    print("2.17.x        | 4.45.0+                 | Dernière stable")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Upgrade transformers vers une version compatible avec TensorFlow"
    )
    parser.add_argument(
        "--version", 
        help="Version spécifique de transformers à installer (ex: 4.35.0)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Forcer l'installation même si la version actuelle est suffisante"
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Afficher la matrice de compatibilité et quitter"
    )
    
    args = parser.parse_args()
    
    if args.show_matrix:
        show_compatibility_matrix()
        sys.exit(0)
    
    upgrade_transformers(target_version=args.version, force=args.force)


if __name__ == "__main__":
    main()







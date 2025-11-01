#!/usr/bin/env python3
"""
Script de vérification de l'état de la restructuration du projet Rakuten.
Vérifie que tous les fichiers nécessaires sont présents et bien configurés.
"""
import sys
from pathlib import Path
from typing import List, Tuple

# Codes couleur pour terminal
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def check_file(path: Path, required: bool = True) -> Tuple[bool, str]:
    """Vérifie si un fichier existe."""
    exists = path.exists()
    if exists:
        return True, f"{GREEN}✓{RESET} {path}"
    elif required:
        return False, f"{RED}✗{RESET} {path} (MANQUANT - REQUIS)"
    else:
        return True, f"{YELLOW}⚠{RESET} {path} (optionnel, absent)"


def check_directory(path: Path) -> Tuple[bool, str]:
    """Vérifie si un répertoire existe."""
    exists = path.is_dir()
    if exists:
        return True, f"{GREEN}✓{RESET} {path}/"
    else:
        return False, f"{RED}✗{RESET} {path}/ (MANQUANT)"


def check_python_imports() -> List[Tuple[bool, str]]:
    """Vérifie que les modules Python peuvent être importés."""
    results = []
    
    # Ajouter le répertoire au path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    imports_to_test = [
        ("src", True),
        ("src.utils.config", True),
        ("src.utils.profiling", True),
        ("src.utils.logging_config", True),
        ("src.data.load_data", True),
        ("src.data.sampling", True),
        ("src.features.text.cleaner", False),  # À adapter
        ("src.features.image.loader", False),  # À adapter
    ]
    
    for module_name, required in imports_to_test:
        try:
            __import__(module_name)
            results.append((True, f"{GREEN}✓{RESET} Import: {module_name}"))
        except ImportError as e:
            if required:
                results.append((False, f"{RED}✗{RESET} Import: {module_name} (ERREUR: {e})"))
            else:
                results.append((True, f"{YELLOW}⚠{RESET} Import: {module_name} (à adapter)"))
    
    return results


def main():
    """Fonction principale de vérification."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Vérification de la Restructuration du Projet Rakuten{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")
    
    project_root = Path(__file__).parent
    all_checks: List[Tuple[bool, str]] = []
    
    # 1. Structure de répertoires
    print(f"{BOLD}1. Structure de répertoires{RESET}")
    print("-" * 70)
    
    directories = [
        "src",
        "src/data",
        "src/features",
        "src/features/text",
        "src/features/image",
        "src/models",
        "src/pipelines",
        "src/visualization",
        "src/utils",
        "config",
        "data",
        "data/raw",
        "data/processed",
        "data/images",
        "models",
        "results",
        "results/predictions",
        "results/metrics",
        "results/figures",
        "scripts",
        "notebooks",
        "tests",
    ]
    
    for dir_name in directories:
        success, msg = check_directory(project_root / dir_name)
        all_checks.append((success, msg))
        print(msg)
    
    # 2. Fichiers de configuration
    print(f"\n{BOLD}2. Fichiers de configuration{RESET}")
    print("-" * 70)
    
    config_files = [
        ("config/config.toml", True),
        ("config/labels_map.json", True),
        ("config/theme_map.json", False),
        ("config/translate_map_starter_from_cleaned.json", False),
    ]
    
    for file_path, required in config_files:
        success, msg = check_file(project_root / file_path, required)
        all_checks.append((success, msg))
        print(msg)
    
    # 3. Fichiers source Python
    print(f"\n{BOLD}3. Fichiers source Python{RESET}")
    print("-" * 70)
    
    source_files = [
        ("src/__init__.py", True),
        ("src/utils/__init__.py", True),
        ("src/utils/config.py", True),
        ("src/utils/profiling.py", True),
        ("src/utils/logging_config.py", True),
        ("src/data/__init__.py", True),
        ("src/data/load_data.py", True),
        ("src/data/sampling.py", True),
        ("src/features/__init__.py", True),
        ("src/features/text/__init__.py", True),
        ("src/features/text/cleaner.py", False),  # Peut nécessiter adaptation
        ("src/features/text/vectorizer.py", False),
        ("src/features/text/stats.py", False),
        ("src/features/image/__init__.py", True),
        ("src/features/image/loader.py", False),
        ("src/features/image/stats.py", False),
        ("src/pipelines/__init__.py", True),
        ("src/pipelines/text_pipeline.py", False),
        ("src/pipelines/image_pipeline.py", False),
        ("src/models/__init__.py", True),
    ]
    
    for file_path, required in source_files:
        success, msg = check_file(project_root / file_path, required)
        all_checks.append((success, msg))
        print(msg)
    
    # 4. Scripts
    print(f"\n{BOLD}4. Scripts exécutables{RESET}")
    print("-" * 70)
    
    script_files = [
        ("scripts/train.py", True),
        ("scripts/predict.py", False),
        ("scripts/evaluate.py", False),
    ]
    
    for file_path, required in script_files:
        success, msg = check_file(project_root / file_path, required)
        all_checks.append((success, msg))
        print(msg)
    
    # 5. Documentation
    print(f"\n{BOLD}5. Documentation{RESET}")
    print("-" * 70)
    
    doc_files = [
        ("README.md", True),
        ("ARCHITECTURE.md", True),
        ("MIGRATION_GUIDE.md", True),
        ("SYNTHESE.md", True),
        ("setup.py", True),
        ("requirements.txt", True),
        (".gitignore", True),
    ]
    
    for file_path, required in doc_files:
        success, msg = check_file(project_root / file_path, required)
        all_checks.append((success, msg))
        print(msg)
    
    # 6. Imports Python
    print(f"\n{BOLD}6. Imports Python{RESET}")
    print("-" * 70)
    
    import_checks = check_python_imports()
    all_checks.extend(import_checks)
    for _, msg in import_checks:
        print(msg)
    
    # 7. Données
    print(f"\n{BOLD}7. Données (optionnel){RESET}")
    print("-" * 70)
    
    data_files = [
        ("data/raw/X_train_update.csv", False),
        ("data/raw/Y_train_CVw08PX.csv", False),
        ("data/raw/X_test_update.csv", False),
    ]
    
    for file_path, required in data_files:
        success, msg = check_file(project_root / file_path, required)
        all_checks.append((success, msg))
        print(msg)
    
    # Résumé
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Résumé{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")
    
    total = len(all_checks)
    passed = sum(1 for success, _ in all_checks if success)
    failed = total - passed
    
    print(f"Total de vérifications : {total}")
    print(f"{GREEN}Réussies : {passed}{RESET}")
    if failed > 0:
        print(f"{RED}Échouées : {failed}{RESET}")
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\nTaux de réussite : {success_rate:.1f}%")
    
    if success_rate == 100:
        print(f"\n{GREEN}{BOLD}✨ Restructuration complète ! Tous les éléments essentiels sont en place.{RESET}")
    elif success_rate >= 70:
        print(f"\n{YELLOW}{BOLD}⚠ Restructuration partielle. Quelques éléments à compléter.{RESET}")
        print("Consultez MIGRATION_GUIDE.md pour les prochaines étapes.")
    else:
        print(f"\n{RED}{BOLD}✗ Restructuration incomplète. Plusieurs éléments manquants.{RESET}")
        print("Consultez SYNTHESE.md et MIGRATION_GUIDE.md pour plus d'informations.")
    
    print(f"\n{BOLD}{'='*70}{RESET}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

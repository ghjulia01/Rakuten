"""
Module de configuration du système de logging pour le projet Rakuten.
===================================================================

Ce module centralise la configuration des logs pour garantir un affichage
cohérent dans toute l'application.

Fonctionnalités :
- Configuration simple du niveau de log (INFO, DEBUG, etc.)
- Sortie vers console et/ou fichier
- Format personnalisable
- Réduction du bruit des bibliothèques tierces

Utilisation typique :
    # Au début de votre script principal
    from src.utils.logging_config import setup_logging
    
    setup_logging(level=logging.INFO)
    
    # Puis partout ailleurs
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Mon message")

"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure le système de logging pour tout le projet.
    
    Cette fonction doit être appelée UNE SEULE FOIS au début du programme
    principal (généralement dans scripts/train.py ou scripts/predict.py).
    
    Args:
        level: Niveau de logging minimum à afficher
               - logging.DEBUG : Tous les messages (très verbeux)
               - logging.INFO : Messages informatifs (recommandé)
               - logging.WARNING : Seulement les avertissements et erreurs
               - logging.ERROR : Seulement les erreurs
               
        log_file: Chemin optionnel vers un fichier de log
                  Si fourni, les logs seront écrits dans ce fichier
                  EN PLUS de la console
                  Exemple: "logs/train_2024-01-15.log"
                  
        format_string: Format personnalisé pour les messages
                      Si None, utilise un format standard
                      
    Exemples:
        # Configuration simple (logs INFO vers console)
        setup_logging()
        
        # Mode debug (logs DEBUG vers console)
        setup_logging(level=logging.DEBUG)
        
        # Logs dans un fichier
        setup_logging(
            level=logging.INFO,
            log_file="logs/train.log"
        )
        
        # Format personnalisé
        setup_logging(
            level=logging.INFO,
            format_string="%(levelname)s - %(message)s"
        )
    """
    # Format par défaut si non spécifié
    # Affiche: 2024-01-15 10:30:45 - module.name - INFO - Mon message
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Liste des handlers (gestionnaires de sortie)
    handlers = []
    
    # 1. Handler pour la console (toujours présent)
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)
    
    # 2. Handler pour fichier (optionnel)
    if log_file:
        # Créer le dossier parent si nécessaire
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handler qui écrit dans le fichier
        # mode='a' = append (ajouter à la fin sans écraser)
        file_handler = logging.FileHandler(
            log_file,
            mode='a',
            encoding='utf-8'
        )
        handlers.append(file_handler)
    
    # Configuration du logger racine
    # Tous les loggers hériteront de cette configuration
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True  # Écrase toute configuration existante
    )
    
    # Réduire la verbosité de certaines bibliothèques tierces
    # Ces bibliothèques ont tendance à être très bavardes
    _reduce_library_verbosity()
    
    # Message de confirmation
    logger = logging.getLogger(__name__)
    logger.info("Système de logging configuré")
    if log_file:
        logger.info(f"Les logs sont également sauvegardés dans: {log_file}")


def _reduce_library_verbosity() -> None:
    """
    Réduit la verbosité des bibliothèques tierces.
    
    Sans cette fonction, certaines bibliothèques affichent beaucoup
    de messages de debug qui polluent les logs.
    
    Bibliothèques concernées :
    - PIL/Pillow : Traitement d'images
    - matplotlib : Graphiques
    - urllib3 : Requêtes HTTP
    - transformers : Modèles HuggingFace (si utilisé)
    """
    # PIL (Pillow) - bibliothèque de traitement d'images
    # Affiche beaucoup de détails sur le chargement des images
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # matplotlib - bibliothèque de graphiques
    # Affiche des messages de debug sur les fonts, backends, etc.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # urllib3 - requêtes HTTP
    # Affiche tous les appels réseau en détail
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # transformers (HuggingFace) - si utilisé pour ViT ou autres modèles
    # Très verbeux sur le téléchargement et chargement des modèles
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    # filelock - utilisé par HuggingFace
    logging.getLogger("filelock").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger avec le nom donné.
    
    Convention : toujours utiliser __name__ pour avoir des logs
    identifiables par module.
    
    Args:
        name: Nom du logger (utilisez __name__)
        
    Returns:
        Logger configuré et prêt à l'emploi
        
    Exemple:
        # Dans votre fichier Python
        from src.utils.logging_config import get_logger
        
        logger = get_logger(__name__)
        logger.info("Module initialisé")
        logger.debug("Détail pour debug")
        logger.warning("Attention !")
        logger.error("Une erreur s'est produite")
    """
    return logging.getLogger(name)


# ============================================================================
# Fonctions avancées pour logging contextuel
# ============================================================================

class LogContext:
    """
    Context manager pour logger le début et la fin d'une opération.
    
    Plus simple que d'écrire manuellement les logs de début/fin.
    
    Exemple:
        with LogContext("Chargement des données"):
            data = load_data()
        
        # Affichera automatiquement:
        # "Début: Chargement des données"
        # "Fin: Chargement des données"
    """
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        """
        Args:
            operation: Nom de l'opération
            logger: Logger à utiliser (si None, utilise le logger racine)
        """
        self.operation = operation
        self.logger = logger or logging.getLogger(__name__)
    
    def __enter__(self):
        self.logger.info(f"Début: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Erreur lors de: {self.operation}")
        else:
            self.logger.info(f"Fin: {self.operation}")


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Démonstration du système de logging")
    print("="*70 + "\n")
    
    # 1. Configuration de base
    print("1. Configuration du logging\n")
    setup_logging(level=logging.DEBUG)
    
    # 2. Obtenir un logger
    logger = get_logger(__name__)
    
    # 3. Différents niveaux de log
    print("\n2. Différents niveaux de log\n")
    logger.debug("Message de DEBUG - utile pour le développement")
    logger.info("Message d'INFO - informations générales")
    logger.warning("Message de WARNING - quelque chose d'inhabituel")
    logger.error("Message d'ERROR - une erreur s'est produite")
    
    # 4. Context manager
    print("\n3. Utilisation du LogContext\n")
    with LogContext("Opération de test", logger):
        logger.info("  -> Travail en cours...")
    
    # 5. Exemple avec fichier
    print("\n4. Configuration avec fichier\n")
    setup_logging(
        level=logging.INFO,
        log_file="test_logs.log"
    )
    logger = get_logger(__name__)
    logger.info("Ce message sera dans le fichier ET la console")
    
    print("\n" + "="*70)
    print("Un fichier 'test_logs.log' a été créé avec les logs")
    print("="*70 + "\n")

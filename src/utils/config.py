"""
Module de gestion de la configuration du projet Rakuten.
=====================================================

Ce module permet de charger et d'accéder facilement à la configuration
depuis le fichier config.toml.

Utilisation simple :
    from src.utils.config import load_config
    
    config = load_config()
    max_features = config.get("text.max_features")

"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Gestion de tomllib/tomli selon la version de Python
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11
    except ImportError:
        raise ImportError(
            "tomli ou tomllib requis. Installez avec: pip install tomli"
        )

import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Gestionnaire de configuration centralisé.
    
    Cette classe charge le fichier config.toml et permet d'accéder
    aux paramètres de manière simple et sécurisée.
    
    Exemples :
        >>> config = Config()
        >>> config.get("text.max_features")
        60000
        >>> config.text["max_features"]
        60000
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise la configuration.
        
        Args:
            config_path: Chemin vers config.toml. Si None, cherche 
                        automatiquement dans les emplacements standards.
        """
        self.config_path = self._find_config(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def _find_config(self, config_path: Optional[str]) -> Path:
        """
        Trouve le fichier de configuration dans les emplacements standards.
        
        Cherche dans l'ordre :
        1. Chemin fourni en paramètre
        2. config/config.toml (depuis le répertoire courant)
        3. ../config/config.toml (un niveau au-dessus)
        4. config/ relatif à ce fichier
        
        Args:
            config_path: Chemin optionnel vers config.toml
            
        Returns:
            Path: Chemin absolu vers config.toml
            
        Raises:
            FileNotFoundError: Si aucun fichier de config n'est trouvé
        """
        # Si un chemin est fourni, l'utiliser
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Config file non trouvé: {config_path}")
        
        # Sinon, chercher dans les emplacements standards
        candidates = [
            Path("config/config.toml"),                                    # Depuis le répertoire courant
            Path("../config/config.toml"),                                 # Un niveau au-dessus
            Path(__file__).parent.parent.parent / "config" / "config.toml",  # Relatif à ce fichier
        ]
        
        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Configuration trouvée: {candidate}")
                return candidate
        
        raise FileNotFoundError(
            "config.toml non trouvé. "
            "Placez-le dans config/ ou spécifiez le chemin."
        )
    
    def load(self) -> None:
        """
        Charge la configuration depuis le fichier TOML.
        
        Cette méthode lit le fichier config.toml et stocke son contenu
        dans un dictionnaire interne.
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            tomli.TOMLDecodeError: Si le fichier TOML est mal formaté
        """
        logger.info(f"Chargement de la configuration depuis {self.config_path}")
        
        with open(self.config_path, "rb") as f:
            self._config = tomllib.load(f)
        
        logger.info("Configuration chargée avec succès")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration par clé avec notation pointée.
        
        Cette méthode permet d'accéder aux valeurs imbriquées dans le TOML
        en utilisant une notation avec des points.
        
        Args:
            key: Clé avec notation pointée (ex: "text.max_features")
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            La valeur trouvée ou la valeur par défaut
            
        Exemples:
            >>> config.get("text.max_features")
            60000
            >>> config.get("unknown.key", 42)
            42
        """
        # Découper la clé en parties (ex: "text.max_features" -> ["text", "max_features"])
        keys = key.split(".")
        value = self._config
        
        # Naviguer dans la structure imbriquée
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """
        Permet l'accès type dictionnaire: config["text"]["max_features"]
        
        Args:
            key: Clé de premier niveau
            
        Returns:
            Valeur associée à la clé
            
        Raises:
            KeyError: Si la clé n'existe pas
        """
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """
        Vérifie si une clé existe: "text" in config
        
        Args:
            key: Clé à vérifier
            
        Returns:
            True si la clé existe, False sinon
        """
        return key in self._config
    
    # ========================================================================
    # Propriétés pratiques pour accès direct aux sections principales
    # ========================================================================
    
    @property
    def paths(self) -> Dict[str, str]:
        """Récupère la section [paths] de la configuration."""
        return self._config.get("paths", {})
    
    @property
    def text(self) -> Dict[str, Any]:
        """Récupère la section [text] de la configuration."""
        return self._config.get("text", {})
    
    @property
    def images(self) -> Dict[str, Any]:
        """Récupère la section [images] de la configuration."""
        return self._config.get("images", {})
    
    @property
    def model(self) -> Dict[str, Any]:
        """Récupère la section [model] de la configuration."""
        return self._config.get("model", {})
    
    @property
    def sampling(self) -> Dict[str, Any]:
        """Récupère la section [sampling] de la configuration."""
        return self._config.get("sampling", {})
    
    @property
    def cv(self) -> Dict[str, Any]:
        """Récupère la section [cv] de la configuration."""
        return self._config.get("cv", {})
    
    @property
    def random_seed(self) -> int:
        """Récupère le seed aléatoire depuis [random]."""
        return self._config.get("random", {}).get("seed", 42)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Retourne la configuration complète sous forme de dictionnaire.
        
        Returns:
            Copie du dictionnaire de configuration
        """
        return self._config.copy()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Fonction pratique pour charger la configuration.
    
    C'est la façon la plus simple d'obtenir un objet Config.
    
    Args:
        config_path: Chemin optionnel vers config.toml
        
    Returns:
        Objet Config chargé
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> print(config.text["max_features"])
    """
    return Config(config_path)


# ============================================================================
# Exemple d'utilisation si on exécute ce fichier directement
# ============================================================================

if __name__ == "__main__":
    # Configuration du logging pour voir les messages
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Afficher quelques valeurs
        print("\n" + "="*60)
        print("Configuration Rakuten chargée")
        print("="*60)
        
        print(f"\nSeed aléatoire: {config.random_seed}")
        print(f"Max features texte: {config.get('text.max_features', 'non défini')}")
        print(f"Taille images: {config.get('images.size', 'non défini')}")
        print(f"Modèle: {config.get('model.name', 'non défini')}")
        
        print("\nChemins des fichiers:")
        for key, value in config.paths.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Assurez-vous que config/config.toml existe.")

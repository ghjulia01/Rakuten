"""
Module de profilage et de débogage pour le projet Rakuten.
========================================================

Ce module fournit des outils pour :
- Mesurer le temps d'exécution des fonctions (@profile_func)
- Chronométrer des blocs de code (Timer)
- Collecter des messages de debug (list_debug_*)

Ces outils sont essentiels pour identifier les goulots d'étranglement
et optimiser les performances du pipeline.

Utilisation :
    from src.utils.profiling import profile_func, Timer
    
    @profile_func
    def ma_fonction():
        # Code qui sera chronométré automatiquement
        pass
    
    with Timer("Mon opération"):
        # Code qui sera chronométré
        pass

"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, List

logger = logging.getLogger(__name__)

# Liste globale pour stocker les messages de debug
# Utile pour tracer les étapes de transformation dans les pipelines sklearn
_DEBUG_LIST: List[str] = []


# ============================================================================
# Gestion de la liste de debug
# ============================================================================

def list_debug_add(msg: str) -> None:
    """
    Ajoute un message à la liste globale de debug.
    
    Utile dans les transformers sklearn pour tracer les étapes
    de transformation sans polluer les logs standards.
    
    Args:
        msg: Message de debug à ajouter
        
    Exemple:
        >>> list_debug_add("Chargement de 1000 images")
        >>> list_debug_add("Transformation TF-IDF effectuée")
    """
    _DEBUG_LIST.append(msg)
    logger.debug(msg)


def list_debug_get() -> List[str]:
    """
    Récupère tous les messages de debug collectés.
    
    Returns:
        Liste de tous les messages ajoutés depuis le début
        
    Exemple:
        >>> messages = list_debug_get()
        >>> for msg in messages:
        ...     print(msg)
    """
    return _DEBUG_LIST.copy()


def list_debug_clear() -> None:
    """
    Efface tous les messages de debug.
    
    Utile pour repartir à zéro entre différents runs.
    
    Exemple:
        >>> list_debug_clear()
        >>> print(len(list_debug_get()))  # Affiche 0
    """
    _DEBUG_LIST.clear()
    logger.debug("Liste de debug effacée")


# ============================================================================
# Décorateur de profilage
# ============================================================================

def profile_func(func: Callable) -> Callable:
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.
    
    Enregistre automatiquement le temps pris par la fonction dans les logs.
    N'affiche que les fonctions qui prennent plus de 0.1 seconde pour éviter
    le bruit dans les logs.
    
    Args:
        func: Fonction à profiler
        
    Returns:
        Fonction wrappée qui mesure automatiquement son temps d'exécution
        
    Exemple:
        @profile_func
        def charger_donnees():
            # Cette fonction sera automatiquement chronométrée
            return pd.read_csv("data.csv")
        
        # Lors de l'exécution, un log sera créé si > 0.1s :
        # "src.data.load_data.charger_donnees took 2.345s"
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Enregistrer le temps de début
        start_time = time.time()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Calculer le temps écoulé
        elapsed = time.time() - start_time
        
        # Logger seulement si le temps est significatif (> 0.1 seconde)
        # Évite de polluer les logs avec des opérations instantanées
        if elapsed > 0.1:
            module_name = func.__module__ or "unknown"
            func_name = func.__name__
            logger.debug(
                f"{module_name}.{func_name} a pris {elapsed:.3f}s"
            )
        
        return result
    
    return wrapper


# ============================================================================
# Context manager pour chronométrer des blocs de code
# ============================================================================

class Timer:
    """
    Context manager pour mesurer le temps d'exécution d'un bloc de code.
    
    Plus flexible que @profile_func car peut être utilisé n'importe où
    dans le code sans décorateur.
    
    Attributes:
        name: Nom de l'opération chronométrée
        log_level: Niveau de log (INFO par défaut)
        start_time: Temps de début (défini à l'entrée du with)
        elapsed: Temps écoulé total (défini à la sortie du with)
        
    Exemples:
        # Utilisation simple
        with Timer("Chargement des données"):
            data = load_data()
        # Affiche: "Chargement des données started..."
        #          "Chargement des données completed in 2.34s"
        
        # Utilisation avec récupération du temps
        with Timer("Entraînement") as t:
            model.fit(X, y)
        print(f"Entraînement terminé en {t.elapsed:.2f} secondes")
        
        # Avec niveau de log personnalisé
        with Timer("Opération critique", log_level=logging.WARNING):
            critical_operation()
    """
    
    def __init__(self, name: str = "Operation", log_level: int = logging.INFO):
        """
        Initialise le timer.
        
        Args:
            name: Nom descriptif de l'opération (affiché dans les logs)
            log_level: Niveau de logging (INFO, DEBUG, WARNING, etc.)
        """
        self.name = name
        self.log_level = log_level
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
    
    def __enter__(self) -> Timer:
        """
        Démarre le chronométrage à l'entrée du bloc 'with'.
        
        Returns:
            self: Permet d'accéder au timer dans le bloc with
        """
        self.start_time = time.time()
        logger.log(self.log_level, f"[START] {self.name}...")
        return self
    
    def __exit__(self, *args: Any) -> None:
        """
        Arrête le chronométrage à la sortie du bloc 'with'.
        
        Calcule et affiche le temps total écoulé.
        Les arguments *args sont les exceptions potentielles
        (non utilisées ici mais requises par le protocole context manager).
        """
        self.elapsed = time.time() - self.start_time
        logger.log(
            self.log_level,
            f"[END] {self.name} terminé en {self.elapsed:.2f}s"
        )


# ============================================================================
# Utilitaires de formatage du temps
# ============================================================================

def format_time(seconds: float) -> str:
    """
    Formate un temps en secondes en format lisible.
    
    Args:
        seconds: Nombre de secondes
        
    Returns:
        Chaîne formatée (ex: "2h 15min 30s" ou "45.2s")
        
    Exemples:
        >>> format_time(45.234)
        '45.2s'
        >>> format_time(3665)
        '1h 1min 5s'
        >>> format_time(125)
        '2min 5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}min {secs}s"
    
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}min {secs}s"


# ============================================================================
# Décorateur pour compter les appels
# ============================================================================

def count_calls(func: Callable) -> Callable:
    """
    Décorateur pour compter le nombre d'appels à une fonction.
    
    Ajoute un attribut 'call_count' à la fonction qui s'incrémente
    à chaque appel.
    
    Args:
        func: Fonction à monitorer
        
    Returns:
        Fonction wrappée avec compteur d'appels
        
    Exemple:
        @count_calls
        def ma_fonction():
            pass
        
        ma_fonction()
        ma_fonction()
        print(ma_fonction.call_count)  # Affiche: 2
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.call_count += 1
        return func(*args, **kwargs)
    
    wrapper.call_count = 0
    return wrapper


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("Démonstration des outils de profilage")
    print("="*60 + "\n")
    
    # 1. Décorateur @profile_func
    print("1. Test du décorateur @profile_func\n")
    
    @profile_func
    def fonction_lente():
        """Fonction qui prend du temps."""
        time.sleep(0.5)
        return "Terminé"
    
    resultat = fonction_lente()
    print(f"Résultat: {resultat}\n")
    
    # 2. Context manager Timer
    print("2. Test du Timer\n")
    
    with Timer("Opération exemple"):
        time.sleep(0.3)
        print("  -> Travail en cours...")
    
    print()
    
    # 3. Timer avec récupération du temps
    print("3. Timer avec accès au temps écoulé\n")
    
    with Timer("Calcul complexe") as t:
        time.sleep(0.2)
    
    print(f"Le calcul a pris exactement {t.elapsed:.3f} secondes\n")
    
    # 4. Liste de debug
    print("4. Test de la liste de debug\n")
    
    list_debug_clear()
    list_debug_add("Premier message")
    list_debug_add("Deuxième message")
    list_debug_add("Troisième message")
    
    messages = list_debug_get()
    print(f"Messages collectés: {len(messages)}")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. {msg}")
    
    print("\n" + "="*60 + "\n")

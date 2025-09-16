# profiling_tools.py
import os
import time
from functools import wraps

import logging

logger = logging.getLogger(__name__)

# Dictionnaire pour stocker les statistiques
function_stats = {}
list_debug = []

def list_debug_add(str):
    list_debug.append(str)

def profile_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialiser les statistiques si la fonction n'a pas encore été appelée
        if func.__qualname__ not in function_stats:
            function_stats[func.__qualname__] = {
                'call_count': 0,
                'total_time': 0.0,
                'max_time': 0.0,
                'min_time': float('inf')
            }

        # Incrémenter le compteur d'appels
        function_stats[func.__qualname__]['call_count'] += 1

        # Mesurer le temps d'exécution
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Calculer le temps écoulé
        elapsed_time = end_time - start_time
        function_stats[func.__qualname__]['total_time'] += elapsed_time

        # Mettre à jour le temps maximum et minimum
        if elapsed_time > function_stats[func.__qualname__]['max_time']:
            function_stats[func.__qualname__]['max_time'] = elapsed_time
        if elapsed_time < function_stats[func.__qualname__]['min_time']:
            function_stats[func.__qualname__]['min_time'] = elapsed_time

        return result
    return wrapper

def print_function_stats():
    logger.info("Statistiques des fonctions :")
    for func_name, stats in function_stats.items():
        logger.info(f"Fonction: {func_name}")
        logger.info(f"  - Nombre d'appels: {stats['call_count']}")
        logger.info(f"  - Temps total: {stats['total_time']:.4f} secondes")
        logger.info(f"  - Temps moyen: {stats['total_time'] / stats['call_count']:.4f} secondes")
        logger.info(f"  - Temps maximum: {stats['max_time']:.4f} secondes")
        logger.info(f"  - Temps minimum: {stats['min_time']:.4f} secondes")
        logger.info("")

def print_list_debug():
    logger.info("Debug :")
    for elt in list_debug:
        logger.info(elt)

def write_function_stats_to_file(outdir="results"):
    """Écrit les statistiques des fonctions dans un fichier sans les afficher dans la console."""
    log_file = os.path.join(outdir, "profiling_stats_time.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Statistiques des fonctions :\n")
        for func_name, stats in function_stats.items():
            message = (
                f"Fonction: {func_name}\n"
                f"  - Nombre d'appels: {stats['call_count']}\n"
                f"  - Temps total: {stats['total_time']:.4f} secondes\n"
                f"  - Temps moyen: {stats['total_time'] / stats['call_count']:.4f} secondes\n"
                f"  - Temps maximum: {stats['max_time']:.4f} secondes\n"
                f"  - Temps minimum: {stats['min_time']:.4f} secondes\n"
            )
            f.write(message + "\n")

def write_list_debug_to_file(outdir="results"):
    log_file = os.path.join(outdir, "profiling_func.log")
    """Écrit la liste de debug dans un fichier sans les afficher dans la console."""
    with open(log_file, "w", encoding="utf-8") as f:
        for elt in list_debug:
            f.write(f"{elt}\n")
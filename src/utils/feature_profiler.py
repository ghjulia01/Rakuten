#!/usr/bin/env python3
"""
Outil de profiling détaillé du pipeline de features.
====================================================

Permet de mesurer le temps et la taille de chaque transformateur
dans le pipeline pour identifier les goulots d'étranglement.

Utilisation:
    from src.utils.feature_profiler import profile_pipeline
    
    results = profile_pipeline(feature_pipeline, X_train, y_train)
    results.print_summary()
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy import sparse
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransformerProfile:
    """Profil d'un transformateur individuel."""
    name: str
    fit_time: float
    transform_time: float
    output_shape: tuple
    output_dtype: str
    is_sparse: bool
    memory_mb: float
    samples_per_sec: float


class FeatureProfiler:
    """Profileur de pipeline de features."""
    
    def __init__(self):
        self.results: List[TransformerProfile] = []
    
    def profile_transformer(
        self,
        name: str,
        transformer: Any,
        X: Any,
        y: Optional[Any] = None
    ) -> TransformerProfile:
        """
        Profile un transformateur individuel.
        
        Args:
            name: Nom du transformateur
            transformer: Transformateur sklearn
            X: Données d'entrée
            y: Labels (optionnel)
            
        Returns:
            TransformerProfile
        """
        # Nombre d'échantillons
        try:
            n_samples = X.shape[0]
        except Exception:
            n_samples = len(X)
        
        # Fit
        start = time.time()
        transformer.fit(X, y)
        fit_time = time.time() - start
        
        # Transform
        start = time.time()
        X_out = transformer.transform(X)
        transform_time = time.time() - start
        
        # Analyser la sortie
        is_sparse = sparse.issparse(X_out)
        
        if is_sparse:
            output_shape = X_out.shape
            output_dtype = str(X_out.dtype)
            memory_mb = (X_out.data.nbytes + X_out.indices.nbytes + X_out.indptr.nbytes) / 1024 / 1024
        else:
            if hasattr(X_out, 'shape'):
                output_shape = X_out.shape
                output_dtype = str(X_out.dtype)
                memory_mb = X_out.nbytes / 1024 / 1024 if hasattr(X_out, 'nbytes') else 0
            else:
                output_shape = (len(X_out), 1)
                output_dtype = "unknown"
                memory_mb = 0
        
        samples_per_sec = n_samples / transform_time if transform_time > 0 else 0
        
        return TransformerProfile(
            name=name,
            fit_time=fit_time,
            transform_time=transform_time,
            output_shape=output_shape,
            output_dtype=output_dtype,
            is_sparse=is_sparse,
            memory_mb=memory_mb,
            samples_per_sec=samples_per_sec
        )
    
    def profile_pipeline(
        self,
        pipeline: Any,
        X: Any,
        y: Optional[Any] = None,
        max_depth: int = 3
    ) -> 'ProfileResults':
        """
        Profile un pipeline complet (récursif).
        
        Args:
            pipeline: Pipeline sklearn
            X: Données d'entrée
            y: Labels (optionnel)
            max_depth: Profondeur maximale de récursion
            
        Returns:
            ProfileResults
        """
        self.results = []
        # Sauvegarder X original pour éviter les problèmes avec sparse matrices
        self._original_X = X
        self._profile_recursive(pipeline, X, y, depth=0, max_depth=max_depth, prefix="")
        return ProfileResults(self.results)
    
    def _profile_recursive(
        self,
        obj: Any,
        X: Any,
        y: Optional[Any],
        depth: int,
        max_depth: int,
        prefix: str
    ):
        """Profile récursivement les transformateurs imbriqués."""
        if depth > max_depth:
            return
        
        # IMPORTANT: Utiliser X original pour éviter les problèmes
        # avec les sparse matrices dans les sous-pipelines
        X_to_use = self._original_X if depth > 0 else X
        
        # Pipeline sklearn
        if hasattr(obj, 'steps'):
            for name, transformer in obj.steps:
                full_name = f"{prefix}{name}" if prefix else name
                logger.info(f"{'  ' * depth}  Profiling: {full_name}")
                
                profile = self.profile_transformer(full_name, transformer, X_to_use, y)
                self.results.append(profile)
                
                # Récursion (limiter la profondeur pour éviter les problèmes)
                if depth < 5 and (hasattr(transformer, 'steps') or hasattr(transformer, 'transformer_list')):
                    self._profile_recursive(
                        transformer, X_to_use, y, depth + 1, max_depth, f"{full_name}."
                    )
        
        # FeatureUnion
        elif hasattr(obj, 'transformer_list'):
            for name, transformer in obj.transformer_list:
                full_name = f"{prefix}{name}" if prefix else name
                logger.info(f"{'  ' * depth}  Profiling: {full_name}")
                
                profile = self.profile_transformer(full_name, transformer, X_to_use, y)
                self.results.append(profile)
                
                # Récursion (limiter la profondeur)
                if depth < 5 and (hasattr(transformer, 'steps') or hasattr(transformer, 'transformer_list')):
                    self._profile_recursive(
                        transformer, X_to_use, y, depth + 1, max_depth, f"{full_name}."
                    )


class ProfileResults:
    """Résultats de profiling."""
    
    def __init__(self, profiles: List[TransformerProfile]):
        self.profiles = profiles
    
    def print_summary(self):
        """Affiche un résumé formaté."""
        print("\n" + "=" * 100)
        print(" PROFILING DES FEATURES - RESUME DETAILLE")
        print("=" * 100)
        
        # Trier par temps de transform
        sorted_profiles = sorted(self.profiles, key=lambda p: p.transform_time, reverse=True)
        
        # Header
        print(f"\n{'Transformateur':<35} {'Fit (s)':<10} {'Transform (s)':<13} {'Shape':<15} {'Mem (MB)':<10} {'Samples/s':<10}")
        print("-" * 100)
        
        total_fit = 0
        total_transform = 0
        total_mem = 0
        
        for profile in sorted_profiles:
            total_fit += profile.fit_time
            total_transform += profile.transform_time
            total_mem += profile.memory_mb
            
            shape_str = f"{profile.output_shape}"
            sparse_marker = " (S)" if profile.is_sparse else ""
            
            print(
                f"{profile.name:<35} "
                f"{profile.fit_time:<10.2f} "
                f"{profile.transform_time:<13.2f} "
                f"{shape_str:<15}{sparse_marker:<3} "
                f"{profile.memory_mb:<10.1f} "
                f"{profile.samples_per_sec:<10.0f}"
            )
        
        # Total
        print("-" * 100)
        print(
            f"{'TOTAL':<35} "
            f"{total_fit:<10.2f} "
            f"{total_transform:<13.2f} "
            f"{'--':<18} "
            f"{total_mem:<10.1f} "
            f"{'--':<10}"
        )
        print("=" * 100)
        
        # Top 3 goulots
        print("\n TOP 3 GOULOTS D'ETRANGLEMENT:")
        for i, profile in enumerate(sorted_profiles[:3], 1):
            pct = (profile.transform_time / total_transform * 100) if total_transform > 0 else 0
            print(f"  {i}. {profile.name}: {profile.transform_time:.2f}s ({pct:.1f}%)")
        
        print("\n MEMOIRE TOTALE:", f"{total_mem:.1f} MB")
        print(" TEMPS TOTAL FIT:", f"{total_fit:.2f}s")
        print(" TEMPS TOTAL TRANSFORM:", f"{total_transform:.2f}s")
        print()
    
    def get_bottlenecks(self, top_n: int = 5) -> List[TransformerProfile]:
        """Retourne les N plus gros goulots."""
        return sorted(self.profiles, key=lambda p: p.transform_time, reverse=True)[:top_n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'profiles': [
                {
                    'name': p.name,
                    'fit_time': p.fit_time,
                    'transform_time': p.transform_time,
                    'output_shape': p.output_shape,
                    'output_dtype': p.output_dtype,
                    'is_sparse': p.is_sparse,
                    'memory_mb': p.memory_mb,
                    'samples_per_sec': p.samples_per_sec
                }
                for p in self.profiles
            ],
            'total_fit_time': sum(p.fit_time for p in self.profiles),
            'total_transform_time': sum(p.transform_time for p in self.profiles),
            'total_memory_mb': sum(p.memory_mb for p in self.profiles)
        }


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def profile_pipeline(
    pipeline: Any,
    X: Any,
    y: Optional[Any] = None,
    max_depth: int = 3
) -> ProfileResults:
    """
    Profile un pipeline complet.
    
    Args:
        pipeline: Pipeline sklearn
        X: Données d'entrée
        y: Labels (optionnel)
        max_depth: Profondeur maximale de récursion
        
    Returns:
        ProfileResults
        
    Exemple:
        >>> from src.utils.feature_profiler import profile_pipeline
        >>> results = profile_pipeline(feature_pipeline, X_sample, y_sample)
        >>> results.print_summary()
    """
    profiler = FeatureProfiler()
    return profiler.profile_pipeline(pipeline, X, y, max_depth)


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import pandas as pd
    
    # Données exemple
    X = pd.DataFrame({
        'text': ['hello world'] * 1000
    })
    
    # Pipeline exemple
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer()),
        ])),
        ('svd', TruncatedSVD(n_components=50))
    ])
    
    # Profiling
    results = profile_pipeline(pipeline, X['text'])
    results.print_summary()








































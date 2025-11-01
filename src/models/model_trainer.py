"""
Module d'entraînement des modèles - Classe ModelTrainer.
=======================================================

Ce module implémente la classe ModelTrainer qui gère l'entraînement
des différents modèles (LR, SVC, XGB, LGBM) avec leurs hyperparamètres.

Inspiré de la structure du projet wine_quality.

Utilisation:
    from src.models.model_trainer import ModelTrainer
    
    trainer = ModelTrainer(config)
    model = trainer.train(X_train, y_train)
    trainer.save_model(model, "models/best_model.joblib")

Auteur: Projet Rakuten
Date: 2024
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Classe pour gérer l'entraînement des modèles de classification.
    
    Cette classe encapsule la logique d'entraînement et de sauvegarde
    des modèles selon la configuration fournie.
    
    Attributes:
        config: Configuration du modèle (dict depuis config.toml)
        random_state: Graine aléatoire pour la reproductibilité
        model: Modèle entraîné (None avant le training)
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> trainer = ModelTrainer(config.model, random_state=42)
        >>> model = trainer.train(X_train, y_train)
        >>> trainer.save_model(model, "models/my_model.joblib")
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        random_state: int = 42
    ):
        """
        Initialise le ModelTrainer.
        
        Args:
            model_config: Configuration du modèle (section [model] du TOML)
            random_state: Graine aléatoire (par défaut 42)
        """
        self.config = model_config
        self.random_state = random_state
        self.model = None
        
        logger.info(f"ModelTrainer initialisé avec modèle: {self.config['name']}")
    
    def create_model(self) -> Any:
        """
        Crée un modèle selon la configuration.
        
        Returns:
            Modèle sklearn non entraîné
            
        Raises:
            ValueError: Si le nom du modèle est inconnu
            
        Exemple:
            >>> trainer = ModelTrainer(config)
            >>> model = trainer.create_model()
            >>> type(model)
            <class 'sklearn.linear_model.LogisticRegression'>
        """
        model_name = self.config["name"].lower()
        
        logger.info(f"Création du modèle: {model_name}")
        
        # ========================================
        # Logistic Regression
        # ========================================
        if model_name == "lr":
            lr_params = self.config.get("lr", {})
            model = LogisticRegression(
                random_state=self.random_state,
                **lr_params
            )
            logger.info(f"  Solver: {lr_params.get('solver', 'default')}")
            logger.info(f"  Penalty: {lr_params.get('penalty', 'default')}")
            logger.info(f"  C: {lr_params.get('C', 1.0)}")
        
        # ========================================
        # Linear SVC
        # ========================================
        elif model_name == "svc":
            svc_params = self.config.get("svc", {})
            model = LinearSVC(
                random_state=self.random_state,
                **svc_params
            )
            logger.info(f"  C: {svc_params.get('C', 1.0)}")
            logger.info(f"  Loss: {svc_params.get('loss', 'squared_hinge')}")
        
        # ========================================
        # XGBoost
        # ========================================
        elif model_name == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError(
                    "XGBoost non installé. Installez avec: pip install xgboost"
                )
            
            xgb_params = self.config.get("xgb", {})
            model = XGBClassifier(
                random_state=self.random_state,
                **xgb_params
            )
            logger.info(f"  N estimators: {xgb_params.get('n_estimators', 100)}")
            logger.info(f"  Learning rate: {xgb_params.get('learning_rate', 0.1)}")
        
        # ========================================
        # LightGBM
        # ========================================
        elif model_name == "lgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError:
                raise ImportError(
                    "LightGBM non installé. Installez avec: pip install lightgbm"
                )
            
            lgbm_params = self.config.get("lgbm", {})
            model = LGBMClassifier(
                random_state=self.random_state,
                **lgbm_params
            )
            logger.info(f"  N estimators: {lgbm_params.get('n_estimators', 100)}")
            logger.info(f"  Learning rate: {lgbm_params.get('learning_rate', 0.1)}")
        
        else:
            raise ValueError(
                f"Modèle inconnu: {model_name}. "
                f"Modèles supportés: 'lr', 'svc', 'xgb', 'lgbm'"
            )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **fit_params
    ) -> Any:
        """
        Entraîne le modèle sur les données fournies.
        
        Args:
            X_train: Features d'entraînement (n_samples, n_features)
            y_train: Labels d'entraînement (n_samples,)
            **fit_params: Paramètres supplémentaires pour fit()
                         (ex: sample_weight, eval_set, etc.)
        
        Returns:
            Modèle entraîné
            
        Exemple:
            >>> trainer = ModelTrainer(config)
            >>> model = trainer.train(X_train, y_train)
            >>> # Avec poids de classe
            >>> model = trainer.train(X_train, y_train, 
            ...                       sample_weight=weights)
        """
        with Timer(f"Entraînement du modèle {self.config['name']}"):
            # Créer le modèle
            self.model = self.create_model()
            
            # Informations sur les données
            logger.info(f"Dimensions d'entraînement: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"Nombre de classes: {len(np.unique(y_train))}")
            
            # Entraîner
            self.model.fit(X_train, y_train, **fit_params)
            
            logger.info("✓ Entraînement terminé")
        
        return self.model
    
    def save_model(
        self,
        model: Any,
        output_path: str
    ) -> None:
        """
        Sauvegarde le modèle dans un fichier .joblib.
        
        Args:
            model: Modèle entraîné à sauvegarder
            output_path: Chemin de sauvegarde (ex: "models/model.joblib")
            
        Exemple:
            >>> trainer.save_model(model, "models/lr_model.joblib")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Sauvegarde du modèle dans: {output_path}")
        joblib.dump(model, output_path)
        
        # Vérifier la taille du fichier
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Modèle sauvegardé ({size_mb:.2f} MB)")
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        """
        Charge un modèle depuis un fichier .joblib.
        
        Args:
            model_path: Chemin vers le fichier .joblib
            
        Returns:
            Modèle chargé
            
        Exemple:
            >>> model = ModelTrainer.load_model("models/best_model.joblib")
            >>> predictions = model.predict(X_test)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        logger.info(f"Chargement du modèle depuis: {model_path}")
        model = joblib.load(model_path)
        logger.info("✓ Modèle chargé")
        
        return model


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Démonstration de ModelTrainer")
    print("="*70 + "\n")
    
    # Configuration exemple
    config = {
        "name": "lr",
        "lr": {
            "solver": "saga",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 1000
        }
    }
    
    # Créer des données exemple
    from sklearn.datasets import make_classification
    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        random_state=42
    )
    
    # Créer et entraîner
    trainer = ModelTrainer(config, random_state=42)
    model = trainer.train(X_train, y_train)
    
    # Sauvegarder
    trainer.save_model(model, "/tmp/test_model.joblib")
    
    # Recharger
    loaded_model = ModelTrainer.load_model("/tmp/test_model.joblib")
    
    print("\n✓ Démonstration terminée")
    print("="*70 + "\n")

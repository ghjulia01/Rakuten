"""
Module d'entraînement des modèles - Classe ModelTrainer.
=======================================================

Ce module implémente la classe ModelTrainer qui gère l'entraînement
des différents modèles (LR, SVC, XGB, LGBM) avec leurs hyperparamètres.

Utilisation:
    from src.models.model_trainer import ModelTrainer
    
    trainer = ModelTrainer(config)
    model = trainer.train(X_train, y_train)
    trainer.save_model(model, "models/best_model.joblib")

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

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
        self.label_encoder = LabelEncoder()  # Pour gérer les classes non-séquentielles
        
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
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **fit_params
    ) -> Any:
        """
        Entraîne le modèle sur les données fournies avec early stopping optionnel.
        
        Args:
            X_train: Features d'entraînement (n_samples, n_features)
            y_train: Labels d'entraînement (n_samples,)
            X_val: Features de validation (optionnel, pour early stopping)
            y_val: Labels de validation (optionnel, pour early stopping)
            **fit_params: Paramètres supplémentaires pour fit()
        
        Returns:
            Modèle entraîné
            
        Exemple:
            >>> trainer = ModelTrainer(config)
            >>> model = trainer.train(X_train, y_train)
            >>> # Avec early stopping
            >>> model = trainer.train(X_train, y_train, X_val, y_val)
        """
        with Timer(f"Entraînement du modèle {self.config['name']}"):
            # Créer le modèle
            self.model = self.create_model()
            
            # Informations sur les données
            logger.info(f"Dimensions d'entraînement: X={X_train.shape}, y={y_train.shape}")
            
            # ========================================
            # ENCODER LES LABELS (10, 40, 50... → 0, 1, 2...)
            # ========================================
            unique_classes = np.unique(y_train)
            logger.info(f"Classes originales: {sorted(unique_classes)}")
            
            # Fit + transform des labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            
            logger.info(f"Classes après encodage: {np.unique(y_train_encoded)}")
            logger.info(f"Nombre de classes: {len(unique_classes)}")
            
            # ========================================
            # EARLY STOPPING pour XGB et LGBM
            # ========================================
            model_name = self.config['name'].lower()
            
            if model_name in ['xgb', 'lgbm']:
                # Vérifier si early_stopping_rounds est configuré
                early_stopping_rounds = None
                if model_name == 'xgb':
                    early_stopping_rounds = self.config.get('xgb', {}).get('early_stopping_rounds')
                elif model_name == 'lgbm':
                    early_stopping_rounds = self.config.get('lgbm', {}).get('early_stopping_rounds')
                
                # Si early stopping activé
                if early_stopping_rounds:
                    # Si pas de validation fournie, créer un split
                    if X_val is None or y_val is None:
                        from sklearn.model_selection import train_test_split
                        logger.info(f" Early stopping activé (rounds={early_stopping_rounds})")
                        logger.info("   Création d'un split validation (20% des données)")
                        
                        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                            X_train, y_train_encoded,
                            test_size=0.2,
                            random_state=self.random_state,
                            stratify=y_train_encoded
                        )
                    else:
                        logger.info(f" Early stopping activé (rounds={early_stopping_rounds})")
                        X_train_split = X_train
                        y_train_split = y_train_encoded
                        X_val_split = X_val
                        y_val_split = self.label_encoder.transform(y_val)
                    
                    # ========================================
                    # XGBoost Early Stopping
                    # ========================================
                    if model_name == 'xgb':
                        self.model.fit(
                            X_train_split, y_train_split,
                            eval_set=[(X_val_split, y_val_split)],
                            verbose=50,  # Log tous les 50 rounds
                            **fit_params
                        )
                        
                        best_iteration = self.model.best_iteration
                        logger.info(f" Early stopping à l'itération {best_iteration}/{self.config['xgb']['n_estimators']}")
                    
                    # ========================================
                    # LightGBM Early Stopping
                    # ========================================
                    elif model_name == 'lgbm':
                        import lightgbm as lgb
                        
                        self.model.fit(
                            X_train_split, y_train_split,
                            eval_set=[(X_val_split, y_val_split)],
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                                lgb.log_evaluation(period=50)
                            ],
                            **fit_params
                        )
                        
                        logger.info(f" Early stopping à l'itération {self.model.best_iteration_}/{self.config['lgbm']['n_estimators']}")
                
                else:
                    # Pas d'early stopping
                    self.model.fit(X_train, y_train_encoded, **fit_params)
            
            else:
                # LR, SVC : pas d'early stopping
                self.model.fit(X_train, y_train_encoded, **fit_params)
            
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
        
        # Sauvegarder le modèle ET le label_encoder ensemble
        model_bundle = {
            'model': model,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_bundle, output_path)
        
        # Vérifier la taille du fichier
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Modèle sauvegardé ({size_mb:.2f} MB)")
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Fait des prédictions et les décode automatiquement.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Prédictions dans les classes originales (10, 40, 50, etc.)
            
        Exemple:
            >>> predictions = trainer.predict(X_test)
            >>> # Retourne [10, 40, 2583, ...] et non [0, 1, 2, ...]
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas entraîné. Appelez train() d'abord.")
        
        # Prédire avec labels encodés (0, 1, 2, ...)
        y_pred_encoded = self.model.predict(X)
        
        # Décoder vers les classes originales (10, 40, 50, ...)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        """
        Charge un modèle depuis un fichier .joblib.
        
        Args:
            model_path: Chemin vers le fichier .joblib
            
        Returns:
            Tuple (model, label_encoder) ou juste model si ancien format
            
        Exemple:
            >>> model_bundle = ModelTrainer.load_model("models/best_model.joblib")
            >>> if isinstance(model_bundle, dict):
            >>>     model = model_bundle['model']
            >>>     label_encoder = model_bundle['label_encoder']
            >>> else:
            >>>     model = model_bundle  # Ancien format
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        logger.info(f"Chargement du modèle depuis: {model_path}")
        loaded = joblib.load(model_path)
        
        # dict avec model + label_encoder
        if isinstance(loaded, dict) and 'model' in loaded:
            logger.info("Modèle + LabelEncoder chargés")
            return loaded
        else:
            # Juste le modèle
            logger.warning("Ancien format détecté (sans LabelEncoder)")
            return loaded


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
    
    print("\n Démonstration terminée")
    print("="*70 + "\n")
"""
Etape 4 : Entrainement du modele (Model Training).
==================================================

Cette etape entraine le modele sur les donnees transformees.

Responsabilites :
- Creer le modele selon la configuration
- Entrainer sur les donnees transformees avec barre de progression
- Validation croisee avec affichage des folds
- Sauvegarder le modele et le pipeline de features

Utilisation:
    from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline
    
    pipeline = ModelTrainingPipeline(config)
    model = pipeline.run(X_train_transformed, y_train_resampled)

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from tqdm import tqdm

from src.models.model_trainer import ModelTrainer
from src.utils.profiling import Timer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Pipeline d'entrainement du modele.
    
    Entraine le modele final sur les donnees transformees avec progression.
    
    Attributes:
        config: Configuration complete du projet
        trainer: Instance de ModelTrainer
        model: Modele entraine (apres run)
        
    Exemple:
        >>> from src.utils.config import load_config
        >>> config = load_config()
        >>> pipeline = ModelTrainingPipeline(config)
        >>> model = pipeline.run(X_train_transformed, y_train_resampled)
    """
    
    def __init__(self, config):
        """
        Initialise le pipeline d'entrainement.
        
        Args:
            config: Objet Config contenant tous les parametres
        """
        self.config = config
        self.trainer = None
        self.model = None
        
        logger.info("=" * 70)
        logger.info("ETAPE 4 : ENTRAINEMENT DU MODELE")
        logger.info("=" * 70)
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """
        Entraine le modele sur les donnees.
        
        Args:
            X_train: Features transformees (n_samples, n_features)
            y_train: Labels (n_samples,)
            
        Returns:
            Modele entraine
        """
        logger.info("\n--- Entrainement du modele ---")
        
        # Creer le trainer
        self.trainer = ModelTrainer(
            model_config=self.config.model,
            random_state=self.config.random_seed
        )
        
        # Informations sur les donnees
        logger.info(f"Donnees d'entrainement: {X_train.shape}")
        logger.info(f"Labels: {y_train.shape}")
        logger.info(f"Nombre de classes: {len(np.unique(y_train))}")
        
        # Type de matrice (sparse ou dense)
        if hasattr(X_train, 'nnz'):
            logger.info(f"Matrice sparse - nnz: {X_train.nnz}")
            density = X_train.nnz / np.prod(X_train.shape)
            logger.info(f"Densite: {density:.4f} ({density*100:.2f}%)")
        else:
            logger.info("Matrice dense")
        
        # ========================================
        # VALIDATION CROISEE (si activee)
        # ========================================
        if self.config.get("cv.enabled", False):
            logger.info("\n--- Validation croisee ---")
            cv_scores = self.perform_cross_validation(X_train, y_train)
            
            logger.info(f"\nCV F1 (weighted) - Moyenne: {cv_scores['mean']:.4f} (+/- {cv_scores['std']:.4f})")
            logger.info(f"CV F1 (weighted) - Scores: {cv_scores['scores']}")
        
        # ========================================
        # ENTRAINEMENT FINAL
        # ========================================
        model_name = self.config.model['name'].upper()
        with Timer(f"Entrainement {model_name}"):
            self.model = self.trainer.train(X_train, y_train)
        
        logger.info("[OK] Entrainement termine")
        
        return self.model
    
    def perform_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """
        Effectue une validation croisee stratifiee avec barre de progression.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dict avec scores, mean, std
        """
        # Parametres CV depuis config
        n_splits = self.config.get("cv.splits", 3)
        shuffle = self.config.get("cv.shuffle", True)
        cv_random_state = self.config.get("cv.random_state", self.config.random_seed)
        show_progress = self.config.get("cv.show_progress", True)
        
        logger.info(f"Parametres CV: {n_splits} folds, shuffle={shuffle}, random_state={cv_random_state}")
        
        # Creer le StratifiedKFold
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=cv_random_state
        )
        
        # Encoder y pour sklearn
        y_encoded = self.trainer.label_encoder.fit_transform(y)
        
        # Liste pour stocker les scores
        cv_scores = []
        
        # ========================================
        # Boucle de CV avec progression
        # ========================================
        with Timer("Validation croisee"):
            # Creer l'iterateur avec ou sans tqdm
            if show_progress:
                fold_iterator = enumerate(
                    tqdm(
                        skf.split(X, y_encoded),
                        total=n_splits,
                        desc="Cross-validation",
                        unit="fold",
                        ncols=80
                    ),
                    start=1
                )
            else:
                fold_iterator = enumerate(skf.split(X, y_encoded), start=1)
            
            # Boucle sur les folds
            for fold_idx, (train_idx, val_idx) in fold_iterator:
                logger.info(f"\n{'='*60}")
                logger.info(f"FOLD {fold_idx}/{n_splits}")
                logger.info(f"{'='*60}")
                logger.info(f"Train: {len(train_idx)} echantillons")
                logger.info(f"Val: {len(val_idx)} echantillons")
                
                # Splitter les donnees
                X_train_fold = X[train_idx]
                y_train_fold = y_encoded[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y_encoded[val_idx]
                
                # Creer un modele pour ce fold
                fold_model = self.trainer.create_model()
                
                # Entrainer
                logger.info("Entrainement du fold...")
                
                # ========================================
                # EARLY STOPPING PAR FOLD
                # ========================================
                if self.config.model['name'] in ['xgb', 'lgbm']:
                    # Vérifier si early_stopping_rounds est configuré
                    early_stopping_rounds = None
                    if self.config.model['name'] == 'xgb':
                        early_stopping_rounds = self.config.model.get('xgb', {}).get('early_stopping_rounds')
                    elif self.config.model['name'] == 'lgbm':
                        early_stopping_rounds = self.config.model.get('lgbm', {}).get('early_stopping_rounds')
                    
                    if early_stopping_rounds:
                        # AVEC early stopping
                        logger.info(f" Early stopping activé pour ce fold (rounds={early_stopping_rounds})")
                        
                        if self.config.model['name'] == 'xgb':
                            fold_model.fit(
                                X_train_fold, y_train_fold,
                                eval_set=[(X_val_fold, y_val_fold)],
                                verbose=False
                            )
                            if hasattr(fold_model, 'best_iteration'):
                                logger.info(f"  Arrêt à l'itération {fold_model.best_iteration}")
                        
                        elif self.config.model['name'] == 'lgbm':
                            import lightgbm as lgb
                            fold_model.fit(
                                X_train_fold, y_train_fold,
                                eval_set=[(X_val_fold, y_val_fold)],
                                callbacks=[
                                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                                ]
                            )
                            if hasattr(fold_model, 'best_iteration_'):
                                logger.info(f"  Arrêt à l'itération {fold_model.best_iteration_}")
                    else:
                        # SANS early stopping
                        fold_model.fit(X_train_fold, y_train_fold)
                else:
                    # LR, SVC : pas d'early stopping
                    fold_model.fit(X_train_fold, y_train_fold)
                
                # Predire sur validation
                y_pred = fold_model.predict(X_val_fold)
                
                # Calculer F1
                score = f1_score(y_val_fold, y_pred, average='weighted')
                cv_scores.append(score)
                
                logger.info(f"F1 Score (fold {fold_idx}): {score:.4f}")
        
        cv_scores = np.array(cv_scores)
        
        return {
            'scores': [float(s) for s in cv_scores],
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std())
        }
    
    def save_model(
        self,
        model: Any,
        output_path: str
    ) -> None:
        """
        Sauvegarde le modele.
        
        Args:
            model: Modele entraine
            output_path: Chemin de sauvegarde
        """
        logger.info("\n--- Sauvegarde du modele ---")
        
        # Creer le dossier parent si necessaire
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        self.trainer.save_model(model, str(output_path))
        
        logger.info(f"[OK] Modele sauvegarde: {output_path}")
    
    def save_full_pipeline(
        self,
        model: Any,
        feature_pipeline: Any,
        output_path: str
    ) -> None:
        """
        Sauvegarde le pipeline complet (features + modele).
        
        Utile pour la prediction : on peut charger tout d'un coup.
        
        Args:
            model: Modele entraine
            feature_pipeline: Pipeline de features (sklearn)
            output_path: Chemin de sauvegarde
        """
        logger.info("\n--- Sauvegarde du pipeline complet ---")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Creer un dict avec tout
        full_pipeline = {
            "feature_pipeline": feature_pipeline,
            "model": model,
            "label_encoder": self.trainer.label_encoder,
            "config": {
                "model_name": self.config.model["name"],
                "random_seed": self.config.random_seed,
            }
        }
        
        # Sauvegarder
        joblib.dump(full_pipeline, output_path)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] Pipeline complet sauvegarde: {output_path}")
        logger.info(f"  Taille: {size_mb:.2f} MB")
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_pipeline: Any = None
    ) -> Any:
        """
        Execute le pipeline d'entrainement complet.
        
        Args:
            X_train: Features transformees
            y_train: Labels
            feature_pipeline: Pipeline de features (optionnel, pour sauvegarde complete)
            
        Returns:
            Modele entraine
        """
        with Timer("Entrainement du modele"):
            
            # ========================================
            # 1. Entrainer le modele
            # ========================================
            self.model = self.train_model(X_train, y_train)
            
            # ========================================
            # 2. Sauvegarder le modele seul
            # ========================================
            model_path = self.config.paths.get("model_out", "models/model.joblib")
            
            # Remplacer les placeholders dans le chemin
            model_name = self.config.model["name"]
            model_path = model_path.replace("{kind}", model_name)
            model_path = model_path.replace("{phase}", "final")
            
            self.save_model(self.model, model_path)
            
            # ========================================
            # 3. Sauvegarder le pipeline complet (si fourni)
            # ========================================
            if feature_pipeline is not None:
                full_pipeline_path = model_path.replace(".joblib", "_full_pipeline.joblib")
                self.save_full_pipeline(
                    self.model,
                    feature_pipeline,
                    full_pipeline_path
                )
            
            # ========================================
            # 4. Resume final
            # ========================================
            logger.info("\n" + "=" * 70)
            logger.info("RESUME DE L'ENTRAINEMENT")
            logger.info("=" * 70)
            logger.info(f"[OK] Modele: {self.config.model['name'].upper()}")
            logger.info(f"[OK] Donnees: {X_train.shape}")
            logger.info(f"[OK] Sauvegarde: {model_path}")
            if feature_pipeline is not None:
                logger.info(f"[OK] Pipeline complet: {full_pipeline_path}")
            logger.info("=" * 70 + "\n")
            
            return self.model


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
    
    setup_logging(level=logging.INFO)
    
    print("\n" + "="*70)
    print("Test de ModelTrainingPipeline")
    print("="*70 + "\n")
    
    try:
        # Charger la configuration
        config = load_config()
        
        # Stage 1: Ingestion
        stage1 = DataIngestionPipeline(config)
        X_train, y_train, X_test = stage1.run()
        
        # Stage 2: Validation
        stage2 = DataValidationPipeline(config)
        validation_ok = stage2.run(X_train, y_train, X_test)
        
        if not validation_ok:
            print("\n[ERROR] Validation echouee - arret du pipeline")
        else:
            # Stage 3: Transformation
            stage3 = DataTransformationPipeline(config)
            X_train_t, y_train_t, X_test_t, feature_pipeline, feature_mapping = stage3.run(
                X_train, y_train, X_test
            )
            
            # Stage 4: Training
            stage4 = ModelTrainingPipeline(config)
            model = stage4.run(X_train_t, y_train_t, feature_pipeline)
            
            print("\n[OK] Entrainement termine avec succes!")
            print(f"  Modele: {type(model).__name__}")
            print(f"  Sauvegarde dans: models/")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Erreur: {e}")
        print("Assurez-vous que les fichiers CSV existent dans data/raw/")
    except Exception as e:
        print(f"\n[ERROR] Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()





























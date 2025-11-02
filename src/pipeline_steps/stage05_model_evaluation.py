"""
Étape 5 : Évaluation du modèle (Model Evaluation) avec libellés originaux et SHAP optionnel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from scipy import sparse

from src.utils.profiling import Timer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers libellés
# ---------------------------------------------------------------------
def _load_labels_map(path: str | Path) -> Dict[int, str]:
    path = Path(path)
    if not path.exists():
        logger.warning(f"labels_map.json introuvable à {path} — noms de classes indisponibles.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Clefs peuvent être str dans le JSON -> cast en int
    return {int(k): v for k, v in raw.items()}


def _decode_to_original_ids(y_enc: np.ndarray, trainer: Any) -> np.ndarray:
    """
    Convertit les labels encodés (0..n-1) vers les IDs originaux (10, 40, 1140, ...).
    Requiert trainer.label_encoder déjà fit sur original_classes.
    """
    if trainer is None or not hasattr(trainer, "label_encoder"):
        raise ValueError("Trainer avec label_encoder requis pour décoder les classes.")
    return trainer.label_encoder.inverse_transform(y_enc)


def _build_label_names(ids: List[int], labels_map: Dict[int, str]) -> List[str]:
    """
    Construit une liste de noms lisibles: 'id - nom' (ou 'id' si nom absent).
    """
    out = []
    for cid in ids:
        name = labels_map.get(cid)
        out.append(f"{cid} - {name}" if name else str(cid))
    return out


def _get_feature_names_from_trainer(trainer: Any, n_features: int) -> List[str]:
    """
    Récupère des noms de features si possible.
    """
    try:
        if hasattr(trainer, "feature_pipeline") and hasattr(
            trainer.feature_pipeline, "get_feature_names_out"
        ):
            names = list(trainer.feature_pipeline.get_feature_names_out())
            if len(names) == n_features:
                return names
    except Exception as e:
        logger.debug(f"Impossible de récupérer les noms de features depuis le pipeline: {e}")

    return [f"f{i}" for i in range(n_features)]


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
class ModelEvaluationPipeline:
    """
    Évalue les performances, génère rapports et matrices de confusion avec labels originaux + noms.
    Gère un export SHAP optionnel (désactivé si SVD actif).
    """

    def __init__(self, config):
        self.config = config
        self.results: Dict[str, Any] = {}

        logger.info("=" * 70)
        logger.info("ÉTAPE 5 : ÉVALUATION DU MODÈLE")
        logger.info("=" * 70)

        # chemins et options
        self.results_dir = Path("results/metrics")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # config paths
        # on tolère config comme objet ou dict
        self.labels_map_path = (
            getattr(self.config.paths, "labels_map", None)
            or self.config.paths.get("labels_map", "config/labels_map.json")
        )
        self.labels_map = _load_labels_map(self.labels_map_path)

        # SHAP config
        eval_cfg = getattr(self.config, "evaluation", {}) or self.config.get("evaluation", {})
        shap_cfg = eval_cfg.get("shap", {}) if isinstance(eval_cfg, dict) else {}
        self.shap_enable: bool = bool(shap_cfg.get("enable", False))
        self.shap_sample_size: int = int(shap_cfg.get("sample_size", 10_000))
        self.shap_outdir = Path(shap_cfg.get("output_dir", "results/shap"))
        self.shap_outdir.mkdir(parents=True, exist_ok=True)

        # SVD actif ?
        feat_cfg = getattr(self.config, "features", {}) or self.config.get("features", {})
        text_cfg = feat_cfg.get("text", {})
        svd_cfg = text_cfg.get("svd", {})
        self.svd_enabled = bool(svd_cfg.get("enable", False))

    # -------------------- prédictions --------------------
    def generate_predictions(
        self,
        model: Any,
        X: np.ndarray,
        dataset_name: str = "validation",
        trainer: Any = None,
    ) -> np.ndarray:
        logger.info(f"\n--- Prédictions sur {dataset_name} ---")
        with Timer(f"Prédiction ({X.shape[0]} échantillons)"):
            if trainer is not None and hasattr(trainer, "predict"):
                y_pred_orig = trainer.predict(X)  # déjà décodé vers IDs originaux
                logger.info("Prédictions décodées vers classes originales")
                return y_pred_orig
            # fallback (labels encodés)
            y_enc = model.predict(X)
            logger.warning(
                "Prédictions en labels encodés (0..n-1). Fournir 'trainer' pour décoder."
            )
            return y_enc

    # -------------------- métriques --------------------
    def calculate_metrics(
        self,
        y_true_orig: np.ndarray,
        y_pred_orig: np.ndarray,
        dataset_name: str = "validation",
    ) -> Dict[str, float]:
        logger.info(f"\n--- Calcul des métriques ({dataset_name}) ---")

        # sklearn attend des entiers (OK)
        metrics = {
            "accuracy": accuracy_score(y_true_orig, y_pred_orig),
            "f1_weighted": f1_score(y_true_orig, y_pred_orig, average="weighted"),
            "f1_macro": f1_score(y_true_orig, y_pred_orig, average="macro"),
            "precision_weighted": precision_score(
                y_true_orig, y_pred_orig, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true_orig, y_pred_orig, average="weighted", zero_division=0
            ),
        }

        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")

        return metrics

    # -------------------- rapports --------------------
    def generate_classification_report(
        self,
        y_true_orig: np.ndarray,
        y_pred_orig: np.ndarray,
    ) -> Tuple[str, List[int], List[str]]:
        """
        Retourne le texte du rapport + la liste ordonnée des IDs + les noms correspondants.
        """
        logger.info("\n--- Rapport de classification ---")

        unique_ids = sorted(np.unique(np.concatenate([y_true_orig, y_pred_orig])))
        target_names = _build_label_names(unique_ids, self.labels_map)

        report_text = classification_report(
            y_true_orig, y_pred_orig, labels=unique_ids, target_names=target_names, zero_division=0
        )

        logger.info("Aperçu du rapport:")
        preview = report_text.splitlines()[:12]
        for line in preview:
            if line.strip():
                logger.info(line)

        return report_text, unique_ids, target_names

    # -------------------- confusion matrix --------------------
    def generate_confusion_matrices(
        self,
        y_true_orig: np.ndarray,
        y_pred_orig: np.ndarray,
        ordered_ids: Optional[List[int]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Produit deux matrices :
          - 'ids' avec index/colonnes = IDs originaux
          - 'named' avec index/colonnes = 'id - nom'
        """
        logger.info("\n--- Matrice de confusion ---")

        if ordered_ids is None:
            ordered_ids = sorted(np.unique(np.concatenate([y_true_orig, y_pred_orig])))

        cm = confusion_matrix(y_true_orig, y_pred_orig, labels=ordered_ids)

        acc_from_cm = cm.diagonal().sum() / cm.sum()
        logger.info(f"Dimensions: {cm.shape}")
        logger.info(f"Diagonal (correctes): {cm.diagonal().sum()}")
        logger.info(f"Total: {cm.sum()}")
        logger.info(f"Accuracy (depuis matrice): {acc_from_cm:.4f}")

        # IDs
        df_ids = pd.DataFrame(cm, index=ordered_ids, columns=ordered_ids)
        df_ids.index.name = "true_id"
        df_ids.columns.name = "pred_id"

        # Noms
        named = _build_label_names(ordered_ids, self.labels_map)
        df_named = pd.DataFrame(cm, index=named, columns=named)
        df_named.index.name = "true_class"
        df_named.columns.name = "pred_class"

        return {"ids": df_ids, "named": df_named}

    # -------------------- sauvegarde --------------------
    def save_results(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        report_text: Optional[str],
        cms: Optional[Dict[str, pd.DataFrame]],
    ) -> None:
        logger.info("\n--- Sauvegarde des résultats ---")

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # métriques
        metrics_file = self.results_dir / f"{model_name}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Métriques: {metrics_file}")

        # rapport
        if report_text:
            report_file = self.results_dir / f"{model_name}_classification_report.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"✓ Rapport: {report_file}")

        # matrices
        if cms:
            ids_file = self.results_dir / f"{model_name}_confusion_matrix_ids.csv"
            cms["ids"].to_csv(ids_file)
            logger.info(f"✓ Matrice (IDs): {ids_file}")

            named_file = self.results_dir / f"{model_name}_confusion_matrix_named.csv"
            cms["named"].to_csv(named_file, encoding="utf-8-sig")
            logger.info(f"✓ Matrice (nominative): {named_file}")

    # -------------------- SHAP (optionnel) --------------------
    def maybe_run_shap(
        self,
        model: Any,
        X: np.ndarray,
        trainer: Any = None,
    ) -> None:
        """
        Génère des artefacts SHAP (si activé dans la config) :
          - shap_values.npz (ou .pkl selon version)
          - mean_abs_shap_importances.csv
          - summary_plot.png
        Désactivé si SVD actif (peu interprétable métier).
        Limité à 'shap_sample_size'.
        """
        if not self.shap_enable:
            logger.info("SHAP désactivé dans la configuration.")
            return

        if self.svd_enabled:
            logger.info("SHAP ignoré (SVD actif) : vecteurs latents peu interprétables métier.")
            return

        try:
            import shap  # noqa: WPS433  (import local volontaire)
            import matplotlib.pyplot as plt  # noqa: WPS433
        except Exception as e:
            logger.warning(f"SHAP indisponible (import): {e}")
            return

        n = min(self.shap_sample_size, X.shape[0])
        if sparse.issparse(X):
            X_small = X[:n].toarray()
        else:
            X_small = np.asarray(X[:n])

        logger.info(f"Lancement SHAP sur {n} observations...")

        with Timer("SHAP computation"):
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                # fallback générique
                explainer = shap.Explainer(model)

            shap_values = explainer(X_small)

        # noms de features
        feat_names = _get_feature_names_from_trainer(trainer, X_small.shape[1])

        # importances (moyenne abs)
        try:
            # shap_values.values: (n_samples, n_features) ou list par classe
            if isinstance(shap_values.values, list):
                vals = np.stack([np.abs(v).mean(axis=0) for v in shap_values.values], axis=0).mean(
                    axis=0
                )
            else:
                vals = np.abs(shap_values.values).mean(axis=0)

            importances = (
                pd.DataFrame({"feature": feat_names, "mean_abs_shap": vals})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            imp_file = self.shap_outdir / "mean_abs_shap_importances.csv"
            importances.to_csv(imp_file, index=False, encoding="utf-8-sig")
            logger.info(f"✓ Importances SHAP: {imp_file}")
        except Exception as e:
            logger.warning(f"Impossible de calculer/sauvegarder les importances SHAP: {e}")

        # sauvegarde brute
        try:
            raw_file = self.shap_outdir / "shap_values.npz"
            # on stocke au format npz simple
            if isinstance(shap_values.values, list):
                # cas multiclass -> empilement par classe
                np.savez_compressed(
                    raw_file,
                    **{f"class_{i}": v for i, v in enumerate(shap_values.values)},
                    base_values=np.asarray(shap_values.base_values),
                )
            else:
                np.savez_compressed(
                    raw_file,
                    values=np.asarray(shap_values.values),
                    base_values=np.asarray(shap_values.base_values),
                )
            logger.info(f"✓ SHAP values: {raw_file}")
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder les valeurs SHAP: {e}")

        # summary plot
        try:
            shap.summary_plot(shap_values, X_small, feature_names=feat_names, show=False)
            fig_path = self.shap_outdir / "summary_plot.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=140)
            plt.close()
            logger.info(f"✓ SHAP summary plot: {fig_path}")
        except Exception as e:
            logger.warning(f"Impossible de générer le summary plot SHAP: {e}")

    # -------------------- run --------------------
    def run(
        self,
        model: Any,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        dataset_name: str = "validation",
        trainer: Any = None,
    ) -> Dict[str, Any]:
        """
        Exécute l'évaluation.
        - Si y_true est fourni, on le **décode** vers IDs originaux avant métriques/rapports.
        - Les prédictions sont **décodées** si 'trainer' est fourni, sinon encodées.
        """
        with Timer("Évaluation du modèle"):
            y_pred_any = self.generate_predictions(model, X, dataset_name, trainer)

            # Si y_true n'est pas fourni, on s'arrête après les prédictions
            if y_true is None:
                logger.info("\nPas de labels fournis - évaluation limitée aux prédictions")
                self.results.update(
                    {
                        "dataset_name": dataset_name,
                        "predictions": y_pred_any,
                    }
                )
                # SHAP éventuellement
                self.maybe_run_shap(model, X, trainer=trainer)
                return self.results

            # y_true est encodé (0..n-1) en sortie des étapes précédentes → on le **décode**
            if trainer is None:
                raise ValueError(
                    "Trainer requis pour décoder y_true/y_pred vers IDs originaux (labels_map)."
                )

            y_true_orig = _decode_to_original_ids(y_true, trainer)
            # y_pred_any peut être déjà original (si trainer.predict) ou encodé
            if y_pred_any.ndim == 1 and np.issubdtype(y_pred_any.dtype, np.integer):
                # heuristique : si valeurs > n_classes, on suppose IDs originaux
                if y_pred_any.max() <= len(trainer.label_encoder.classes_) - 1:
                    y_pred_orig = _decode_to_original_ids(y_pred_any, trainer)
                else:
                    y_pred_orig = y_pred_any
            else:
                y_pred_orig = y_pred_any

            # Métriques
            metrics = self.calculate_metrics(y_true_orig, y_pred_orig, dataset_name)

            # Rapport + listes ordonnées
            report_text, ordered_ids, _target_names = self.generate_classification_report(
                y_true_orig, y_pred_orig
            )

            # Matrices
            cms = self.generate_confusion_matrices(y_true_orig, y_pred_orig, ordered_ids)

            # Sauvegarde
            model_name = self.config.model["name"]
            metrics_to_save = {
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"],
                "f1_macro": metrics["f1_macro"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
            }
            self.save_results(model_name, metrics_to_save, report_text, cms)

            # Résultats en mémoire
            self.results.update(
                {
                    "dataset_name": dataset_name,
                    "predictions": y_pred_orig,
                    **metrics_to_save,
                    "classification_report": report_text,
                    "confusion_matrix_ids": cms["ids"].values,
                    "confusion_matrix_named": cms["named"].values,
                    "ordered_class_ids": ordered_ids,
                }
            )

            # SHAP optionnel (après évaluation)
            self.maybe_run_shap(model, X, trainer=trainer)

            # Résumé
            logger.info("\n" + "=" * 70)
            logger.info("RÉSUMÉ DE L'ÉVALUATION")
            logger.info("=" * 70)
            logger.info(f" Dataset: {dataset_name}")
            logger.info(f" Prédictions: {len(y_pred_orig)}")
            logger.info(f" Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f" F1 (weighted): {metrics['f1_weighted']:.4f}")
            logger.info(f" F1 (macro): {metrics['f1_macro']:.4f}")
            logger.info(f" Résultats sauvegardés dans: {self.results_dir.as_posix()}")
            logger.info("=" * 70 + "\n")

            return self.results


# -----------------------------------------------------------------------------
# Exécution autonome rapide (debug)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    from src.utils.config import load_config
    from src.pipeline_steps.stage01_data_ingestion import DataIngestionPipeline
    from src.pipeline_steps.stage02_data_validation import DataValidationPipeline
    from src.pipeline_steps.stage03_data_transformation import DataTransformationPipeline
    from src.pipeline_steps.stage04_model_training import ModelTrainingPipeline

    setup_logging(level=logging.INFO)

    print("\n" + "=" * 70)
    print("Test de ModelEvaluationPipeline")
    print("=" * 70 + "\n")

    try:
        cfg = load_config()

        # 1. Ingestion
        stage1 = DataIngestionPipeline(cfg)
        X_train, y_train, X_test = stage1.run()

        # 2. Validation
        stage2 = DataValidationPipeline(cfg)
        if not stage2.run(X_train, y_train, X_test):
            raise RuntimeError("Validation échouée.")

        # 3. Transformation
        stage3 = DataTransformationPipeline(cfg)
        X_train_t, y_train_t, X_test_t, feature_pipeline = stage3.run(
            X_train, y_train, X_test
        )

        # 4. Training
        stage4 = ModelTrainingPipeline(cfg)
        model = stage4.run(X_train_t, y_train_t, feature_pipeline)
        trainer = stage4.trainer  # pour décodage + feature names

        # 5. Évaluation (sur train pour exemple)
        stage5 = ModelEvaluationPipeline(cfg)
        _ = stage5.run(model, X_train_t, y_train_t, dataset_name="train", trainer=trainer)

        print("\n Évaluation terminée.")

    except Exception as e:
        import traceback

        print(f"\n Erreur: {e}")
        traceback.print_exc()
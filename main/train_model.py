"""
Classification Rakuten – Baselines & Pipeline (version commentée)

Objectif général :
- Construire une pipeline multimodale (texte + image) avec rééchantillonnage pour traiter le déséquilibre des classes.
- Comparer rapidement deux classifieurs (LogisticRegression et LinearSVC) en validation croisée.
- Entraîner le modèle complet sur X_train/Y_train puis prédire sur X_test.

Baselines :
- B0 — Naïf (majoritaire) : DummyClassifier(strategy="most_frequent").
- B1 — Naïf (aléatoire stratifié) : DummyClassifier(strategy="stratified").
- B2 — Texte seul (TF-IDF → LR) : branche texte sans rééchantillonnage (référence).
- B3 — Image seule (ImageLoader → flatten → PCA → LR) : sans texte.
- B4 — Multimodal complet : Texte (TF-IDF) + Images (pixels + stats optionnelles).
      Rééchantillonnage : under-sampling adaptatif par fold + over-sampling ciblé.
      (Under-sampling “capé” par classe, recalculé à chaque fold → CV-safe.)

Fonctionnalités CLI :
- --model {lr,svc} : définir le classifieur final (défaut : lr).
- --compare : comparer LR et LinearSVC via validation croisée stratifiée (F1-macro)
              sur X_train/Y_train, en conservant le split Rakuten d’origine.
- --config : lire un fichier TOML pour centraliser chemins & hyperparams.

Détails pipeline :
- Fusionner texte+image avec FeatureUnion.
- Branche images “pixels” : charger RGB, redimensionner, aplatir, (option) PCA/SVD.
- Branche images “stats” : ImageStatsFeaturizer (width, height, occupancy, white_ratio, black_ratio).
  Les seuils venir du TOML : [images.stats] white_threshold, black_threshold, min_area.
  Si out_prefix="auto", faire refléter ces seuils dans les noms de colonnes (ex. img_w230_b25_*).
- Rééchantillonnage :
  * Under-sampling : AdaptiveUnderSampler(cap_dict) → pour chaque fold, fixer la cible à min(plafond, effectif_du_fold).
  * Over-sampling : RandomOverSampler(sampling_strategy=...).
- Placer StandardScaler(with_mean=False) APRÈS under/over pour scaler la distribution réellement vue par le modèle.
- Sérialiser avec Joblib, sauvegarder les prédictions X_test (index produit), journaliser les étapes.

Exécution :
# Baselines
python -m main.train_model --config features/config.toml --baseline b0
python -m main.train_model --config features/config.toml --baseline b1
python -m main.train_model --config features/config.toml --baseline b2
python -m main.train_model --config features/config.toml --baseline b3

# Modèle multimodal “B4” (entraînement + prédictions)
python -m main.train_model --config features/config.toml

# Comparaison LR vs SVC (CV stratifiée F1-macro, avec under adaptatif)
python -m main.train_model --config features/config.toml --compare
"""

# === Importer les bibliothèques standard et ML ==================================
import os
import argparse
import joblib
import time
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline, Pipeline as SkPipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import SamplerMixin

# === Définir le chemin par défaut du TOML ======================================
# Définir un chemin par défaut vers features/config.toml (relatif à ce fichier)
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "features" / "config.toml"

# === Charger la configuration TOML =============================================
# Utiliser tomllib (Py>=3.11) sinon rétroporter sur tomli
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # fallback si environnement < 3.11
    import tomli as tomllib


def load_config(config_path: str | Path | None = None) -> dict:
    """Charger le fichier TOML et retourner un dict Python."""
    cfg_path = Path(config_path) if config_path else DEFAULT_CFG
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg

# === Importer les pipelines texte / image et le featurizer de stats ============
# Importer les fabriques de branches (définies ailleurs dans le repo)
from models.text_pipeline import create_text_pipeline            # construire la branche texte
from models.image_pipeline import create_image_pipeline          # construire la branche pixels
from features.image_stats import ImageStatsFeaturizer            # extraire des stats d'objets


# === Initialiser les graines de hasard (reproductibilité) =======================
def init_seeds(seed: int) -> None:
    """Fixer les graines pour numpy, random et Python hash."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# === Construire les stratégies d'échantillonnage ================================
def make_sampling_strategies(y_train: pd.Series,
                             major_class: int = 2583,
                             major_cap: int = 6000,
                             tail_min: int = 1500):
    """
    Construire deux stratégies pour imblearn :
      - under : plafonner uniquement la classe major_class à major_cap
      - over  : remonter toutes les classes d'effectif < tail_min à tail_min
    Retourner (under_dict, over_dict).
    """
    vc = y_train.value_counts()
    under = {
        int(cls): (min(int(cnt), int(major_cap)) if int(cls) == int(major_class) else int(cnt))
        for cls, cnt in vc.items()
    }
    over = {int(cls): int(tail_min) for cls, cnt in vc.items() if int(cnt) < int(tail_min)}
    return under, over


# === Définir un under-sampler adaptatif (CV-safe) ===============================
class AdaptiveUnderSampler(BaseEstimator, SamplerMixin):
    """
    Implémenter un under-sampler adaptatif : pour chaque fold de CV,
    recalculer une cible par classe en la bornant à min(plafond, effectif_du_fold).
    cap_dict : {classe: plafond_max} (ex. {2583: 6000}).
    """
    def __init__(self, cap_dict=None, random_state=None):
        self.cap_dict = cap_dict or {}
        self.random_state = random_state

    def fit_resample(self, X, y):
        # Compter les effectifs observés dans CE fold
        cnt = Counter(y)
        # Construire une stratégie feasible par classe
        sampling_strategy = {cls: min(n, self.cap_dict.get(cls, n)) for cls, n in cnt.items()}
        # Déléguer à RandomUnderSampler avec la stratégie clipée
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
        Xr, yr = rus.fit_resample(X, y)
        return Xr, yr


# === Fabriquer le classifieur à partir de la config =============================
def build_classifier(cfg: dict, seed: int):
    """Construire le classifieur (LR ou LinearSVC) en lisant [model] dans le TOML."""
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", "lr")).lower()
    use_class_weight = bool(model_cfg.get("use_class_weight", False))
    solver = str(model_cfg.get("solver", "saga"))
    C = float(model_cfg.get("C", 1.0))
    max_iter = int(model_cfg.get("max_iter", 3000))
    tol = float(model_cfg.get("tol", 1e-3))
    ovr = bool(model_cfg.get("ovr", False))
    cw = "balanced" if use_class_weight else None

    if name == "svc":
        # Retourner un SVM linéaire (souvent performant sur TF-IDF sparse)
        return LinearSVC(class_weight=cw)

    # Configurer une LogisticRegression sparse-friendly (saga), 
    # sans multi_class explicite (déprécié)
    # Lire n_jobs depuis le TOML (section [compute])
    n_jobs = int(cfg.get("compute", {}).get("n_jobs", 1))

    base_lr = LogisticRegression(
        solver=solver,
        penalty="l2",
        C=C,
        max_iter=max_iter,
        tol=tol,
        class_weight=cw,
        random_state=seed,
        n_jobs=1,
    )
    if ovr:
    # Envelopper en One-vs-Rest explicite et paralléliser les K classifieurs
        from sklearn.multiclass import OneVsRestClassifier
        return OneVsRestClassifier(base_lr, n_jobs=n_jobs)

    # Sinon, retourner la LR multinomiale “native”
    return base_lr


# === Construire les pipelines baselines sans rééchantillonnage ==================
def build_baseline_pipeline(kind: str, cfg: dict, seed: int):
    """
    Construire une pipeline baseline SANS rééchantillonnage.
      - b0: Dummy most_frequent
      - b1: Dummy stratified
      - b2: Texte seul (TF-IDF -> LR)
      - b3: Image seule (Image -> flatten -> PCA -> LR)
    Retourner (pipe, need_cols) où need_cols est la liste des colonnes nécessaires de X.
    """
    need_cols = ["designation", "description", "productid", "imageid"]

    if kind == "b0":
        # Prédire toujours la classe majoritaire
        pipe = DummyClassifier(strategy="most_frequent")
        return pipe, ["designation"]  # importer n'importe quelle colonne non vide

    if kind == "b1":
        # Échantillonner des prédictions selon la distribution des classes
        pipe = DummyClassifier(strategy="stratified", random_state=seed)
        return pipe, ["designation"]

    if kind == "b2":
        # Construire la branche texte seule (sans under/over)
        text_cfg = cfg.get("text", {})
        text_branch = create_text_pipeline(
            max_features=text_cfg.get("max_features", 5000),
            translate_map_path=text_cfg.get("translate_map_path", None),
            use_stem=bool(text_cfg.get("use_stem", True)),
            min_df=text_cfg.get("min_df", 0.0),
            max_df=text_cfg.get("max_df", 1.0),
            sublinear_tf=bool(text_cfg.get("sublinear_tf", True)),
            norm=text_cfg.get("norm", "l2"),
            trip_accents=text_cfg.get("strip_accents", "unicode"),
            stop_words=text_cfg.get("stop_words", None),
        )
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=3000,
            tol=1e-3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        pipe = SkPipeline([("text", text_branch), ("clf", clf)])
        return pipe, need_cols

    if kind == "b3":
        # Construire la branche image seule : charger -> flatten -> PCA -> LR
        img_size = tuple(cfg.get("images", {}).get("size", [64, 64]))
        img_dir = cfg["images"]["train_dir"]
        img_branch = create_image_pipeline(
            image_dir=img_dir,
            image_size=img_size,
            dim_reduction={"enabled": False},  # placer la PCA juste après
        )
        pca_n = int(cfg.get("images", {}).get("dim_reduction", {}).get("n_components", 100))
        img_pca = make_pipeline(img_branch, PCA(n_components=pca_n, random_state=seed))
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=3000,
            tol=1e-3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        pipe = SkPipeline([("img", img_pca), ("clf", clf)])
        return pipe, need_cols

    raise ValueError(f"Baseline inconnue: {kind}")


# === Exécuter une baseline et écrire un rapport CV ==============================
def run_baseline_and_report(kind: str, X_train: pd.DataFrame, y_train: pd.Series, cfg: dict, outdir: str = "results"):
    """
    Exécuter la baseline 'kind' en CV, puis écrire un résumé clair :
      - results/baseline_results_summary.csv (append)
      - results/report_{kind}_cv.txt        (rapport complet par classe)
    """
    os.makedirs(outdir, exist_ok=True)

    seed = int(cfg.get("random", {}).get("seed", 42))
    splits = int(cfg.get("cv", {}).get("splits", 3))
    shuffle = bool(cfg.get("cv", {}).get("shuffle", True))
    cv_seed = int(cfg.get("cv", {}).get("random_state", seed))

    pipe, need_cols = build_baseline_pipeline(kind, cfg, seed)

    # Construire une CV stratifiée
    cv = StratifiedKFold(n_splits=splits, shuffle=shuffle, random_state=cv_seed)

    # Produire des prédictions out-of-fold pour obtenir un rapport global
    t0 = time.time()
    y_pred_cv = cross_val_predict(pipe, X_train[need_cols], y_train, cv=cv, n_jobs=1, method="predict")
    dt = time.time() - t0

    # Calculer les métriques
    f1_macro = f1_score(y_train, y_pred_cv, average="macro")
    f1_weighted = f1_score(y_train, y_pred_cv, average="weighted")
    report = classification_report(y_train, y_pred_cv, digits=4)

    # Sauvegarder le résumé
    summary_csv = os.path.join(outdir, "baseline_results_summary.csv")
    row = pd.DataFrame([{
        "baseline": kind.upper(),
        "cv_splits": splits,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_infer_time_sec": round(dt, 3),
        "notes": f"text_only={kind=='b2'} | image_only={kind=='b3'} | dummy={kind in ['b0','b1']}"
    }])
    row.to_csv(summary_csv, index=False, mode="a", header=not os.path.exists(summary_csv))

    # Sauvegarder le rapport détaillé
    with open(os.path.join(outdir, f"report_{kind}_cv.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== Baseline {kind.upper()} — CV (n_splits={splits}) ===\n")
        f.write(f"F1-macro     : {f1_macro:.4f}\n")
        f.write(f"F1-weighted  : {f1_weighted:.4f}\n")
        f.write(f"Time (sec)   : {dt:.2f}\n\n")
        f.write(report)

    # Afficher un résumé console
    print(f"[{kind.upper()}] F1-macro={f1_macro:.4f} | F1-weighted={f1_weighted:.4f} | time={dt:.1f}s")
    print(f"> Résumé: {summary_csv}")
    print(f"> Rapport: {os.path.join(outdir, f'report_{kind}_cv.txt')}")


# === Construire la pipeline multimodale complète ================================
def create_combined_pipeline(cfg: dict, under_strategy: dict, over_strategy: dict, seed: int):
    """Construire la pipeline texte+image, rééchantillonner, scaler, et ajouter le modèle final."""
    # Construire la branche TEXTE
    text_cfg = cfg.get("text", {})
    text_branch = create_text_pipeline(
        max_features=text_cfg.get("max_features", 5000),
        translate_map_path=text_cfg.get("translate_map_path", None),
        use_stem=bool(text_cfg.get("use_stem", True)),
        min_df=text_cfg.get("min_df", 0.0),
        max_df=text_cfg.get("max_df", 1.0),
    )

    # Construire la branche IMAGES (pixels)
    image_train_dir = cfg["images"]["train_dir"]
    image_size = tuple(cfg.get("images", {}).get("size", [64, 64]))
    image_pixels = create_image_pipeline(
        image_dir=image_train_dir,
        image_size=image_size,
        dim_reduction=cfg.get("images", {}).get("dim_reduction", {}),
    )

    # Ajouter éventuellement la branche IMAGES (stats)
    transformers = [("text", text_branch), ("image_pixels", image_pixels)]
    stats_cfg = cfg.get("images", {}).get("stats", {})
    if bool(stats_cfg.get("enabled", False)):
        image_stats = ImageStatsFeaturizer(
            image_dir=image_train_dir,
            imgid_col="imageid",
            pid_col="productid",
            white_threshold=int(stats_cfg.get("white_threshold", 230)),
            black_threshold=int(stats_cfg.get("black_threshold", 25)),
            min_area=int(stats_cfg.get("min_area", 16)),
            out_prefix=str(stats_cfg.get("out_prefix", "auto")),
        )
        transformers.append(("image_stats", image_stats))

    # Fusionner les branches
    features = FeatureUnion(transformer_list=transformers)

    # Construire le classifieur final
    model = build_classifier(cfg, seed)

    # Construire la pipeline Imbalanced-Learn
    pipe = ImbPipeline(steps=[
        ("features", features),
        ("under", AdaptiveUnderSampler(cap_dict=under_strategy, random_state=seed)),
        ("over", RandomOverSampler(sampling_strategy=over_strategy, random_state=seed)),
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model),
    ])
    return pipe


# === Entraîner sur le train, prédire sur le test ================================
def train_and_predict_on_test(X_train, y_train, X_test, cfg: dict):
    """Entraîner la pipeline complète sur X_train/Y_train puis prédire y_test (labels) pour X_test."""
    # Lire les seeds
    seed = int(cfg.get("random", {}).get("seed", 42))

    # Construire les stratégies d'échantillonnage à partir de y_train
    under, over = make_sampling_strategies(
        y_train,
        major_class=cfg["sampling"]["major_class"],
        major_cap=cfg["sampling"]["major_cap"],
        tail_min=cfg["sampling"]["tail_min"],
    )

    # Construire la pipeline complète
    pipe = create_combined_pipeline(cfg, under, over, seed)

    # Entraîner la pipeline
    print(">> Entraîner la pipeline complète…")
    pipe.fit(X_train, y_train)

    # Re-pointer les dossiers images vers TEST pour l'inférence
    print(">> Re-pointer les lecteurs d'images vers le dossier TEST…")
    feat_union = pipe.named_steps["features"]
    image_test_dir = cfg["images"]["test_dir"]

    # Parcourir les transformeurs et mettre à jour les répertoires d'images
    new_list = []
    for name, sub in feat_union.transformer_list:
        if name == "image_pixels":
            # sub est une sklearn.Pipeline (loader → transformations)
            if "loader" in sub.named_steps:
                loader = sub.named_steps["loader"]
                if hasattr(loader, "set_image_dir"):
                    loader.set_image_dir(image_test_dir)
                elif hasattr(loader, "image_dir"):
                    loader.image_dir = image_test_dir
                else:
                    raise RuntimeError("ImageLoader n'avoir ni 'set_image_dir' ni attribut 'image_dir'.")
            new_list.append((name, sub))
        elif name == "image_stats":
            if hasattr(sub, "set_image_dir"):
                sub.set_image_dir(image_test_dir)
            new_list.append((name, sub))
        else:
            new_list.append((name, sub))
    feat_union.transformer_list = new_list

    # Prédire sur X_test
    print(">> Prédire sur X_test…")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This Pipeline instance is not fitted yet",
            category=FutureWarning,
            module=r"sklearn\.pipeline",
        )
        y_pred = pipe.predict(X_test)

    return pipe, y_pred


# === Comparer LR et SVC en CV stratifiée =======================================
def compare_models_cv(X_train, y_train, cfg: dict):
    """Comparer LR et LinearSVC avec la même pipeline (sauf le classifieur), en CV F1-macro."""
    seed = int(cfg.get("random", {}).get("seed", 42))
    n_jobs = int(cfg.get("compute", {}).get("n_jobs", 1))
    cv_splits = int(cfg.get("cv", {}).get("splits", 3))
    shuffle = bool(cfg.get("cv", {}).get("shuffle", True))
    cv_seed = int(cfg.get("cv", {}).get("random_state", seed))

    # Construire les stratégies under/over à partir de y_train
    under, over = make_sampling_strategies(
        y_train,
        major_class=cfg["sampling"]["major_class"],
        major_cap=cfg["sampling"]["major_cap"],
        tail_min=cfg["sampling"]["tail_min"],
    )

    # Définir la CV
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=shuffle, random_state=cv_seed)

    # Évaluer les deux modèles
    rows = []
    for name in ["lr", "svc"]:
        cfg_local = {**cfg, "model": {**cfg.get("model", {}), "name": name}}
        pipe = create_combined_pipeline(cfg_local, under, over, seed)
        scores = cross_val_score(
            pipe, X_train, y_train,
            scoring="f1_macro", cv=cv, n_jobs=n_jobs
        )
        print(f"   - {name.upper()} | F1-macro = {scores.mean():.4f} ± {scores.std():.4f} | {scores}")
        rows.append({
            "model": name.upper(),
            "cv_mean_f1_macro": scores.mean(),
            "cv_std": scores.std(),
            "cv_scores": scores.tolist(),
        })

    return pd.DataFrame(rows)


# === Parser les arguments CLI ===================================================
def parse_args():
    """Définir les arguments CLI et retourner l'objet Namespace."""
    p = argparse.ArgumentParser(description="Entraîner la pipeline texte+image avec rééchantillonnage ; comparer LR vs SVC en option")
    p.add_argument("--config", default=str(DEFAULT_CFG), help="Chemin vers le TOML (défaut: features/config.toml)")
    p.add_argument("--compare", action="store_true", help="Comparer LR vs SVC via CV sur X_train (F1-macro)")
    p.add_argument("--baseline", choices=["b0", "b1", "b2", "b3"], help="Exécuter une baseline simple et sortir")
    p.add_argument("--model", choices=["lr", "svc"], default=None, help="Forcer le modèle (écraser [model].name dans le TOML)")
    return p.parse_args()


# === Fonction principale ========================================================
def main():
    # Lire les arguments et la configuration
    args = parse_args()
    cfg = load_config(args.config)

    # Initialiser les graines pour la reproductibilité
    seed = int(cfg.get("random", {}).get("seed", 42))
    init_seeds(seed)

    # Charger les chemins de données depuis le TOML
    print(">> Charger les données…")
    X_train = pd.read_csv(cfg["paths"]["x_train_csv"], index_col=0)
    y_train = pd.read_csv(cfg["paths"]["y_train_csv"], index_col=0).squeeze()
    X_test = pd.read_csv(cfg["paths"]["x_test_csv"], index_col=0)

    # Vérifier les colonnes nécessaires
    needed = ["designation", "description", "productid", "imageid"]
    for col in needed:
        if col not in X_train.columns:
            raise ValueError(f"Colonne manquante dans X_train : '{col}'")
        if col not in X_test.columns:
            raise ValueError(f"Colonne manquante dans X_test : '{col}'")

    # Exécuter une baseline et sortir si demandé
    if args.baseline:
        run_baseline_and_report(args.baseline, X_train, y_train, cfg, outdir="results")
        return

    # Comparer LR vs SVC en CV si demandé
    if args.compare:
        print(">> Comparer LR vs SVC (CV F1-macro)…")
        df_cmp = compare_models_cv(X_train[needed], y_train, cfg)
        out_cmp = cfg["outputs"]["compare_out"]
        os.makedirs(os.path.dirname(out_cmp), exist_ok=True)
        df_cmp.to_csv(out_cmp, index=False)
        print(f">> Sauvegarder les résultats de comparaison : {out_cmp}")

    # Entraîner la pipeline complète et prédire sur X_test
    pipe, y_pred = train_and_predict_on_test(X_train[needed], y_train, X_test[needed], cfg)

    # Sauvegarder le modèle et les prédictions
    os.makedirs(os.path.dirname(cfg["outputs"]["model_out"]), exist_ok=True)
    joblib.dump(pipe, cfg["outputs"]["model_out"])
    print(f">> Sauvegarder le modèle : {cfg['outputs']['model_out']}")

    pred_df = pd.DataFrame(y_pred, index=X_test.index, columns=["predicted_label"])
    pred_df.to_csv(cfg["outputs"]["pred_out"])
    print(f">> Sauvegarder les prédictions : {cfg['outputs']['pred_out']}")


# === Entrée script ==============================================================
if __name__ == "__main__":
    main()

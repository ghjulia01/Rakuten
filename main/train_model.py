# main/train_model.py
"""
Pipeline texte+image avec under/over-sampling (imblearn) et comparaison LR vs SVC.

Fonctionnalités  :
- --model {lr,svc} : permet le choix du classifieur final (par défaut: lr).
- --compare : évalue LR et SVC via validation croisée (F1-macro) 
sur X_train/Y_train (sans toucher au split Rakuten).
- --config : lecture optionnelle d'un fichier TOML pour centraliser 
chemins & hyperparams.
- Journalisation claire des étapes pour faciliter le rapport.

- Sauvegarde du pipeline entraîné et des prédictions sur X_test.
- Prédictions sauvegardées dans un CSV avec index produit.
- Utilisation de joblib pour la sérialisation du modèle.
- Pipelines combinés texte+image avec FeatureUnion.
- Stratégies de rééchantillonnage dynamiques basées sur y_train.
- Le split Rakuten est respecté : X_train/Y_train pour l'entraînement, 
X_test pour la prédiction finale.
- Dossiers images séparés train/test.
"""
"""
Pipeline texte + image (pixels + stats objet) avec under/over-sampling.
- Lecture d'un config TOML (chemins, hyperparams, seuils image, etc.)
- Option --compare : CV stratifiée pour LR vs LinearSVC (F1-macro)
- Respect strict du split Rakuten proposé (X_train/Y_train vs X_test)

Script exécutable via :
python -m main.train_model --config config.toml        # entraînement + prédiction
python -m main.train_model --config config.toml --compare  # + comparaison LR vs SVC (CV)
"""

import os
import argparse
import joblib
import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from pathlib import Path

# Résoudre le chemin du TOML en relatif au fichier
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "features" / "config.toml"

# Py 3.11+ : tomllib (standard). Sinon, fallback sur tomli si installé.
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli (si Python < 3.11)

def load_config(config_path: str | Path | None = None) -> dict:
    cfg_path = Path(config_path) if config_path else DEFAULT_CFG
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg


from models.text_pipeline import create_text_pipeline
from models.image_pipeline import create_image_pipeline  
# pixels (resize->flatten->sparse)
from features.image_stats import ImageStatsFeaturizer     
# stats objet: width/height/occ AVANT resize


# ---------------------------- Sampling ----------------------------
def make_sampling_strategies(y_train: pd.Series, 
                             major_class=2583, 
                             major_cap=6000, 
                             tail_min=1500):
    """
    Construit deux stratégies pour imblearn:
      - under: seule la classe major_class plafonnée à major_cap
      - over : toutes classes avec effectif < tail_min remontées à tail_min
    """
    vc = y_train.value_counts()
    under = {
        int(cls): (min(int(cnt), 
                       int(major_cap)) if int(cls) == int(major_class) else int(cnt))
        for cls, cnt in vc.items()
    }
    over = {int(cls): int(tail_min) for cls, 
            cnt in vc.items() if int(cnt) < int(tail_min)}
    return under, over


# ------------------------- Classifier factory ---------------------
def build_classifier(name: str, use_class_weight: bool):
    cw = "balanced" if use_class_weight else None
    name = (name or "lr").lower()
    if name == "svc":
        return LinearSVC(class_weight=cw)
    return LogisticRegression(max_iter=3000, solver="saga", class_weight=cw, n_jobs=1)


# ---------------------- Pipeline construction ---------------------
def create_combined_pipeline(cfg: dict, under_strategy: dict, 
                             over_strategy: dict):
    # --- TEXT ---
    text_cfg = cfg.get("text", {}) 
    text_branch = create_text_pipeline(
        max_features=cfg.get("text", {}).get("max_features", 5000),
        translate_map_path=cfg.get("text", {}).get("translate_map_path", None),
        use_stem=bool(cfg.get("text", {}).get("use_stem", True)),
        min_df=text_cfg.get("min_df", 0.0),   
        max_df=text_cfg.get("max_df", 1.0)
    )

    # --- IMAGES: pixels (dimxdim-> flatten -> sparse) ---
   # récupère les paramètres depuis le TOML
    image_train_dir = cfg["images"]["train_dir"]
    image_size = tuple(cfg["images"].get("size", [64, 64]))

    image_pixels = create_image_pipeline(
        image_dir=image_train_dir,
        image_size=image_size,
        dim_reduction=cfg.get("images", {}).get("dim_reduction", {})
    )

    # --- IMAGES: stats (width/height/occupancy hors blanc/noir, 
    # AVANT resize), 
    # activable via TOML ---
    transformers = [("text", text_branch), ("image_pixels", image_pixels)]
    stats_cfg = cfg.get("images", {}).get("stats", {})
    if bool(stats_cfg.get("enabled", False)):
        image_stats = ImageStatsFeaturizer(
            image_dir=image_train_dir,          # <— ICI aussi on passe le train_dir
            imgid_col="imageid",
            pid_col="productid",
            white_threshold=int(stats_cfg.get("white_threshold", 230)),
            black_threshold=int(stats_cfg.get("black_threshold", 25)),
            min_area=int(stats_cfg.get("min_area", 16)),
            out_prefix=str(stats_cfg.get("out_prefix", "img_w230_b25_")),
        )
        transformers.append(("image_stats", image_stats))

    features = FeatureUnion(transformer_list=transformers)

    model = build_classifier(
        name=cfg.get("model", {}).get("name", "lr"),
        use_class_weight=bool(cfg.get("model", {}).get("use_class_weight", True)),
    )

    pipe = ImbPipeline(steps=[
        ("features", features),
        ("scaler", StandardScaler(with_mean=False)),
        ("under", RandomUnderSampler(sampling_strategy=under_strategy, 
                                     random_state=42)),
        ("over", RandomOverSampler(sampling_strategy=over_strategy, 
                                   random_state=42)),
        ("model", model),
    ])
    return pipe


# ---------------- Train on train, predict on test -----------------
def train_and_predict_on_test(X_train, y_train, X_test, cfg: dict):
    under, over = make_sampling_strategies(
        y_train,
        major_class=cfg["sampling"]["major_class"],
        major_cap=cfg["sampling"]["major_cap"],
        tail_min=cfg["sampling"]["tail_min"],
    )
    pipe = create_combined_pipeline(cfg, under, over)

    print(">> Entraînement…")
    pipe.fit(X_train, y_train)

    print(">> Prédiction sur X_test…")
    # Repointer les branches images vers dossier TEST (pixels + stats)
    feat_union = pipe.named_steps["features"]
    new_list = []
    image_test_dir = cfg["images"]["test_dir"]
    image_size = tuple(cfg["images"].get("size", [64, 64]))
    for name, sub in feat_union.transformer_list:
        if name == "image_pixels":
            new_list.append((name, 
                             create_image_pipeline(image_dir=image_test_dir, 
                                                   image_size=image_size)))
        elif name == "image_stats":
            sub.set_image_dir(image_test_dir)  # méthode prévue dans le featurizer
            new_list.append((name, sub))
        else:
            new_list.append((name, sub))
    feat_union.transformer_list = new_list

    y_pred = pipe.predict(X_test)
    return pipe, y_pred


# ----------------------- CV comparison (opt) ----------------------
def compare_models_cv(X_train, y_train, cfg: dict, cv_splits=3, random_state=42):
    """
    Évalue LR et LinearSVC avec exactement la même pipeline (sauf le classifieur),
    CV stratifiée (F1-macro). On ne touche pas à X_test.
    """
    print(">> Comparaison LR vs SVC (CV F1-macro)…")
    under, over = make_sampling_strategies(
        y_train,
        major_class=cfg["sampling"]["major_class"],
        major_cap=cfg["sampling"]["major_cap"],
        tail_min=cfg["sampling"]["tail_min"],
    )
    cv = StratifiedKFold(n_splits=int(cv_splits), 
                         shuffle=True, 
                         random_state=random_state)

    res = []
    for name in ["lr", "svc"]:
        cfg_local = {**cfg, "model": {**cfg.get("model", {}), "name": name}}
        pipe = create_combined_pipeline(cfg_local, under, over)
        scores = cross_val_score(pipe, X_train, 
                                 y_train, scoring="f1_macro", cv=cv, n_jobs=1)
        print(f"   - {name.upper()} | F1-macro = {scores.mean():.4f} ± {scores.std():.4f} | {scores}")
        res.append({"model": name.upper(), 
                    "cv_mean_f1_macro": scores.mean(), 
                    "cv_std": scores.std(), 
                    "cv_scores": scores.tolist()})
    return pd.DataFrame(res)


# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train (text+image) with sampling; optional LR vs SVC comparison")
    p.add_argument("--config", default="config.toml", help="Chemin vers le TOML (défaut: config.toml)")
    p.add_argument("--compare", action="store_true", help="Compare LR vs SVC via CV sur X_train (F1-macro)")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    print(">> Chargement des données…")
    X_train = pd.read_csv(cfg["paths"]["x_train_csv"], index_col=0)
    y_train = pd.read_csv(cfg["paths"]["y_train_csv"], index_col=0).squeeze()
    X_test  = pd.read_csv(cfg["paths"]["x_test_csv"],  index_col=0)

    # Colonnes requises côté features (respect split Rakuten)
    needed = ["designation", "description", "productid", "imageid"]
    for col in needed:
        if col not in X_train.columns:
            raise ValueError(f"Colonne manquante dans X_train : '{col}'")
        if col not in X_test.columns:
            raise ValueError(f"Colonne manquante dans X_test : '{col}'")

    # Option: comparaison LR vs SVC (ne touche pas X_test)
    if args.compare:
        df_cmp = compare_models_cv(
            X_train[needed], y_train,
            cfg=cfg,
            cv_splits=cfg.get("cv", {}).get("splits", 3)
        )
        out_cmp = cfg["outputs"]["compare_out"]
        os.makedirs(os.path.dirname(out_cmp), exist_ok=True)
        df_cmp.to_csv(out_cmp, index=False)
        print(f">> Résultats comparison sauvegardés : {out_cmp}")

    # Entraînement complet + prédiction sur X_test
    pipe, y_pred = train_and_predict_on_test(X_train[needed], y_train, X_test[needed], cfg)

    # Sauvegardes
    os.makedirs(os.path.dirname(cfg["outputs"]["model_out"]), exist_ok=True)
    joblib.dump(pipe, cfg["outputs"]["model_out"])
    print(f">> Modèle sauvegardé : {cfg['outputs']['model_out']}")

    pred_df = pd.DataFrame(y_pred, index=X_test.index, columns=["predicted_label"])
    pred_df.to_csv(cfg["outputs"]["pred_out"])
    print(f">> Prédictions sauvegardées : {cfg['outputs']['pred_out']}")


if __name__ == "__main__":
    main()
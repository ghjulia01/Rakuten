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

# TOML: compatibilité py>=3.11 (tomllib) et py<3.11 (tomli)
try:
    import tomllib as tomli  # py311+ si py<3.11 tomli
except Exception:  # pragma: no cover
    import tomli  # type: ignore

from models.text_pipeline import create_text_pipeline
from models.image_pipeline import create_image_pipeline  
# DataFrame -> select_pid + loader + flatten + to_sparse


# ---------------------------------------------------------------------
# Stratégies de rééchantillonnage
# ---------------------------------------------------------------------
def make_sampling_strategies(y_train: pd.Series, major_class=2583, major_cap=6000, tail_min=1500):
    """
    Construit deux stratégies de rééquilibrage à passer à imblearn :
      - under : seule la classe major_class est réduite à major_cap 
      si besoin (les autres inchangées)
      - over  : toutes les classes avec effectif < tail_min sont remontées à tail_min
    """
    vc = y_train.value_counts()

    # undersampling ciblé: seule la classe major_class est potentiellement réduite
    under = {
        int(cls): (min(int(cnt), int(major_cap)) if int(cls) == int(major_class) else int(cnt))
        for cls, cnt in vc.items()
    }

    # oversampling des petites classes: toute classe < tail_min -> tail_min
    over = {int(cls): int(tail_min) for cls, cnt in vc.items() if int(cnt) < int(tail_min)}

    return under, over


# ---------------------------------------------------------------------
# Fabrique du modèle (LR ou SVC)
# ---------------------------------------------------------------------
def build_classifier(model_name: str, use_class_weight: bool):
    """
    Retourne un estimateur sklearn selon --model.
    - 'lr'  -> LogisticRegression(max_iter=1000, solver='lbfgs', 
    class_weight='balanced' optionnel)
    - 'svc' -> LinearSVC(class_weight='balanced' optionnel)
    """
    cw = "balanced" if use_class_weight else None

    if model_name.lower() == "svc":
        # LinearSVC : efficace en haute dimension, adapté aux matrices 
        # creuses TF-IDF + pixels aplatis
        return LinearSVC(class_weight=cw)
    # par défaut : LR
    return LogisticRegression(max_iter=1000, solver="lbfgs", class_weight=cw, n_jobs=None)


# ---------------------------------------------------------------------
# Pipeline combiné (texte + image) avec under + over sampling
# ---------------------------------------------------------------------
def create_combined_pipeline(image_dir, image_size=(64, 64),
                             under_strategy=None, over_strategy=None,
                             model_name: str = "lr", use_class_weight: bool = True):
    """
    - Texte : create_text_pipeline() 
    (nettoyage + vectorisation de designation+description)
    - Image : create_image_pipeline(image_dir) 
    (productid -> ImageLoader -> resize -> flatten -> sparse)
    - Concat : FeatureUnion
    - Scale : StandardScaler(with_mean=False pour sparse)
    - Under : RandomUnderSampler (pour 2583 seulement)
    - Over  : RandomOverSampler  (pour toutes classes < seuil)
    - Modèle : LR ou LinearSVC (class_weight='balanced' optionnel)

    NB : L'ordre scaler -> under -> over garantit que le modèle 
    voit des features normalisées
    et que l'échantillonnage se fait dans l'espace standardisé.
    """
    text_branch = create_text_pipeline()
    image_branch = create_image_pipeline(image_dir=image_dir, image_size=image_size)

    features = FeatureUnion([
        ("text", text_branch),
        ("image", image_branch),
    ])

    model = build_classifier(model_name, use_class_weight)

    return ImbPipeline(steps=[
        ("features", features),
        ("scaler", StandardScaler(with_mean=False)),
        ("under",  RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
        ("over",   RandomOverSampler(sampling_strategy=over_strategy,  random_state=42)),
        ("model",  model),
    ])


# ---------------------------------------------------------------------
# Entraîner sur train + prédire sur test (Rakuten split respecté)
# ---------------------------------------------------------------------
def train_and_predict_on_test(X_train, y_train, X_test,
                              image_train_dir, image_test_dir,
                              image_size=(64, 64),
                              major_class=2583, major_cap=6000, tail_min=1500,
                              use_class_weight=True,
                              model_name: str = "lr"):
    # 1) Construire les stratégies dynamiques depuis y_train
    under, over = make_sampling_strategies(
        y_train, major_class=major_class, major_cap=major_cap, tail_min=tail_min
    )

    # 2) Pipeline configuré sur le dossier d'images d'entraînement
    pipe = create_combined_pipeline(
        image_dir=image_train_dir,
        image_size=image_size,
        under_strategy=under,
        over_strategy=over,
        model_name=model_name,
        use_class_weight=use_class_weight
    )

    print(f">> Entraînement sur X_train avec modèle = {model_name.upper()} …")
    pipe.fit(X_train, y_train)

    print(">> Prédiction sur X_test …")
    # 3) Re-pointer la branche image vers le dossier test (sans refit global)
    #    On modifie uniquement la sous-branche image de la FeatureUnion
    text_step_name, text_pipe = pipe.named_steps["features"].transformer_list[0]
    image_step_name, _ = pipe.named_steps["features"].transformer_list[1]
    pipe.named_steps["features"].transformer_list = [
        (text_step_name, text_pipe),
        (image_step_name, create_image_pipeline(image_dir=image_test_dir, 
                                                image_size=image_size)),
    ]
    y_pred = pipe.predict(X_test)
    return pipe, y_pred


# ---------------------------------------------------------------------
# Évaluation optionnelle : comparaison LR vs SVC par CV sur X_train/Y_train
# ---------------------------------------------------------------------
def compare_models_cv(X_train, y_train,
                      image_train_dir, image_size,
                      major_class, major_cap, tail_min,
                      use_class_weight=True, cv_splits=3, random_state=42):
    """
    Construit deux pipelines identiques (sauf le classifieur), 
    puis évalue F1-macro en CV stratifiée.
    On ne touche pas à X_test. Sert à informer le choix de --model 
    pour l'entraînement final.
    """
    print(">> Comparaison LR vs SVC par validation croisée (F1-macro) …")

    # Stratégies de sampling fixées à partir du y_train complet (cohérent pour les folds)
    under, over = make_sampling_strategies(
        y_train, major_class=major_class, major_cap=major_cap, tail_min=tail_min
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results = []
    for model_name in ["lr", "svc"]:
        pipe = create_combined_pipeline(
            image_dir=image_train_dir,
            image_size=image_size,
            under_strategy=under,
            over_strategy=over,
            model_name=model_name,
            use_class_weight=use_class_weight
        )
        scores = cross_val_score(
            pipe, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=1
        )
        results.append({
            "model": model_name.upper(),
            "cv_mean_f1_macro": scores.mean(),
            "cv_std": scores.std(),
            "cv_scores": scores.tolist(),
        })
        print(f"   - {model_name.upper():>3} | F1-macro (moy ± std) = {scores.mean():.4f} ± {scores.std():.4f} | scores = {scores}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train (text+image) with targeted under/over-sampling, " \
        "optionally compare LR vs SVC on X_train"
    )
    # Config
    p.add_argument("--config", default=None, help="Chemin vers un fichier TOML (optionnel)")
    # CSVs
    p.add_argument("--x_train_csv", default=os.path.join("data", "X_train_update.csv"))
    p.add_argument("--y_train_csv", default=os.path.join("data", "Y_train_CVw08PX.csv"))
    p.add_argument("--x_test_csv",  default=os.path.join("data", "X_test_update.csv"))
    # Images
    p.add_argument("--image_train_dir", 
                   default=os.path.join("data", 
                                        "images", "images", "image_train"))
    p.add_argument("--image_test_dir",  
                   default=os.path.join("data", "images", "images", "image_test"))
    p.add_argument("--image_size", type=int, nargs=2, default=[64, 64])
    # Règles de rééquilibrage
    p.add_argument("--major_class", type=int, default=2583, 
                   help="Classe majoritaire à réduire")
    p.add_argument("--major_cap",   type=int, default=6000, 
                   help="Taille cible pour la classe majoritaire")
    p.add_argument("--tail_min",    type=int, default=1500, 
                   help="Seuil minimum pour oversampler les petites classes")
    p.add_argument("--no_class_weight", action="store_true", 
                   help="Désactive class_weight='balanced'")
    # Modèle et comparaison
    p.add_argument("--model", choices=["lr", "svc"], 
                   default="lr", help="Choix du classifieur final")
    p.add_argument("--compare", action="store_true", 
                   help="Compare LR vs SVC via CV sur X_train (F1-macro)")
    p.add_argument("--cv_splits", type=int, default=3, 
                   help="Nombre de folds pour la CV")
    # Sorties
    p.add_argument("--model_out", default=os.path.join("models", 
                                                       "text_image_classifier.joblib"))
    p.add_argument("--pred_out",  default=os.path.join("models", 
                                                       "y_test_pred.csv"))
    p.add_argument("--compare_out", default=os.path.join("models", 
                                                         "compare_cv_results.csv"))
    return p.parse_args()


def load_config(path: str | None):
    """Charge un fichier TOML s'il est présent. Retourne un dict ."""
    if path is None:
        # essai auto sur 'config.toml' à côté du script
        auto = os.path.join(os.path.dirname(__file__), "config.toml")
        if os.path.isfile(auto):
            path = auto
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return tomli.load(f)
    return {}  # aucun fichier


def override_from_cfg(args, cfg: dict):
    """
    Ecrase les valeurs d'args si présentes dans le TOML.
    Sections attendues :
      [paths]   x_train_csv, y_train_csv, x_test_csv
      [images]  train_dir, test_dir, size
      [sampling] major_class, major_cap, tail_min
      [model]   name ('lr'|'svc'), use_class_weight (bool)
      [cv]      splits (int)
      [outputs] model_out, pred_out, compare_out
    """
    # Paths
    paths = cfg.get("paths", {})
    setattr(args, "x_train_csv", paths.get("x_train_csv", args.x_train_csv))
    setattr(args, "y_train_csv", paths.get("y_train_csv", args.y_train_csv))
    setattr(args, "x_test_csv",  paths.get("x_test_csv",  args.x_test_csv))

    # Images
    images = cfg.get("images", {})
    setattr(args, "image_train_dir", images.get("train_dir", args.image_train_dir))
    setattr(args, "image_test_dir",  images.get("test_dir",  args.image_test_dir))
    if "size" in images and isinstance(images["size"], (list, tuple)) and len(images["size"]) == 2:
        setattr(args, "image_size", images["size"])

    # Sampling
    sampling = cfg.get("sampling", {})
    setattr(args, "major_class", sampling.get("major_class", args.major_class))
    setattr(args, "major_cap",   sampling.get("major_cap",   args.major_cap))
    setattr(args, "tail_min",    sampling.get("tail_min",    args.tail_min))

    # Model
    model_cfg = cfg.get("model", {})
    setattr(args, "model", model_cfg.get("name", args.model))
    if "use_class_weight" in model_cfg:
        # Si le TOML dit False -> activer --no_class_weight
        setattr(args, "no_class_weight", not bool(model_cfg["use_class_weight"]))

    # CV
    cv_cfg = cfg.get("cv", {})
    setattr(args, "cv_splits", cv_cfg.get("splits", args.cv_splits))

    # Outputs
    outs = cfg.get("outputs", {})
    setattr(args, "model_out",    outs.get("model_out",    args.model_out))
    setattr(args, "pred_out",     outs.get("pred_out",     args.pred_out))
    setattr(args, "compare_out",  outs.get("compare_out",  args.compare_out))

    return args


def main():
    args = parse_args()

    # Lecture optionnelle du TOML
    cfg = load_config(args.config)
    if cfg:
        print(f">> Chargement de la configuration TOML …")
        args = override_from_cfg(args, cfg)

    print(">> Chargement des données …")
    X_train = pd.read_csv(args.x_train_csv, index_col=0)
    y_train = pd.read_csv(args.y_train_csv, index_col=0).squeeze()
    X_test  = pd.read_csv(args.x_test_csv,  index_col=0)

    needed = ["designation", "description", "productid"]
    for col in needed:
        if col not in X_train.columns:
            raise ValueError(f"Colonne manquante dans X_train : '{col}'")
        if col not in X_test.columns:
            raise ValueError(f"Colonne manquante dans X_test : '{col}'")

    image_size = tuple(map(int, args.image_size))
    use_class_weight = (not args.no_class_weight)

    # Étape optionnelle : comparaison LR vs SVC (CV sur X_train/Y_train)
    if args.compare:
        df_compare = compare_models_cv(
            X_train[needed], y_train,
            image_train_dir=args.image_train_dir,
            image_size=image_size,
            major_class=args.major_class,
            major_cap=args.major_cap,
            tail_min=args.tail_min,
            use_class_weight=use_class_weight,
            cv_splits=int(args.cv_splits)
        )
        os.makedirs(os.path.dirname(args.compare_out), exist_ok=True)
        df_compare.to_csv(args.compare_out, index=False)
        print(f">> Résultats de comparaison sauvegardés : {args.compare_out}")

    # Entraînement final sur tout X_train / prédiction sur X_test avec le modèle choisi
    pipe, y_pred = train_and_predict_on_test(
        X_train[needed], y_train,
        X_test[needed],
        image_train_dir=args.image_train_dir,
        image_test_dir=args.image_test_dir,
        image_size=image_size,
        major_class=args.major_class,
        major_cap=args.major_cap,
        tail_min=args.tail_min,
        use_class_weight=use_class_weight,
        model_name=args.model,
    )

    # Sauvegardes
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f">> Modèle sauvegardé : {args.model_out}")

    pred_df = pd.DataFrame(y_pred, index=X_test.index, columns=["predicted_label"])
    pred_df.to_csv(args.pred_out)
    print(f">> Prédictions sauvegardées : {args.pred_out}")


if __name__ == "__main__":
    main()

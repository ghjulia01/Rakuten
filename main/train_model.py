# main/train_model.py
import os
import argparse
import joblib
import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from models.text_pipeline import create_text_pipeline
from models.image_pipeline import create_image_pipeline  # DataFrame -> select_pid + loader + flatten + to_sparse

# ---------------------------------------------------------------------
# Stratégies de rééchantillonnage
# ---------------------------------------------------------------------
def make_sampling_strategies(y_train: pd.Series, major_class=2583, major_cap=6000, tail_min=1500):
    """
    Construit deux stratégies : under (uniquement la classe major_class ramenée à major_cap)
    et over (toutes les classes < tail_min remontées à tail_min).
    """
    vc = y_train.value_counts()

    # undersampling ciblé: seule la classe major_class est potentiellement réduite
    under = {
        cls: (min(cnt, major_cap) if cls == major_class else cnt)
        for cls, cnt in vc.items()
    }

    # oversampling des petites classes: toute classe < tail_min -> tail_min
    over = {cls: tail_min for cls, cnt in vc.items() if cnt < tail_min}

    return under, over

# ---------------------------------------------------------------------
# Pipeline combiné (texte + image) avec under + over sampling
# ---------------------------------------------------------------------
def create_combined_pipeline(image_dir, image_size=(64, 64),
                             under_strategy=None, over_strategy=None,
                             use_class_weight=True):
    """
    - Texte : create_text_pipeline() (TextCleaner combine déjà designation+description)
    - Image : create_image_pipeline(image_dir) (productid -> ImageLoader -> flatten -> sparse)
    - Concat : FeatureUnion
    - Scale : StandardScaler(with_mean=False et sparse)
    - Under : RandomUnderSampler (pour 2583 seulement)
    - Over  : RandomOverSampler  (pour toutes classes < seuil)
    - Modèle : LogisticRegression (optionnellement class_weight='balanced')
    """
    text_branch = create_text_pipeline()
    image_branch = create_image_pipeline(image_dir=image_dir, image_size=image_size)

    features = FeatureUnion([
        ("text", text_branch),
        ("image", image_branch),
    ])

    model = LogisticRegression(
        max_iter=1000,
        class_weight=("balanced" if use_class_weight else None),
        solver="lbfgs"
    )

    return ImbPipeline(steps=[
        ("features", features),
        ("scaler", StandardScaler(with_mean=False)),
        ("under",  RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)),
        ("over",   RandomOverSampler(sampling_strategy=over_strategy,  random_state=42)),
        ("model",  model),
    ])

# ---------------------------------------------------------------------
# Entraîner sur train + prédire sur test (pas de y_test)
# ---------------------------------------------------------------------
def train_and_predict_on_test(X_train, y_train, X_test,
                              image_train_dir, image_test_dir,
                              image_size=(64, 64),
                              major_class=2583, major_cap=6000, tail_min=1500,
                              use_class_weight=True):
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
        use_class_weight=use_class_weight
    )

    print(">> Entraînement sur X_train…")
    pipe.fit(X_train, y_train)

    print(">> Prédiction sur X_test…")
    # 3) Re-pointer la branche image vers le dossier test (sans refit)
    pipe.named_steps["features"].transformer_list = [
        ("text",  pipe.named_steps["features"].transformer_list[0][1]),
        ("image", create_image_pipeline(image_dir=image_test_dir, image_size=image_size)),
    ]
    y_pred = pipe.predict(X_test)
    return pipe, y_pred

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train (text+image) with targeted under/over-sampling, then predict on test")
    # CSVs
    p.add_argument("--x_train_csv", default=os.path.join("data", "X_train_update.csv"))
    p.add_argument("--y_train_csv", default=os.path.join("data", "Y_train_CVw08PX.csv"))
    p.add_argument("--x_test_csv",  default=os.path.join("data", "X_test_update.csv"))
    # Images
    p.add_argument("--image_train_dir", default=os.path.join("data", "images", "images", "image_train"))
    p.add_argument("--image_test_dir",  default=os.path.join("data", "images", "images", "image_test"))
    p.add_argument("--image_size", type=int, nargs=2, default=[64, 64])
    # Règles de rééquilibrage
    p.add_argument("--major_class", type=int, default=2583, help="Classe majoritaire à réduire")
    p.add_argument("--major_cap",   type=int, default=6000, help="Taille cible pour la classe majoritaire")
    p.add_argument("--tail_min",    type=int, default=1500, help="Seuil minimum pour oversampler les petites classes")
    p.add_argument("--no_class_weight", action="store_true", help="Désactive class_weight='balanced'")
    # Sorties
    p.add_argument("--model_out", default=os.path.join("models", "text_image_logreg.joblib"))
    p.add_argument("--pred_out",  default=os.path.join("models", "y_test_pred.csv"))
    return p.parse_args()

def main():
    args = parse_args()

    print(">> Chargement des données…")
    X_train = pd.read_csv(args.x_train_csv, index_col=0)
    y_train = pd.read_csv(args.y_train_csv, index_col=0).squeeze()
    X_test  = pd.read_csv(args.x_test_csv,  index_col=0)

    needed = ["designation", "description", "productid"]
    for col in needed:
        if col not in X_train.columns:
            raise ValueError(f"Colonne manquante dans X_train : '{col}'")
        if col not in X_test.columns:
            raise ValueError(f"Colonne manquante dans X_test : '{col}'")

    image_size = tuple(args.image_size)

    pipe, y_pred = train_and_predict_on_test(
        X_train[needed], y_train,
        X_test[needed],
        image_train_dir=args.image_train_dir,
        image_test_dir=args.image_test_dir,
        image_size=image_size,
        major_class=args.major_class,
        major_cap=args.major_cap,
        tail_min=args.tail_min,
        use_class_weight=(not args.no_class_weight),
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
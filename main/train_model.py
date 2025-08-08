# main/train_model.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# --- Nos bloques réutilisables ---
from models.text_pipeline import create_text_pipeline
from models.image_pipeline import create_image_pipeline
from features.image_loader import ImageLoader  

# Image loader a une version indexée pour charger les images
# et les redimensionner en (h, w, 3) pour les images RGB 
# plus rapidement


# ---------------------------------------
# 1) Pipeline combiné (texte + image)
# ---------------------------------------
def create_combined_pipeline(image_dir, image_size=(64, 64)):
    """
    - Branche texte : create_text_pipeline() (TextCleaner combine déjà designation+description)
    - Branche image : productid -> ImageLoader -> flatten
    - Concat via FeatureUnion
    - StandardScaler (with_mean=False) adapté aux matrices clairsemées
    - LogisticRegression équilibrée
    """
    text_branch = create_text_pipeline()
    image_branch = create_image_pipeline(image_dir=image_dir, image_size=image_size)

    features = FeatureUnion([
        ("text", text_branch),
        ("image", image_branch),
    ])

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )

    return Pipeline([
        ("features", features),
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model),
    ])

# ---------------------------------------
# 2) Entraînement + évaluation
# ---------------------------------------
def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       image_train_dir: str, image_test_dir: str,
                       image_size=(64, 64)):
    """
    Entraîne sur X_train / y_train et évalue sur X_test / y_test.
    Les répertoires d'images sont séparés (train / test).
    """
    # Création du pipeline avec le dossier images d'entraînement
    pipe = create_combined_pipeline(image_dir=image_train_dir, image_size=image_size)

    print(">> Entraînement...")
    pipe.fit(X_train, y_train)

    print(">> Évaluation sur test...")
    # Changer le dossier images dans la branche image avant prédiction
    pipe.named_steps["features"].transformer_list = [
        ("text", pipe.named_steps["features"].transformer_list[0][1]),
        ("image", create_image_pipeline(image_dir=image_test_dir, 
                                        image_size=image_size)),
    ]
    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("F1 pondéré :", f1_score(y_test, y_pred, average="weighted"))

    return pipe

# ---------------------------------------
# 3) CLI / Point d’entrée
# ---------------------------------------
def parse_args():
    p.add_argument("--x_train_csv", default=os.path.join("data", "X_train_update.csv"))
    p.add_argument("--y_train_csv", default=os.path.join("data", "Y_train_CVw08PX.csv"))
    p.add_argument("--x_test_csv",  default=os.path.join("data", "X_test_update.csv"))
    p.add_argument("--y_test_csv",  default="")  # optional: leave empty if you don't have labels

    p.add_argument("--image_train_dir", default=os.path.join("data", "images", "images", "image_train"))
    p.add_argument("--image_test_dir",  default=os.path.join("data", "images", "images", "image_test"))
    p.add_argument("--image_size", type=int, nargs=2, default=[64, 64])

    p.add_argument("--model_out", default=os.path.join("models", "text_image_logreg.joblib"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.2)
    return p.parse_args()

def main():
    args = parse_args()

    # Chargement
    print(">> Chargement des données…")
    X_train = pd.read_csv(args.x_train_csv, index_col=0)
    y_train = pd.read_csv(args.y_train_csv, index_col=0).squeeze()

    # Petite vérification pour s'assurer que les colonnes attendues sont présentes
    for col in ["designation", "description", "productid"]:
        if col not in X_train.columns:
            raise ValueError(f"Missing column in X_train: '{col}'")

    # Entraînement
    use_real_test = False
    X_test, y_test = None, None
    if os.path.isfile(args.x_test_csv) and len(args.x_test_csv) > 0:
        X_test = pd.read_csv(args.x_test_csv, index_col=0)
        missing = [c for c in ["designation", "description", "productid"] if c not in X_test.columns]
        if missing:
            raise ValueError(f"Missing columns in X_test: {missing}")

        if args.y_test_csv and os.path.isfile(args.y_test_csv) and os.path.getsize(args.y_test_csv) > 0:
            y_test = pd.read_csv(args.y_test_csv, index_col=0).squeeze()
            use_real_test = True

    image_size = tuple(args.image_size)

    if use_real_test:
        pipe = train_and_eval_on_test(
            X_train[["designation", "description", "productid"]],
            y_train,
            X_test[["designation", "description", "productid"]],
            y_test,
            image_train_dir=args.image_train_dir,
            image_test_dir=args.image_test_dir,
            image_size=image_size
        )
    else:
        print(">> No y_test provided — using a validation split on training data.")
        pipe = train_and_eval_with_split(
            X_train[["designation", "description", "productid"]],
            y_train,
            image_train_dir=args.image_train_dir,
            image_size=image_size,
            test_size=args.val_size,
            seed=args.seed
        )

    # Sauvegarde
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f">> Modèle sauvegardé: {args.model_out}")

if __name__ == "__main__":
    main()
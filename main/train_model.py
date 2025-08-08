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
def train_and_evaluate(X_df: pd.DataFrame, y: pd.Series,
                       image_dir: str, image_size=(64, 64),
                       test_size=0.2, seed=42):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_df, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = create_combined_pipeline(image_dir=image_dir, image_size=image_size)

    print(">> Entraînement...")
    pipe.fit(X_tr, y_tr)

    print(">> Validation...")
    y_pred = pipe.predict(X_va)
    print(classification_report(y_va, y_pred))
    print("F1 pondéré :", f1_score(y_va, y_pred, average="weighted"))

    return pipe

# ---------------------------------------
# 3) CLI / Point d’entrée
# ---------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal (texte + image)")
    p.add_argument("--x_train_csv", default=os.path.join("data", "X_train.csv"))
    p.add_argument("--y_train_csv", default=os.path.join("data", "Y_train.csv"))
    p.add_argument("--image_dir", default=os.path.join("data", "images", "images", "image_train"))
    p.add_argument("--image_size", type=int, nargs=2, default=[64, 64])
    p.add_argument("--model_out", default=os.path.join("models", "text_image_logreg.joblib"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    # Chargement
    print(">> Chargement des données…")
    X = pd.read_csv(args.x_train_csv)
    y = pd.read_csv(args.y_train_csv).squeeze()

    # Sanity check minimal
    for col in ["designation", "description", "productid"]:
        if col not in X.columns:
            raise ValueError(f"Colonne manquante dans X_train : '{col}'")

    # Entraînement
    pipe = train_and_evaluate(
        X_df=X[["designation", "description", "productid"]],
        y=y,
        image_dir=args.image_dir,
        image_size=tuple(args.image_size),
        seed=args.seed
    )

    # Sauvegarde
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f">> Modèle sauvegardé : {args.model_out}")

if __name__ == "__main__":
    main()
# main/train_model.py
import os
import argparse
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from models.text_pipeline import create_text_pipeline
from models.image_pipeline import create_image_pipeline  # DataFrame -> (select_pid+loader+flatten+to_sparse)

# --------------------------
# Pipeline combiné
# --------------------------
def create_combined_pipeline(image_dir, image_size=(64, 64)):
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

# --------------------------
# Entraîner sur train, prédire sur test (sans y_test)
# --------------------------
def train_and_predict_on_test(X_train, y_train, X_test,
                              image_train_dir, image_test_dir,
                              image_size=(64, 64)):
    # 1) pipeline avec dossier images d'entraînement
    pipe = create_combined_pipeline(image_dir=image_train_dir, image_size=image_size)

    print(">> Entraînement sur X_train…")
    pipe.fit(X_train, y_train)

    print(">> Prédiction sur X_test…")
    # 2) on remplace la branche image pour pointer vers le dossier test (pas de refit)
    pipe.named_steps["features"].transformer_list = [
        ("text", pipe.named_steps["features"].transformer_list[0][1]),
        ("image", create_image_pipeline(image_dir=image_test_dir, image_size=image_size)),
    ]
    y_pred = pipe.predict(X_test)
    return pipe, y_pred

# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train (text+image) and predict on test set without y_test")
    # Tes fichiers
    p.add_argument("--x_train_csv", default=os.path.join("data", "X_train_update.csv"))
    p.add_argument("--y_train_csv", default=os.path.join("data", "Y_train_CVw08PX.csv"))
    p.add_argument("--x_test_csv",  default=os.path.join("data", "X_test_update.csv"))
    # Dossiers images
    p.add_argument("--image_train_dir", default=os.path.join("data", "images", "images", "image_train"))
    p.add_argument("--image_test_dir",  default=os.path.join("data", "images", "images", "image_test"))
    # Taille des images
    p.add_argument("--image_size", type=int, nargs=2, default=[64, 64])
    # Sorties
    p.add_argument("--model_out", default=os.path.join("models", "text_image_logreg.joblib"))
    p.add_argument("--pred_out", default=os.path.join("models", "y_test_pred.csv"))
    return p.parse_args()

def main():
    args = parse_args()

    print(">> Chargement des données…")
    X_train = pd.read_csv(args.x_train_csv, index_col=0)
    y_train = pd.read_csv(args.y_train_csv, index_col=0).squeeze()
    X_test  = pd.read_csv(args.x_test_csv, index_col=0)

    needed = ["designation", "description", "productid"]
    for col in needed:
        if col not in X_train.columns:
            raise ValueError(f"Colonne manquante dans X_train : '{col}'")
        if col not in X_test.columns:
            raise ValueError(f"Colonne manquante dans X_test : '{col}'")

    image_size = tuple(args.image_size)

    # Train + Predict
    pipe, y_pred = train_and_predict_on_test(
        X_train[needed], y_train,
        X_test[needed],
        image_train_dir=args.image_train_dir,
        image_test_dir=args.image_test_dir,
        image_size=image_size
    )

    # Sauvegardes
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f">> Modèle sauvegardé : {args.model_out}")

    # CSV de prédictions (index aligné à X_test)
    pred_df = pd.DataFrame(y_pred, index=X_test.index, columns=["predicted_label"])
    pred_df.to_csv(args.pred_out)
    print(f">> Prédictions sauvegardées : {args.pred_out}")

if __name__ == "__main__":
    main()
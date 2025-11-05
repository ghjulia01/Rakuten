"""
Script de production complète pour le projet Rakuten.

Ce script orchestre toutes les étapes:
1. Chargement des données
2. Extraction des features (texte + image)
3. Entraînement du modèle
4. Évaluation et métriques
5. Sauvegarde des artefacts (modèle + feature_mapping.json)

Usage:
    python tools/run_full_production.py [--test-mode]
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


def load_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données train et test."""
    print("\n[1/6] Chargement des données...")
    
    data_path = Path(data_dir)
    train_df = pd.read_csv(data_path / "X_train.csv")
    test_df = pd.read_csv(data_path / "X_test.csv")
    
    print(f"  ✓ Train: {len(train_df):,} échantillons")
    print(f"  ✓ Test:  {len(test_df):,} échantillons")
    
    return train_df, test_df


def extract_text_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 50000
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]], object]:
    """
    Extrait les features textuelles avec mapping détaillé.
    
    Returns:
        X_train_text, X_test_text, feature_mapping, vectorizer
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("\n[2/6] Extraction des features textuelles...")
    
    # Préparer le texte
    train_df['text_combined'] = (
        train_df['designation'].fillna('') + ' ' + 
        train_df['description'].fillna('')
    ).str.lower()
    
    test_df['text_combined'] = (
        test_df['designation'].fillna('') + ' ' + 
        test_df['description'].fillna('')
    ).str.lower()
    
    # TF-IDF
    print("  → TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    tfidf_train = vectorizer.fit_transform(train_df['text_combined'])
    tfidf_test = vectorizer.transform(test_df['text_combined'])
    
    # Feature mapping - initialisation
    feature_mapping = {}
    current_idx = 0
    
    # Bloc 1: TF-IDF
    tfidf_size = tfidf_train.shape[1]
    feature_mapping['text_tfidf'] = (current_idx, current_idx + tfidf_size)
    current_idx += tfidf_size
    
    # Features binaires et stats
    print("  → Features additionnelles (has_desc, stats)...")
    
    # Bloc 2: Has description (1 feature)
    has_desc_train = (~train_df['description'].isna()).astype(int).values.reshape(-1, 1)
    has_desc_test = (~test_df['description'].isna()).astype(int).values.reshape(-1, 1)
    feature_mapping['text_has_desc'] = (current_idx, current_idx + 1)
    current_idx += 1
    
    # Bloc 3: Text stats (5 features)
    def compute_text_stats(df):
        stats = np.column_stack([
            df['designation'].fillna('').str.len().values,
            df['description'].fillna('').str.len().values,
            df['designation'].fillna('').str.split().str.len().fillna(0).values,
            df['description'].fillna('').str.split().str.len().fillna(0).values,
            df['productid'].values
        ])
        return stats
    
    text_stats_train = compute_text_stats(train_df)
    text_stats_test = compute_text_stats(test_df)
    feature_mapping['text_stats'] = (current_idx, current_idx + 5)
    current_idx += 5
    
    # Combiner toutes les features textuelles
    X_train_text = np.hstack([
        tfidf_train.toarray(),
        has_desc_train,
        text_stats_train
    ])
    
    X_test_text = np.hstack([
        tfidf_test.toarray(),
        has_desc_test,
        text_stats_test
    ])
    
    print(f"  ✓ Features textuelles: {X_train_text.shape[1]:,}")
    print(f"    - TF-IDF: {tfidf_size:,}")
    print(f"    - Has description: 1")
    print(f"    - Text stats: 5")
    
    return X_train_text, X_test_text, feature_mapping, vectorizer


def extract_image_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_mapping: Dict[str, Tuple[int, int]],
    image_dir: str = "data/images"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Extrait les features images (CNN + couleurs) avec mapping.
    
    Returns:
        X_train_img, X_test_img, updated_feature_mapping
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
    import cv2
    
    print("\n[3/6] Extraction des features images...")
    
    # Charger ResNet50
    print("  → Chargement de ResNet50...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    def extract_cnn_features(image_ids, img_dir):
        """Extrait features CNN pour une liste d'IDs."""
        features = []
        img_path = Path(img_dir)
        
        for img_id in tqdm(image_ids, desc="  → CNN features"):
            img_file = img_path / f"image_{img_id}_product_{img_id}.jpg"
            
            if img_file.exists():
                try:
                    img = image.load_img(img_file, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    feat = model.predict(img_array, verbose=0)
                    features.append(feat.flatten())
                except Exception as e:
                    features.append(np.zeros(2048))
            else:
                features.append(np.zeros(2048))
        
        return np.array(features)
    
    def extract_color_features(image_ids, img_dir):
        """Extrait features couleur (histogrammes RGB)."""
        features = []
        img_path = Path(img_dir)
        
        for img_id in tqdm(image_ids, desc="  → Color features"):
            img_file = img_path / f"image_{img_id}_product_{img_id}.jpg"
            
            if img_file.exists():
                try:
                    img = cv2.imread(str(img_file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Histogrammes pour chaque canal (16 bins)
                    hist_features = []
                    for i in range(3):
                        hist = cv2.calcHist([img], [i], None, [16], [0, 256])
                        hist = hist.flatten() / hist.sum()
                        hist_features.extend(hist)
                    
                    features.append(hist_features)
                except Exception as e:
                    features.append(np.zeros(48))
            else:
                features.append(np.zeros(48))
        
        return np.array(features)
    
    # Extraction pour train
    cnn_train = extract_cnn_features(train_df['imageid'].values, image_dir)
    color_train = extract_color_features(train_df['imageid'].values, image_dir)
    
    # Extraction pour test
    cnn_test = extract_cnn_features(test_df['imageid'].values, image_dir)
    color_test = extract_color_features(test_df['imageid'].values, image_dir)
    
    # Mise à jour du mapping
    current_idx = max(end for _, (_, end) in feature_mapping.items())
    
    # Bloc 4: CNN features (2048)
    feature_mapping['image_cnn'] = (current_idx, current_idx + 2048)
    current_idx += 2048
    
    # Bloc 5: Color features (48)
    feature_mapping['image_color'] = (current_idx, current_idx + 48)
    current_idx += 48
    
    # Combiner
    X_train_img = np.hstack([cnn_train, color_train])
    X_test_img = np.hstack([cnn_test, color_test])
    
    print(f"  ✓ Features images: {X_train_img.shape[1]:,}")
    print(f"    - CNN (ResNet50): 2048")
    print(f"    - Couleurs (RGB hist): 48")
    
    return X_train_img, X_test_img, feature_mapping


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_mode: bool = False
) -> RandomForestClassifier:
    """Entraîne le modèle Random Forest."""
    print("\n[4/6] Entraînement du modèle...")
    
    if test_mode:
        print("  MODE TEST: paramètres réduits")
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    else:
        print("  → MODE PRODUCTION: paramètres complets")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    print(f"  ✓ Entraînement terminé en {elapsed/60:.1f} minutes")
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "results"
) -> Dict:
    """Évalue le modèle et sauvegarde les métriques."""
    print("\n[5/6] Évaluation du modèle...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  ✓ Accuracy:    {accuracy:.4f}")
    print(f"  ✓ F1-score:    {f1_weighted:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Sauvegarder les résultats
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Métriques JSON
    metrics = {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }
    
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Prédictions
    np.save(output_path / "y_pred.npy", y_pred)
    np.save(output_path / "y_proba.npy", y_proba)
    np.save(output_path / "y_test.npy", y_test)
    
    print(f"  ✓ Métriques sauvegardées dans {output_dir}/")
    
    return metrics


def save_artifacts(
    model: RandomForestClassifier,
    vectorizer: object,
    feature_mapping: Dict[str, Tuple[int, int]],
    X_test: np.ndarray,
    artifacts_dir: str = "artifacts"
):
    """Sauvegarde tous les artefacts nécessaires."""
    print("\n[6/6] Sauvegarde des artefacts...")
    
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    # Modèle
    with open(artifacts_path / "model_final.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Modèle: {artifacts_dir}/model_final.pkl")
    
    # Vectorizer
    with open(artifacts_path / "tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  ✓ Vectorizer: {artifacts_dir}/tfidf_vectorizer.pkl")
    
    # Feature mapping (CRITIQUE pour SHAP)
    with open(artifacts_path / "feature_mapping.json", 'w') as f:
        json.dump(feature_mapping, f, indent=2)
    print(f"  ✓ Feature mapping: {artifacts_dir}/feature_mapping.json")
    
    # Features de test (pour SHAP)
    np.savez_compressed(
        artifacts_path / "features_test.npz",
        features=X_test
    )
    print(f"  ✓ Features test: {artifacts_dir}/features_test.npz")
    
    # Résumé du mapping
    print("\n   Résumé du feature mapping:")
    total_features = 0
    for block_name, (start, end) in feature_mapping.items():
        n_features = end - start
        total_features += n_features
        print(f"    - {block_name:20s}: [{start:6d}, {end:6d}] → {n_features:6,} features")
    print(f"    {'TOTAL':20s}:                  → {total_features:6,} features")


def main():
    parser = argparse.ArgumentParser(description="Production complète Rakuten")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Mode test avec paramètres réduits")
    parser.add_argument("--data-dir", default="data", 
                       help="Répertoire des données")
    parser.add_argument("--image-dir", default="data/images", 
                       help="Répertoire des images")
    parser.add_argument("--output-dir", default="results", 
                       help="Répertoire de sortie")
    parser.add_argument("--artifacts-dir", default="artifacts", 
                       help="Répertoire des artefacts")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" PRODUCTION COMPLÈTE - RAKUTEN CLASSIFICATION")
    print("=" * 80)
    
    if args.test_mode:
        print("\n  MODE TEST ACTIVÉ - Paramètres réduits\n")
    
    # 1. Chargement
    train_df, test_df = load_data(args.data_dir)
    y_train = train_df['prdtypecode'].values
    y_test = test_df['prdtypecode'].values
    
    # 2. Features textuelles
    X_train_text, X_test_text, feature_mapping, vectorizer = extract_text_features(
        train_df, test_df
    )
    
    # 3. Features images
    X_train_img, X_test_img, feature_mapping = extract_image_features(
        train_df, test_df, feature_mapping, args.image_dir
    )
    
    # Combiner toutes les features
    X_train = np.hstack([X_train_text, X_train_img])
    X_test = np.hstack([X_test_text, X_test_img])
    
    print(f"\n✓ Features finales: {X_train.shape[1]:,} dimensions")
    
    # 4. Entraînement
    model = train_model(X_train, y_train, test_mode=args.test_mode)
    
    # 5. Évaluation
    metrics = evaluate_model(model, X_test, y_test, args.output_dir)
    
    # 6. Sauvegarde
    save_artifacts(
        model, vectorizer, feature_mapping, X_test, args.artifacts_dir
    )
    
    print("\n" + "=" * 80)
    print(" PRODUCTION TERMINÉE AVEC SUCCÈS!")
    print("=" * 80)
    print(f"\n Fichiers générés:")
    print(f"  - Modèle:          {args.artifacts_dir}/model_final.pkl")
    print(f"  - Feature mapping: {args.artifacts_dir}/feature_mapping.json")
    print(f"  - Métriques:       {args.output_dir}/metrics.json")
    print(f"  - Prédictions:     {args.output_dir}/y_pred.npy")
    print(f"\nProchaine étape:")
    print(f"  python tools/compute_shap.py \\")
    print(f"      --model {args.artifacts_dir}/model_final.pkl \\")
    print(f"      --features {args.artifacts_dir}/features_test.npz \\")
    print(f"      --output results/shap/")


if __name__ == "__main__":
    main()
# tools/exemple_gradcam_min.py
# Exemple d'utilisation de Grad-CAM avec une tête entraînée
# python tools/example_gradcam_min.py
# Nécessite d'avoir entraîné une tête et sauvegardé le modèle dans 'artifacts/head_ft.pt'

import pandas as pd
from models.cnn_features import CNNFeaturizer
from gradcam_resnet_head import ResNetGradCAMWithHead

fe = CNNFeaturizer(
    image_dir="C:/.../data/images/images",
    arch="resnet50",
    device="auto",
)
fe.load_head("artifacts/head_ft.pt")

df = pd.read_csv("notebooks/df.csv")
img_path = fe._path_from_row(df.iloc[0])

gradcam = ResNetGradCAMWithHead(featurizer=fe)
heatmap, overlay, idx, label, logits = gradcam.generate(img_path, class_idx=None, overlay_alpha=0.45, show=True)

print("Classe prédite:", label, "(idx:", idx, ")")
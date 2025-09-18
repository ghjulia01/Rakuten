# tools/run_gradcam.py
# Visualisation Grad-CAM pour une image donnée (avec tête FT)
# python tools/run_gradcam.py --images_dir data/images/images --head_path artifacts/head_ft.pt --csv notebooks/df.csv --row 0 --arch resnet50 --alpha 0.45
import argparse
import pandas as pd
from models.cnn_features import CNNFeaturizer
from gradcam_resnet_head import ResNetGradCAMWithHead

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Dossier des images Rakuten (…/data/images/images)")
    ap.add_argument("--head_path", required=True, help="Fichier tête FT sauvegardée (ex: artifacts/head_ft.pt)")
    ap.add_argument("--csv", required=True, help="CSV avec colonnes imageid,productid (val/test)")
    ap.add_argument("--row", type=int, default=0, help="Index de ligne dans le CSV à visualiser")
    ap.add_argument("--arch", default="resnet50", help="Backbone torchvision (resnet50 par défaut)")
    ap.add_argument("--alpha", type=float, default=0.45, help="Opacité de l’overlay Grad-CAM")
    args = ap.parse_args()

    # 1) Featurizer + tête FT
    fe = CNNFeaturizer(image_dir=args.images_dir, arch=args.arch, device="auto")
    fe.load_head(args.head_path)

    # 2) Sélection d’une image à partir du CSV
    df = pd.read_csv(args.csv)
    row = df.iloc[args.row]
    img_path = fe._path_from_row(row)  # construit image_{imageid}_product_{productid}.jpg

    # 3) Grad-CAM
    gradcam = ResNetGradCAMWithHead(featurizer=fe)
    heatmap, overlay, idx, label, logits = gradcam.generate(
        img_path, class_idx=None, overlay_alpha=args.alpha, show=True
    )
    print(f"Classe prédite : {label} (idx: {idx})")
    # Option: sauvegarder la heatmap/overlay
    heatmap.save("artifacts/gradcam_heatmap.png"); overlay.save("artifacts/gradcam_overlay.png")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from pathlib import Path
from models.cnn_features import ARCH_REGISTRY


def parse_args():
    """Parser les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description="Grad-CAM pour visualiser les zones importantes d'une image.")
    parser.add_argument("img_path", type=str, help="Chemin vers l'image à analyser.")
    parser.add_argument("model_path", type=str, help="Chemin vers le modèle sauvegardé.")
    parser.add_argument("--output_dir", type=str, default="results/gradcam", help="Répertoire de sortie pour enregistrer les images.")
    parser.add_argument("--display", action="store_true", help="Afficher les images à l'écran.")
    parser.add_argument("--save", action="store_true", help="Enregistrer les images dans un répertoire.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Facteur de transparence de la heatmap (0 à 1).")
    return parser.parse_args()

def list_images_in_directory(directory):
    """Lister les fichiers d'images dans un répertoire."""
    valid_extensions = ['.jpg']
    image_files = []

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_files.append(file_path)

    return image_files

def get_filename_from_path(img_path):
    """Extraire le nom de fichier sans le chemin et l'extension."""
    base = os.path.basename(img_path)
    filename_without_ext = os.path.splitext(base)[0]
    return filename_without_ext

def load_model_for_gradcam(model_path):
    """Recharger le modèle pour Grad-CAM."""
    state = torch.load(model_path)

    # Reconstruire le modèle
    arch_key = state['arch'].lower()
    if arch_key not in ARCH_REGISTRY:
        raise ValueError(f"Architecture inconnue: {state['arch']}")

    ctor, weights_enum, _ = ARCH_REGISTRY[arch_key]
    model = ctor(weights=None)  # Ne pas charger les poids pré-entraînés
    model.fc = nn.Identity()   # Remplacer la couche fc par Identity

    # Charger les poids entraînés
    model.load_state_dict(state['state_dict'])
    model.eval()

    return model, state['use_imagenet_norm']

def preprocess_image(img_path, use_imagenet_norm=True):
    """Prétraiter une image pour Grad-CAM."""
    img = Image.open(img_path).convert('RGB')

    if use_imagenet_norm:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Ajouter une dimension batch

    return img_tensor, np.array(img)

def generate_gradcam(model, img_tensor, target_layer="layer4"):
    """Générer une heatmap Grad-CAM."""
    model.eval()
    img_tensor.requires_grad = True

    # Enregistrer les activations et gradients de la couche cible
    activations = None
    gradients = None

    def full_backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    # Obtenir la couche cible
    for name, module in model.named_modules():
        if name == target_layer:
            target = module
            break

    # Enregistrer les hooks
    forward_handle = target.register_forward_hook(forward_hook)
    full_backward_handle = target.register_full_backward_hook(full_backward_hook)

    # Passer l'image à travers le modèle pour obtenir les activations
    output = model(img_tensor)

    # Choisir la classe avec la plus haute prédiction
    pred = output.argmax(dim=1)
    pred_class = pred.item()

    # Calculer les gradients
    model.zero_grad()
    output[:, pred_class].backward()

    # Retirer les hooks
    forward_handle.remove()
    full_backward_handle.remove()

    # Moyenne des gradients sur les dimensions spatiales
    pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)

    # Pondérer les activations par les gradients
    weighted_activations = activations * pooled_gradients
    heatmap = torch.mean(weighted_activations, dim=1).squeeze()

    # Normaliser la heatmap
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    return heatmap


def display_save_gradcam(img, heatmap, args, img_filename, alpha=0.7):
    """Afficher et sauvegarder la heatmap Grad-CAM superposée à l'image."""
    # Redimensionner la heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_img = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

    # Superposer la heatmap sur l'image
    superimposed_img = cv2.addWeighted(img, 1, heatmap_img, alpha, 0)
    superimposed_img = np.clip(superimposed_img, 0, 255)
    superimposed_img = superimposed_img.astype(np.uint8)

    # Créer la figure
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Image originale")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Heatmap Grad-CAM")
    plt.imshow(heatmap_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Superposition")
    plt.imshow(superimposed_img)
    plt.axis('off')

    # Sauvegarder la figure avant de l'afficher
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{img_filename}_combined.jpg")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Image combinée enregistrée : {output_path}")

    # Afficher la figure
    if args.display:
        plt.show()

    plt.close()


def main():
    args = parse_args()

    # 1. Recharger le modèle
    model, use_imagenet_norm = load_model_for_gradcam(args.model_path)

    # Vérifier si img_path est un répertoire ou un fichier
    if os.path.isdir(args.img_path):
        # Si c'est un répertoire, traiter toutes les images
        image_files = list_images_in_directory(args.img_path)
        for img_file in image_files:
            print(f"Traitement de l'image : {img_file}")
            img_filename = get_filename_from_path(img_file)

            # 2. Charger et prétraiter l'image
            img_tensor, img = preprocess_image(img_file, use_imagenet_norm)

            # 3. Générer la heatmap Grad-CAM
            heatmap = generate_gradcam(model, img_tensor)

            # 4. Afficher et sauvegarder les résultats
            display_save_gradcam(img, heatmap, args, img_filename=img_filename, alpha=args.alpha)
    else:
        # Si c'est un fichier, traiter uniquement cette image
        img_filename = get_filename_from_path(args.img_path)

        # 2. Charger et prétraiter l'image
        img_tensor, img = preprocess_image(args.img_path, use_imagenet_norm)

        # 3. Générer la heatmap Grad-CAM
        heatmap = generate_gradcam(model, img_tensor)

        # 4. Afficher et sauvegarder les résultats
        display_save_gradcam(img, heatmap, args, img_filename=img_filename, alpha=args.alpha)
def _build_path(image_dir, row):
    # Construit image_{imageid}_product_{productid}.jpg
    return os.path.join(
        image_dir,
        f"image_{int(row['imageid'])}_product_{int(row['productid'])}.jpg"
    )

def main_grad_cam(*, backbone, image_dir, samples, labels, outdir, device="cpu", mode="resnet"):
    """
    Appel programmatique depuis train_model.py
    - backbone : modèle torch déjà en mémoire (ResNet ou ViT)
    - image_dir : dossier des images train
    - samples : DataFrame avec colonnes ['imageid','productid'] (panel)
    - labels  : array/Series des vraies classes pour annoter (optionnel)
    - outdir  : dossier de sortie
    - device  : 'cpu' / 'cuda' / 'dml'
    - mode    : 'resnet' (Grad-CAM) ou 'vit' (attention rollout via utils)
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Si ViT: délègue à un runner dédié (si dispo)
    if mode == "vit":
        try:
            from tools.gradcam_utils import run_vit_attention_rollout_batch
            return run_vit_attention_rollout_batch(
                backbone=backbone,
                image_dir=image_dir,
                samples=samples,
                labels=labels,
                outdir=outdir,
                device=device,
                head_fusion="mean",
                discard_ratio=0.0,
            )
        except Exception as e:
            print(f"[GradCAM] ViT rollout indisponible ({e}) → skip")
            return None

    # Sinon: chemin ResNet (Grad-CAM classique) en utilisant les helpers locaux
    try:
        backbone.eval()
        if device and hasattr(backbone, "to"):
            backbone.to(device)

        for i, row in samples.iterrows():
            img_path = _build_path(image_dir, row)
            if not os.path.exists(img_path):
                print(f"[GradCAM] Image manquante: {img_path} — skip")
                continue

            # Prétraitement & heatmap
            img_tensor, img_np = preprocess_image(img_path, use_imagenet_norm=True)
            if device == "cuda":
                img_tensor = img_tensor.cuda()
            heatmap = generate_gradcam(backbone, img_tensor, target_layer="layer4")

            # Nom de fichier propre
            fn = f"img{int(row['imageid'])}_prod{int(row['productid'])}"
            args = argparse.Namespace(save=True, display=False, output_dir=outdir, alpha=0.7)
            display_save_gradcam(img_np, heatmap, args, img_filename=fn, alpha=args.alpha)

        print(f"[GradCAM] Résultats sous: {outdir}")
        return outdir
    except Exception as e:
        print(f"[GradCAM] ResNet indisponible ({e}) → skip")
        return None
    
if __name__ == "__main__":
    main()
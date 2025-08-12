import os
import re
import numpy as np
from PIL import Image  # Pour la manipulation d'images
from sklearn.base import BaseEstimator, TransformerMixin


# La classe ImageLoader est utilisée pour charger et 
# prétraiter les images dans le pipeline.
# Le pattern utilisé permet de repérer les noms de 
# fichiers comme image_216810030_product_6342052.jpg
# (\d+) : capture un ou plusieurs chiffres
# .group(2) retournera l'id du produit

class ImageLoader(BaseEstimator, TransformerMixin):
    def __init__(self, image_dir, image_size=(64, 64)):
        self.image_dir = image_dir
        # Définit le répertoire où les images sont stockées
        self.image_size = image_size
        # Définit la taille des images à redimensionner
        self.pattern = re.compile(r"image_(\d+)_product_(\d+)\.jpg")
        # Compile le pattern pour matcher les noms de fichiers d'images
        self.index = {}
        # Dictionnaire pour indexer les images par product_id

# Cette méthode est obligatoire pour les objets scikit-learn.
# Elle ne fait rien ici (pas d’apprentissage à faire), 
# mais elle permet l’intégration dans un pipeline.
# X est la liste des product_id.

    def fit(self, X, y=None):
        """Construit un index productid -> chemin image pour gagner 
        du temps lors de la transformation."""
        for filename in os.listdir(self.image_dir):
            match = self.pattern.match(filename)
            if match:
                product_id = match.group(2)
                self.index[product_id] = os.path.join(self.image_dir, filename)
        return self


# Pour chaque product_id, il charge l'image correspondante
# Si l’image existe, elle est transformée en tableau numpy
# Sinon, un tableau vide (image noire) est retourné
# Le résultat est un tableau de forme 
# # (n_samples, height, width, channels)

    def transform(self, X):
        """
        Charge et transforme les images pour chaque product_id.
        Retourne un tableau numpy de forme (n_samples, height, width, channels),
        avec les pixels normalisés dans [0, 1].
        Si l'image est manquante, retourne une image noire.
        """
        H, W = self.image_size
        imgs = [
             self._process_image(self.index[str(pid)])
             if str(pid) in self.index
             else np.zeros((H, W, 3), dtype=np.float32)
             for pid in X
        ]
        return np.stack(imgs, axis=0).astype(np.float32)
        


# Cette méthode traite l'image : redimensionnement et normalisation
# Ouvre l’image et la convertit en couleur et la redimensionne
# la transforme en tableau numpy ((64, 64, 3)) avec des valeurs entre 0 et 1

    def _process_image(self, image_path):
        H, W = self.image_size
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # Pillow attend (width, height)
            img = img.resize((W, H))
            arr = np.asarray(img, dtype=np.float32) / 255.0
        # sécurité forme
        if arr.shape != (H, W, 3):
            raise ValueError(f"Got {arr.shape}, expected {(H, W, 3)}")
        return arr
        #  Normalise les pixels dans [0, 1]
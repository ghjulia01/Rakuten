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
        self.image_size = image_size
        self.pattern = re.compile(r"image_(\d+)_product_(\d+)\.jpg")

# Cette méthode est obligatoire pour les objets scikit-learn.
# Elle ne fait rien ici (pas d’apprentissage à faire), 
# mais elle permet l’intégration dans un pipeline.
# X est la liste des product_id.

    def fit(self, X, y=None):
        return self


# Pour chaque product_id, il charge l'image correspondante
# Si l’image existe, elle est transformée en tableau numpy
# Sinon, un tableau vide (image noire) est retourné
# Le résultat est un tableau de forme 
# # (n_samples, height, width, channels)

    def transform(self, X):
        images = []
        for product_id in X:
            img_array = self._load_image_for_product(product_id)
            images.append(img_array)
        return np.array(images)

# Cette méthode charge l'image pour un product_id donné
# Si l'image n'est pas trouvée, elle retourne une image vide (noire)
#  (np.zeros) si aucune image ne correspond à product_id.
    def _load_image_for_product(self, product_id):
        for filename in os.listdir(self.image_dir):
            match = self.pattern.match(filename)
            if match and match.group(2) == str(product_id):
                path = os.path.join(self.image_dir, filename)
                return self._process_image(path)
        return np.zeros((*self.image_size, 3), dtype=np.uint8)  # Si image absente

# Cette méthode traite l'image : redimensionnement et normalisation
# Ouvre l’image et la convertit en couleur et la redimensionne
# la transforme en tableau numpy ((64, 64, 3)) avec des valeurs entre 0 et 1

    def _process_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Assure que l'image est en RGB
            img = img.resize(self.image_size) # Redimensionne à (64, 64)
            return np.array(img) / 255.0  #  Normalise les pixels dans [0, 1]
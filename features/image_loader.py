# features/image_loader.py
# ------------------------------------------------------------
# Rôle : charger et prétraiter les images pour un pipeline sklearn.
# Format de fichier attendu : image_{imageid}_product_{productid}.jpg
# Exemple : image_216810030_product_6342052.jpg
# ------------------------------------------------------------
import os
import re
import time
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Union, Iterable
import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from functools import lru_cache

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageLoader(BaseEstimator, TransformerMixin):
    """
    Chargeur d'images optimisé pour le pipeline Rakuten.
    
    Caractéristiques:
    - Cache mémoire et disque pour optimisation
    - Traitement par lots pour économiser la RAM
    - Gestion robuste des erreurs
    - Prétraitement configurable
    - Support des images manquantes
    
    Arguments:
        image_dir (str): Répertoire contenant les images
        image_size (tuple): Dimensions cibles (hauteur, largeur)
        batch_size (int): Taille des lots pour le traitement
        cache_dir (str): Répertoire pour le cache disque
        use_cache (bool): Activer/désactiver le cache
    """
    def __init__(self,
                 image_dir: str,
                 image_size: tuple[int, int] = (128, 128),
                 batch_size: int = 64,
                 cache_dir: str = ".cache",
                 use_cache: bool = True):
        self.image_dir = image_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.pattern = re.compile(r"^image_(\d+)_product_(\d+)\.jpg$")
        self.index: Dict[str, str] = {}
        
        if use_cache:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info(f"Cache activé dans {cache_dir}")

    def _normalize_path(self, path: str) -> str:
        """Normalise un chemin Windows avec préfixe UNC si nécessaire."""
        path = os.path.abspath(path)
        if os.name == "nt" and not path.startswith("\\\\?\\"):
            return "\\\\?\\" + path
        return path

    @lru_cache(maxsize=1000)
    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """
        Charge et prétraite une image avec double cache (mémoire + disque).
        
        Args:
            path (str): Chemin de l'image
            
        Returns:
            np.ndarray: Image prétraitée [0-1] de forme (H, W, 3)
        """
        try:
            # Vérifier le cache disque si activé
            if self.use_cache:
                cache_path = self._get_cache_path(path)
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)

            # Charger et prétraiter l'image
            H, W = self.image_size
            with Image.open(path) as img:
                # Conversion en RGB pour uniformisation
                img = img.convert("RGB")
                # Redimensionnement avec LANCZOS pour meilleure qualité
                img = img.resize((W, H), Image.Resampling.LANCZOS)
                # Normalisation [0-1] et conversion float32
                arr = np.asarray(img, dtype=np.float32) / 255.0

            # Sauvegarder dans le cache disque
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(arr, f)

            return arr

        except Exception as e:
            logger.error(f"Erreur pour {path}: {str(e)}")
            return np.zeros((*self.image_size, 3), dtype=np.float32)

    def _get_cache_path(self, image_path: str) -> Path:
        """Génère un chemin unique dans le cache pour une image."""
        return self.cache_dir / f"{hash(image_path)}.pkl"

    def fit(self, X=None, y=None) -> "ImageLoader":
        """
        Indexe les images du répertoire pour accès rapide.
        
        Returns:
            self: Pour chaînage
        """
        logger.info(f"Indexation des images dans {self.image_dir}")
        start_time = time.time()
        self.index.clear()
        
        root = Path(self.image_dir)
        n_files = 0
        
        for p in root.glob("*.jpg"):
            n_files += 1
            if m := self.pattern.match(p.name):
                product_id = m.group(2)
                self.index[product_id] = self._normalize_path(str(p))
                
        duration = time.time() - start_time
        logger.info(f"Indexation: {len(self.index)}/{n_files} images en {duration:.1f}s")
        return self

    def transform(self, X) -> np.ndarray:
        """
        Charge et transforme les images par lots.
        
        Args:
            X: DataFrame avec 'productid' ou liste de product_ids
            
        Returns:
            np.ndarray: Batch d'images (N, H, W, 3) normalisées [0-1]
        """
        start_time = time.time()
        logger.info("Début du chargement des images")

        # Extraire les product IDs
        if hasattr(X, "productid"):
            product_ids = X.productid.astype(str).tolist()
        else:
            product_ids = [str(x) for x in X]

        # Initialiser le tableau de sortie
        H, W = self.image_size
        n_samples = len(product_ids)
    
    # Traitement par lots pour économiser la mémoire
        batches = []
        for i in range(0, n_samples, self.batch_size):
            batch_ids = product_ids[i:i + self.batch_size]
        
        # Résoudre les chemins d'images
            batch_paths = []
            for pid in batch_ids:
                path = self.index.get(pid)
                if path is None:
                    logger.warning(f"Image manquante pour product_id {pid}")
                batch_paths.append(path if path else "")
        
        # Charger et prétraiter le lot
            batch_images = [
                self._load_and_preprocess(p) if p else 
                np.zeros((H, W, 3), dtype=np.float32)
                for p in batch_paths
            ]
        
        # Empiler le lot
            batches.append(np.stack(batch_images))
        
            if (i + self.batch_size) % 1000 == 0:
                logger.info(f"Traité {i + self.batch_size}/{n_samples} images")

    # Concaténer tous les lots
        result = np.concatenate(batches, axis=0)
    
        duration = time.time() - start_time
        logger.info(f"Chargement terminé en {duration:.1f}s")
        return result
        
# ------------------------------
# Utilitaires
# ------------------------------
def _win_norm(p: str) -> str:
    """Normaliser un chemin pour Windows (préfixe UNC si besoin)."""
    p = os.path.abspath(p)
    if os.name == "nt":
        return p if p.startswith("\\\\?\\") else "\\\\?\\" + p
    return p


# ------------------------------
# Transformeur sklearn
# ------------------------------
class ImageLoader(BaseEstimator, TransformerMixin):
    """
    Charger les images à partir d'un répertoire selon le motif :
    image_(\\d+)_product_(\\d+)\\.jpg  -> groupe(1)=imageid, groupe(2)=productid

    - Indexer les chemins par productid (accès rapide).
    - Accepter X sous forme de DataFrame (colonne 'productid' ou 'imageid'),
      de Series ou d'itérable simple de productid.
    - Convertir les images en RGB, redimensionner à (H, W) et normaliser sur [0,1].
    - Tolérer les fichiers manquants/illisibles (remplir par image noire).
    """

    def __init__(self, image_dir: str, image_size: tuple[int, int] = (64, 64)) -> None:
        """Initialiser le répertoire d’images, la taille et le motif de nommage."""
        self.image_dir: str = image_dir
        self.image_size: tuple[int, int] = image_size
        self.pattern = re.compile(r"^image_(\d+)_product_(\d+)\.jpg$")
        self.index: dict[str, str] = {}  # productid -> chemin normalisé

    # ----------------------------------------------------------
    # API de confort
    # ----------------------------------------------------------
    def set_image_dir(self, new_dir: str) -> "ImageLoader":
        """Re-pointer le dossier d’images (ex. pour X_test) et réinitialiser l’index."""
        self.image_dir = new_dir
        self.index = {}
        return self

    # ----------------------------------------------------------
    # API sklearn
    # ----------------------------------------------------------
    def fit(self, X=None, y=None) -> "ImageLoader":
        """Indexer les fichiers du répertoire par productid pour accélérer la résolution."""
        self.index = {}
        root = Path(self.image_dir)
        # Parcourir uniquement les .jpg compatibles avec le motif
        for p in root.glob("*.jpg"):
            m = self.pattern.match(p.name)
            if not m:
                continue
            product_id = m.group(2)
            self.index[product_id] = _win_norm(str(p))
        return self

    def transform(self, X) -> np.ndarray:
        """
        Résoudre les chemins puis charger / prétraiter les images.

        X : DataFrame avec 'productid' ou 'imageid', ou bien Series/itérable.
            - Si 'imageid' existe et correspond déjà au *stem* (sans extension),
              construire le chemin comme <image_dir>/<imageid>.jpg
            - Sinon, utiliser 'productid' et résoudre par index puis par glob().
        Retour : np.ndarray de forme (n_samples, H, W, 3), dtype float32 dans [0,1].
        """
        H, W = self.image_size
        root = Path(self.image_dir)

        # 1) Extraire les clés (stems ou productid)
        stems: Optional[List[str]] = None
        pids: Optional[List[str]] = None

        # Cas DataFrame
        if hasattr(X, "columns"):
            cols = set(getattr(X, "columns", []))
            if "imageid" in cols:
                # Utiliser directement les stems : image_{iid}_product_{pid}
                stems = [str(v).strip() for v in X["imageid"].tolist()]
            elif "productid" in cols:
                pids = [str(v).strip() for v in X["productid"].tolist()]
            else:
                raise ValueError("DataFrame attendu avec colonne 'imageid' ou 'productid'.")
        else:
            # Cas Series/itérable -> considérer des productid
            try:
                pids = [str(v).strip() for v in X]  # type: ignore
            except Exception as e:
                raise ValueError("Entrée non supportée : fournir DataFrame/Series/itérable.") from e

        # 2) Construire les chemins
        paths: List[Optional[str]] = []

        if stems is not None:
            # Construire le chemin à partir des stems (sans extension)
            for s in stems:
                p = root / f"{s}.jpg"
                paths.append(_win_norm(str(p)) if p.exists() else None)

        else:
            # Résoudre par productid : d’abord via l’index, sinon via glob pattern
            for pid in pids or []:
                p = self.index.get(pid)
                if p is None:
                    m = next(root.glob(f"image_*_product_{pid}.jpg"), None)
                    p = _win_norm(str(m)) if m is not None else None
                paths.append(p)

        # 3) Charger / prétraiter les images (remplir par noir si manquantes)
        imgs: List[np.ndarray] = []
        for p in paths:
            if p is None:
                # Créer une image noire si le fichier est manquant
                imgs.append(np.zeros((H, W, 3), dtype=np.float32))
                continue
            try:
                imgs.append(self._process_image(p))
            except OSError:
                # Tolérer les fichiers illisibles/lockés
                imgs.append(np.zeros((H, W, 3), dtype=np.float32))

        # 4) Empiler les images en batch
        return np.stack(imgs, axis=0).astype(np.float32)

    # ----------------------------------------------------------
    # Détails d’implémentation
    # ----------------------------------------------------------
    def _process_image(self, image_path: str) -> np.ndarray:
        """Ouvrir, convertir en RGB, redimensionner (W, H), normaliser sur [0,1]."""
        H, W = self.image_size
        image_path = _win_norm(image_path)  # Sécuriser le chemin Windows
        with Image.open(image_path) as img:
            # Convertir en RGB
            img = img.convert("RGB")
            # Redimensionner (Pillow attend (width, height))
            img = img.resize((W, H))
            # Convertir en numpy et normaliser
            arr = np.asarray(img, dtype=np.float32) / 255.0

        # Valider la forme
        if arr.shape != (H, W, 3):
            raise ValueError(f"Forme lue {arr.shape}, attendu {(H, W, 3)}")

        return arr
        #  Normalise les pixels dans [0, 1]
        

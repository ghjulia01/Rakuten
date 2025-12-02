"""
BERT Feature Extractor avec Cache Intelligent.

Optimisé pour Rakuten Challenge :
- Cache des embeddings (évite recalcul)
- Batch processing (GPU efficient)
- Camembert-base (français)


"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Union
import pickle
import hashlib

logger = logging.getLogger(__name__)


class BertFeaturizer:
    """
    Extracteur de features BERT avec cache intelligent.
    
    Utilise CamemBERT pour le français, optimisé pour Rakuten.
    
    Attributes:
        model_name: Nom du modèle (défaut: camembert-base)
        max_length: Longueur max tokens (défaut: 128)
        batch_size: Taille des batchs (défaut: 32)
        use_cache: Activer le cache (défaut: True)
        cache_dir: Dossier de cache
        
    Example:
        >>> bert = BertFeaturizer(use_cache=True)
        >>> embeddings = bert.transform(df['designation'])
        >>> embeddings.shape
        (84916, 768)
    """
    
    def __init__(
        self,
        model_name: str = 'camembert-base',
        max_length: int = 128,
        batch_size: int = 32,
        use_cache: bool = True,
        cache_dir: str = 'artifacts/cache/bert'
    ):
        """
        Initialise le BERT featurizer.
        
        Args:
            model_name: Modèle HuggingFace (camembert-base recommandé)
            max_length: Tokens max (128 suffisant pour Rakuten)
            batch_size: Batch size (32 optimal pour GPU)
            use_cache: Activer cache (fortement recommandé)
            cache_dir: Dossier de cache
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy loading (uniquement si nécessaire)
        self.tokenizer = None
        self.model = None
        self.device = None
        
        logger.info(f"BertFeaturizer initialisé : {model_name}")
        if use_cache:
            logger.info(f"  Cache activé: {self.cache_dir}")
    
    def _load_model(self):
        """Charge le modèle BERT (lazy loading)."""
        if self.model is not None:
            return
        
        logger.info(f"Chargement du modèle {self.model_name}...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Détection GPU
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model.eval()  # Mode évaluation
            
            logger.info(f"✓ Modèle chargé sur {self.device}")
            
        except ImportError:
            logger.error("Transformers non installé : pip install transformers torch")
            raise
    
    def _get_cache_key(self, texts: pd.Series) -> str:
        """
        Génère une clé de cache unique basée sur les textes.
        
        Args:
            texts: Série de textes
            
        Returns:
            Clé MD5 unique
        """
        # Hash basé sur le contenu des textes
        content = ''.join(texts.astype(str).tolist()[:100])  # Sample des 100 premiers
        content += f"|{len(texts)}|{self.model_name}|{self.max_length}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Charge les embeddings depuis le cache.
        
        Args:
            cache_key: Clé de cache
            
        Returns:
            Embeddings ou None si pas en cache
        """
        cache_file = self.cache_dir / f"bert_{cache_key}.pkl"
        
        if cache_file.exists():
            logger.info(f" Chargement depuis cache : {cache_file.name}")
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"✓ Cache chargé : {embeddings.shape}")
            return embeddings
        
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray):
        """
        Sauvegarde les embeddings dans le cache.
        
        Args:
            cache_key: Clé de cache
            embeddings: Embeddings à sauvegarder
        """
        cache_file = self.cache_dir / f"bert_{cache_key}.pkl"
        
        logger.info(f" Sauvegarde dans cache : {cache_file.name}")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Afficher taille du fichier
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        logger.info(f" Cache sauvegardé : {size_mb:.1f} MB")
    
    def _compute_embeddings(self, texts: pd.Series) -> np.ndarray:
        """
        Calcule les embeddings BERT (sans cache).
        
        Args:
            texts: Série de textes
            
        Returns:
            Embeddings de shape (n_samples, 768)
        """
        import torch
        from tqdm import tqdm
        
        self._load_model()
        
        # Préparer les données
        texts_list = texts.fillna('').astype(str).tolist()
        n_samples = len(texts_list)
        
        logger.info(f"Calcul des embeddings BERT pour {n_samples:,} textes...")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Device: {self.device}")
        
        embeddings_list = []
        
        # Traiter par batch
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, self.batch_size), 
                         desc="BERT encoding", total=n_batches):
                batch_texts = texts_list[i:i + self.batch_size]
                
                # Tokenization
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Forward pass
                outputs = self.model(**encoded)
                
                # Pooling: utiliser [CLS] token (premier token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(cls_embeddings)
        
        # Concaténer tous les batchs
        embeddings = np.vstack(embeddings_list)
        
        logger.info(f"✓ Embeddings calculés : {embeddings.shape}")
        
        return embeddings
    
    def fit(self, X, y=None):
        """
        Fit (requis par sklearn, ne fait rien).
        
        Args:
            X: Features (ignoré)
            y: Labels (ignoré)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        Transforme les textes en embeddings BERT.
        
        Args:
            X: Série ou DataFrame avec textes
            
        Returns:
            Embeddings de shape (n_samples, 768)
        """
        # Gérer DataFrame
        if isinstance(X, pd.DataFrame):
        # Concaténer titre + description
            if 'designation' in X.columns and 'description' in X.columns:
                texts = X['designation'].fillna('') + ' ' + X['description'].fillna('')
            else:
                texts = X['designation']
        else:
            texts = X
        
        # Vérifier cache
        if self.use_cache:
            cache_key = self._get_cache_key(texts)
            cached_embeddings = self._load_from_cache(cache_key)
            
            if cached_embeddings is not None:
                return cached_embeddings
        
        # Calculer embeddings
        embeddings = self._compute_embeddings(texts)
        
        # Sauvegarder dans cache
        if self.use_cache:
            self._save_to_cache(cache_key, embeddings)
        
        return embeddings
    
    def get_feature_names_out(self, input_features=None):
        """
        Retourne les noms de features (requis par sklearn).
        
        Returns:
            Liste de noms ['bert_0', 'bert_1', ..., 'bert_767']
        """
        return [f'bert_{i}' for i in range(768)]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Test du BERT featurizer.
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Créer données test
    texts = pd.Series([
        "Console de jeux vidéo Nintendo Switch",
        "Livre pour enfants Harry Potter",
        "Carte mémoire SD 64GB Samsung"
    ])
    
    # Test 1: Premier calcul (avec cache)
    print("\n" + "=" * 60)
    print("TEST 1: Premier calcul (création cache)")
    print("=" * 60)
    
    bert = BertFeaturizer(use_cache=True)
    embeddings1 = bert.transform(texts)
    
    print(f"\nEmbeddings shape: {embeddings1.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings1[0, :5]}")
    
    # Test 2: Deuxième calcul (lecture cache)
    print("\n" + "=" * 60)
    print("TEST 2: Deuxième calcul (lecture cache)")
    print("=" * 60)
    
    bert2 = BertFeaturizer(use_cache=True)
    embeddings2 = bert2.transform(texts)
    
    # Vérifier identité
    assert np.allclose(embeddings1, embeddings2), "Cache invalide!"
    print("✓ Cache fonctionne correctement!")
    
    # Test 3: Sans cache
    print("\n" + "=" * 60)
    print("TEST 3: Sans cache (recalcul)")
    print("=" * 60)
    
    bert3 = BertFeaturizer(use_cache=False)
    embeddings3 = bert3.transform(texts)
    
    print(f"\nEmbeddings shape: {embeddings3.shape}")
    
    print("\n Tous les tests passés!")

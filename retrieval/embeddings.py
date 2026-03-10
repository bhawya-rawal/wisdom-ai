import logging
import numpy as np
from typing import List, Union
from config.settings import settings

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    torch = None

class EmbeddingService:
    """Manages the SentenceTransformer model for generating text embeddings"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.enabled = EMBEDDINGS_AVAILABLE
        self.model = None
        
        if self.enabled:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self.model = SentenceTransformer(self.model_name)
                self.model.to(self.device)
                logger.info("✓ Embeddings model loaded on %s: %s", self.device, self.model_name)
            except Exception as e:
                logger.error("Error loading embedding model: %s. Switching to fallback mode.", e)
                self.enabled = False
        else:
            logger.warning("sentence-transformers is not available. Embedding service running in mock mode.")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate a single normalized embedding for text"""
        if not self.enabled or not self.model:
            # Generate a reproducible dummy vector based on hash of text
            h_val = float(hash(text) % 1000) / 1000.0
            vector = np.zeros(384, dtype='float32')
            vector[0] = h_val
            vector[1] = 1.0 - h_val
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector
            
        embedding = self.model.encode(text)
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts"""
        if not self.enabled or not self.model:
            vectors = []
            for t in texts:
                vectors.append(self.get_embedding(t))
            return np.array(vectors, dtype='float32')
            
        embeddings = self.model.encode(texts)
        # Normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

# Singleton instance
embedding_service = EmbeddingService()

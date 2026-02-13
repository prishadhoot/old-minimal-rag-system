"""
Embedding Module

Model: all-MiniLM-L6-v2

Properties:
- 384 dimensional vectors
- L2 normalized
- CPU-compatible

Responsibilities:
- Embed list of texts
- Embed single query
- No disk writes
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    """
    Embedding generator using sentence-transformers.
    
    Constraints:
    - Model loaded once in constructor
    - No disk writes
    - No global state
    - Always normalized
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Name of sentence-transformers model
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Normalized embeddings of shape (n, 384)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # L2 normalize
        embeddings = self._normalize(embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text.
        
        Args:
            text: Query string to embed
            
        Returns:
            Normalized embedding of shape (1, 384)
        """
        embedding = self.model.encode([text], convert_to_numpy=True)
        
        # L2 normalize
        embedding = self._normalize(embedding)
        
        return embedding
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

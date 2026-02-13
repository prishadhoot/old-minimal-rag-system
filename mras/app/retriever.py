"""
Retriever Module

Phase 1: Semantic search only

Steps:
1. Embed query
2. Search FAISS
3. Return top_k chunks

Constraints:
- No LLM calls
- No prompt formatting
- No retry logic
- Single responsibility: retrieval
"""

from typing import List
from app.models import Chunk
from app.embedding import Embedder
from app.vector_store import VectorStore


class Retriever:
    """
    Semantic retrieval using embeddings and vector search.
    
    Constraints:
    - No LLM calls
    - No prompt formatting
    - No retry logic
    """
    
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        """
        Initialize retriever.
        
        Args:
            embedder: Embedder instance for query encoding
            vector_store: VectorStore instance for search
        """
        self.embedder = embedder
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of Chunk objects ranked by relevance
        """
        # Embed query
        query_vector = self.embedder.embed_query(query)
        
        # Search vector store
        chunks = self.vector_store.search(query_vector, top_k)
        
        return chunks

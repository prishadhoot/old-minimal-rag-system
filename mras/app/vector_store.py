"""
Vector Store Module

Index: FAISS IndexFlatIP

Storage:
- FAISS index file
- SQLite metadata database

Responsibilities:
- Add vectors + metadata
- Search vectors
- Persist and reload

Invariants:
- Vector count == metadata row count
- All vectors normalized
"""

import numpy as np
import faiss
import sqlite3
from typing import List
from pathlib import Path
from app.models import Chunk


class VectorStore:
    """
    Vector storage and retrieval using FAISS.
    
    Constraints:
    - Uses FAISS IndexFlatIP
    - Must persist FAISS index and SQLite metadata
    - Must raise RuntimeError if search called before load or add
    - Must preserve insertion order mapping
    """
    
    def __init__(self, index_path: str, db_path: str):
        """
        Initialize vector store.
        
        Args:
            index_path: Path to FAISS index file
            db_path: Path to SQLite database file
        """
        self.index_path = Path(index_path)
        self.db_path = Path(db_path)
        self.index = None
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        """
        Add vectors and metadata.
        
        Args:
            vectors: Numpy array of shape (n, 384)
            chunks: List of Chunk objects (length must equal vectors)
            
        Raises:
            ValueError: If length mismatch between vectors and chunks
        """
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Vector count ({len(vectors)}) must equal chunk count ({len(chunks)})"
            )
        
        # Initialize index if needed
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors to FAISS
        self.index.add(vectors.astype('float32'))
        
        # Store metadata in SQLite
        self._store_metadata(chunks)
    
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Chunk]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding of shape (1, 384)
            top_k: Number of results to return
            
        Returns:
            List of top_k Chunk objects sorted by similarity
            
        Raises:
            RuntimeError: If search called before index is initialized
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call load() or add() first.")
        
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call add() first.")
        
        # Search FAISS index
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        # Retrieve metadata for results
        chunks = self._retrieve_metadata(indices[0])
        
        return chunks
    
    def save(self) -> None:
        """
        Persist index and metadata to disk.
        
        Raises:
            RuntimeError: If index not initialized
        """
        if self.index is None:
            raise RuntimeError("Cannot save: index not initialized")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
    
    def load(self) -> None:
        """
        Load index and metadata from disk.
        
        If files don't exist, initializes empty index.
        """
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            # Initialize empty index
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Ensure metadata DB exists
        if not self.db_path.exists():
            self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                idx INTEGER PRIMARY KEY,
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def _store_metadata(self, chunks: List[Chunk]) -> None:
        """Store chunk metadata in SQLite"""
        # Initialize DB if needed
        if not self.db_path.exists():
            self._init_database()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current max index
        cursor.execute("SELECT COALESCE(MAX(idx), -1) FROM chunks")
        start_idx = cursor.fetchone()[0] + 1
        
        # Insert chunks
        for i, chunk in enumerate(chunks):
            cursor.execute(
                "INSERT INTO chunks (idx, id, document_id, text) VALUES (?, ?, ?, ?)",
                (start_idx + i, chunk.id, chunk.document_id, chunk.text)
            )
        
        conn.commit()
        conn.close()
    
    def _retrieve_metadata(self, indices: np.ndarray) -> List[Chunk]:
        """Retrieve chunk metadata from SQLite by indices"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        chunks = []
        for idx in indices:
            cursor.execute(
                "SELECT id, document_id, text FROM chunks WHERE idx = ?",
                (int(idx),)
            )
            row = cursor.fetchone()
            if row:
                chunks.append(Chunk(
                    id=row[0],
                    document_id=row[1],
                    text=row[2]
                ))
        
        conn.close()
        return chunks

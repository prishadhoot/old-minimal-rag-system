"""
Chunking Module

Strategy:
- 500 word chunks
- 100 word overlap
- Deterministic ID format: document_id_index

Pure function:
- Input: document_id, raw_text
- Output: list of Chunk objects
"""

from typing import List
from app.models import Chunk
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_document(document_id: str, text: str) -> List[Chunk]:
    """
    Split text into overlapping chunks.
    
    Args:
        document_id: Unique identifier for the document
        text: Raw text to chunk
        
    Returns:
        List of Chunk objects with format: {document_id}_{chunk_index}
        
    Constraints:
        - Chunk size: 500 words
        - Overlap: 100 words
        - Pure function (no side effects)
    """
    words = text.split()
    chunks = []
    
    if len(words) == 0:
        return chunks
    
    chunk_index = 0
    position = 0
    
    while position < len(words):
        # Extract chunk
        end_position = min(position + CHUNK_SIZE, len(words))
        chunk_words = words[position:end_position]
        chunk_text = ' '.join(chunk_words)
        
        # Create Chunk object
        chunk = Chunk(
            id=f"{document_id}_{chunk_index}",
            document_id=document_id,
            text=chunk_text
        )
        chunks.append(chunk)
        
        # Move position forward by (chunk_size - overlap)
        position += (CHUNK_SIZE - CHUNK_OVERLAP)
        chunk_index += 1
        
        # Break if we've reached the end
        if end_position >= len(words):
            break
    
    return chunks

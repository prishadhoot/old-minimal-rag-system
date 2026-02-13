"""
Tests for chunking module
"""

import pytest
from app.chunking import chunk_document
from app.models import Chunk


def test_chunk_document_basic():
    """Test basic chunking functionality"""
    text = " ".join(["word"] * 600)  # 600 words
    chunks = chunk_document("test_doc", text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.document_id == "test_doc" for chunk in chunks)


def test_chunk_document_ids():
    """Test chunk ID format"""
    text = " ".join(["word"] * 600)
    chunks = chunk_document("doc123", text)
    
    assert chunks[0].id == "doc123_0"
    assert chunks[1].id == "doc123_1"


def test_chunk_document_empty():
    """Test chunking empty text"""
    chunks = chunk_document("empty_doc", "")
    assert len(chunks) == 0


def test_chunk_document_small():
    """Test chunking text smaller than chunk size"""
    text = " ".join(["word"] * 50)  # 50 words
    chunks = chunk_document("small_doc", text)
    
    assert len(chunks) == 1
    assert chunks[0].text == text

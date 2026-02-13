"""
Ingestion Module

Responsibilities:
- Load .txt, .md, .pdf files
- Extract raw text
- Return (document_id, text) tuples

Must NOT:
- Chunk
- Embed
- Store
"""

import os
from pathlib import Path
from typing import List, Tuple


def load_documents(folder_path: str) -> List[Tuple[str, str]]:
    """
    Load documents from a folder.
    
    Args:
        folder_path: Path to folder containing documents
        
    Returns:
        List of tuples: (document_id, raw_text)
        where document_id = filename without extension
        
    Raises:
        ValueError: If unsupported file type encountered
    """
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    for file_path in folder.iterdir():
        if file_path.is_file():
            extension = file_path.suffix.lower()
            document_id = file_path.stem
            
            if extension == '.txt':
                text = _load_txt(file_path)
            elif extension == '.md':
                text = _load_md(file_path)
            elif extension == '.pdf':
                text = _load_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file type: {extension} for file {file_path}")
            
            documents.append((document_id, text))
    
    return documents


def _load_txt(file_path: Path) -> str:
    """Load text from .txt file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def _load_md(file_path: Path) -> str:
    """Load text from .md file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_document_from_bytes(filename: str, content: bytes) -> Tuple[str, str]:
    """
    Load a single document from in-memory bytes.
    
    Args:
        filename: Original filename (e.g., "doc.txt")
        content: Raw file content as bytes
        
    Returns:
        Tuple of (document_id, raw_text) where document_id = filename without extension
        
    Raises:
        ValueError: If unsupported file type
    """
    path = Path(filename)
    extension = path.suffix.lower()
    document_id = path.stem
    
    if extension == '.txt':
        text = content.decode('utf-8')
    elif extension == '.md':
        text = content.decode('utf-8')
    elif extension == '.pdf':
        import io
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                raise RuntimeError("pypdf or PyPDF2 is required to load PDF files. Install with: pip install pypdf")
        reader = PdfReader(io.BytesIO(content))
        text_parts = [page.extract_text() or '' for page in reader.pages]
        text = '\n'.join(text_parts) if text_parts else ''
    else:
        raise ValueError(f"Unsupported file type: {extension} for file {filename}")
    
    return (document_id, text)


def _load_pdf(file_path: Path) -> str:
    """Load text from .pdf file"""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise RuntimeError("pypdf or PyPDF2 is required to load PDF files. Install with: pip install pypdf")
    
    text_parts = []
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or '')
    
    return '\n'.join(text_parts)

"""
Configuration settings for MRAS
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
STATIC_DIR = BASE_DIR / "static"
INDEX_PATH = DATA_DIR / "faiss.index"
DB_PATH = DATA_DIR / "metadata.db"
EVAL_PATH = DATA_DIR / "eval.json"

# Chunking settings
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 100  # words

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval settings
DEFAULT_TOP_K = 5

# LLM settings (OpenRouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
# Fallback models when primary is rate-limited (429)
OPENROUTER_FALLBACK_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]

# Agent settings
RETRY_MULTIPLIER = 2
MAX_RETRIES = 1

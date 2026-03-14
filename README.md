# MRAS - Minimal RAG Agent System (OLD)

A local, production-style Retrieval-Augmented Generation (RAG) system built with simplicity and determinism in mind.

## Overview

MRAS is a proof-of-concept RAG system that demonstrates:
- Document ingestion and chunking
- Local embedding generation
- Vector storage with FAISS
- Semantic retrieval
- LLM-powered reasoning (Gemini)
- Minimal agent loop
- Evaluation harness

## Architecture

```
Document Ingestion
→ Chunking
→ Embedding
→ FAISS Vector Index
→ Retriever
→ LLM (Gemini)
→ Agent Loop
→ FastAPI API Layer
```

## Tech Stack

- **Language**: Python 3.11
- **API**: FastAPI + Uvicorn
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (CPU version)
- **LLM**: Gemini API
- **Storage**: SQLite + Local filesystem

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Gemini API Key (Legacy)

```bash
export GEMINI_API_KEY="your-api-key-here"
```

On Windows:
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```

### Ingest Documents
```bash
POST /ingest
{
    "folder_path": "path/to/documents"
}
```

### Query
```bash
POST /query
{
    "query": "What is the capital of France?",
    "top_k": 5
}
```

### Evaluate
```bash
POST /evaluate
```

## Usage Example

### 1. Add documents to the data/documents folder
```bash
# Copy your .txt, .md, or .pdf files to data/documents/
```

### 2. Ingest documents
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "data/documents"}'
```

### 3. Query the system
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 5}'
```

## Project Structure

```
mras/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models.py            # Pydantic data models
│   ├── ingestion.py         # Document loading
│   ├── chunking.py          # Text chunking
│   ├── embedding.py         # Embedding generation
│   ├── vector_store.py      # FAISS vector storage
│   ├── retriever.py         # Semantic retrieval
│   ├── llm.py               # Gemini LLM client
│   ├── agent.py             # RAG agent loop
│   └── evaluation.py        # Evaluation metrics
├── data/
│   ├── documents/           # Source documents
│   ├── faiss.index          # FAISS index (generated)
│   ├── metadata.db          # SQLite metadata (generated)
│   └── eval.json            # Evaluation dataset
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Design Principles

1. **Minimal moving parts** - No unnecessary abstractions
2. **Clear module boundaries** - Each module has a single responsibility
3. **Deterministic execution** - No random behavior
4. **No overengineering** - Simple solutions over complex ones
5. **Local deployment** - Everything runs locally

## Constraints

- Fully local deployment
- Free tooling only
- Simplicity prioritized over features
- No LangChain or LlamaIndex
- No async complexity
- Deterministic behavior

## Evaluation

Create an `eval.json` file in the `data/` directory with the following format:

```json
[
    {
        "question": "What is machine learning?",
        "expected_keywords": ["algorithm", "data", "prediction"]
    },
    {
        "question": "How does neural network work?",
        "expected_keywords": ["neuron", "layer", "activation"]
    }
]
```

Then call the `/evaluate` endpoint to get metrics:
- Average keyword match score
- Average response latency

## Module Interfaces

### Ingestion
- Loads `.txt`, `.md`, `.pdf` files
- Returns `(document_id, text)` tuples

### Chunking
- 500 word chunks with 100 word overlap
- Deterministic chunk IDs: `{document_id}_{index}`

### Embedding
- all-MiniLM-L6-v2 model
- 384 dimensional vectors
- L2 normalized

### Vector Store
- FAISS IndexFlatIP
- SQLite metadata storage
- Persistent index

### Retriever
- Semantic search only
- Returns top-k chunks

### LLM Client
- Gemini API integration
- NOT_FOUND protocol enforcement

### Agent
- Deterministic loop with one retry
- Returns answer + source chunk IDs

## Skills Demonstrated

- Embedding space reasoning
- Cosine similarity
- Vector search fundamentals
- Modular architecture
- API engineering
- LLM prompt control
- Evaluation mindset
- Deterministic system design

## License


## Author
Prisha Dhoot

(Built as a proof-of-concept for understanding RAG systems from first principles.)

! This README has been AI-generated

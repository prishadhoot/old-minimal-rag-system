"""
FastAPI Application

Required Endpoints:
- POST /ingest
- POST /query
- POST /evaluate
- GET /health

State Initialization Contract:
At startup:
1. Instantiate Embedder
2. Instantiate VectorStore
3. Attempt load()
4. Instantiate Retriever
5. Instantiate LLMClient
6. Instantiate Agent
"""

# Load .env before config so OPENROUTER_API_KEY is available
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    _env_path = _Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import json
from pathlib import Path

from app.config import (
    INDEX_PATH, DB_PATH, EVAL_PATH, DOCUMENTS_DIR, STATIC_DIR,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODELS,
)
from app.models import QueryRequest, QueryResponse, EvalSample
from app.embedding import Embedder
from app.vector_store import VectorStore
from app.retriever import Retriever
from app.llm import LLMClient
from app.agent import Agent
from app.ingestion import load_documents, load_document_from_bytes
from app.chunking import chunk_document
from app.evaluation import evaluate


# Initialize FastAPI app
app = FastAPI(title="MRAS - Minimal RAG Agent System")

# Global state (initialized at startup)
embedder = None
vector_store = None
retriever = None
llm_client = None
agent = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize system components at startup.
    No lazy initialization.
    """
    global embedder, vector_store, retriever, llm_client, agent
    
    # 1. Instantiate Embedder
    embedder = Embedder()
    
    # 2. Instantiate VectorStore
    vector_store = VectorStore(str(INDEX_PATH), str(DB_PATH))
    
    # 3. Attempt load()
    vector_store.load()
    
    # 4. Instantiate Retriever
    retriever = Retriever(embedder, vector_store)
    
    # 5. Instantiate LLMClient (OpenRouter)
    llm_client = LLMClient(
        OPENROUTER_API_KEY, OPENROUTER_MODEL,
        fallback_models=OPENROUTER_FALLBACK_MODELS,
    )
    
    # 6. Instantiate Agent
    agent = Agent(retriever, llm_client)


class IngestRequest(BaseModel):
    folder_path: str


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest documents from a folder.
    
    Steps:
    1. Load documents
    2. Chunk documents
    3. Embed chunks
    4. Add to vector store
    5. Save index
    """
    try:
        # Load documents
        documents = load_documents(request.folder_path)
        
        # Chunk and embed
        all_chunks = []
        for doc_id, text in documents:
            chunks = chunk_document(doc_id, text)
            all_chunks.extend(chunks)
        
        # Extract texts for embedding
        chunk_texts = [chunk.text for chunk in all_chunks]
        
        # Embed
        vectors = embedder.embed_texts(chunk_texts)
        
        # Add to vector store
        vector_store.add(vectors, all_chunks)
        
        # Save
        vector_store.save()
        
        return IngestResponse(
            status="success",
            chunks_indexed=len(all_chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@app.get("/documents")
async def list_documents():
    """
    List documents in the documents directory (uploaded files).
    """
    if not DOCUMENTS_DIR.exists():
        return {"documents": []}
    docs = []
    for f in sorted(DOCUMENTS_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            docs.append({"name": f.name, "stem": f.stem})
    return {"documents": docs}


@app.post("/upload", response_model=IngestResponse)
async def upload(files: List[UploadFile] = File(...)):
    """
    Upload documents and ingest them into the vector store.
    
    Accepts .txt, .md, .pdf files.
    Saves to data/documents/, then chunks, embeds, and adds to index.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    
    try:
        for upload_file in files:
            filename = upload_file.filename or "unnamed"
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {ext}. Allowed: .txt, .md, .pdf"
                )
            
            content = await upload_file.read()
            
            # Save to disk for persistence
            dest = DOCUMENTS_DIR / filename
            dest.write_bytes(content)
            
            # Load and chunk
            doc_id, text = load_document_from_bytes(filename, content)
            chunks = chunk_document(doc_id, text)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return IngestResponse(status="success", chunks_indexed=0)
        
        chunk_texts = [chunk.text for chunk in all_chunks]
        vectors = embedder.embed_texts(chunk_texts)
        vector_store.add(vectors, all_chunks)
        vector_store.save()
        
        return IngestResponse(
            status="success",
            chunks_indexed=len(all_chunks)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    """
    try:
        response = agent.answer(request.query, request.top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EvaluateResponse(BaseModel):
    avg_keyword_score: float
    avg_latency: float


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint():
    """
    Evaluate agent performance using eval.json.
    """
    try:
        # Load evaluation samples
        with open(EVAL_PATH, 'r') as f:
            eval_data = json.load(f)
        
        eval_samples = [EvalSample(**sample) for sample in eval_data]
        
        # Run evaluation
        results = evaluate(agent, eval_samples)
        
        return EvaluateResponse(**results)
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="eval.json not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Redirect to UI"""
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


# Mount static files for UI (must be after routes)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

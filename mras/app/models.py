"""
Data Models (Canonical Schemas)
All modules must use these Pydantic models.
"""

from pydantic import BaseModel
from typing import List


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class EvalSample(BaseModel):
    question: str
    expected_keywords: List[str]

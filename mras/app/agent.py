"""
Agent Module

Deterministic loop:
1. Retrieve top_k
2. Build context string
3. Call LLM
4. If NOT_FOUND → retrieve top_k * 2
5. Retry once
6. Return answer + source chunk IDs

Constraints:
- Only one retry
- No recursion
- No dynamic planning
- Must return sources = chunk_ids used
"""

from typing import List
from app.models import QueryResponse, Chunk
from app.retriever import Retriever
from app.llm import LLMClient


class Agent:
    """
    RAG agent with deterministic retrieval-generation loop.
    
    Constraints:
    - Only one retry
    - No recursion
    - No dynamic planning
    """
    
    def __init__(self, retriever: Retriever, llm_client: LLMClient):
        """
        Initialize agent.
        
        Args:
            retriever: Retriever instance
            llm_client: LLMClient instance
        """
        self.retriever = retriever
        self.llm_client = llm_client
    
    def answer(self, query: str, top_k: int = 5) -> QueryResponse:
        """
        Answer a query using RAG.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve initially
            
        Returns:
            QueryResponse with answer and source chunk IDs
        """
        # First attempt: retrieve top_k
        chunks = self.retriever.retrieve(query, top_k)
        context = self._build_context(chunks)
        answer = self.llm_client.generate(context, query)
        
        # If NOT_FOUND, retry with more chunks
        if "NOT_FOUND" in answer:
            # Retrieve top_k * 2
            chunks = self.retriever.retrieve(query, top_k * 2)
            context = self._build_context(chunks)
            answer = self.llm_client.generate(context, query)
        
        # Extract source IDs
        source_ids = [chunk.id for chunk in chunks]
        
        return QueryResponse(
            answer=answer,
            sources=source_ids
        )
    
    def _build_context(self, chunks: List[Chunk]) -> str:
        """
        Build context string from chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}] {chunk.text}")
        
        return "\n\n".join(context_parts)

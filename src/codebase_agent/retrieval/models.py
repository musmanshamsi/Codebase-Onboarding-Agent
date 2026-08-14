"""Data models for hybrid retrieval, graph expansion, and context assembly (FR-5)."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Chunk record retrieved via semantic vector search or structural graph expansion."""

    chunk_id: str = Field(description="Unique chunk ID")
    document: str = Field(description="Source code text snippet")
    file_path: str = Field(description="Relative file path")
    symbol_name: str = Field(description="Symbol name")
    symbol_type: str = Field(description="Symbol type: function, method, class, module")
    start_line: int = Field(description="1-indexed start line")
    end_line: int = Field(description="1-indexed end line")
    graph_node_id: str = Field(description="Foreign key to knowledge graph node")
    similarity_score: float = Field(default=0.0, description="Similarity score (0.0 to 1.0)")
    source: str = Field(default="semantic", description="Retrieval source: 'semantic' or 'graph_expansion'")


class RetrievalResult(BaseModel):
    """Aggregated output from Hybrid Retrieval Engine."""

    query: str = Field(description="Original user question")
    semantic_chunks: List[RetrievedChunk] = Field(default_factory=list)
    expanded_chunks: List[RetrievedChunk] = Field(default_factory=list)
    final_context_chunks: List[RetrievedChunk] = Field(default_factory=list)
    total_tokens: int = Field(default=0, description="Estimated total tokens in assembled context")
    sufficient_context: bool = Field(default=True, description="True if top match satisfies similarity_threshold")
    message: Optional[str] = Field(default=None, description="Explanation if context is insufficient")

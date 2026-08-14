"""Data models for answer generation and evidence citation (FR-5.4, FR-5.5)."""

from typing import List, Optional
from pydantic import BaseModel, Field
from codebase_agent.retrieval.models import RetrievedChunk


class Citation(BaseModel):
    """Verifiable evidence citation referencing source code file and line range."""

    file_path: str = Field(description="Relative file path")
    start_line: int = Field(description="1-indexed start line")
    end_line: int = Field(description="1-indexed end line")
    symbol_name: str = Field(description="Symbol name")
    graph_node_id: str = Field(description="Graph node ID")


class QueryResponse(BaseModel):
    """Complete query response containing synthesized answer and citations."""

    query: str = Field(description="User question")
    answer: str = Field(description="Synthesized natural-language answer")
    citations: List[Citation] = Field(default_factory=list, description="Verifiable evidence citations")
    sufficient_context: bool = Field(default=True, description="True if context was sufficient")
    model_name: str = Field(default="qwen2.5-coder:7b", description="LLM model name used")
    total_tokens: int = Field(default=0, description="Tokens in assembled context")

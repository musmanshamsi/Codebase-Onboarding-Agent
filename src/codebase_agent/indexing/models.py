"""Data models and node ID generators for vector indexing (Database Design Section 3)."""

from typing import List, Optional
from pydantic import BaseModel, Field


def make_graph_node_id(file_path: str, symbol_name: Optional[str] = None) -> str:
    """
    Shared graph node ID generator matching GraphBuilder & Database Design Section 4.3.
    File node ID: {file_path}
    Symbol node ID: {file_path}::{symbol_name}
    """
    file_clean = file_path.replace("\\", "/")
    if symbol_name:
        return f"{file_clean}::{symbol_name}"
    return file_clean


def make_chunk_id(file_path: str, symbol_name: str, start_line: int) -> str:
    """
    Unique vector chunk record ID generator (Database Design Section 3.3).
    Format: {file_path}::{symbol_name}::{start_line}
    """
    file_clean = file_path.replace("\\", "/")
    return f"{file_clean}::{symbol_name}::{start_line}"


class CodeChunk(BaseModel):
    """Syntax-aligned code chunk record for vector embedding and graph referencing."""

    id: str = Field(description="Unique chunk ID format: {file_path}::{symbol_name}::{start_line}")
    document: str = Field(description="Raw source code snippet text")
    file_path: str = Field(description="Relative file path")
    symbol_name: str = Field(description="Owning symbol name or module name")
    symbol_type: str = Field(description="Symbol type: function, method, class, module")
    start_line: int = Field(description="1-indexed start line in file")
    end_line: int = Field(description="1-indexed end line in file")
    language: str = Field(default="python", description="Language tag")
    graph_node_id: str = Field(description="Foreign key into graph store: {file_path}::{symbol_name}")
    content_hash: str = Field(description="SHA-256 hash of chunk text")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding float array")

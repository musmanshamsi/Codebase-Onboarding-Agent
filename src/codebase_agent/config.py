"""Configuration module for Codebase Onboarding Agent (FR-8)."""

from typing import List
from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Configuration settings for repository ingestion (FR-1, FR-8.4)."""

    include_patterns: List[str] = Field(
        default=["**/*"],
        description="Glob patterns of files to include in ingestion."
    )
    exclude_patterns: List[str] = Field(
        default=[
            "node_modules/**",
            ".venv/**",
            "venv/**",
            "dist/**",
            "build/**",
            ".git/**",
            "__pycache__/**",
            ".agent_index/**",
            "*.pyc",
            "*.pyo",
            "*.bin",
            "*.exe",
            "*.dll",
            "*.so",
            "*.dylib",
            "*.zip",
            "*.tar",
            "*.gz",
            "*.db",
            "*.sqlite",
            "*.graphml",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.ico",
            "*.svg",
            "*.pdf"
        ],
        description="Glob patterns of files/directories to exclude."
    )
    supported_extensions: List[str] = Field(
        default=[".py"],
        description="File extensions considered for source code parsing."
    )
    max_file_size_bytes: int = Field(
        default=1_048_576,  # 1 MB
        description="Maximum file size in bytes; larger files will be skipped."
    )
    index_dir_name: str = Field(
        default=".agent_index",
        description="Directory name storing agent vector DB, graph, and metadata cache."
    )
    max_line_length_minified: int = Field(
        default=500,
        description="Max average line length threshold before treating file as minified."
    )


class AgentConfig(BaseModel):
    """Overall system configuration wrapper."""

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)

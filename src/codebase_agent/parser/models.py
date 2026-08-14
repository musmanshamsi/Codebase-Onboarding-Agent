"""Data models for code parser and symbol extraction (FR-2)."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SymbolType(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    VARIABLE = "variable"
    MODULE = "module"


class Symbol(BaseModel):
    """Extracted code entity (function, class, method, variable)."""

    id: str = Field(description="Unique node ID format: {file_path}::{symbol_name}")
    name: str = Field(description="Unqualified symbol name")
    type: SymbolType = Field(description="Symbol classification")
    file_path: str = Field(description="Relative file path in repository")
    start_line: int = Field(description="1-indexed start line in source code")
    end_line: int = Field(description="1-indexed end line in source code")
    parent_symbol: Optional[str] = Field(default=None, description="Parent class or scope if nested")
    language: str = Field(default="python", description="Programming language tag")


class ImportStatement(BaseModel):
    """Extracted module import statement."""

    source_module: str = Field(description="Imported module name or relative path dots")
    imported_symbols: List[str] = Field(default_factory=list, description="Specific imported names e.g., ['process_payment']")
    alias_map: Dict[str, str] = Field(default_factory=dict, description="Mapping of original name to alias if imported as")
    is_relative: bool = Field(default=False, description="True if relative import (e.g. from ..service import X)")
    level: int = Field(default=0, description="Relative dot count (0 for absolute, 1 for ., 2 for ..)")
    line_number: int = Field(description="1-indexed line number of import statement")


class CallSite(BaseModel):
    """Extracted function or method call expression site."""

    caller_symbol_id: Optional[str] = Field(default=None, description="ID of owning function/method symbol containing this call")
    function_name: str = Field(description="Callee function/method name e.g., charge_card")
    line_number: int = Field(description="1-indexed line number of call expression")
    args_count: int = Field(default=0, description="Number of arguments passed")


class FileParseResult(BaseModel):
    """Aggregation of all symbols, imports, and call sites extracted from one file."""

    file_path: str = Field(description="Relative file path")
    language: str = Field(description="Detected language tag")
    symbols: List[Symbol] = Field(default_factory=list)
    imports: List[ImportStatement] = Field(default_factory=list)
    call_sites: List[CallSite] = Field(default_factory=list)
    parse_status: str = Field(default="success", description="parse status: 'success' or 'failed'")
    error_message: Optional[str] = Field(default=None, description="Error details if parse_status = 'failed'")

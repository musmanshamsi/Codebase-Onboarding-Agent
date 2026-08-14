"""Code-aware syntax chunker component implementing Algorithm 3.4 (FR-4.1)."""

import hashlib
from typing import List, Optional
from codebase_agent.indexing.models import CodeChunk, make_chunk_id, make_graph_node_id
from codebase_agent.parser.models import FileParseResult, Symbol, SymbolType

try:
    from llama_index.core.node_parser import CodeSplitter
except ImportError:
    CodeSplitter = None


class Chunker:
    """Splits source files into syntax-aligned embedding chunks bound to graph node IDs."""

    def __init__(self, max_chunk_lines: int = 40, overlap_lines: int = 10):
        self.max_chunk_lines = max_chunk_lines
        self.overlap_lines = overlap_lines

    def chunk_file(
        self,
        file_path: str,
        source_code: str,
        parse_result: FileParseResult,
        content_hash: str
    ) -> List[CodeChunk]:
        """
        Implements Algorithm 3.4 (chunk_file).
        Splits file into embedding-ready units aligned to function/class boundaries.
        """
        chunks: List[CodeChunk] = []
        lines = source_code.splitlines(keepends=True)
        total_lines = len(lines)

        if total_lines == 0:
            return chunks

        # Process function, method, and class symbols
        processed_symbols: List[Symbol] = [
            s for s in parse_result.symbols
            if s.type in (SymbolType.FUNCTION, SymbolType.METHOD, SymbolType.CLASS)
        ]

        if processed_symbols:
            for sym in processed_symbols:
                s_line = max(1, sym.start_line)
                e_line = min(total_lines, sym.end_line)

                symbol_lines = lines[s_line - 1 : e_line]
                chunk_text = "".join(symbol_lines)
                symbol_line_count = len(symbol_lines)

                graph_node_id = make_graph_node_id(file_path, sym.name)

                if symbol_line_count <= self.max_chunk_lines:
                    chunk_id = make_chunk_id(file_path, sym.name, s_line)
                    c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                    chunks.append(CodeChunk(
                        id=chunk_id,
                        document=chunk_text,
                        file_path=file_path,
                        symbol_name=sym.name,
                        symbol_type=sym.type.value,
                        start_line=s_line,
                        end_line=e_line,
                        language=sym.language,
                        graph_node_id=graph_node_id,
                        content_hash=c_hash
                    ))
                else:
                    # Sub-chunk large symbols with overlap
                    sub_chunks = self._split_large_symbol(
                        file_path=file_path,
                        sym=sym,
                        lines=symbol_lines,
                        base_start_line=s_line,
                        graph_node_id=graph_node_id
                    )
                    chunks.extend(sub_chunks)

        # Fallback if no functions/classes found (or for whole module chunking)
        if not chunks:
            chunk_text = source_code
            graph_node_id = make_graph_node_id(file_path)
            chunk_id = make_chunk_id(file_path, "module", 1)
            c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunks.append(CodeChunk(
                id=chunk_id,
                document=chunk_text,
                file_path=file_path,
                symbol_name="module",
                symbol_type="module",
                start_line=1,
                end_line=total_lines,
                language=parse_result.language,
                graph_node_id=graph_node_id,
                content_hash=c_hash
            ))

        return chunks

    def _split_large_symbol(
        self,
        file_path: str,
        sym: Symbol,
        lines: List[str],
        base_start_line: int,
        graph_node_id: str
    ) -> List[CodeChunk]:
        """Splits symbol exceeding max_chunk_lines into overlapping sub-chunks."""
        chunks: List[CodeChunk] = []
        step = max(1, self.max_chunk_lines - self.overlap_lines)
        total_sym_lines = len(lines)

        for i in range(0, total_sym_lines, step):
            sub_lines = lines[i : i + self.max_chunk_lines]
            if not sub_lines:
                break
            
            sub_text = "".join(sub_lines)
            s_line = base_start_line + i
            e_line = base_start_line + i + len(sub_lines) - 1

            chunk_id = make_chunk_id(file_path, f"{sym.name}_part{i//step + 1}", s_line)
            c_hash = hashlib.sha256(sub_text.encode("utf-8")).hexdigest()

            chunks.append(CodeChunk(
                id=chunk_id,
                document=sub_text,
                file_path=file_path,
                symbol_name=sym.name,
                symbol_type=sym.type.value,
                start_line=s_line,
                end_line=e_line,
                language=sym.language,
                graph_node_id=graph_node_id,
                content_hash=c_hash
            ))

        return chunks

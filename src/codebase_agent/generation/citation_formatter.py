"""Citation Formatter component (Architecture Section 4.12, FR-5.5)."""

from typing import List, Tuple
from codebase_agent.generation.models import Citation, QueryResponse
from codebase_agent.retrieval.models import RetrievedChunk


class CitationFormatter:
    """Formats LLM answers alongside verifiable file paths and line ranges as evidence."""

    @staticmethod
    def format_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
        """Extracts unique evidence citations from retrieved chunks."""
        seen: set = set()
        citations: List[Citation] = []

        for chunk in chunks:
            key = (chunk.file_path, chunk.start_line, chunk.end_line)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    symbol_name=chunk.symbol_name,
                    graph_node_id=chunk.graph_node_id
                ))
        return citations

    @staticmethod
    def render_output(response: QueryResponse) -> str:
        """Renders CLI output text pairing LLM answer with verifiable citations."""
        lines: List[str] = [
            "=== Answer ===",
            response.answer,
            ""
        ]

        if response.citations:
            lines.append("Citations:")
            for idx, cit in enumerate(response.citations, 1):
                lines.append(f"  [{idx}] {cit.file_path}:{cit.start_line}-{cit.end_line} (Symbol: {cit.symbol_name})")
        else:
            lines.append("Citations: None (insufficient context or general guidance)")

        return "\n".join(lines)

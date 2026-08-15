"""Context Assembler component implementing Algorithm 3.7 (FR-5.6)."""

from typing import Dict, List, Set, Optional
from codebase_agent.retrieval.models import RetrievedChunk, RetrievalResult


class ContextAssembler:
    """Deduplicates, token-budgets, and checks sufficiency score threshold for prompt context."""

    def __init__(self, max_context_tokens: int = 4096, similarity_threshold: float = 0.38):
        self.max_context_tokens = max_context_tokens
        self.similarity_threshold = similarity_threshold

    def assemble(
        self,
        query: str,
        semantic_chunks: List[RetrievedChunk],
        expanded_chunks: List[RetrievedChunk]
    ) -> RetrievalResult:
        """
        Implements Algorithm 3.7 (assemble_context) and sufficiency check (FR-5.6).
        """
        # 1. Deduplicate chunks by chunk_id
        seen_ids: Set[str] = set()
        unique_chunks: List[RetrievedChunk] = []

        for chunk in semantic_chunks + expanded_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        # 2. Sort by similarity score descending
        unique_chunks.sort(key=lambda c: c.similarity_score, reverse=True)

        # 3. Check Sufficiency Threshold (FR-5.6)
        top_similarity = semantic_chunks[0].similarity_score if semantic_chunks else 0.0
        sufficient_context = top_similarity >= self.similarity_threshold and len(unique_chunks) > 0

        if not sufficient_context:
            return RetrievalResult(
                query=query,
                semantic_chunks=semantic_chunks,
                expanded_chunks=expanded_chunks,
                final_context_chunks=[],
                total_tokens=0,
                sufficient_context=False,
                message=(
                    f"Insufficient relevant code context found in repository "
                    f"(top similarity score {top_similarity:.3f} < threshold {self.similarity_threshold})."
                )
            )

        # 4. Token Budgeting (Algorithm 3.7)
        final_chunks: List[RetrievedChunk] = []
        accumulated_tokens = 0

        for chunk in unique_chunks:
            chunk_tokens = self.estimate_tokens(chunk.document)
            if accumulated_tokens + chunk_tokens > self.max_context_tokens:
                break
            final_chunks.append(chunk)
            accumulated_tokens += chunk_tokens

        return RetrievalResult(
            query=query,
            semantic_chunks=semantic_chunks,
            expanded_chunks=expanded_chunks,
            final_context_chunks=final_chunks,
            total_tokens=accumulated_tokens,
            sufficient_context=True,
            message=None
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimator (approx 4 chars per token)."""
        return max(1, len(text) // 4)

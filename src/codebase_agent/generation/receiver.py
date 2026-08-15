"""Query Receiver component coordinating the Query Pipeline (Architecture Section 4.7, Database Design Section 5.4)."""

import json
from pathlib import Path
from typing import Optional

from codebase_agent.config import IngestionConfig
from codebase_agent.generation.citation_formatter import CitationFormatter
from codebase_agent.generation.generator import LLMGenerator
from codebase_agent.generation.models import QueryResponse
from codebase_agent.graph.store import GraphStore
from codebase_agent.retrieval.context_assembler import ContextAssembler
from codebase_agent.retrieval.graph_expander import GraphExpander
from codebase_agent.retrieval.vector_retriever import VectorRetriever
from codebase_agent.storage.metadata_cache import MetadataCache


class QueryReceiver:
    """Coordinates query intake, hybrid retrieval, LLM synthesis, and query logging."""

    def __init__(
        self,
        repo_dir: Path,
        model_name: str = "qwen2.5-coder:1.5b",
        top_k: int = 3,
        hops: int = 1,
        similarity_threshold: float = 0.38
    ):
        self.repo_dir = repo_dir
        self.model_name = model_name
        self.top_k = top_k
        self.hops = hops
        self.similarity_threshold = similarity_threshold

        config = IngestionConfig()
        self.index_dir = self.repo_dir / config.index_dir_name
        self.chroma_dir = self.index_dir / "chroma"
        self.graph_path = self.index_dir / "graph" / "repo_graph.graphml"
        self.db_path = self.index_dir / "cache.db"

        self._retriever: Optional[VectorRetriever] = None
        self._expander: Optional[GraphExpander] = None
        self._assembler: Optional[ContextAssembler] = None
        self._generator: Optional[LLMGenerator] = None
        self._cache: Optional[MetadataCache] = None
        self._G = None
        self._graph_mtime = None

    def _get_components(self):
        if self._retriever is None:
            self._retriever = VectorRetriever(chroma_dir=self.chroma_dir)
        if self._expander is None:
            self._expander = GraphExpander(chroma_dir=self.chroma_dir)
        if self._assembler is None or self._assembler.similarity_threshold != self.similarity_threshold:
            self._assembler = ContextAssembler(max_context_tokens=4096, similarity_threshold=self.similarity_threshold)
        if self._generator is None or self._generator.model_name != self.model_name:
            self._generator = LLMGenerator(model_name=self.model_name)
        if self._cache is None:
            self._cache = MetadataCache(db_path=self.db_path)

        if self.graph_path.exists():
            mtime = self.graph_path.stat().st_mtime
            if self._G is None or self._graph_mtime != mtime:
                self._G = GraphStore.load_graph(self.graph_path)
                self._graph_mtime = mtime

        return self._retriever, self._expander, self._assembler, self._generator, self._cache, self._G

    def process_query(self, question: str) -> QueryResponse:
        """Executes full query pipeline end-to-end (FR-5.1 - FR-5.6)."""
        if not self.chroma_dir.exists() or not self.graph_path.exists():
            return QueryResponse(
                query=question,
                answer=f"Index directories missing at {self.index_dir}. Please index the repository first.",
                citations=[],
                sufficient_context=False,
                model_name=self.model_name
            )

        # 1. Fetch Cached Pipeline Components
        retriever, expander, assembler, generator, cache, G = self._get_components()

        # 2. Hybrid Retrieval (Semantic + Graph Expansion)
        semantic_chunks = retriever.retrieve(query_text=question, top_k=self.top_k)
        expanded_chunks = expander.expand(semantic_chunks=semantic_chunks, G=G if G is not None else GraphStore.load_graph(self.graph_path), hops=self.hops)

        # 3. Context Assembly & Sufficiency Check (FR-5.6)
        retrieval_res = assembler.assemble(query=question, semantic_chunks=semantic_chunks, expanded_chunks=expanded_chunks)

        if not retrieval_res.sufficient_context:
            answer_text = retrieval_res.message or "I do not have sufficient context in the codebase to answer this question."
            citations = []
            sufficient_context = False
        else:
            # 4. LLM Generation
            answer_text = generator.generate_answer(query=question, chunks=retrieval_res.final_context_chunks)
            citations = CitationFormatter.format_citations(retrieval_res.final_context_chunks)
            sufficient_context = True

        response = QueryResponse(
            query=question,
            answer=answer_text,
            citations=citations,
            sufficient_context=sufficient_context,
            model_name=self.model_name,
            total_tokens=retrieval_res.total_tokens
        )

        # 5. Log Query to SQLite (Database Design Section 5.4)
        try:
            self._log_query(cache, question, semantic_chunks, expanded_chunks, response)
        except Exception:
            pass

        return response

    @staticmethod
    def _log_query(
        cache: MetadataCache,
        question: str,
        semantic_chunks,
        expanded_chunks,
        response: QueryResponse
    ):
        """Writes query record to query_log table in SQLite cache database."""
        with cache._get_connection() as conn:
            cursor = conn.cursor()
            sem_ids = json.dumps([c.chunk_id for c in semantic_chunks])
            exp_ids = json.dumps([c.chunk_id for c in expanded_chunks])

            cursor.execute("""
                INSERT INTO query_log (
                    question, asked_at, top_k_chunk_ids, graph_expanded_ids,
                    answer_returned, insufficient_context
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                question,
                cache.get_last_run().get("completed_at", "") if cache.get_last_run() else "",
                sem_ids,
                exp_ids,
                response.answer,
                0 if response.sufficient_context else 1
            ))
            conn.commit()

"""Vector Retriever component implementing Algorithm 3.5 (FR-5.2)."""

from pathlib import Path
from typing import List, Optional
import chromadb

from chromadb.config import Settings
from codebase_agent.indexing.embedder import Embedder
from codebase_agent.retrieval.models import RetrievedChunk


class VectorRetriever:
    """Performs semantic vector similarity search against ChromaDB collection."""

    def __init__(self, chroma_dir: Path, embedder: Optional[Embedder] = None):
        self.chroma_dir = chroma_dir
        self.embedder = embedder or Embedder()
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="codebase_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query_text: str, top_k: int = 3) -> List[RetrievedChunk]:
        """
        Implements Algorithm 3.5 (semantic_retrieve).
        Fetches top_k nearest neighbor chunks for query_text.
        """
        if self.collection.count() == 0:
            return []

        query_vector = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks: List[RetrievedChunk] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return retrieved_chunks

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            similarity = max(0.0, 1.0 - dist)
            retrieved_chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                document=doc,
                file_path=meta.get("file_path", ""),
                symbol_name=meta.get("symbol_name", ""),
                symbol_type=meta.get("symbol_type", "function"),
                start_line=meta.get("start_line", 1),
                end_line=meta.get("end_line", 1),
                graph_node_id=meta.get("graph_node_id", ""),
                similarity_score=round(similarity, 4),
                source="semantic"
            ))

        # Sort descending by similarity score
        retrieved_chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        return retrieved_chunks

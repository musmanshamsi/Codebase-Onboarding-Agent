"""Graph Expander component implementing Algorithm 3.6 (FR-5.3)."""

from pathlib import Path
from typing import List, Set
import chromadb
from chromadb.config import Settings
import networkx as nx

from codebase_agent.graph.builder import GraphBuilder
from codebase_agent.retrieval.models import RetrievedChunk


class GraphExpander:
    """Enriches semantically retrieved chunks with structurally connected neighborhood code."""

    def __init__(self, chroma_dir: Path):
        self.chroma_dir = chroma_dir
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="codebase_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def expand(
        self,
        semantic_chunks: List[RetrievedChunk],
        G: nx.DiGraph,
        hops: int = 1
    ) -> List[RetrievedChunk]:
        """
        Implements Algorithm 3.6 (graph_expand).
        Traverses 1-hop callers/callees/importers from seed semantic chunks.
        """
        if not semantic_chunks or G.number_of_nodes() == 0:
            return []

        existing_chunk_ids: Set[str] = {c.chunk_id for c in semantic_chunks}
        seed_node_ids: List[str] = [c.graph_node_id for c in semantic_chunks if c.graph_node_id]

        builder = GraphBuilder(G)
        neighbor_node_ids = builder.get_neighborhood(seed_node_ids, hops=hops)

        # Identify neighbor graph node IDs that were not retrieved by semantic search
        new_neighbor_node_ids = [n_id for n_id in neighbor_node_ids if n_id not in seed_node_ids]

        if not new_neighbor_node_ids:
            return []

        expanded_chunks: List[RetrievedChunk] = []

        try:
            # Batch query ChromaDB for chunks matching neighbor graph_node_ids
            res = self.collection.get(
                where={"graph_node_id": {"$in": new_neighbor_node_ids}},
                include=["documents", "metadatas"]
            )

            if res and res.get("ids"):
                ids = res["ids"]
                docs = res["documents"]
                metas = res["metadatas"]

                for chunk_id, doc, meta in zip(ids, docs, metas):
                    if chunk_id in existing_chunk_ids:
                        continue

                    # Give structural neighbor chunks a baseline relevance score tied to seed matches
                    seed_score = max((c.similarity_score for c in semantic_chunks), default=0.5) * 0.8

                    expanded_chunks.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        document=doc,
                        file_path=meta.get("file_path", ""),
                        symbol_name=meta.get("symbol_name", ""),
                        symbol_type=meta.get("symbol_type", "function"),
                        start_line=meta.get("start_line", 1),
                        end_line=meta.get("end_line", 1),
                        graph_node_id=meta.get("graph_node_id", ""),
                        similarity_score=round(seed_score, 4),
                        source="graph_expansion"
                    ))
        except Exception:
            pass

        return expanded_chunks

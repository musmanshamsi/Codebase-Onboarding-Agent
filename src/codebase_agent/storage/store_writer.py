"""Store Writer component (Architecture Section 4.6, Database Design Section 6, FR-4.3, FR-4.4)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings

from codebase_agent.indexing.models import CodeChunk


class StoreWriter:
    """Sole persistence coordinator managing ChromaDB vector store writes and synchronization."""

    def __init__(self, chroma_dir: Path):
        self.chroma_dir = chroma_dir
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "codebase_chunks"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: List[CodeChunk]):
        """
        Upserts code chunks into ChromaDB persistent collection (Database Design Section 3.5).
        """
        if not chunks:
            return

        now_iso = datetime.now(timezone.utc).isoformat()

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in chunks:
            if c.embedding is None:
                continue

            ids.append(c.id)
            embeddings.append(c.embedding)
            documents.append(c.document)
            metadatas.append({
                "file_path": c.file_path,
                "symbol_name": c.symbol_name,
                "symbol_type": c.symbol_type,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "language": c.language,
                "graph_node_id": c.graph_node_id,
                "content_hash": c.content_hash,
                "last_indexed_at": now_iso
            })

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def delete_file_chunks(self, file_path: str):
        """Removes all vector chunks belonging to a deleted file path (Database Design Section 3.5)."""
        try:
            self.collection.delete(where={"file_path": file_path})
        except Exception:
            pass

    def inspect_collection(self) -> Dict[str, Any]:
        """Returns collection stats for CLI store inspect command."""
        count = self.collection.count()
        sample = None
        dimension = None

        if count > 0:
            res = self.collection.get(limit=1, include=["documents", "metadatas", "embeddings"])
            if res and res.get("documents"):
                sample = {
                    "id": res["ids"][0],
                    "document": res["documents"][0],
                    "metadata": res["metadatas"][0]
                }
                if res.get("embeddings") is not None and len(res["embeddings"]) > 0:
                    dimension = len(res["embeddings"][0])

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "dimension": dimension,
            "sample_record": sample,
            "chroma_dir": str(self.chroma_dir)
        }

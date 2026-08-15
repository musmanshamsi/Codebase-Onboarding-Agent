"""Local Embedder component (Architecture Section 4.5, FR-4.2, FR-8.2)."""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from codebase_agent.indexing.models import CodeChunk

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


class Embedder:
    """Generates dense vector representations for code chunks locally via SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        self.model = _MODEL_CACHE[model_name]

    def embed_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Generates vector embeddings for a list of CodeChunk objects in-place."""
        if not chunks:
            return chunks

        documents = [c.document for c in chunks]
        embeddings = self.model.encode(documents, show_progress_bar=False, convert_to_numpy=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        return chunks

    def embed_query(self, query_text: str) -> List[float]:
        """Encodes a natural language query string into query vector."""
        emb = self.model.encode(query_text, show_progress_bar=False, convert_to_numpy=True)
        return emb.tolist()


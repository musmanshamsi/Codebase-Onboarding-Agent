"""FastAPI Web UI Backend Server directly invoking agent core components (FR-7.2, FR-8.1..8.4)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from codebase_agent.config import IngestionConfig
from codebase_agent.generation.receiver import QueryReceiver
from codebase_agent.graph.store import GraphStore
from codebase_agent.ingestion.manager import IngestionManager
from codebase_agent.ingestion.reindexer import Reindexer
from codebase_agent.storage.metadata_cache import MetadataCache


def create_app(repo_root: Optional[Path] = None) -> FastAPI:
    """Factory function creating FastAPI app instance bound to target repository."""
    app = FastAPI(title="Codebase Onboarding Agent Web UI", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    resolved_repo = (repo_root or Path(".")).resolve()
    config = IngestionConfig()
    index_dir = resolved_repo / config.index_dir_name
    db_path = index_dir / "cache.db"
    graph_path = index_dir / "graph" / "repo_graph.graphml"
    chroma_dir = index_dir / "chroma"

    static_dir = Path(__file__).parent / "static"

    class QueryRequest(BaseModel):
        question: str = Field(description="Natural language question")
        model_name: str = Field(default="qwen2.5-coder:1.5b")
        top_k: int = Field(default=3, ge=1, le=10)
        similarity_threshold: float = Field(default=0.38, ge=0.0, le=1.0)
        hops: int = Field(default=1, ge=0, le=3)

    class IndexRequest(BaseModel):
        full: bool = Field(default=False)

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        """Returns indexing status, file count, graph nodes/edges, and last run timestamp."""
        cache = MetadataCache(db_path=db_path)
        last_run = cache.get_last_run()
        indexed_files = cache.get_all_file_states()

        nodes_count = 0
        edges_count = 0
        if graph_path.exists():
            try:
                G = GraphStore.load_graph(graph_path)
                nodes_count = G.number_of_nodes()
                edges_count = G.number_of_edges()
            except Exception:
                pass

        total_symbols = sum(f.get("symbol_count", 0) for f in indexed_files.values())

        return {
            "repo_path": str(resolved_repo),
            "indexed_files_count": len(indexed_files),
            "total_symbols": total_symbols,
            "graph_nodes": nodes_count,
            "graph_edges": edges_count,
            "index_directory": str(index_dir),
            "last_run": last_run
        }

    @app.get("/api/models")
    def get_available_models() -> Dict[str, Any]:
        """Fetches list of available local Ollama models."""
        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=3)
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name") for m in data.get("models", [])]
                return {"models": models or ["qwen2.5-coder:7b", "qwen2.5-coder:1.5b"], "ollama_online": True}
        except Exception:
            pass
        return {"models": ["qwen2.5-coder:7b", "qwen2.5-coder:1.5b"], "ollama_online": False}

    @app.post("/api/index")
    def trigger_index(req: IndexRequest) -> Dict[str, Any]:
        """Triggers repository indexing/re-indexing pipeline."""
        try:
            manager = IngestionManager(repo_target=str(resolved_repo), config=config)
            discovered = manager.discover_files()
            cache = MetadataCache(db_path=db_path)

            if req.full:
                to_process = [
                    {
                        "abs_path": str(abs_p),
                        "rel_path": rel_p,
                        "content_hash": manager.compute_content_hash(abs_p),
                        "language": manager.detect_language(abs_p)
                    }
                    for abs_p, rel_p in discovered
                ]
                to_delete = []
            else:
                to_process, to_delete = manager.compute_incremental_diff(discovered, cache)

            reindexer = Reindexer(repo_root=resolved_repo, config=config)
            summary = reindexer.execute_incremental(
                to_process=to_process,
                to_delete=to_delete,
                discovered_files=discovered
            )
            return {"status": "success", "summary": summary}
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))

    receiver_cache: Dict[tuple, QueryReceiver] = {}

    @app.post("/api/query")
    def query_codebase(req: QueryRequest) -> Dict[str, Any]:
        """Invokes QueryReceiver to synthesize grounded answer with citations."""
        try:
            key = (str(resolved_repo), req.model_name, req.top_k, req.hops, req.similarity_threshold)
            if key not in receiver_cache:
                receiver_cache[key] = QueryReceiver(
                    repo_dir=resolved_repo,
                    model_name=req.model_name,
                    top_k=req.top_k,
                    hops=req.hops,
                    similarity_threshold=req.similarity_threshold
                )
            receiver = receiver_cache[key]
            response = receiver.process_query(req.question)
            return response.model_dump()
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))

    @app.get("/api/graph")
    def get_graph_data() -> Dict[str, Any]:
        """Exports GraphML store as D3/Vis JSON format for network visualization."""
        if not graph_path.exists():
            return {"nodes": [], "links": []}

        try:
            G = GraphStore.load_graph(graph_path)
            nodes = []
            for n_id, data in G.nodes(data=True):
                nodes.append({
                    "id": n_id,
                    "label": n_id.split("::")[-1],
                    "type": data.get("type", "node"),
                    "file": data.get("file_path", "")
                })

            links = []
            for u, v, data in G.edges(data=True):
                links.append({
                    "source": u,
                    "target": v,
                    "type": data.get("relation", "calls")
                })
            return {"nodes": nodes, "links": links}
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex))

    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app

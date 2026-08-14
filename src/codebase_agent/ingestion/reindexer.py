"""Incremental Re-Indexer and Orphan Cleanup Manager (FR-6.1, FR-6.2, NFR-3)."""

from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx

from codebase_agent.config import IngestionConfig
from codebase_agent.graph.builder import GraphBuilder
from codebase_agent.graph.store import GraphStore
from codebase_agent.indexing.chunker import Chunker
from codebase_agent.indexing.embedder import Embedder
from codebase_agent.parser.ast_parser import Parser
from codebase_agent.parser.models import FileParseResult
from codebase_agent.storage.metadata_cache import MetadataCache
from codebase_agent.storage.store_writer import StoreWriter


class Reindexer:
    """Manages incremental file modification/deletion diffs and orphan chunk cleanup."""

    def __init__(self, repo_root: Path, config: Optional[IngestionConfig] = None):
        self.repo_root = repo_root
        self.config = config or IngestionConfig()
        self.index_dir = self.repo_root / self.config.index_dir_name
        self.db_path = self.index_dir / "cache.db"
        self.graph_path = self.index_dir / "graph" / "repo_graph.graphml"
        self.chroma_dir = self.index_dir / "chroma"

    def execute_incremental(
        self,
        to_process: List[Dict[str, str]],
        to_delete: List[str],
        discovered_files: List[Tuple[Path, str]]
    ) -> Dict[str, int]:
        """
        Executes incremental updates: re-parses changed files, evicts orphan chunks,
        updates GraphML nodes/edges, and updates SQLite metadata (FR-6.1, FR-6.2).
        """
        cache = MetadataCache(db_path=self.db_path)
        parser = Parser()
        chunker = Chunker()
        store_writer = StoreWriter(chroma_dir=self.chroma_dir)

        run_id = cache.start_index_run(run_type="incremental")

        # 1. Handle File Deletions (Orphan Eviction)
        for del_rel_path in to_delete:
            store_writer.delete_file_chunks(file_path=del_rel_path)
            cache.delete_file_state(file_path=del_rel_path)

        # 2. Handle Modified & New Files
        all_parse_results: Dict[str, FileParseResult] = {}
        new_chunks = []
        processed_count = 0
        failed_count = 0

        for abs_p, rel_p in discovered_files:
            parse_res = parser.parse_file(file_path=rel_p, repo_root=self.repo_root)
            all_parse_results[rel_p] = parse_res

            if parse_res.parse_status == "success":
                processed_count += 1
            else:
                failed_count += 1

            is_changed = any(item["rel_path"] == rel_p for item in to_process)
            if is_changed and parse_res.parse_status == "success":
                # Evict previous chunks for modified file
                store_writer.delete_file_chunks(file_path=rel_p)

                try:
                    with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                        src_text = f.read()

                    item_info = next(item for item in to_process if item["rel_path"] == rel_p)
                    f_chunks = chunker.chunk_file(
                        file_path=rel_p,
                        source_code=src_text,
                        parse_result=parse_res,
                        content_hash=item_info["content_hash"]
                    )
                    new_chunks.extend(f_chunks)
                except Exception:
                    pass

        # Embed & Upsert new chunks
        if new_chunks:
            embedder = Embedder()
            embedder.embed_chunks(new_chunks)
            store_writer.upsert_chunks(new_chunks)

        # 3. Rebuild & Save Knowledge Graph
        graph_builder = GraphBuilder()
        G = graph_builder.build_graph(all_parse_results)
        GraphStore.save_graph(G, self.graph_path)

        # 4. Update SQLite File Index State
        for item in to_process:
            p_res = all_parse_results.get(item["rel_path"])
            sym_cnt = len(p_res.symbols) if p_res else 0
            p_stat = p_res.parse_status if p_res else "failed"
            p_err = p_res.error_message if p_res else "Parse omitted"

            cache.upsert_file_state(
                file_path=item["rel_path"],
                content_hash=item["content_hash"],
                language=item["language"],
                symbol_count=sym_cnt,
                parse_status=p_stat,
                parse_error=p_err
            )

        cache.finish_index_run(
            run_id=run_id,
            files_processed=processed_count,
            files_failed=failed_count,
            status="completed"
        )

        return {
            "run_id": run_id,
            "processed_files": len(to_process),
            "deleted_files": len(to_delete),
            "new_chunks": len(new_chunks),
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges()
        }

"""Comprehensive unit tests for Phase 7 Reindexer, orphan cleanup, and GitHookInstaller."""

import time
from pathlib import Path
import pytest

from codebase_agent.graph.builder import GraphBuilder
from codebase_agent.graph.store import GraphStore
from codebase_agent.hooks.installer import GitHookInstaller
from codebase_agent.indexing.chunker import Chunker
from codebase_agent.indexing.embedder import Embedder
from codebase_agent.ingestion.manager import IngestionManager
from codebase_agent.ingestion.reindexer import Reindexer
from codebase_agent.parser.ast_parser import Parser
from codebase_agent.storage.metadata_cache import MetadataCache
from codebase_agent.storage.store_writer import StoreWriter


@pytest.fixture
def repo_with_index(tmp_path):
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # Initial file: app.py
    code1 = "def start_app():\n    print('Starting')\n"
    (repo_dir / "app.py").write_text(code1, encoding="utf-8")

    # Initial file: helper.py
    code2 = "def helper_func():\n    return True\n"
    (repo_dir / "helper.py").write_text(code2, encoding="utf-8")

    # Perform initial index
    manager = IngestionManager(repo_target=str(repo_dir))
    discovered = manager.discover_files()
    reindexer = Reindexer(repo_root=repo_dir)

    to_process = [
        {
            "abs_path": str(abs_p),
            "rel_path": rel_p,
            "content_hash": manager.compute_content_hash(abs_p),
            "language": "python"
        }
        for abs_p, rel_p in discovered
    ]

    reindexer.execute_incremental(to_process=to_process, to_delete=[], discovered_files=discovered)
    return repo_dir, reindexer, manager


def test_incremental_file_deletion_and_orphan_cleanup(repo_with_index):
    repo_dir, reindexer, manager = repo_with_index

    # 1. Inspect ChromaDB before deletion
    store_writer = StoreWriter(chroma_dir=reindexer.chroma_dir)
    assert store_writer.inspect_collection()["total_chunks"] == 2

    # 2. Delete helper.py
    (repo_dir / "helper.py").unlink()

    discovered = manager.discover_files()
    assert len(discovered) == 1

    cache = MetadataCache(db_path=reindexer.db_path)
    to_process, to_delete = manager.compute_incremental_diff(discovered, cache)

    assert "helper.py" in to_delete

    # 3. Execute incremental reindex
    summary = reindexer.execute_incremental(to_process=to_process, to_delete=to_delete, discovered_files=discovered)

    assert summary["deleted_files"] == 1
    # Verify ChromaDB chunk evicted
    assert store_writer.inspect_collection()["total_chunks"] == 1

    # Verify GraphML updated
    G = GraphStore.load_graph(reindexer.graph_path)
    assert not G.has_node("helper.py::helper_func")


def test_git_hook_installer(tmp_path):
    repo_dir = tmp_path / "git_repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    installer = GitHookInstaller(repo_root=repo_dir)
    assert installer.is_git_repository() is True
    assert installer.check_status() is False

    # Install
    success, msg = installer.install_hook()
    assert success is True
    assert installer.check_status() is True
    assert installer.hook_file.exists()

    # Uninstall
    u_success, u_msg = installer.uninstall_hook()
    assert u_success is True
    assert installer.check_status() is False


def test_incremental_reindex_performance_benchmark(repo_with_index):
    repo_dir, reindexer, manager = repo_with_index

    # Modify app.py
    (repo_dir / "app.py").write_text("def start_app():\n    print('Updated app')\n", encoding="utf-8")

    start_time = time.time()
    discovered = manager.discover_files()
    cache = MetadataCache(db_path=reindexer.db_path)
    to_process, to_delete = manager.compute_incremental_diff(discovered, cache)
    reindexer.execute_incremental(to_process=to_process, to_delete=to_delete, discovered_files=discovered)
    elapsed = time.time() - start_time

    # Assert execution completed in sub-15 seconds (NFR-3)
    assert elapsed < 15.0, f"Incremental re-indexing took {elapsed:.3f}s, exceeding 15s NFR-3 target!"

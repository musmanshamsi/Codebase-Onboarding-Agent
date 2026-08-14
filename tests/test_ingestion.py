"""Unit tests for Phase 1 Ingestion Manager and Metadata Cache."""

import shutil
import tempfile
from pathlib import Path
import pytest

from codebase_agent.config import IngestionConfig
from codebase_agent.ingestion.manager import IngestionManager
from codebase_agent.storage.metadata_cache import MetadataCache


@pytest.fixture
def temp_repo():
    """Creates a temporary test repository structure."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_repo_"))
    
    # Python source files
    main_py = temp_dir / "main.py"
    main_py.write_text("print('hello')\n", encoding="utf-8")
    
    sub_dir = temp_dir / "pkg"
    sub_dir.mkdir(parents=True)
    mod_py = sub_dir / "mod.py"
    mod_py.write_text("def foo(): pass\n", encoding="utf-8")
    
    # Binary file
    bin_file = temp_dir / "data.bin"
    bin_file.write_bytes(b"\x00\x01\x02HEADER\x00BYTES")

    # Excluded folder file
    node_dir = temp_dir / "node_modules"
    node_dir.mkdir(parents=True)
    (node_dir / "node_file.py").write_text("print('ignored')", encoding="utf-8")

    yield temp_dir
    
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_discovery_and_filtering(temp_repo):
    config = IngestionConfig()
    manager = IngestionManager(repo_target=str(temp_repo), config=config)
    
    discovered = manager.discover_files()
    rel_paths = [rel for _, rel in discovered]

    assert "main.py" in rel_paths
    assert "pkg/mod.py" in rel_paths
    assert "data.bin" not in rel_paths
    assert "node_modules/node_file.py" not in rel_paths


def test_metadata_cache_lifecycle(temp_repo):
    db_path = temp_repo / ".agent_index" / "cache.db"
    cache = MetadataCache(db_path=db_path)
    
    run_id = cache.start_index_run(run_type="full")
    assert run_id > 0
    
    cache.upsert_file_state(
        file_path="main.py",
        content_hash="abc123hash",
        language="python",
        symbol_count=1,
        parse_status="success"
    )
    
    state = cache.get_file_state("main.py")
    assert state is not None
    assert state["content_hash"] == "abc123hash"
    assert state["language"] == "python"

    cache.finish_index_run(run_id=run_id, files_processed=1, files_failed=0, status="completed")
    last_run = cache.get_last_run()
    assert last_run["run_id"] == run_id
    assert last_run["status"] == "completed"

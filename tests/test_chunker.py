"""Comprehensive unit tests for Phase 4 Chunker, Embedder, StoreWriter, and shared Node/Chunk IDs."""

import shutil
import tempfile
from pathlib import Path
import pytest

from codebase_agent.graph.builder import GraphBuilder
from codebase_agent.indexing.chunker import Chunker
from codebase_agent.indexing.embedder import Embedder
from codebase_agent.indexing.models import make_chunk_id, make_graph_node_id
from codebase_agent.parser.ast_parser import Parser
from codebase_agent.storage.store_writer import StoreWriter


@pytest.fixture
def sample_code_file(tmp_path):
    srv_dir = tmp_path / "services"
    srv_dir.mkdir(parents=True, exist_ok=True)
    code = '''def process_payment(order_id: str, amount: float) -> bool:
    print("Processing payment")
    return True

def charge_card(order_id: str, amount: float) -> bool:
    print("Charging card")
    return True
'''
    file_path = srv_dir / "payment.py"
    file_path.write_text(code, encoding="utf-8")
    return file_path, tmp_path


def test_shared_node_id_consistency(sample_code_file):
    """
    Verifies that Chunker and GraphBuilder generate identical graph_node_ids
    for foreign key joinability between vector store metadata and graph nodes.
    """
    file_path, repo_root = sample_code_file
    rel_path = "services/payment.py"

    parser = Parser()
    parse_result = parser.parse_file(rel_path, repo_root=repo_root)
    assert parse_result.parse_status == "success"

    # 1. Build Graph Node
    builder = GraphBuilder()
    G = builder.build_graph({rel_path: parse_result})

    # 2. Build Chunks
    chunker = Chunker()
    chunks = chunker.chunk_file(
        file_path=rel_path,
        source_code=file_path.read_text(),
        parse_result=parse_result,
        content_hash="dummyhash"
    )

    assert len(chunks) == 2
    for chunk in chunks:
        # Assert chunk.graph_node_id exists directly as a node in the graph
        assert G.has_node(chunk.graph_node_id), f"Graph node ID '{chunk.graph_node_id}' missing in Graph!"


def test_chunker_sub_chunking(tmp_path):
    # Generate a large function exceeding max_chunk_lines=10
    large_func_lines = ["def large_function():\n"] + [f"    x = {i}\n" for i in range(25)]
    code = "".join(large_func_lines)

    file_path = tmp_path / "large.py"
    file_path.write_text(code, encoding="utf-8")

    parser = Parser()
    parse_res = parser.parse_file("large.py", repo_root=tmp_path)

    chunker = Chunker(max_chunk_lines=10, overlap_lines=2)
    chunks = chunker.chunk_file("large.py", code, parse_res, "hash123")

    assert len(chunks) > 1
    assert "part1" in chunks[0].id
    assert chunks[0].graph_node_id == "large.py::large_function"


def test_store_writer_chroma_persistence(tmp_path):
    chroma_dir = tmp_path / "chroma"
    writer = StoreWriter(chroma_dir=chroma_dir)

    # Inspect empty collection
    info = writer.inspect_collection()
    assert info["total_chunks"] == 0

    # Embed and upsert chunk
    chunker = Chunker()
    parser = Parser()
    code = "def foo(): pass"
    f_path = tmp_path / "foo.py"
    f_path.write_text(code)
    p_res = parser.parse_file("foo.py", repo_root=tmp_path)

    chunks = chunker.chunk_file("foo.py", code, p_res, "h1")
    embedder = Embedder()
    embedder.embed_chunks(chunks)

    writer.upsert_chunks(chunks)

    info_after = writer.inspect_collection()
    assert info_after["total_chunks"] == 1
    assert info_after["dimension"] == 384
    assert info_after["sample_record"]["metadata"]["symbol_name"] == "foo"

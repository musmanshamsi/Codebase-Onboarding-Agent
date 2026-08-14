"""Comprehensive unit tests for Phase 5 VectorRetriever, GraphExpander, and ContextAssembler."""

from pathlib import Path
import networkx as nx
import pytest

from codebase_agent.graph.builder import GraphBuilder
from codebase_agent.indexing.chunker import Chunker
from codebase_agent.indexing.embedder import Embedder
from codebase_agent.parser.ast_parser import Parser
from codebase_agent.retrieval.context_assembler import ContextAssembler
from codebase_agent.retrieval.graph_expander import GraphExpander
from codebase_agent.retrieval.models import RetrievedChunk
from codebase_agent.retrieval.vector_retriever import VectorRetriever
from codebase_agent.storage.store_writer import StoreWriter


@pytest.fixture
def indexed_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    # File 1: services/payment.py
    srv_dir = repo_dir / "services"
    srv_dir.mkdir()
    payment_code = '''def process_payment(order_id: str, amount: float) -> bool:
    print("Processing payment")
    return charge_card(order_id, amount)

def charge_card(order_id: str, amount: float) -> bool:
    print("Charging card")
    return True
'''
    (srv_dir / "payment.py").write_text(payment_code, encoding="utf-8")

    # File 2: routes/checkout.py
    routes_dir = repo_dir / "routes"
    routes_dir.mkdir()
    checkout_code = '''from ..services.payment import process_payment

def checkout_handler(order_id: str):
    process_payment(order_id, 99.99)
'''
    (routes_dir / "checkout.py").write_text(checkout_code, encoding="utf-8")

    parser = Parser()
    chunker = Chunker()
    embedder = Embedder()

    all_parse = {
        "services/payment.py": parser.parse_file("services/payment.py", repo_root=repo_dir),
        "routes/checkout.py": parser.parse_file("routes/checkout.py", repo_root=repo_dir)
    }

    builder = GraphBuilder()
    G = builder.build_graph(all_parse)

    chroma_dir = repo_dir / ".agent_index" / "chroma"
    writer = StoreWriter(chroma_dir=chroma_dir)

    all_chunks = []
    for rel_p, p_res in all_parse.items():
        abs_p = repo_dir / rel_p
        chunks = chunker.chunk_file(rel_p, abs_p.read_text(), p_res, "hash123")
        all_chunks.extend(chunks)

    embedder.embed_chunks(all_chunks)
    writer.upsert_chunks(all_chunks)

    return repo_dir, chroma_dir, G


def test_vector_retriever_ranking(indexed_repo):
    repo_dir, chroma_dir, G = indexed_repo
    retriever = VectorRetriever(chroma_dir=chroma_dir)

    # Query about payments
    chunks = retriever.retrieve("How to process payments?", top_k=2)
    assert len(chunks) == 2
    assert chunks[0].similarity_score >= chunks[1].similarity_score
    assert any("payment" in c.symbol_name.lower() for c in chunks)


def test_graph_expander_neighbors(indexed_repo):
    repo_dir, chroma_dir, G = indexed_repo
    retriever = VectorRetriever(chroma_dir=chroma_dir)
    expander = GraphExpander(chroma_dir=chroma_dir)

    # Seed chunk: services/payment.py::charge_card
    seed_chunks = retriever.retrieve("charge card", top_k=1)
    assert len(seed_chunks) == 1

    expanded = expander.expand(seed_chunks, G=G, hops=1)
    # 1-hop neighbor process_payment should be pulled in
    assert len(expanded) >= 1
    expanded_sources = {c.source for c in expanded}
    assert "graph_expansion" in expanded_sources


def test_context_assembler_budget_and_sufficiency():
    assembler = ContextAssembler(max_context_tokens=100, similarity_threshold=0.3)

    # 1. Test Sufficiency Pass
    sem_chunk = RetrievedChunk(
        chunk_id="c1",
        document="def process_payment(): pass",
        file_path="payment.py",
        symbol_name="process_payment",
        symbol_type="function",
        start_line=1,
        end_line=2,
        graph_node_id="payment.py::process_payment",
        similarity_score=0.75,
        source="semantic"
    )
    res_pass = assembler.assemble("payment question", [sem_chunk], [])
    assert res_pass.sufficient_context is True
    assert len(res_pass.final_context_chunks) == 1

    # 2. Test Sufficiency Fail (FR-5.6)
    weak_chunk = RetrievedChunk(
        chunk_id="c2",
        document="print('unrelated')",
        file_path="main.py",
        symbol_name="main",
        symbol_type="function",
        start_line=1,
        end_line=2,
        graph_node_id="main.py::main",
        similarity_score=0.12,
        source="semantic"
    )
    res_fail = assembler.assemble("quantum baking question", [weak_chunk], [])
    assert res_fail.sufficient_context is False
    assert res_fail.message is not None
    assert "Insufficient relevant code context" in res_fail.message

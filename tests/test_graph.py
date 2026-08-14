"""Comprehensive unit tests for Phase 3 Knowledge Graph Builder, Resolver, and GraphStore."""

import shutil
import tempfile
from pathlib import Path
import pytest

from codebase_agent.graph.builder import GraphBuilder
from codebase_agent.graph.resolver import CallResolver
from codebase_agent.graph.store import GraphStore
from codebase_agent.parser.ast_parser import Parser


@pytest.fixture
def sample_parsed_repo(tmp_path):
    """Creates a multi-file Python repository structure and parses it."""
    repo_dir = tmp_path / "graph_repo"
    repo_dir.mkdir()

    # File 1: services/payment.py
    srv_dir = repo_dir / "services"
    srv_dir.mkdir()
    payment_code = '''def process_payment(order_id: str, amount: float) -> bool:
    return charge_card(order_id, amount)

def charge_card(order_id: str, amount: float) -> bool:
    print("Charged")
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
    all_parse = {
        "services/payment.py": parser.parse_file("services/payment.py", repo_root=repo_dir),
        "routes/checkout.py": parser.parse_file("routes/checkout.py", repo_root=repo_dir)
    }

    return repo_dir, all_parse


def test_two_pass_graph_construction(sample_parsed_repo):
    repo_dir, all_parse = sample_parsed_repo
    builder = GraphBuilder()
    G = builder.build_graph(all_parse)

    # FR-3.1: Check nodes exist for files and functions
    assert G.has_node("services/payment.py")
    assert G.has_node("routes/checkout.py")
    assert G.has_node("services/payment.py::process_payment")
    assert G.has_node("services/payment.py::charge_card")
    assert G.has_node("routes/checkout.py::checkout_handler")

    # FR-3.2 & FR-3.3: Check edge relations and resolution across files
    # 1. defined_in edge
    assert G.has_edge("services/payment.py::process_payment", "services/payment.py")
    assert G.get_edge_data("services/payment.py::process_payment", "services/payment.py")["relation"] == "defined_in"

    # 2. imports edge (routes/checkout.py -> services/payment.py)
    assert G.has_edge("routes/checkout.py", "services/payment.py")
    assert G.get_edge_data("routes/checkout.py", "services/payment.py")["relation"] == "imports"

    # 3. resolved calls edge across files (checkout_handler -> process_payment)
    assert G.has_edge("routes/checkout.py::checkout_handler", "services/payment.py::process_payment")
    assert G.get_edge_data("routes/checkout.py::checkout_handler", "services/payment.py::process_payment")["resolved"] is True

    # 4. resolved calls edge within same file (process_payment -> charge_card)
    assert G.has_edge("services/payment.py::process_payment", "services/payment.py::charge_card")
    assert G.get_edge_data("services/payment.py::process_payment", "services/payment.py::charge_card")["resolved"] is True


def test_graph_queries(sample_parsed_repo):
    repo_dir, all_parse = sample_parsed_repo
    builder = GraphBuilder()
    G = builder.build_graph(all_parse)

    # FR-3.5: Query callers
    callers = builder.get_callers("services/payment.py::process_payment")
    caller_ids = [c_id for c_id, _ in callers]
    assert "routes/checkout.py::checkout_handler" in caller_ids

    # Query callees
    callees = builder.get_callees("services/payment.py::process_payment")
    callee_ids = [c_id for c_id, _ in callees]
    assert "services/payment.py::charge_card" in callee_ids

    # Query 1-hop neighborhood
    neighborhood = builder.get_neighborhood(["services/payment.py::process_payment"], hops=1)
    assert "routes/checkout.py::checkout_handler" in neighborhood
    assert "services/payment.py::charge_card" in neighborhood
    assert "services/payment.py" in neighborhood


def test_graphml_persistence(sample_parsed_repo, tmp_path):
    repo_dir, all_parse = sample_parsed_repo
    builder = GraphBuilder()
    G = builder.build_graph(all_parse)

    graph_file = tmp_path / "repo_graph.graphml"
    GraphStore.save_graph(G, graph_file)
    assert graph_file.exists()

    loaded_G = GraphStore.load_graph(graph_file)
    assert loaded_G.number_of_nodes() == G.number_of_nodes()
    assert loaded_G.number_of_edges() == G.number_of_edges()
    assert loaded_G.has_edge("routes/checkout.py::checkout_handler", "services/payment.py::process_payment")

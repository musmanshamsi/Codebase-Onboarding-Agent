"""Comprehensive unit tests for Phase 8 Web UI FastAPI Server."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from codebase_agent.ui.server import create_app


@pytest.fixture
def test_client(tmp_path):
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("def main(): pass", encoding="utf-8")

    app = create_app(repo_root=repo_dir)
    return TestClient(app), repo_dir


def test_ui_status_endpoint(test_client):
    client, repo_dir = test_client
    res = client.get("/api/status")

    assert res.status_code == 200
    data = res.json()
    assert "repo_path" in data
    assert data["repo_path"] == str(repo_dir.resolve())
    assert "graph_nodes" in data


def test_ui_models_endpoint(test_client):
    client, _ = test_client
    res = client.get("/api/models")

    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_ui_graph_endpoint(test_client):
    client, _ = test_client
    res = client.get("/api/graph")

    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "links" in data


def test_ui_query_endpoint_missing_index(test_client):
    client, _ = test_client
    payload = {
        "question": "How to start app?",
        "model_name": "qwen2.5-coder:1.5b",
        "top_k": 3,
        "similarity_threshold": 0.3
    }
    res = client.post("/api/query", json=payload)

    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "sufficient_context" in data
    assert data["sufficient_context"] is False

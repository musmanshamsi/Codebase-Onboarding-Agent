"""Comprehensive unit tests for Phase 6 LLMGenerator, CitationFormatter, and QueryReceiver."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from codebase_agent.generation.citation_formatter import CitationFormatter
from codebase_agent.generation.generator import LLMGenerator, SYSTEM_PROMPT
from codebase_agent.generation.models import QueryResponse
from codebase_agent.generation.receiver import QueryReceiver
from codebase_agent.retrieval.models import RetrievedChunk
from codebase_agent.storage.metadata_cache import MetadataCache


@pytest.fixture
def sample_chunks():
    return [
        RetrievedChunk(
            chunk_id="services/payment.py::process_payment::3",
            document="def process_payment(order_id, amount):\n    return charge_card(order_id, amount)",
            file_path="services/payment.py",
            symbol_name="process_payment",
            symbol_type="function",
            start_line=3,
            end_line=5,
            graph_node_id="services/payment.py::process_payment",
            similarity_score=0.85,
            source="semantic"
        ),
        RetrievedChunk(
            chunk_id="services/payment.py::charge_card::7",
            document="def charge_card(order_id, amount):\n    print('Charged')",
            file_path="services/payment.py",
            symbol_name="charge_card",
            symbol_type="function",
            start_line=7,
            end_line=9,
            graph_node_id="services/payment.py::charge_card",
            similarity_score=0.72,
            source="graph_expansion"
        )
    ]


def test_prompt_formatting_and_grounding_rules(sample_chunks):
    generator = LLMGenerator(model_name="qwen2.5-coder:7b")
    prompt = generator.format_context_prompt("How does payment work?", sample_chunks)

    assert "User Question: How does payment work?" in prompt
    assert "services/payment.py:3-5" in prompt
    assert "services/payment.py:7-9" in prompt
    assert "def process_payment" in prompt
    assert "SYSTEM_PROMPT" in globals()
    assert "Use ONLY the provided code snippets" in SYSTEM_PROMPT


def test_citation_formatter(sample_chunks):
    citations = CitationFormatter.format_citations(sample_chunks)
    assert len(citations) == 2
    assert citations[0].file_path == "services/payment.py"
    assert citations[0].start_line == 3
    assert citations[0].end_line == 5
    assert citations[0].symbol_name == "process_payment"

    resp = QueryResponse(
        query="How to charge cards?",
        answer="process_payment calls charge_card.",
        citations=citations,
        sufficient_context=True,
        model_name="qwen2.5-coder:7b"
    )
    rendered = CitationFormatter.render_output(resp)
    assert "=== Answer ===" in rendered
    assert "process_payment calls charge_card." in rendered
    assert "[1] services/payment.py:3-5 (Symbol: process_payment)" in rendered
    assert "[2] services/payment.py:7-9 (Symbol: charge_card)" in rendered


@patch("requests.post")
def test_generator_mocked_ollama_success(mock_post, sample_chunks):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "process_payment invokes charge_card to complete transaction."}
    }
    mock_post.return_value = mock_response

    generator = LLMGenerator(model_name="qwen2.5-coder:7b")
    answer = generator.generate_answer("How to charge?", sample_chunks)

    assert "process_payment invokes charge_card" in answer
    assert mock_post.called


def test_query_receiver_sqlite_logging(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    index_dir = repo_dir / ".agent_index"
    index_dir.mkdir()

    db_path = index_dir / "cache.db"
    cache = MetadataCache(db_path=db_path)

    # Test logging query
    receiver = QueryReceiver(repo_dir=repo_dir)
    resp = QueryResponse(
        query="Test query?",
        answer="Test answer",
        citations=[],
        sufficient_context=True
    )
    receiver._log_query(cache, "Test query?", [], [], resp)

    with cache._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM query_log")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["question"] == "Test query?"
        assert rows[0]["answer_returned"] == "Test answer"

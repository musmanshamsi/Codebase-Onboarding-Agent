"""Comprehensive unit tests for Phase 2 Parser, Extractors, and ModuleResolver."""

import shutil
import tempfile
from pathlib import Path
import pytest

from codebase_agent.parser.ast_parser import Parser
from codebase_agent.parser.models import SymbolType, ImportStatement
from codebase_agent.parser.module_resolver import ModuleResolver
from codebase_agent.parser.python_extractor import PythonExtractor


@pytest.fixture
def sample_code_file(tmp_path):
    code = '''"""Sample module for parser testing."""

import os
from services.payment import process_payment as pay, refund_payment
from ..models.order import Order

class PaymentProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def process(self, order_id: str, amount: float) -> bool:
        print("Processing...")
        res = pay(order_id, amount)
        self.log_transaction(order_id)
        return res

    def log_transaction(self, order_id: str):
        pass

async def async_checkout(order_id: str):
    proc = PaymentProcessor("key_123")
    proc.process(order_id, 100.0)
'''
    file_path = tmp_path / "processor.py"
    file_path.write_text(code, encoding="utf-8")
    return file_path


def test_python_extractor_symbols_and_lines(sample_code_file):
    parser = Parser()
    result = parser.parse_file(file_path=str(sample_code_file))

    assert result.parse_status == "success"
    symbols = result.symbols

    symbol_names = {s.name: s for s in symbols}
    assert "PaymentProcessor" in symbol_names
    assert symbol_names["PaymentProcessor"].type == SymbolType.CLASS

    assert "process" in symbol_names
    assert symbol_names["process"].type == SymbolType.METHOD
    assert symbol_names["process"].parent_symbol == "PaymentProcessor"

    assert "async_checkout" in symbol_names
    assert symbol_names["async_checkout"].type == SymbolType.FUNCTION

    # Check 1-indexed line numbers
    assert symbol_names["PaymentProcessor"].start_line == 7
    assert symbol_names["process"].start_line == 11


def test_python_extractor_imports(sample_code_file):
    parser = Parser()
    result = parser.parse_file(file_path=str(sample_code_file))

    imports = result.imports
    assert len(imports) >= 2

    # Verify alias mapping
    from_payment = next(i for i in imports if "payment" in i.source_module)
    assert "process_payment" in from_payment.imported_symbols
    assert from_payment.alias_map.get("process_payment") == "pay"

    # Verify relative import (from ..models.order import Order)
    from_models = next(i for i in imports if i.is_relative)
    assert from_models.level == 2
    assert from_models.source_module == "models.order"


def test_python_extractor_call_sites(sample_code_file):
    parser = Parser()
    result = parser.parse_file(file_path=str(sample_code_file))

    call_sites = result.call_sites
    call_names = [cs.function_name for cs in call_sites]

    assert "pay" in call_names
    assert "log_transaction" in call_names
    assert "process" in call_names


def test_module_resolver():
    known_files = {
        "app/services/payment.py",
        "app/models/order.py",
        "app/models/__init__.py"
    }

    # Test absolute import resolution
    imp1 = ImportStatement(
        source_module="app.services.payment",
        imported_symbols=["process_payment"],
        line_number=1
    )
    resolved1 = ModuleResolver.resolve_import_to_file(
        calling_file_path="app/routes/checkout.py",
        import_stmt=imp1,
        known_files=known_files
    )
    assert resolved1 == "app/services/payment.py"

    # Test relative import resolution (from ..models.order import Order from app/services/handler.py)
    imp2 = ImportStatement(
        source_module="models.order",
        imported_symbols=["Order"],
        is_relative=True,
        level=2,
        line_number=2
    )
    resolved2 = ModuleResolver.resolve_import_to_file(
        calling_file_path="app/services/handler.py",
        import_stmt=imp2,
        known_files=known_files
    )
    assert resolved2 == "app/models/order.py"


def test_fault_isolation_malformed_file(tmp_path):
    parser = Parser()

    # Non-existent file
    res1 = parser.parse_file(file_path=str(tmp_path / "non_existent.py"))
    assert res1.parse_status == "failed"
    assert "File not found" in res1.error_message

    # Unsupported extension file existing on disk
    doc_file = tmp_path / "document.txt"
    doc_file.write_text("Sample plain text content", encoding="utf-8")
    res2 = parser.parse_file(file_path=str(doc_file))
    assert res2.parse_status == "failed"
    assert "Unsupported language" in res2.error_message


def test_syntax_error_graceful_recovery(tmp_path):
    parser = Parser()
    broken_code = """def good_function():
    return 42

def bad_function(
    print("missing closing paren")
"""
    broken_file = tmp_path / "broken.py"
    broken_file.write_text(broken_code, encoding="utf-8")

    result = parser.parse_file(file_path=str(broken_file))
    assert result.parse_status == "success"
    # Verify good_function was extracted despite syntax error further down
    symbol_names = [s.name for s in result.symbols]
    assert "good_function" in symbol_names

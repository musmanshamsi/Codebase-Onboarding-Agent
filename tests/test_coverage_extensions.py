"""Coverage Extension Unit Tests targeting previously uncovered branches."""

import os
from pathlib import Path
from codebase_agent.hooks.installer import GitHookInstaller
from codebase_agent.retrieval.context_assembler import ContextAssembler
from codebase_agent.retrieval.models import RetrievedChunk
from codebase_agent.graph.resolver import CallResolver
from codebase_agent.parser.models import CallSite, FileParseResult, ImportStatement, Symbol, SymbolType


def test_installer_non_git_repo(tmp_path: Path):
    installer = GitHookInstaller(repo_root=tmp_path)
    success, msg = installer.install_hook()
    assert success is False
    assert "not a Git repository" in msg


def test_installer_uninstall_and_status(tmp_path: Path):
    git_dir = tmp_path / ".git" / "hooks"
    git_dir.mkdir(parents=True)

    installer = GitHookInstaller(repo_root=tmp_path)

    # 1. Uninstall when not installed
    success, msg = installer.uninstall_hook()
    assert success is True
    assert "is not installed" in msg
    assert installer.check_status() is False

    # 2. Install then uninstall
    install_ok, _ = installer.install_hook()
    assert install_ok is True
    assert installer.check_status() is True

    uninstall_ok, uninst_msg = installer.uninstall_hook()
    assert uninstall_ok is True
    assert "Successfully uninstalled" in uninst_msg
    assert installer.check_status() is False


def test_context_assembler_token_budget_overflow():
    assembler = ContextAssembler(max_context_tokens=50, similarity_threshold=0.3)

    # Each chunk has document of 120 chars (~30 tokens)
    chunk1 = RetrievedChunk(
        chunk_id="chunk_1",
        document="A" * 120,
        file_path="a.py",
        symbol_name="sym1",
        symbol_type="function",
        start_line=1,
        end_line=10,
        graph_node_id="a.py::sym1",
        similarity_score=0.9,
        source="semantic"
    )

    chunk2 = RetrievedChunk(
        chunk_id="chunk_2",
        document="B" * 120,
        file_path="b.py",
        symbol_name="sym2",
        symbol_type="function",
        start_line=1,
        end_line=10,
        graph_node_id="b.py::sym2",
        similarity_score=0.8,
        source="semantic"
    )

    result = assembler.assemble(
        query="test query",
        semantic_chunks=[chunk1, chunk2],
        expanded_chunks=[]
    )

    assert result.sufficient_context is True
    assert len(result.final_context_chunks) == 1
    assert result.final_context_chunks[0].chunk_id == "chunk_1"
    assert result.total_tokens <= 50


def test_call_resolver_uncovered_branches():
    # 1. Missing caller parse result
    target_id, resolved = CallResolver.resolve_call(
        caller_file_path="nonexistent.py",
        call_site=CallSite(line_number=5, function_name="foo"),
        all_parse_results={},
        known_files=set()
    )
    assert resolved is False
    assert target_id == "foo"

    # 2. Import with alias match
    caller_parse = FileParseResult(
        file_path="caller.py",
        language="python",
        symbols=[],
        imports=[
            ImportStatement(
                line_number=1,
                source_module="target_mod",
                imported_symbols=["original_func"],
                alias_map={"original_func": "aliased_func"}
            )
        ],
        call_sites=[CallSite(line_number=10, function_name="aliased_func")]
    )

    target_parse = FileParseResult(
        file_path="target_mod.py",
        language="python",
        symbols=[
            Symbol(
                id="target_mod.py::original_func",
                name="original_func",
                type=SymbolType.FUNCTION,
                file_path="target_mod.py",
                start_line=1,
                end_line=10
            )
        ],
        imports=[],
        call_sites=[]
    )

    all_parses = {"caller.py": caller_parse, "target_mod.py": target_parse}
    known_files = {"caller.py", "target_mod.py"}

    target_id, resolved = CallResolver.resolve_call(
        caller_file_path="caller.py",
        call_site=CallSite(line_number=10, function_name="aliased_func"),
        all_parse_results=all_parses,
        known_files=known_files
    )
    assert resolved is True
    assert target_id == "target_mod.py::original_func"

    # 3. Global unique symbol fallback
    unique_parse = FileParseResult(
        file_path="unique_file.py",
        language="python",
        symbols=[
            Symbol(
                id="unique_file.py::unique_global_symbol",
                name="unique_global_symbol",
                type=SymbolType.FUNCTION,
                file_path="unique_file.py",
                start_line=1,
                end_line=5
            )
        ],
        imports=[],
        call_sites=[]
    )
    all_parses_unique = {"caller.py": caller_parse, "unique_file.py": unique_parse}

    target_id, resolved = CallResolver.resolve_call(
        caller_file_path="caller.py",
        call_site=CallSite(line_number=15, function_name="unique_global_symbol"),
        all_parse_results=all_parses_unique,
        known_files={"caller.py", "unique_file.py"}
    )
    assert resolved is True
    assert target_id == "unique_file.py::unique_global_symbol"

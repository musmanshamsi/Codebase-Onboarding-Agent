"""CLI entry point for Codebase Onboarding Agent (FR-7.1)."""

import json
import sys
import time
from pathlib import Path
import click

from codebase_agent.config import IngestionConfig
from codebase_agent.graph.builder import GraphBuilder
from codebase_agent.graph.store import GraphStore
from codebase_agent.ingestion.manager import IngestionManager
from codebase_agent.parser.ast_parser import Parser
from codebase_agent.storage.metadata_cache import MetadataCache


@click.group()
def main():
    """Codebase Onboarding Agent - Semantic + Structural Repository Intelligence."""
    pass


@main.command()
@click.option("--repo", "-r", default=".", help="Path to local repository or remote Git URL.")
@click.option("--dry-run", is_flag=True, help="Perform discovery and parsing without updating database or graph.")
@click.option("--full", is_flag=True, help="Force full re-indexing instead of incremental diff.")
def index(repo: str, dry_run: bool, full: bool):
    """Index a code repository (Phases 1-3: Ingestion, Parsing, Graph)."""
    start_time = time.time()
    click.echo(f"=== Codebase Indexing (Phases 1-3) ===")
    click.echo(f"Target repository: {repo}")
    click.echo(f"Mode: {'Dry-Run' if dry_run else 'Full Index' if full else 'Incremental'}")

    config = IngestionConfig()
    try:
        manager = IngestionManager(repo_target=repo, config=config)
    except Exception as e:
        click.echo(f"Error initializing IngestionManager: {e}", err=True)
        sys.exit(1)

    try:
        repo_root = manager.repo_path
        click.echo(f"Resolved root: {repo_root}")

        index_dir = repo_root / config.index_dir_name
        db_path = index_dir / "cache.db"
        graph_path = index_dir / "graph" / "repo_graph.graphml"

        cache = MetadataCache(db_path=db_path)
        parser = Parser()

        discovered = manager.discover_files()
        click.echo(f"Discovered source files: {len(discovered)}")

        if full:
            to_process = [
                {
                    "abs_path": str(abs_p),
                    "rel_path": rel_p,
                    "content_hash": manager.compute_content_hash(abs_p),
                    "language": manager.detect_language(abs_p)
                }
                for abs_p, rel_p in discovered
            ]
            to_delete = []
        else:
            to_process, to_delete = manager.compute_incremental_diff(discovered, cache)

        click.echo(f"Files to process (new/changed): {len(to_process)}")
        click.echo(f"Files to remove: {len(to_delete)}")

        # --- Phase 2: AST Parsing across all discovered files ---
        all_parse_results = {}
        processed_count = 0
        failed_count = 0

        for abs_p, rel_p in discovered:
            res = parser.parse_file(file_path=rel_p, repo_root=repo_root)
            all_parse_results[rel_p] = res
            if res.parse_status == "success":
                processed_count += 1
            else:
                failed_count += 1

        total_symbols = sum(len(r.symbols) for r in all_parse_results.values())
        click.echo(f"Parsed files: {len(all_parse_results)} ({total_symbols} symbols extracted)")

        # --- Phase 3: Graph Construction ---
        graph_builder = GraphBuilder()
        G = graph_builder.build_graph(all_parse_results)
        click.echo(f"Knowledge Graph Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if not dry_run:
            run_id = cache.start_index_run(run_type="full" if full else "incremental")

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

            for del_path in to_delete:
                cache.delete_file_state(del_path)

            cache.finish_index_run(
                run_id=run_id,
                files_processed=processed_count,
                files_failed=failed_count,
                status="completed"
            )

            # Persist GraphML
            GraphStore.save_graph(G, graph_path)

            click.echo("\n--- Indexing Output Summary ---")
            click.echo(f"Run ID: {run_id}")
            click.echo(f"SQLite Cache: {db_path}")
            click.echo(f"GraphML Store: {graph_path}")

        elapsed = time.time() - start_time
        click.echo(f"\nIndexing pipeline completed in {elapsed:.3f} seconds.")

    finally:
        manager.cleanup()


@main.command()
@click.option("--file", "-f", required=True, help="Path to source code file to parse.")
@click.option("--json-out", is_flag=True, help="Output raw JSON structure.")
def parse(file: str, json_out: bool):
    """Parse a single source file and display extracted symbols, imports, and call sites (FR-2)."""
    parser = Parser()
    result = parser.parse_file(file_path=file)

    if json_out:
        click.echo(result.model_dump_json(indent=2))
        return

    click.echo(f"=== Parse Results: {file} ===")
    click.echo(f"Status: {result.parse_status}")
    if result.parse_status == "failed":
        click.echo(f"Error: {result.error_message}", err=True)
        return

    click.echo(f"\nSymbols Extracted ({len(result.symbols)}):")
    for sym in result.symbols:
        click.echo(f"  - [{sym.type.value.upper()}] {sym.name} (Lines {sym.start_line}-{sym.end_line}) -> ID: {sym.id}")

    click.echo(f"\nImport Statements ({len(result.imports)}):")
    for imp in result.imports:
        symbols_str = ", ".join(imp.imported_symbols) if imp.imported_symbols else "*"
        alias_str = f" as {imp.alias_map}" if imp.alias_map else ""
        rel_str = f" (relative level {imp.level})" if imp.is_relative else ""
        click.echo(f"  - Line {imp.line_number}: from '{imp.source_module}' import [{symbols_str}]{alias_str}{rel_str}")

    click.echo(f"\nCall Sites ({len(result.call_sites)}):")
    for cs in result.call_sites:
        caller = cs.caller_symbol_id or "global"
        click.echo(f"  - Line {cs.line_number}: call {cs.function_name}() inside caller '{caller}' (args: {cs.args_count})")


@main.group()
def graph():
    """Knowledge Graph Subcommands (FR-3.5)."""
    pass


@graph.command("query")
@click.option("--repo", "-r", default=".", help="Path to repository.")
@click.option("--node", "-n", required=True, help="Graph node ID to query e.g. 'main.py::main'.")
@click.option("--relation", "-rel", type=click.Choice(["callers", "callees", "imports", "neighborhood"]), default="callers", help="Relationship traversal type.")
def graph_query(repo: str, node: str, relation: str):
    """Query structural graph relationships (callers, callees, imports, neighborhood) (FR-3.5)."""
    config = IngestionConfig()
    graph_path = Path(repo) / config.index_dir_name / "graph" / "repo_graph.graphml"

    if not graph_path.exists():
        click.echo(f"GraphML store not found at {graph_path}. Please run 'index' first.", err=True)
        sys.exit(1)

    G = GraphStore.load_graph(graph_path)
    builder = GraphBuilder(G)

    click.echo(f"=== Graph Query: Node '{node}' ({relation.upper()}) ===")

    if not G.has_node(node):
        click.echo(f"Node '{node}' not found in knowledge graph.", err=True)
        click.echo(f"Available sample nodes: {list(G.nodes)[:5]}")
        return

    if relation == "callers":
        callers = builder.get_callers(node)
        click.echo(f"Direct Callers ({len(callers)}):")
        for caller_id, data in callers:
            click.echo(f"  - {caller_id} (Type: {data.get('type')}, File: {data.get('file_path')})")

    elif relation == "callees":
        callees = builder.get_callees(node)
        click.echo(f"Direct Callees ({len(callees)}):")
        for callee_id, data in callees:
            click.echo(f"  - {callee_id} (Type: {data.get('type')}, File: {data.get('file_path')})")

    elif relation == "imports":
        imports = builder.get_imports(node)
        click.echo(f"Imported Files ({len(imports)}):")
        for imp_id, data in imports:
            click.echo(f"  - {imp_id} (Language: {data.get('language')})")

    elif relation == "neighborhood":
        neighbors = builder.get_neighborhood([node], hops=1)
        click.echo(f"1-Hop Structural Neighborhood ({len(neighbors)} nodes):")
        for n_id in sorted(neighbors):
            n_data = G.nodes.get(n_id, {})
            click.echo(f"  - {n_id} (Type: {n_data.get('type', 'unknown')})")


if __name__ == "__main__":
    main()

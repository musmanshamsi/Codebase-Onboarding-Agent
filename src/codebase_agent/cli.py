"""CLI entry point for Codebase Onboarding Agent (FR-7.1)."""

import json
import sys
import time
from pathlib import Path
import click

from codebase_agent.config import IngestionConfig
from codebase_agent.generation.citation_formatter import CitationFormatter
from codebase_agent.generation.receiver import QueryReceiver
from codebase_agent.graph.builder import GraphBuilder
from codebase_agent.graph.store import GraphStore
from codebase_agent.hooks.installer import GitHookInstaller
from codebase_agent.indexing.chunker import Chunker
from codebase_agent.indexing.embedder import Embedder
from codebase_agent.ingestion.manager import IngestionManager
from codebase_agent.ingestion.reindexer import Reindexer
from codebase_agent.parser.ast_parser import Parser
from codebase_agent.retrieval.context_assembler import ContextAssembler
from codebase_agent.retrieval.graph_expander import GraphExpander
from codebase_agent.retrieval.vector_retriever import VectorRetriever
from codebase_agent.storage.metadata_cache import MetadataCache
from codebase_agent.storage.store_writer import StoreWriter


@click.group()
def main():
    """Codebase Onboarding Agent - Semantic + Structural Repository Intelligence."""
    pass


@main.command()
@click.option("--repo", "-r", default=".", help="Path to local repository or remote Git URL.")
@click.option("--dry-run", is_flag=True, help="Perform discovery, parsing, chunking without updating DB or vector store.")
@click.option("--full", is_flag=True, help="Force full re-indexing instead of incremental diff.")
def index(repo: str, dry_run: bool, full: bool):
    """Index a code repository (Phases 1-4: Ingestion, Parsing, Graph, Vector Embeddings)."""
    start_time = time.time()
    click.echo(f"=== Codebase Indexing ===")
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
        chroma_dir = index_dir / "chroma"

        cache = MetadataCache(db_path=db_path)

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

        if not dry_run:
            reindexer = Reindexer(repo_root=repo_root, config=config)
            summary = reindexer.execute_incremental(
                to_process=to_process,
                to_delete=to_delete,
                discovered_files=discovered
            )

            click.echo("\n--- Indexing Output Summary ---")
            click.echo(f"Run ID: {summary['run_id']}")
            click.echo(f"Processed Files: {summary['processed_files']}")
            click.echo(f"Evicted Deleted Files: {summary['deleted_files']}")
            click.echo(f"New Chunks Stored: {summary['new_chunks']}")
            click.echo(f"Graph Nodes: {summary['graph_nodes']}, Edges: {summary['graph_edges']}")
            click.echo(f"SQLite Cache: {db_path}")
            click.echo(f"GraphML Store: {graph_path}")
            click.echo(f"ChromaDB Store: {chroma_dir}")

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


@main.group()
def store():
    """Vector Store Subcommands (FR-4.4)."""
    pass


@store.command("inspect")
@click.option("--repo", "-r", default=".", help="Path to repository.")
@click.option("--collection", "-c", default="codebase_chunks", help="ChromaDB collection name.")
def store_inspect(repo: str, collection: str):
    """Inspect ChromaDB collection status, total chunks, vector dimension, and sample metadata (FR-4.4)."""
    config = IngestionConfig()
    chroma_dir = Path(repo) / config.index_dir_name / "chroma"

    if not chroma_dir.exists():
        click.echo(f"ChromaDB directory not found at {chroma_dir}. Please run 'index' first.", err=True)
        sys.exit(1)

    writer = StoreWriter(chroma_dir=chroma_dir)
    info = writer.inspect_collection()

    click.echo(f"=== Vector Store Inspection: {info['collection_name']} ===")
    click.echo(f"Storage Path: {info['chroma_dir']}")
    click.echo(f"Total Chunks Stored: {info['total_chunks']}")
    click.echo(f"Vector Dimension: {info['dimension'] or 'N/A'}")

    sample = info.get("sample_record")
    if sample:
        click.echo("\n--- Sample Chunk Record ---")
        click.echo(f"Chunk ID: {sample['id']}")
        click.echo(f"Document Snippet (first 100 chars): {repr(sample['document'][:100])}")
        click.echo("Metadata Fields:")
        for k, v in sample["metadata"].items():
            click.echo(f"  - {k}: {v}")


@main.command()
@click.argument("query_text")
@click.option("--repo", "-r", default=".", help="Path to repository.")
@click.option("--top-k", "-k", default=3, help="Top semantic vector matches count.")
@click.option("--hops", default=1, help="Structural graph expansion hops.")
@click.option("--threshold", default=0.3, help="Minimum similarity threshold score.")
def retrieve(query_text: str, repo: str, top_k: int, hops: int, threshold: float):
    """Perform hybrid retrieval (semantic search + graph expansion + context assembly) (FR-5.1 - FR-5.3, FR-5.6)."""
    config = IngestionConfig()
    index_dir = Path(repo) / config.index_dir_name
    chroma_dir = index_dir / "chroma"
    graph_path = index_dir / "graph" / "repo_graph.graphml"

    if not chroma_dir.exists() or not graph_path.exists():
        click.echo(f"Index directories missing at {index_dir}. Please run 'index' first.", err=True)
        sys.exit(1)

    retriever = VectorRetriever(chroma_dir=chroma_dir)
    expander = GraphExpander(chroma_dir=chroma_dir)
    assembler = ContextAssembler(max_context_tokens=4096, similarity_threshold=threshold)

    G = GraphStore.load_graph(graph_path)

    semantic_chunks = retriever.retrieve(query_text=query_text, top_k=top_k)
    expanded_chunks = expander.expand(semantic_chunks=semantic_chunks, G=G, hops=hops)
    result = assembler.assemble(query=query_text, semantic_chunks=semantic_chunks, expanded_chunks=expanded_chunks)

    click.echo(f"=== Hybrid Retrieval Engine Output ===")
    click.echo(f"Query: '{query_text}'")
    click.echo(f"Sufficient Context: {'YES' if result.sufficient_context else 'NO'}")

    if not result.sufficient_context:
        click.echo(f"\n[INSUFFICIENT CONTEXT WARNING]: {result.message}")
        return

    click.echo(f"Assembled Context Tokens: ~{result.total_tokens} tokens")
    click.echo(f"\nFinal Assembled Chunks ({len(result.final_context_chunks)}):")
    for idx, chunk in enumerate(result.final_context_chunks, 1):
        click.echo(f"\n[{idx}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (Source: {chunk.source}, Score: {chunk.similarity_score})")
        click.echo(f"    Graph Node ID: {chunk.graph_node_id}")
        click.echo(f"    Snippet: {repr(chunk.document[:80])}...")


@main.command("query")
@click.argument("question")
@click.option("--repo", "-r", default=".", help="Path to target repository.")
@click.option("--model", "-m", default="qwen2.5-coder:7b", help="Local Ollama LLM model name (FR-8.1).")
@click.option("--top-k", "-k", default=3, help="Top-K vector retrieval count.")
@click.option("--threshold", default=0.3, help="Minimum similarity threshold score.")
def query_command(question: str, repo: str, model: str, top_k: int, threshold: float):
    """Answer natural language questions about codebase with verifiable file/line citations (FR-5.1 - FR-5.5)."""
    receiver = QueryReceiver(
        repo_dir=Path(repo),
        model_name=model,
        top_k=top_k,
        similarity_threshold=threshold
    )

    response = receiver.process_query(question)

    click.echo(f"=== Codebase Question Answering (Model: {response.model_name}) ===")
    click.echo(f"Question: {response.query}\n")

    output_text = CitationFormatter.render_output(response)
    click.echo(output_text)


@main.group()
def hook():
    """Git Hooks Subcommands for Automatic Incremental Indexing (FR-6.3)."""
    pass


@hook.command("install")
@click.option("--repo", "-r", default=".", help="Path to target Git repository.")
def hook_install(repo: str):
    """Install .git/hooks/post-commit script to trigger automatic incremental re-indexing on commits (FR-6.3)."""
    installer = GitHookInstaller(repo_root=Path(repo))
    success, msg = installer.install_hook()
    if success:
        click.echo(f"Success: {msg}")
    else:
        click.echo(f"Error: {msg}", err=True)
        sys.exit(1)


@hook.command("uninstall")
@click.option("--repo", "-r", default=".", help="Path to target Git repository.")
def hook_uninstall(repo: str):
    """Uninstall .git/hooks/post-commit script (FR-6.3)."""
    installer = GitHookInstaller(repo_root=Path(repo))
    success, msg = installer.uninstall_hook()
    if success:
        click.echo(f"Success: {msg}")
    else:
        click.echo(f"Error: {msg}", err=True)
        sys.exit(1)


@hook.command("status")
@click.option("--repo", "-r", default=".", help="Path to target Git repository.")
def hook_status(repo: str):
    """Check post-commit hook status."""
    installer = GitHookInstaller(repo_root=Path(repo))
    installed = installer.check_status()
    click.echo(f"Git Post-Commit Hook Installed: {'YES' if installed else 'NO'}")
    click.echo(f"Hook File Path: {installer.hook_file}")


if __name__ == "__main__":
    main()

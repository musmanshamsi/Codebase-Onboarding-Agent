"""CLI entry point for Codebase Onboarding Agent (FR-7.1)."""

import json
import sys
import time
from pathlib import Path
import click

from codebase_agent.config import IngestionConfig
from codebase_agent.ingestion.manager import IngestionManager
from codebase_agent.parser.ast_parser import Parser
from codebase_agent.storage.metadata_cache import MetadataCache


@click.group()
def main():
    """Codebase Onboarding Agent - Semantic + Structural Repository Intelligence."""
    pass


@main.command()
@click.option("--repo", "-r", default=".", help="Path to local repository or remote Git URL.")
@click.option("--dry-run", is_flag=True, help="Perform discovery and hashing without updating database.")
@click.option("--full", is_flag=True, help="Force full re-indexing instead of incremental diff.")
def index(repo: str, dry_run: bool, full: bool):
    """Index a code repository (Phase 1 Ingestion & Metadata Cache)."""
    start_time = time.time()
    click.echo(f"=== Codebase Ingestion (Phase 1) ===")
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

        db_path = repo_root / config.index_dir_name / "cache.db"
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

        click.echo("\n--- Discovered File Details ---")
        for item in to_process:
            click.echo(f"  [+] {item['rel_path']} (Lang: {item['language']}, SHA256: {item['content_hash'][:12]}...)")

        for del_path in to_delete:
            click.echo(f"  [-] {del_path} (Deleted from repo)")

        if not dry_run:
            run_id = cache.start_index_run(run_type="full" if full else "incremental")
            processed_count = 0
            failed_count = 0

            for item in to_process:
                try:
                    cache.upsert_file_state(
                        file_path=item["rel_path"],
                        content_hash=item["content_hash"],
                        language=item["language"],
                        symbol_count=0,
                        parse_status="success",
                        parse_error=None
                    )
                    processed_count += 1
                except Exception as ex:
                    failed_count += 1
                    cache.upsert_file_state(
                        file_path=item["rel_path"],
                        content_hash=item["content_hash"],
                        language=item["language"],
                        symbol_count=0,
                        parse_status="failed",
                        parse_error=str(ex)
                    )

            for del_path in to_delete:
                cache.delete_file_state(del_path)

            cache.finish_index_run(
                run_id=run_id,
                files_processed=processed_count,
                files_failed=failed_count,
                status="completed"
            )

            last_run = cache.get_last_run()
            click.echo("\n--- Metadata Cache Status ---")
            click.echo(f"Run ID: {last_run['run_id']}")
            click.echo(f"Status: {last_run['status']}")
            click.echo(f"Processed: {last_run['files_processed']}, Failed: {last_run['files_failed']}")
            click.echo(f"Database persisted at: {db_path}")

        elapsed = time.time() - start_time
        click.echo(f"\nIngestion phase completed in {elapsed:.3f} seconds.")

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


if __name__ == "__main__":
    main()

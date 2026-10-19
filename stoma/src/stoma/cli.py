"""
Stoma Command Line Interface.
Provides CLI commands for:
- ingest: one-shot ingestion of files or directories
- query: retrieval from LTHCM topological memory with full provenance
- watch: real-time filesystem observation and debounced reconciliation
- reconcile: full manual reconciliation of indexed files against disk
- status: inspect local index, manifest, model info, and passage counts
- interchange: test index compatibility against Dianoia's load requirements
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click

from stoma.config import StomaConfig, verify_runtime_dependencies
from stoma.indexer import StomaIndexer
from stoma.interchange import verify_dianoia_interchange
from stoma.watcher import StomaWatcher


def _format_timestamp(ts: float) -> str:
    """Format an epoch-second float to ISO UTC string."""
    try:
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return f"{ts:.2f}"


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=False), help="Path to config YAML.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose debug logging.")
@click.pass_context
def cli(ctx: click.Context, config_path: Optional[str], verbose: bool):
    """Stoma — ingestion pipeline and topological memory mouth for LTHCM."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Strictly enforce exact imported submodule SHAs and paths before proceeding
    verify_runtime_dependencies()
    cfg_p = Path(config_path) if config_path else None
    ctx.obj = StomaConfig.load(cfg_p)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--force", "-f", is_flag=True, help="Force re-indexing even if file hash is unchanged.")
@click.pass_obj
def ingest(config: StomaConfig, path: str, force: bool):
    """Ingest a file or directory into LTHCM with full provenance."""
    target = Path(path).resolve()
    indexer = StomaIndexer(config=config)

    if target.is_file():
        click.echo(f"Ingesting file: {target}")
        report = indexer.ingest_file(target, force=force)
        click.echo(f"Status:   {report.status}")
        click.echo(f"Passages: {report.passages_count}")
        click.echo(f"Duration: {report.duration_sec:.3f}s")
        if report.error_message:
            click.secho(f"Error:    {report.error_message}", fg="red")
        if report.warning_message:
            click.secho(f"Warning:  {report.warning_message}", fg="yellow")
    elif target.is_dir():
        click.echo(f"Scanning and ingesting directory: {target}")
        summary = indexer.scan_and_ingest_directory(target)
        click.echo("\n--- Ingestion Summary ---")
        click.echo(f"Files Scanned:        {summary.total_files_scanned}")
        click.echo(f"Files Indexed:        {summary.files_indexed}")
        click.echo(f"Skipped Unchanged:    {summary.files_skipped_unchanged}")
        click.echo(f"Skipped Unsupported:  {summary.files_skipped_unsupported}")
        click.echo(f"Failed:               {summary.files_failed}")
        click.echo(f"Passages Indexed:     {summary.total_passages_indexed}")
        click.echo(f"Duration:             {summary.duration_sec:.3f}s")

        # Show skipped or failed details
        for r in summary.reports:
            if r.status == "skipped_unsupported" and r.error_message:
                click.secho(f"  [UNSUPPORTED] {r.rel_path}: {r.error_message}", fg="yellow")
            elif r.status == "failed":
                click.secho(f"  [FAILED] {r.rel_path}: {r.error_message}", fg="red")


@cli.command()
@click.argument("query_text")
@click.option("--top-k", "-k", default=5, help="Number of passages to retrieve.")
@click.option("--domain", "-d", default="general", help="Retrieval domain.")
@click.pass_obj
def query(config: StomaConfig, query_text: str, top_k: int, domain: str):
    """Query LTHCM topological memory and display passages with full provenance."""
    indexer = StomaIndexer(config=config)
    q_time = time.time()
    res = indexer.query(query_text=query_text, top_k=top_k)

    click.echo(f"\nQuery: '{query_text}'")
    click.echo(f"Query Timestamp:  {q_time:.4f} ({_format_timestamp(q_time)})")
    click.echo(f"Candidates Found: {res['candidates_count']}")
    click.echo(f"Causally Valid:   {res['causally_valid_count']}")
    click.echo(f"Causal Fallback:  {res['causal_fallback']}")
    click.echo(f"Results Returned: {len(res['results'])}\n" + "=" * 60)

    for i, item in enumerate(res["results"], start=1):
        prov = item.get("provenance") or {}
        loc = prov.get("location") or {}
        loc_str = ", ".join(f"{k}={v}" for k, v in loc.items()) if loc else "n/a"
        file_hash = prov.get("file_content_hash", "n/a")
        ingest_ts = prov.get("ingested_at", 0.0)

        click.secho(f"\n[{i}] Passage ID: {item['doc_id']}", fg="cyan", bold=True)
        click.echo(f"    Source Path:      {prov.get('rel_path', 'unknown')}")
        click.echo(f"    Location:         {loc_str}")
        click.echo(f"    Content SHA-256:  {file_hash}")
        click.echo(f"    Ingested At:      {ingest_ts:.4f} ({_format_timestamp(ingest_ts)})")
        if item.get("gauge_sim") is not None:
            click.echo(f"    Gauge Similarity: {item['gauge_sim']:.4f}")

        click.echo("\n    Content:")
        for line in item["content"].splitlines()[:10]:
            click.echo(f"      {line}")
        if len(item["content"].splitlines()) > 10:
            click.echo("      [... truncated ...]")
        click.echo("-" * 60)


@cli.command()
@click.argument("path", type=click.Path(exists=True), required=False)
@click.option("--debounce", default=0.5, help="Debounce window in seconds.")
@click.pass_obj
def watch(config: StomaConfig, path: Optional[str], debounce: float):
    """Watch directory in real-time and incrementally ingest/reconcile changes."""
    indexer = StomaIndexer(config=config)
    watcher = StomaWatcher(indexer=indexer, debounce_sec=debounce)
    watch_target = [Path(path).resolve()] if path else None

    click.echo("Starting Stoma file watcher (Ctrl+C to stop)...")
    watcher.start(watch_paths=watch_target)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        click.echo("\nStopping watcher...")
        watcher.stop()
        click.echo("Watcher stopped.")


@cli.command()
@click.pass_obj
def reconcile(config: StomaConfig):
    """Perform a full manual reconciliation between disk and index."""
    indexer = StomaIndexer(config=config)
    watcher = StomaWatcher(indexer=indexer)
    summary = watcher.reconcile()

    click.echo("\n--- Reconciliation Summary ---")
    click.echo(f"Files Scanned on Disk:    {summary['total_files_scanned']}")
    click.echo(f"Files Indexed:            {summary.get('files_indexed', 0)}")
    click.echo(f"Files Unchanged:          {summary.get('files_unchanged', 0)}")
    click.echo(f"Skipped Unsupported:      {summary.get('files_skipped_unsupported', 0)}")
    click.echo(f"Failed Files:             {summary.get('files_failed', 0)}")
    click.echo(f"Deleted Files Reconciled: {summary['missing_files_cleaned']}")
    click.echo(f"Active Manifest Sources:  {summary['active_manifest_sources']}")
    click.echo(f"Active Indexed Passages:  {summary['active_manifest_passages']}")

    if summary.get("failed_details"):
        click.secho("\nFailed Files Details:", fg="red", bold=True)
        for detail in summary["failed_details"]:
            click.secho(f"  * {detail.get('path', 'unknown')}: {detail.get('error', 'unknown error')}", fg="red")
        dropped = summary.get("failed_details_dropped", 0)
        if dropped > 0:
            click.secho(f"  ... and {dropped} additional failure details omitted (diagnostic memory capped).", fg="yellow")


@cli.command()
@click.pass_obj
def status(config: StomaConfig):
    """Display current Stoma storage, model identity, and index state."""
    indexer = StomaIndexer(config=config)
    m = indexer.manifest
    meta = m.model_meta

    click.echo("\n=== Stoma System Status ===")
    click.echo(f"Storage Directory:  {config.storage_dir}")
    click.echo(f"Active Generation:  gen_{indexer.current_generation:06d}")
    click.echo(f"Index File:         {config.index_path} ({'Exists' if config.index_path.is_file() else 'Missing'})")
    click.echo(f"Manifest File:      {config.manifest_path} ({'Exists' if config.manifest_path.is_file() else 'Missing'})")
    click.echo(f"Total Sources:      {len(m.sources)}")
    click.echo(f"Total Passages:     {len(m.passages)}")
    click.echo(f"Index d_model:      {m.d_model}")
    click.echo(f"Index num_facets:   {m.num_facets}")

    if meta:
        click.echo("\n--- Encoder Model Info ---")
        click.echo(f"Model Name:         {meta.model_name}")
        click.echo(f"d_model:            {meta.d_model}")
        click.echo(f"Max Sequence Len:   {meta.max_seq_length}")
        click.echo(f"Tokenizer:          {meta.tokenizer_name}")
        click.echo(f"Model Revision:     {meta.revision or 'n/a'}")
        click.echo(f"Weights Checksum:   {meta.weights_checksum or 'n/a'}")
        click.echo(f"Effective Max Tok:  {meta.effective_max_tokens}")

    click.echo("\n--- Permitted Ingestion Roots ---")
    for r in config.allowed_roots:
        click.echo(f"  * {r}")


@cli.command()
@click.pass_obj
def interchange(config: StomaConfig):
    """Test direct index interchange with Dianoia's load requirements."""
    res = verify_dianoia_interchange(config.index_path)

    click.echo("\n=== Dianoia Interchange Verification ===")
    click.echo(f"Index Path:               {config.index_path}")
    click.echo(f"Serialization Match:      {res.get('serialization_match')}")
    click.echo(f"Is LTHCMEngine:           {res.get('is_lthcm_engine')}")
    click.echo(f"d_model:                  {res.get('d_model')}")
    click.echo(f"num_facets:               {res.get('num_facets')}")
    click.echo(f"Dimension Match:          {res.get('dimension_match')}")
    click.echo(f"Graph Edge Count:         {res.get('edge_count', 0)}")
    click.echo(f"Status:                   {res.get('status')}")

    if res.get("compatible"):
        click.secho("\nIndex is 100% compatible with Dianoia load and query paths!", fg="green")
    else:
        click.secho(f"\nReported Status: {res.get('status')}", fg="yellow", bold=True)
        if "reason" in res:
            click.echo(f"Reason: {res['reason']}")
        elif "message" in res:
            click.echo(f"Message: {res['message']}")


@cli.command("setup-model")
@click.option("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name to prefetch.")
@click.pass_obj
def setup_model(config: StomaConfig, model: str):
    """Explicitly prefetch model weights to populate the offline cache."""
    from stoma.embedder import prefetch_model
    click.echo(f"Prefetching model '{model}' to local HuggingFace cache...")
    try:
        prefetch_model(model_name=model)
        click.secho(f"Successfully prefetched '{model}'. Offline runtime is now prepared.", fg="green")
    except Exception as e:
        click.secho(f"Failed to prefetch model: {e}", fg="red")
        sys.exit(1)


@cli.group("graph")
def graph_group():
    """Knowledge graph inspection and offline edge construction."""
    pass


@graph_group.command("build")
@click.option("--index", "index_path", type=click.Path(exists=True), help="Path to index file (defaults to config index_path).")
@click.option("--threshold", "-t", default=0.75, type=float, help="Cosine similarity threshold tau for edge creation (default 0.75).")
@click.option("--k-neighbors", "-k", default=5, type=int, help="Maximum number of nearest neighbors per passage (default 5).")
@click.pass_obj
def build_graph(config: StomaConfig, index_path: Optional[str], threshold: float, k_neighbors: int):
    """Build semantic cosine similarity edges across chunk embeddings into the wormhole graph."""
    from stoma.graph_builder import build_knn_knowledge_graph
    target_path = Path(index_path) if index_path else config.index_path
    click.echo(f"Building knowledge graph for: {target_path} (tau>={threshold}, k={k_neighbors})...")
    res = build_knn_knowledge_graph(
        index_path=target_path,
        similarity_threshold=threshold,
        max_k_neighbors=k_neighbors,
    )
    click.echo(f"Nodes in index: {res['nodes_count']}")
    click.echo(f"Edges added:    {res['edges_added']}")
    click.echo(f"Total edges:    {res['total_edges']}")
    click.secho(f"\n{res['message']}", fg="green" if res['edges_added'] > 0 else "yellow")


def main():
    cli()


if __name__ == "__main__":
    main()

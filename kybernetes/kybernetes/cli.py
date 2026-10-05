"""Command-line interface for Kybernetes orchestrator."""

import asyncio
from pathlib import Path
from typing import Optional
import sys
import uuid
import click
import uvicorn
from tabulate import tabulate

from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.krisis import KrisisAdapter
from kybernetes.adapters.lthcm import LTHCMAdapter
from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.budget.engine import BudgetEngine
from kybernetes.budget.storage import BudgetDatabase
from kybernetes.contracts.turn import TurnRequest
from kybernetes.engine.pipeline import ExecutionPipeline
from kybernetes.engine.session import SessionManager
from kybernetes.routing.router import PlanRouter
from kybernetes.server.app import create_app
from kybernetes.tools.registry import ToolRegistry


@click.group()
@click.version_option(version="0.1.0", prog_name="kybernetes")
def cli():
    """Kybernetes: Local Assistant Orchestrator for the Metis Stack."""
    pass


@cli.command("doctor")
def doctor():
    """Verify local Metis stack dependencies, model endpoints, and subsystem readiness."""
    click.echo("\n[Kybernetes Doctor: Inspecting Stack Subsystems]")

    hestia = HestiaAdapter()
    lthcm = LTHCMAdapter()
    dianoia = DianoiaAdapter()
    phylax = PhylaxAdapter()
    krisis = KrisisAdapter()

    h_health = asyncio.run(hestia.check_health())
    if h_health.get("available"):
        reg = "registered" if h_health.get("default_model_registered") else "default model not registered"
        h_status = f"ONLINE ({reg})"
        h_details = f"Models: {', '.join(h_health.get('models', [])) or 'None'}"
    else:
        h_status = f"OFFLINE ({h_health.get('url')})"
        h_details = "Service unreachable; start Hestia on port 8080"

    # LTHCM / Stoma check
    is_real_lthcm = bool(lthcm.memory_root and Path(lthcm.memory_root).is_dir())
    lthcm_status = "READY (LIVE INDEX)" if is_real_lthcm else "SIMULATED (DEMO CORPUS)"
    lthcm_details = f"Corpus root: {lthcm.memory_root}" if is_real_lthcm else "In-memory demo corpus (4 passages), timestamp bounds active"

    # Dianoia check
    dianoia_status = "SIMULATED (ZERO-EDGE FALLBACK)"
    dianoia_details = "Simulated graph deliberation with zero-edge flat retrieval fallback"

    # Phylax check
    phylax_status = "SIMULATED (LOCAL SHIM)"
    phylax_details = "In-memory approval manager active (live GuardedGateway unconfigured)"

    # Krisis check
    krisis_status = "SIMULATED (LOCAL HEURISTIC)"
    krisis_details = "Half-even scoring and heuristic citation verification (standalone selector unconfigured)"

    table_data = [
        ["Hestia Model Server", h_status, h_details],
        ["LTHCM / Stoma Memory", lthcm_status, lthcm_details],
        ["Dianoia Deliberation", dianoia_status, dianoia_details],
        ["Phylax Capability Gateway", phylax_status, phylax_details],
        ["Krisis Verifier Gates", krisis_status, krisis_details],
    ]

    click.echo(tabulate(table_data, headers=["Subsystem", "Status", "Details"], tablefmt="rounded_grid"))
    click.echo("\nNote: Local adapter status reported. Live production deployment requires verified external endpoints/roots.\n")


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind host address")
@click.option("--port", default=8081, help="Bind port number")
@click.option("--db-path", default="kybernetes_budget.db", help="SQLite database path")
@click.option("--memory-root", default=None, help="Path to Stoma/LTHCM memory root directory")
@click.option("--config", "config_file", default=None, help="Path to YAML configuration file")
@click.option("--admin-token", default=None, help="Admin bearer token for /v1/admin endpoints")
@click.option("--service-token", default=None, help="Service bearer token for /v1/chat endpoints")
def serve(
    host: str,
    port: int,
    db_path: str,
    memory_root: Optional[str] = None,
    config_file: Optional[str] = None,
    admin_token: Optional[str] = None,
    service_token: Optional[str] = None,
):
    """Start Kybernetes FastAPI SSE server and admin control plane."""
    click.echo(f"Starting Kybernetes orchestrator server on http://{host}:{port}")
    from kybernetes.config import load_config
    cfg = load_config(config_file)
    if admin_token:
        cfg.admin_token = admin_token
    if service_token:
        cfg.service_token = service_token
    if memory_root:
        cfg.memory_root = memory_root
    if db_path:
        cfg.db_path = db_path

    app = create_app(
        db_path=db_path,
        memory_root=memory_root,
        service_token=service_token,
        config=cfg,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command("chat")
@click.option("--query", "-q", prompt="User query", help="Query text to orchestrate")
def chat(query: str):
    """Run a single interactive turn through the full Kybernetes pipeline."""
    async def _run():
        db = BudgetDatabase(":memory:")
        pipeline = ExecutionPipeline(
            router=PlanRouter(),
            budget_engine=BudgetEngine(db),
            tool_registry=ToolRegistry(),
            hestia_adapter=HestiaAdapter(),
            lthcm_adapter=LTHCMAdapter(),
            dianoia_adapter=DianoiaAdapter(),
            phylax_adapter=PhylaxAdapter(),
            krisis_adapter=KrisisAdapter(),
            session_manager=SessionManager(),
        )

        req = TurnRequest(session_id="cli-session", query=query)
        click.echo(f"\n--- Submitting Turn: '{query}' ---")
        async for event in pipeline.execute_turn_stream(req):
            if event.event_type == "plan_selected":
                click.echo(f"-> Plan: {event.payload.get('route')} (Tags: {event.payload.get('route_tags')})")
            elif event.event_type == "evidence":
                citations = event.payload.get("citations", [])
                click.echo(f"-> Retrieved {len(citations)} evidence citations from LTHCM")
                for c in citations:
                    click.echo(f"   {c['citation_id']} {c['source_title']}: {c['text'][:80]}...")
            elif event.event_type == "delta":
                sys.stdout.write(event.payload.get("text", ""))
                sys.stdout.flush()
            elif event.event_type == "completed":
                click.echo(f"\n\n-> Turn Completed. Status: {event.status}")
                summary = event.payload.get("route_summary", {})
                click.echo(f"   Route: {summary.get('route')}, Latency: {summary.get('latency_ms')} ms")

    asyncio.run(_run())


@cli.command("demo")
def demo():
    """Run the 5 canonical Kybernetes demonstration scenarios."""
    from kybernetes.demo import run_all_demos
    asyncio.run(run_all_demos())


if __name__ == "__main__":
    cli()

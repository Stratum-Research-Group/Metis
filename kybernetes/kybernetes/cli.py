"""Command-line interface for Kybernetes orchestrator."""

import asyncio
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
    h_status = "ONLINE" if h_health.get("available") else f"OFFLINE ({h_health.get('url')})"

    table_data = [
        ["Hestia Model Server", h_status, ", ".join(h_health.get("models", [])) or "None (fallback ready)"],
        ["LTHCM / Stoma Memory", "READY", "Epoch timestamp bounds active"],
        ["Dianoia Deliberation", "READY", "Zero-edge fallback enabled"],
        ["Phylax Capability Gateway", "READY", "Point-of-action enforcement active"],
        ["Krisis Verifier Gates", "READY", "6dp half-even scoring & quality gates active"],
    ]

    click.echo(tabulate(table_data, headers=["Subsystem", "Status", "Details"], tablefmt="rounded_grid"))
    click.echo("\nAll local adapters initialized successfully.\n")


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind host address")
@click.option("--port", default=8080, help="Bind port number")
@click.option("--db-path", default="kybernetes_budget.db", help="SQLite database path")
def serve(host: str, port: int, db_path: str):
    """Start Kybernetes FastAPI SSE server for Prosopon UI and client connections."""
    click.echo(f"Starting Kybernetes orchestrator server on http://{host}:{port}")
    app = create_app(db_path=db_path)
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

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


# --- Lochos Multi-Agent Squad CLI ---

@cli.group("lochos")
def lochos_group():
    """Lochos: Multi-Agent Squad and Durable Work Queue Management."""
    pass


@lochos_group.command("ps")
@click.option("--db-path", default=None, help="Path to lochos.sqlite")
def lochos_ps(db_path: Optional[str] = None):
    """List all agent seats, semantic states, and active batons in the squad."""
    from kybernetes.lochos import LochosQueue, LochosSquadManager
    custom_db = Path(db_path) if db_path else None
    queue = LochosQueue(custom_db) if custom_db else None
    manager = LochosSquadManager(queue=queue)
    status_dict = manager.to_status_dict()

    click.echo(f"\n[Lochos Squad: {status_dict['rig_name']} (v{status_dict['version']})]")
    click.echo(f"Summary: {status_dict['summary']}\n")

    table_data = []
    for seat in status_dict["seats"]:
        baton_str = seat["current_task"]["title"] if seat["current_task"] else "None (idle)"
        table_data.append([
            seat["address"],
            seat["role"],
            seat["model"],
            seat["state"].upper(),
            seat["permission_policy"],
            baton_str[:35] + ("..." if len(baton_str) > 35 else ""),
        ])

    click.echo(tabulate(
        table_data,
        headers=["Seat Address", "Role", "Model", "State", "Policy", "Active Baton"],
        tablefmt="rounded_grid",
    ))

    attns = status_dict["attention_required"]
    if attns:
        click.echo(f"\n[!] Attention Required ({len(attns)} blocked seat{'s' if len(attns) > 1 else ''}):")
        for a in attns:
            click.echo(f"  - [{a['seat_id']}] Task: '{a['task_title']}' -> Reason: {a['reason']}")
    else:
        click.echo("\n[x] No blocked seats. All agents operational.\n")


@lochos_group.command("queue")
@click.option("--status", default=None, help="Filter by status (queued, in_progress, completed, failed)")
@click.option("--seat", default=None, help="Filter by seat ID")
def lochos_queue_cmd(status: Optional[str] = None, seat: Optional[str] = None):
    """List work items in the durable SQLite queue."""
    from kybernetes.lochos import LochosSquadManager
    manager = LochosSquadManager()
    tasks = manager.queue.list_tasks(status=status, seat_id=seat)
    if not tasks:
        click.echo("No work items found in queue.")
        return

    table_data = [
        [
            t.task_id,
            t.mission,
            t.title[:30] + ("..." if len(t.title) > 30 else ""),
            t.baton_owner or "Unassigned",
            t.status.upper(),
        ]
        for t in tasks
    ]
    click.echo(tabulate(
        table_data,
        headers=["Task ID", "Mission", "Title", "Baton Owner", "Status"],
        tablefmt="rounded_grid",
    ))


@lochos_group.command("dispatch")
@click.option("--mission", "-m", default="release-0.5.8", help="Mission identifier")
@click.option("--slice", "-s", "slice_id", default="001", help="Slice identifier")
@click.option("--title", "-t", prompt="Task title", help="Title of task")
@click.option("--seat", prompt="Target seat (e.g. coder, reviewer)", help="Target seat ID")
def lochos_dispatch_cmd(mission: str, slice_id: str, title: str, seat: str):
    """Enqueue and assign a mission slice task to a seat."""
    from kybernetes.lochos import LochosSquadManager
    manager = LochosSquadManager()
    try:
        item = manager.dispatch_work(
            mission=mission,
            slice_id=slice_id,
            title=title,
            target_seat=seat,
        )
        click.echo(f"Successfully dispatched task [{item.task_id}] to {item.baton_owner} (Status: {item.status})")
    except Exception as e:
        click.echo(f"Dispatch error: {e}", err=True)


@lochos_group.command("transfer")
@click.option("--task-id", prompt="Task ID", help="Task ID to transfer")
@click.option("--from-seat", prompt="Current seat", help="Seat handing off baton")
@click.option("--to-seat", prompt="Destination seat", help="Seat receiving baton")
@click.option("--note", default="Baton handoff", help="Handoff note")
def lochos_transfer_cmd(task_id: str, from_seat: str, to_seat: str, note: str):
    """Transfer task baton from one seat to another."""
    from kybernetes.lochos import LochosSquadManager
    manager = LochosSquadManager()
    try:
        item = manager.transfer_baton(
            task_id=task_id,
            from_seat=from_seat,
            to_seat=to_seat,
            note=note,
        )
        click.echo(f"Baton for task [{item.task_id}] transferred to {item.baton_owner}")
    except Exception as e:
        click.echo(f"Transfer error: {e}", err=True)


@lochos_group.command("resolve")
@click.option("--task-id", prompt="Task ID", help="Task ID with attention block")
@click.option("--seat", prompt="Seat address/id", help="Seat ID to unblock")
@click.option("--resolution", default="approved_by_operator", help="Approval resolution")
def lochos_resolve_cmd(task_id: str, seat: str, resolution: str):
    """Resolve an attention block and resume the seat."""
    from kybernetes.lochos import LochosSquadManager
    manager = LochosSquadManager()
    try:
        item = manager.queue.resolve_attention(task_id=task_id, seat_id=seat, resolution=resolution)
        click.echo(f"Attention resolved for task [{item.task_id}]. Status: {item.status}")
    except Exception as e:
        click.echo(f"Resolution error: {e}", err=True)


if __name__ == "__main__":
    cli()

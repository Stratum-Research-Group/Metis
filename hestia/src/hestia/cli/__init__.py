"""Hestia CLI — manage the model serving layer."""

from __future__ import annotations

import sys

import click
import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8080"


@click.group()
@click.option(
    "--url",
    default=DEFAULT_BASE_URL,
    envvar="HESTIA_URL",
    help="Hestia API base URL.",
)
@click.pass_context
def cli(ctx: click.Context, url: str) -> None:
    """Hestia — local AI model serving layer."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url.rstrip("/")


@cli.command()
@click.pass_context
def list(ctx: click.Context) -> None:
    """List all registered models and their status."""
    url = ctx.obj["url"]
    try:
        resp = httpx.get(f"{url}/models", timeout=10.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to Hestia at {url}", err=True)
        sys.exit(1)

    data = resp.json()
    models = data.get("models", [])

    if not models:
        click.echo("No models registered.")
        return

    # Table header
    click.echo(f"{'NAME':<30} {'TYPE':<12} {'STATE':<10} {'VRAM (MB)':<10} {'PORT':<8}")
    click.echo("─" * 75)

    for m in models:
        port = m.get("port") or "—"
        click.echo(
            f"{m['name']:<30} {m['type']:<12} {m['state']:<10} "
            f"{m['vram_estimate_mb']:<10} {port:<8}"
        )


@cli.command()
@click.argument("name")
@click.pass_context
def load(ctx: click.Context, name: str) -> None:
    """Load a model by name."""
    url = ctx.obj["url"]
    click.echo(f"Loading {name}...")

    try:
        resp = httpx.post(
            f"{url}/manage/load",
            json={"model": name},
            timeout=180.0,
        )
        if resp.status_code == 404:
            click.echo(f"Error: Model '{name}' not found.", err=True)
            sys.exit(1)
        if resp.status_code == 503:
            detail = resp.json().get("detail", "Unknown error")
            click.echo(f"Error: {detail}", err=True)
            sys.exit(1)
        resp.raise_for_status()
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to Hestia at {url}", err=True)
        sys.exit(1)

    click.echo(f"✓ {name} loaded successfully.")


@cli.command()
@click.argument("name")
@click.pass_context
def unload(ctx: click.Context, name: str) -> None:
    """Unload a model by name."""
    url = ctx.obj["url"]

    try:
        resp = httpx.post(f"{url}/manage/unload", json={"model": name}, timeout=60.0)
        if resp.status_code == 404:
            click.echo(f"Error: Model '{name}' not found.", err=True)
            sys.exit(1)
        resp.raise_for_status()
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to Hestia at {url}", err=True)
        sys.exit(1)

    click.echo(f"✓ {name} unloaded.")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show detailed status: VRAM use per model and GPU utilisation."""
    url = ctx.obj["url"]

    try:
        resp = httpx.get(f"{url}/health", timeout=10.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to Hestia at {url}", err=True)
        sys.exit(1)

    data = resp.json()
    click.echo(f"Service: {data['status']}")
    click.echo()

    models = data.get("models", [])
    loaded = [m for m in models if m["state"] == "loaded"]
    total_vram = sum(m["vram_estimate_mb"] for m in loaded)

    # GPU summary
    gpu_info = data.get("gpu", {})
    gpu_name = gpu_info.get("name") or "RTX 5060 Ti 16 GB"
    vram_free = gpu_info.get("vram_free_mb", 0)
    vram_total = gpu_info.get("vram_total_mb", 0)

    click.echo(f"GPU: {gpu_name}")
    click.echo(f"  Models loaded: {len(loaded)}")
    click.echo(f"  VRAM reserved: ~{total_vram} MB")
    if vram_total:
        click.echo(f"  VRAM live: {vram_free} MB free / {vram_total} MB total")
    click.echo()

    if models:
        click.echo(f"{'NAME':<30} {'TYPE':<12} {'STATE':<10} {'VRAM (MB)':<10} {'IN-FLIGHT':<10}")
        click.echo("─" * 75)
        for m in models:
            click.echo(
                f"{m['name']:<30} {m['type']:<12} {m['state']:<10} "
                f"{m['vram_estimate_mb']:<10} {m.get('inflight_requests', 0):<10}"
            )


@cli.command()
@click.pass_context
def serve(ctx: click.Context) -> None:
    """Start the Hestia API server."""
    import uvicorn

    from hestia.api.app import create_app

    app = create_app()
    config_path = "hestia.yaml"

    # Read listen port from config
    from pathlib import Path

    from hestia.config import ServiceConfig

    try:
        svc_config = ServiceConfig.from_yaml(Path(config_path))
    except FileNotFoundError:
        click.echo(f"Config not found: {config_path}", err=True)
        sys.exit(1)

    click.echo(f"Starting Hestia on {svc_config.listen_host}:{svc_config.listen_port}...")
    uvicorn.run(app, host=svc_config.listen_host, port=svc_config.listen_port, log_level="info")


if __name__ == "__main__":
    cli()

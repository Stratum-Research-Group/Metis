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
@click.option(
    "--token",
    default=None,
    envvar="HESTIA_SERVICE_TOKEN",
    help="Hestia service bearer token.",
)
@click.pass_context
def cli(ctx: click.Context, url: str, token: str | None) -> None:
    """Hestia — local AI model serving layer."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url.rstrip("/")
    ctx.obj["token"] = token


def _get_headers(ctx: click.Context) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = ctx.obj.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@cli.command()
@click.pass_context
def list(ctx: click.Context) -> None:
    """List all registered models and their status."""
    url = ctx.obj["url"]
    try:
        resp = httpx.get(f"{url}/models", headers=_get_headers(ctx), timeout=10.0)
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
            headers=_get_headers(ctx),
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
        resp = httpx.post(
            f"{url}/manage/unload",
            json={"model": name},
            headers=_get_headers(ctx),
            timeout=60.0,
        )
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
@click.option("--host", type=str, default=None, help="Host to listen on")
@click.option("--port", type=int, default=None, help="Port to listen on")
@click.option("--config", "config_path", type=str, default=None, help="Path to hestia.yaml config")
@click.option("--models", "models_path", type=str, default=None, help="Path to models.yaml registry")
@click.pass_context
def serve(
    ctx: click.Context,
    host: Optional[str] = None,
    port: Optional[int] = None,
    config_path: Optional[str] = None,
    models_path: Optional[str] = None,
) -> None:
    """Start the Hestia API server."""
    import uvicorn
    from pathlib import Path
    from hestia.config import ServiceConfig
    from hestia.api.app import create_app

    # Resolve config file path
    resolved_config: Optional[Path] = None
    candidates = [
        config_path,
        "hestia.yaml",
        "hestia/hestia.yaml",
        str(Path(__file__).resolve().parent.parent.parent.parent / "hestia.yaml"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            resolved_config = Path(c).resolve()
            break

    if not resolved_config:
        click.echo(f"Config not found: {config_path or 'hestia.yaml'}", err=True)
        sys.exit(1)

    # Resolve models file path
    resolved_models: Optional[Path] = None
    m_candidates = [
        models_path,
        "models.yaml",
        "hestia/models.yaml",
        str(Path(__file__).resolve().parent.parent.parent.parent / "models.yaml"),
    ]
    for m in m_candidates:
        if m and Path(m).is_file():
            resolved_models = Path(m).resolve()
            break

    models_arg = str(resolved_models) if resolved_models else "models.yaml"

    try:
        svc_config = ServiceConfig.from_yaml(resolved_config)
    except FileNotFoundError:
        click.echo(f"Config not found: {resolved_config}", err=True)
        sys.exit(1)

    listen_host = host or svc_config.listen_host
    listen_port = port or svc_config.listen_port

    app = create_app(config_path=str(resolved_config), models_path=models_arg)

    click.echo(f"Starting Hestia on {listen_host}:{listen_port}...")
    uvicorn.run(app, host=listen_host, port=listen_port, log_level="info")


if __name__ == "__main__":
    cli()

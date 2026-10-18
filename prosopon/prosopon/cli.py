"""Command-Line Interface for Prosopon."""

from pathlib import Path
from typing import Optional
import click
import uvicorn

from prosopon.config import ProsoponConfig
from prosopon.auth.pairing import PairingManager
from prosopon.auth.tls import generate_self_signed_cert
from prosopon.ops.doctor import print_doctor_report
from prosopon.server import create_app


@click.group()
def cli():
    """Prosopon: local text and voice interface for Metis."""
    pass


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 for LAN mode)")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option("--lan", is_flag=True, help="Enable LAN mode (bind 0.0.0.0 with pairing requirement)")
@click.option("--tls", is_flag=True, help="Enable HTTPS with auto-generated self-signed certificate")
@click.option("--mock/--no-mock", default=True, help="Use mock Kybernetes orchestrator for local testing")
@click.option("--kybernetes-url", default=None, help="Base URL for versioned Kybernetes API (e.g. http://127.0.0.1:8081/v1)")
@click.option("--data-dir", default="./data", help="Directory for persistent sessions and tokens")
def serve(host: str, port: int, lan: bool, tls: bool, mock: bool, kybernetes_url: Optional[str], data_dir: str):
    """Start the Prosopon local web and voice interface server."""
    bind_host = "0.0.0.0" if lan else host
    cfg = ProsoponConfig(
        host=bind_host,
        port=port,
        enable_tls=tls,
        mock_kybernetes=mock,
        kybernetes_api_url=kybernetes_url or "http://127.0.0.1:8081/v1",
        data_dir=Path(data_dir),
    )

    cert_path = None
    key_path = None
    if tls:
        cert_path, key_path = generate_self_signed_cert(cfg.data_dir / "tls")
        cfg.cert_file = str(cert_path)
        cfg.key_file = str(key_path)

    app = create_app(cfg)

    # Print startup banner
    proto = "https" if tls else "http"
    click.echo(f"\n=======================================================")
    click.echo(f"  Prosopon Local Interface (v0.1.0)")
    click.echo(f"  Mode: {'LAN (Phone Ready)' if lan else 'Local Loopback'}")
    click.echo(f"  URL: {proto}://{bind_host}:{port}")
    if lan:
        mgr = getattr(app.state, "pairing_mgr", None) or PairingManager(cfg.data_dir)
        pin = mgr.generate_pairing_pin()
        click.echo(f"  Single-Owner Phone Pairing PIN: {pin} (valid 5 min)")
    click.echo(f"=======================================================\n")

    uvicorn.run(
        app,
        host=bind_host,
        port=port,
        ssl_certfile=str(cert_path) if cert_path else None,
        ssl_keyfile=str(key_path) if key_path else None,
        log_level="info",
    )


@cli.command("run", hidden=True)
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option("--lan", is_flag=True, help="Enable LAN mode")
@click.option("--tls", is_flag=True, help="Enable HTTPS")
@click.option("--mock/--no-mock", default=True, help="Use mock Kybernetes")
@click.option("--kybernetes-url", default=None, help="Base URL for Kybernetes API")
@click.option("--data-dir", default="./data", help="Directory for persistent data")
@click.pass_context
def run(ctx: click.Context, **kwargs):
    """Run alias for serve."""
    ctx.forward(serve)


@cli.command()
@click.option("--data-dir", default="./data", help="Path to data directory")
def pair(data_dir: str):
    """Generate a 6-digit pairing PIN for authorizing a phone browser."""
    mgr = PairingManager(Path(data_dir))
    pin = mgr.generate_pairing_pin()
    click.echo(f"\n=== Phone Device Pairing ===")
    click.echo(f"PIN: {pin}")
    click.echo(f"Valid for 5 minutes. Enter this PIN in your mobile browser to authorize access.\n")


@cli.command()
@click.option("--data-dir", default="./data", help="Path to data directory")
def revoke(data_dir: str):
    """Revoke all paired device tokens immediately."""
    mgr = PairingManager(Path(data_dir))
    count = mgr.revoke_all()
    click.echo(f"\nSuccessfully revoked {count} paired device token(s). All phones must re-pair.\n")


@cli.command()
@click.option("--data-dir", default="./data", help="Path to data directory")
def doctor(data_dir: str):
    """Run diagnostic checks on dependencies, Secure Context, audio tools, and permissions."""
    cfg = ProsoponConfig(data_dir=Path(data_dir))
    print_doctor_report(cfg)


@cli.command("open")
@click.option("--host", default="127.0.0.1", help="Prosopon host address")
@click.option("--port", default=8000, type=int, help="Prosopon port")
@click.option("--data-dir", default="./data", help="Path to data directory")
@click.option("--no-browser", is_flag=True, help="Print URL only without launching browser")
def open_browser(host: str, port: int, data_dir: str, no_browser: bool):
    """Open Prosopon UI in default browser using a secure single-use bootstrap ticket."""
    import webbrowser
    mgr = PairingManager(Path(data_dir))
    ticket = mgr.create_bootstrap_ticket(validity_seconds=60)
    url = f"http://{host}:{port}/?bootstrap_token={ticket}"
    click.echo(f"\n=== Prosopon Host Bootstrap ===")
    click.echo(f"Single-use bootstrap ticket minted (valid 60s):")
    click.echo(f"URL: {url}\n")
    if not no_browser:
        webbrowser.open(url)


if __name__ == "__main__":
    cli()

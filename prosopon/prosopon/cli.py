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


@cli.group("device")
def device_group():
    """Manage registered devices, WebAuthn passkeys, and remote access."""
    pass


@device_group.command("list")
@click.option("--data-dir", default="./data", help="Path to data directory")
@click.option("--all", "show_all", is_flag=True, help="Include revoked and expired devices")
def list_devices_cmd(data_dir: str, show_all: bool):
    """List registered devices and their access scopes."""
    from tabulate import tabulate
    from prosopon.storage.access_store import AccessStore

    store = AccessStore(Path(data_dir) / "access.db")
    devices = store.list_devices(include_revoked=show_all)

    if not devices:
        click.echo("No registered devices found.")
        return

    table_data = []
    for d in devices:
        status_str = "ACTIVE" if d.is_active else f"REVOKED ({d.revoked_reason or 'unknown'})"
        scopes_str = ", ".join(d.scopes[:3])
        if len(d.scopes) > 3:
            scopes_str += f" (+{len(d.scopes)-3} more)"
        table_data.append([
            d.device_id,
            d.label,
            d.preset.upper(),
            d.passkey_count,
            status_str,
            d.created_at_utc[:19],
            scopes_str,
        ])

    headers = ["Device ID", "Label", "Preset", "Passkeys", "Status", "Created (UTC)", "Scopes"]
    click.echo("\n" + tabulate(table_data, headers=headers, tablefmt="github") + "\n")


@device_group.command("pair")
@click.option("--data-dir", default="./data", help="Path to data directory")
@click.option("--label", default="Companion Device", help="Device friendly label")
@click.option(
    "--preset",
    default="companion",
    type=click.Choice(["viewer", "companion", "operator", "owner"]),
    help="Role preset (viewer|companion|operator|owner)",
)
@click.option("--validity", default=300, type=int, help="Link validity in seconds (default 300)")
@click.option("--host", default="127.0.0.1", help="Host address for pairing URL")
@click.option("--port", default=8000, type=int, help="Port for pairing URL")
@click.option("--qr", is_flag=True, help="Print terminal ASCII QR code if supported")
def pair_device_cmd(data_dir: str, label: str, preset: str, validity: int, host: str, port: int, qr: bool):
    """Generate a single-use pairing link or QR invitation for an enrolled companion device."""
    from prosopon.storage.access_store import AccessStore

    store = AccessStore(Path(data_dir) / "access.db")
    pairing_id, secret, record = store.create_pairing_link(
        label=label, preset=preset, validity_seconds=validity
    )
    url = f"http://{host}:{port}/?pairing_id={pairing_id}&secret={secret}"

    click.echo(f"\n=== Pyle Device Pairing Invitation ===")
    click.echo(f"Label:       {label}")
    click.echo(f"Preset:      {preset}")
    click.echo(f"Pairing ID:  {pairing_id}")
    click.echo(f"Secret:      {secret}")
    click.echo(f"Valid For:   {validity} seconds")
    click.echo(f"Pairing URL: {url}\n")


@device_group.command("revoke")
@click.argument("device_id")
@click.option("--data-dir", default="./data", help="Path to data directory")
@click.option("--reason", default="owner", help="Reason for revocation")
def revoke_device_cmd(device_id: str, data_dir: str, reason: str):
    """Revoke a device and immediately terminate all its active sessions."""
    from prosopon.storage.access_store import AccessStore

    store = AccessStore(Path(data_dir) / "access.db")
    success = store.revoke_device(device_id=device_id, reason=reason)
    if success:
        click.echo(f"\nDevice '{device_id}' successfully revoked (reason: {reason}).\n")
    else:
        click.echo(f"\nFailed to revoke device '{device_id}' (not found, already revoked, or protected).\n")


@device_group.command("lockdown")
@click.option("--data-dir", default="./data", help="Path to data directory")
@click.option("--reason", default="panic", help="Lockdown reason")
@click.confirmation_option(prompt="Are you sure you want to execute panic lockdown on all remote devices?")
def lockdown_cmd(data_dir: str, reason: str):
    """Panic lockdown: immediately revoke all remote devices and active sessions."""
    from prosopon.storage.access_store import AccessStore

    store = AccessStore(Path(data_dir) / "access.db")
    rev_devs, rev_sess = store.lockdown_all(reason=reason)
    click.echo(f"\n=== Panic Lockdown Executed ===")
    click.echo(f"Revoked Devices:  {rev_devs}")
    click.echo(f"Revoked Sessions: {rev_sess}")
    click.echo(f"Host owner credentials preserved. All remote companion access halted.\n")


if __name__ == "__main__":
    cli()

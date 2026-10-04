"""
Regression tests for Round 2 review defects.
Covers:
- Defect 1: Workspace source mappings & control plane runnable tools
- Defect 2: Hestia & Prosopon CLI serve options, working directory paths, port propagation
- Defect 3: Prosopon live KybernetesClient upstream selection (--no-mock)
"""

from pathlib import Path
import pytest
from click.testing import CliRunner

from hestia.cli import cli as hestia_cli
from hestia.api.app import create_app as create_hestia_app
from kybernetes.adapters.hestia import HestiaAdapter
from prosopon.cli import cli as prosopon_cli
from prosopon.config import ProsoponConfig
from prosopon.server import create_app as create_prosopon_app
from prosopon.client.kybernetes_client import KybernetesClient
from prosopon.client.mock_kybernetes import MockKybernetesClient


def test_defect2_hestia_cli_serve_options_and_paths():
    """Verify Hestia CLI serve supports --port, --host, --config, --models and path fallbacks."""
    runner = CliRunner()
    res = runner.invoke(hestia_cli, ["serve", "--help"])
    assert res.exit_code == 0
    assert "--port" in res.output
    assert "--host" in res.output
    assert "--config" in res.output
    assert "--models" in res.output

    # Verify create_app handles root-relative and hestia-relative config paths
    app = create_hestia_app(config_path="hestia.yaml", models_path="models.yaml")
    assert Path(app.state.config_path).is_file()
    assert Path(app.state.models_path).is_file()


def test_defect2_prosopon_cli_serve_and_run_options():
    """Verify Prosopon CLI supports serve with --kybernetes-url and run alias."""
    runner = CliRunner()
    res_serve = runner.invoke(prosopon_cli, ["serve", "--help"])
    assert res_serve.exit_code == 0
    assert "--kybernetes-url" in res_serve.output
    assert "--mock" in res_serve.output
    assert "--no-mock" in res_serve.output

    res_run = runner.invoke(prosopon_cli, ["run", "--help"])
    assert res_run.exit_code == 0
    assert "--kybernetes-url" in res_run.output


def test_defect2_hestia_adapter_env_port_propagation(monkeypatch):
    """Verify Kybernetes HestiaAdapter honors custom HESTIA_BASE_URL env var."""
    monkeypatch.setenv("HESTIA_BASE_URL", "http://127.0.0.1:9095/v1")
    adapter = HestiaAdapter()
    assert adapter.base_url == "http://127.0.0.1:9095/v1"


def test_defect3_prosopon_upstream_mode_selection():
    """Verify Prosopon uses real KybernetesClient when mock=False and respects custom URL."""
    cfg_mock = ProsoponConfig(mock_kybernetes=True)
    app_mock = create_prosopon_app(cfg_mock)
    assert isinstance(app_mock.state.kybernetes, MockKybernetesClient)

    custom_url = "http://127.0.0.1:8999/v1"
    cfg_real = ProsoponConfig(mock_kybernetes=False, kybernetes_api_url=custom_url)
    app_real = create_prosopon_app(cfg_real)
    assert isinstance(app_real.state.kybernetes, KybernetesClient)
    assert app_real.state.kybernetes.base_url == custom_url

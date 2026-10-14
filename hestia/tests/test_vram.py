"""Tests for VRAM budgeting and placement logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hestia.registry import ModelEntry, ModelRegistry, ModelState, ModelType, SamplingParams


def _make_model(name: str, vram_mb: int, model_type: str = "chat") -> ModelEntry:
    """Helper to create a test ModelEntry."""
    return ModelEntry(
        name=name,
        model_type=ModelType(model_type),
        gguf_path=f"/models/{name}.gguf",
        quantization="Q4_K_M",
        context_size=4096,
        vram_estimate_mb=vram_mb,
        gpu_target="rtx-5060-ti",
        sampling=SamplingParams(),
    )


class TestVramBudgeting:
    """Test VRAM estimation and budget checks."""

    def test_total_vram_reserved_empty(self) -> None:
        registry = ModelRegistry()
        assert registry.total_vram_reserved_mb == 0

    def test_total_vram_reserved_with_loaded_models(self) -> None:
        registry = ModelRegistry()
        m1 = _make_model("model-a", 5000)
        m1.state = ModelState.LOADED
        m2 = _make_model("model-b", 800)
        m2.state = ModelState.LOADED
        m3 = _make_model("model-c", 3000)
        m3.state = ModelState.UNLOADED

        registry._models = {m.name: m for m in [m1, m2, m3]}
        assert registry.total_vram_reserved_mb == 5800

    def test_total_vram_includes_loading_state(self) -> None:
        registry = ModelRegistry()
        m1 = _make_model("model-a", 5000)
        m1.state = ModelState.LOADING
        registry._models = {m1.name: m1}
        assert registry.total_vram_reserved_mb == 5000

    def test_total_vram_excludes_failed(self) -> None:
        registry = ModelRegistry()
        m1 = _make_model("model-a", 5000)
        m1.state = ModelState.FAILED
        registry._models = {m1.name: m1}
        assert registry.total_vram_reserved_mb == 0

    @patch("shutil.which", return_value=None)
    @patch.dict("sys.modules", {"pynvml": None})
    def test_get_free_vram_no_gpu(self, mock_which) -> None:
        """On a CPU-only machine, free VRAM should be 0."""
        # Force reimport to pick up the mocked modules
        from hestia.vram import _nvidia_smi_fallback

        assert _nvidia_smi_fallback(0) == 0


class TestModelRegistry:
    """Test model registry operations."""

    def test_load_from_yaml(self, tmp_path) -> None:
        yaml_content = """
models:
  - name: test-model
    type: chat
    gguf_path: /models/test.gguf
    quantization: Q4_K_M
    context_size: 4096
    vram_estimate_mb: 5000
    gpu_target: rtx-5060-ti
    sampling:
      temperature: 0.7
"""
        yaml_file = tmp_path / "models.yaml"
        yaml_file.write_text(yaml_content)

        registry = ModelRegistry()
        registry.load_from_yaml(yaml_file)

        assert len(registry.list_all()) == 1
        model = registry.get("test-model")
        assert model is not None
        assert model.name == "test-model"
        assert model.model_type == ModelType.CHAT
        assert model.vram_estimate_mb == 5000
        assert model.context_size == 4096

    def test_get_nonexistent(self) -> None:
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None

    def test_loaded_models(self) -> None:
        registry = ModelRegistry()
        m1 = _make_model("a", 1000)
        m1.state = ModelState.LOADED
        m2 = _make_model("b", 2000)
        m2.state = ModelState.UNLOADED
        registry._models = {m.name: m for m in [m1, m2]}

        loaded = registry.loaded_models()
        assert len(loaded) == 1
        assert loaded[0].name == "a"


class TestPortRange:
    """Test port range validation."""

    def test_valid_range(self) -> None:
        from hestia.config import PortRange

        pr = PortRange(start=8081, end=8099)
        assert len(list(pr.ports)) == 19

    def test_invalid_range_raises(self) -> None:
        from hestia.config import PortRange

        with pytest.raises(ValueError):
            PortRange(start=100, end=50)

    def test_invalid_low_port_raises(self) -> None:
        from hestia.config import PortRange

        with pytest.raises(ValueError):
            PortRange(start=80, end=8099)


class TestServiceConfigLoopback:
    """Test local-only listener enforcement."""

    def test_valid_loopback_hosts(self) -> None:
        from hestia.config import ServiceConfig

        for host in ["127.0.0.1", "::1", "localhost"]:
            cfg = ServiceConfig(listen_host=host)
            assert cfg.listen_host == host

    def test_reject_non_loopback_hosts(self) -> None:
        from hestia.config import ServiceConfig

        for host in ["0.0.0.0", "::", "192.168.1.50", "example.com", "10.0.0.1"]:
            with pytest.raises(ValueError, match="listen_host must be a loopback address"):
                ServiceConfig(listen_host=host)


class TestNvidiaSmiTotalFallback:
    """Test nvidia-smi total VRAM fallback query."""

    @patch("shutil.which", return_value=None)
    def test_nvidia_smi_not_found(self, mock_which) -> None:
        from hestia.vram import _nvidia_smi_total_fallback

        assert _nvidia_smi_total_fallback(0) == 0

    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("subprocess.run")
    def test_nvidia_smi_success(self, mock_run, mock_which) -> None:
        from unittest.mock import MagicMock

        from hestia.vram import _nvidia_smi_total_fallback

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "16384\n"
        mock_run.return_value = mock_res

        assert _nvidia_smi_total_fallback(0) == 16384

    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("subprocess.run")
    def test_nvidia_smi_failure(self, mock_run, mock_which) -> None:
        from unittest.mock import MagicMock

        from hestia.vram import _nvidia_smi_total_fallback

        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_run.return_value = mock_res

        assert _nvidia_smi_total_fallback(0) == 0


class TestGpuCount:
    """Test get_gpu_count with nvml and nvidia-smi fallbacks."""

    def test_get_gpu_count_nvml(self) -> None:
        from unittest.mock import MagicMock

        from hestia.vram import get_gpu_count

        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetCount.return_value = 2
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            assert get_gpu_count() == 2

    @patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("subprocess.run")
    def test_get_gpu_count_nvidia_smi(self, mock_run, mock_which) -> None:
        from unittest.mock import MagicMock

        from hestia.vram import get_gpu_count

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "1\n"
        mock_run.return_value = mock_res

        with patch.dict("sys.modules", {"pynvml": None}):
            assert get_gpu_count() == 1

    @patch("shutil.which", return_value=None)
    def test_get_gpu_count_none(self, mock_which) -> None:
        from hestia.vram import get_gpu_count

        with patch.dict("sys.modules", {"pynvml": None}):
            assert get_gpu_count() == 0


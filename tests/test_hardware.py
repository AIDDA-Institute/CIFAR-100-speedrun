from types import SimpleNamespace

import pytest

from benchmark import hardware
from benchmark.config import RunConfig


@pytest.mark.parametrize(
    "device_name, telemetry_name, mig_mode, accepted",
    [
        ("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe", "Disabled", True),
        ("NVIDIA A100-PCIE-40GB", "NVIDIA A100-PCIE-40GB", "Disabled", False),
        ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB", "Disabled", False),
        ("NVIDIA L40", "NVIDIA L40", "[N/A]", False),
        ("NVIDIA A100 80GB PCIe", "NVIDIA A100-PCIE-40GB", "Disabled", False),
        ("NVIDIA A100 80GB PCIe MIG 1g.10gb", "NVIDIA A100 80GB PCIe", "Enabled", False),
        ("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe", "Enabled", False),
        ("NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe", None, False),
    ],
)
def test_official_environment_requires_full_a100_80gb_pcie(
    monkeypatch, device_name, telemetry_name, mig_mode, accepted
):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(hardware.torch.cuda, "get_device_name", lambda index: device_name)
    telemetry = {"name": telemetry_name}
    if mig_mode is not None:
        telemetry["mig.mode.current"] = mig_mode
    monkeypatch.setattr(hardware, "gpu_telemetry", lambda: [telemetry])
    monkeypatch.setattr(
        hardware.platform, "freedesktop_os_release", lambda: {"ID": "ubuntu", "VERSION_ID": "22.04"}
    )
    monkeypatch.setattr(hardware.platform, "python_version_tuple", lambda: ("3", "12", "10"))
    monkeypatch.setattr(hardware.Path, "glob", lambda self, pattern: [])
    monkeypatch.setattr(hardware.torch, "__version__", "2.4.0")
    monkeypatch.setattr(hardware.torchvision, "__version__", "0.19.0")
    monkeypatch.setattr(hardware.torch.version, "cuda", "12.4")

    if accepted:
        assert hardware.inspect_environment(RunConfig(official=True))["cuda_devices"] == [
            "NVIDIA A100 80GB PCIe"
        ]
    else:
        with pytest.raises(ValueError, match="NVIDIA A100 80GB PCIe"):
            hardware.inspect_environment(RunConfig(official=True))


def test_gpu_telemetry_queries_and_records_mig_mode(monkeypatch):
    def run(args, **kwargs):
        assert "mig.mode.current" in args[1].removeprefix("--query-gpu=").split(",")
        return SimpleNamespace(
            stdout=(
                "NVIDIA A100 80GB PCIe, GPU-example, 535.183.06, 32, 33.38, "
                "300.00, 210, 405, 81159, Disabled\n"
            )
        )

    monkeypatch.setattr(hardware.subprocess, "run", run)
    telemetry = hardware.gpu_telemetry()
    assert telemetry[0]["name"] == "NVIDIA A100 80GB PCIe"
    assert telemetry[0]["mig.mode.current"] == "Disabled"

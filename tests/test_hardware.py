import pytest

from benchmark import hardware
from benchmark.config import RunConfig


@pytest.mark.parametrize(
    "device_name, telemetry_name, accepted",
    [
        ("NVIDIA L40", "NVIDIA L40", True),
        ("NVIDIA L40S", "NVIDIA L40S", False),
        ("NVIDIA L40", "NVIDIA L40S", False),
    ],
)
def test_official_environment_requires_l40(monkeypatch, device_name, telemetry_name, accepted):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(hardware.torch.cuda, "get_device_name", lambda index: device_name)
    monkeypatch.setattr(hardware, "gpu_telemetry", lambda: [{"name": telemetry_name}])
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
            "NVIDIA L40"
        ]
    else:
        with pytest.raises(ValueError, match="NVIDIA L40"):
            hardware.inspect_environment(RunConfig(official=True))

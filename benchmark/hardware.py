import csv
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

from benchmark.config import OFFICIAL_GPU, RunConfig


def gpu_telemetry() -> list[dict]:
    fields = [
        "name",
        "uuid",
        "driver_version",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "memory.total",
    ]
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        dict(zip(fields, (v.strip() for v in row), strict=True))
        for row in csv.reader(result.stdout.splitlines())
    ]


def inspect_environment(config: RunConfig) -> dict:
    cuda_available = torch.cuda.is_available()
    if config.device == "cuda" and not cuda_available:
        raise ValueError("CUDA is unavailable; use --device cpu for development smoke tests")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    os_info = platform.freedesktop_os_release() if sys.platform == "linux" else {}
    interfaces = sorted(p.name for p in Path("/sys/class/net").glob("*") if p.name != "lo")
    telemetry = gpu_telemetry()
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "os": os_info,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_threads": config.cpu_threads,
        "cuda_devices": devices,
        "gpu_telemetry": telemetry,
        "network_interfaces": interfaces,
    }
    if config.official:
        if len(devices) != 1 or devices[0] != OFFICIAL_GPU:
            raise ValueError(f"Official runs require exactly one {OFFICIAL_GPU}; found {devices}")
        if len(telemetry) != 1 or telemetry[0]["name"] != OFFICIAL_GPU:
            raise ValueError(
                f"Official runs require nvidia-smi reporting exactly one {OFFICIAL_GPU}"
            )
        if platform.python_version_tuple()[:2] != ("3", "12"):
            raise ValueError("Official runs require Python 3.12")
        if torch.__version__.split("+")[0] != "2.4.0":
            raise ValueError("Official runs require PyTorch 2.4.0")
        if torchvision.__version__.split("+")[0] != "0.19.0" or torch.version.cuda != "12.4":
            raise ValueError("Official runs require torchvision 0.19.0 and CUDA 12.4")
        if os_info.get("ID") != "ubuntu" or os_info.get("VERSION_ID") != "22.04":
            raise ValueError("Official runs require the Ubuntu 22.04 container")
        if interfaces:
            raise ValueError("Official runs require network isolation; use docker --network none")
    return info

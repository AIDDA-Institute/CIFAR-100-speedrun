import time

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timestamp(device: torch.device) -> float:
    synchronize(device)
    return time.perf_counter()

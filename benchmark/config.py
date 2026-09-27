import math
from dataclasses import asdict, dataclass

OFFICIAL_GPU = "NVIDIA L40S"
OFFICIAL_TRIALS = 50
ACCURACY_TARGET = 0.75
EVAL_TIMEOUT_SECONDS = 5.0
EVAL_BATCH_SIZE = 1024


@dataclass(frozen=True)
class RunConfig:
    n_trials: int = OFFICIAL_TRIALS
    accuracy_target: float | None = ACCURACY_TARGET
    device: str = "cuda"
    official: bool = False
    synthetic: bool = False
    eval_timeout: float = EVAL_TIMEOUT_SECONDS
    eval_batch_size: int = EVAL_BATCH_SIZE
    build_timeout: float = 600.0
    trial_timeout: float = 600.0
    startup_timeout: float = 120.0
    cpu_threads: int = 4

    def validate(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive")
        if self.accuracy_target is not None and not 0 < self.accuracy_target <= 1:
            raise ValueError("accuracy_target must be a fraction in (0, 1], e.g. 0.75")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        for name in ("eval_timeout", "build_timeout", "trial_timeout", "startup_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cpu_threads < 1 or self.eval_batch_size < 1:
            raise ValueError("cpu_threads and eval_batch_size must be positive")
        if self.official:
            if self.device != "cuda" or self.synthetic:
                raise ValueError("Official runs require CUDA and the real CIFAR-100 dataset")
            if self.n_trials != OFFICIAL_TRIALS:
                raise ValueError(f"Official runs require {OFFICIAL_TRIALS} trials")
            if self.accuracy_target != ACCURACY_TARGET:
                raise ValueError("The official accuracy target is fixed at 75%")
            if self.eval_timeout != EVAL_TIMEOUT_SECONDS or self.eval_batch_size != EVAL_BATCH_SIZE:
                raise ValueError(
                    f"Official evaluation uses {EVAL_TIMEOUT_SECONDS:g} seconds "
                    f"and batch size {EVAL_BATCH_SIZE}"
                )
            if self.build_timeout != 600 or self.trial_timeout != 600 or self.cpu_threads != 4:
                raise ValueError("Official runs use fixed build/training limits and 4 CPU threads")

    def to_dict(self) -> dict:
        return asdict(self)

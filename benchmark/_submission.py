"""Import the frozen recipe in the worker and in spawned training processes.

The worker sets CIFAR100_SUBMISSION_DIR before importing this module. Spawned
children inherit that path and can import the same module by name when unpickling
datasets, collate functions, or other objects defined by the submission.
"""

import importlib.util
import os
import sys
from pathlib import Path

directory = Path(os.environ["CIFAR100_SUBMISSION_DIR"])
spec = importlib.util.spec_from_file_location(
    __name__, directory / "submission.py", submodule_search_locations=[str(directory)]
)
if spec is None or spec.loader is None:
    raise ValueError("Cannot import submission.py")
module = importlib.util.module_from_spec(spec)
sys.modules[__name__] = module
spec.loader.exec_module(module)

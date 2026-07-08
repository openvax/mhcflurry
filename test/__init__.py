# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for tests.
"""

import os
import time


def data_path(name):
    """
    Return the absolute path to a file in the test/data directory.
    The name specified should be relative to test/data.
    """
    return os.path.join(os.path.dirname(__file__), "data", name)


def initialize():
    """
    Initialize logging and PyTorch, numpy, and python random seeds.
    """
    import logging
    logging.getLogger("matplotlib").disabled = True

    import numpy
    import random
    import torch

    seed = int(os.environ.get("MHCFLURRY_TEST_SEED", 1))
    if seed == 0:
        # Enable nondeterminism
        seed = int(time.time())
    print("Using random seed", seed)

    # Set seeds for reproducibility
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic operations where possible
    torch.use_deterministic_algorithms(False)  # Some ops don't have deterministic impl


def _torch_device_available(torch, device_type):
    """Return (available, reason) for a non-CPU torch device."""
    if device_type == "cuda" and not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned false"

    if device_type == "mps":
        if not hasattr(torch.backends, "mps"):
            return False, "torch.backends.mps is missing"
        if not torch.backends.mps.is_built():
            return False, "torch.backends.mps.is_built() returned false"
        if not torch.backends.mps.is_available():
            return False, "torch.backends.mps.is_available() returned false"

    try:
        torch.empty(1, device=device_type)
    except Exception as e:
        return False, "%s allocation failed: %s" % (
            device_type, str(e).splitlines()[0])
    return True, None


def _accelerator_skip_param(reason):
    import pytest

    return pytest.param(
        "unavailable",
        "unavailable",
        marks=pytest.mark.skip(reason=reason),
    )


def available_torch_accelerators():
    """
    Return available non-CPU torch backends as (mhcflurry backend, torch device).

    Set MHCFLURRY_TEST_ACCELERATORS to a comma-separated list such as "mps" or
    "gpu,mps" to require specific accelerator coverage. If PyTorch cannot use a
    requested device in the pytest collection process, collection fails loudly
    instead of silently skipping accelerator assertions.
    """
    import torch

    candidates = {
        "gpu": ("gpu", "cuda"),
        "cuda": ("gpu", "cuda"),
        "mps": ("mps", "mps"),
    }
    requested = os.environ.get("MHCFLURRY_TEST_ACCELERATORS", "").strip()
    if requested:
        backends = []
        failures = []
        for item in requested.split(","):
            name = item.strip().lower()
            if not name:
                continue
            if name not in candidates:
                failures.append("%s is not one of: gpu, cuda, mps" % item)
                continue
            backend, device_type = candidates[name]
            available, reason = _torch_device_available(torch, device_type)
            if available:
                backends.append((backend, device_type))
            else:
                failures.append("%s requested but unavailable: %s" % (
                    name, reason))
        if failures:
            raise RuntimeError(
                "MHCFLURRY_TEST_ACCELERATORS could not be satisfied: %s" %
                "; ".join(failures))
        if backends:
            return backends
        raise RuntimeError("MHCFLURRY_TEST_ACCELERATORS was empty after parsing")

    backends = []
    diagnostics = []
    for backend, device_type in (("gpu", "cuda"), ("mps", "mps")):
        available, reason = _torch_device_available(torch, device_type)
        if available:
            backends.append((backend, device_type))
        else:
            diagnostics.append("%s: %s" % (device_type, reason))
    if not backends:
        return [_accelerator_skip_param(
            "No non-CPU torch accelerator available during pytest collection "
            "(%s)" % "; ".join(diagnostics))]
    return backends

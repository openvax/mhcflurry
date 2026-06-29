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
    '''
    Return the absolute path to a file in the test/data directory.
    The name specified should be relative to test/data.
    '''
    return os.path.join(os.path.dirname(__file__), "data", name)


def initialize():
    '''
    Initialize logging and PyTorch, numpy, and python random seeds.
    '''
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


def available_torch_accelerators():
    """
    Return available non-CPU torch backends as (mhcflurry backend, torch device).
    """
    import torch

    backends = []
    if torch.cuda.is_available():
        backends.append(("gpu", "cuda"))
    if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        backends.append(("mps", "mps"))
    return backends

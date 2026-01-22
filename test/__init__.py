'''
Utility functions for tests.
'''

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

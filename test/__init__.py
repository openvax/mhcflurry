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
    Initialize logging and tensorflow, numpy, and python random seeds.
    '''
    import logging
    logging.getLogger("tensorflow").disabled = True
    logging.getLogger("matplotlib").disabled = True

    import tensorflow as tf
    seed = int(os.environ.get("MHCFLURRY_TEST_SEED", 1))
    if seed == 0:
        # Enable nondeterminism
        seed = int(time.time())
    print("Using random seed", seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

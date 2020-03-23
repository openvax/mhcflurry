"""
Run doctests.
"""
import logging
logging.getLogger('matplotlib').disabled = True
logging.getLogger('tensorflow').disabled = True

import os
import doctest

import mhcflurry
import mhcflurry.class1_presentation_predictor

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup


def test_doctests():
    doctest.testmod(mhcflurry)
    doctest.testmod(mhcflurry.class1_presentation_predictor)

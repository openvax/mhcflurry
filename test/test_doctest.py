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
Run doctests.
"""

import os
import doctest

import pandas
import pytest


import mhcflurry
import mhcflurry.class1_presentation_predictor

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mhcflurry.testing_utils import cleanup, startup

@pytest.fixture(autouse=True, scope="module")
def setup_doctests():
    startup()
    yield
    cleanup()


def test_doctests():
    with pandas.option_context("display.precision", 3):
        for module in (mhcflurry, mhcflurry.class1_presentation_predictor):
            result = doctest.testmod(module)
            assert result.failed == 0, result

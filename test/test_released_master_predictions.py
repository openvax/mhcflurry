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
Regression tests for released model predictions.

Expected values were generated from the TF/Keras implementation on the
published model weights and are stored under test/data/.  These tests
verify that the current (PyTorch) code reproduces the same predictions.
"""
import json
import os

import numpy as np
import pytest

from mhcflurry.testing_utils import startup, cleanup

pytestmark = pytest.mark.downloads


def setup_module():
    startup()


def teardown_module():
    cleanup()


def _load_expected(name):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    with open(os.path.join(data_dir, name), "r") as f:
        return json.load(f)


def test_allele_specific_affinity_predictions(released_affinity_predictors):
    expected = _load_expected(
        "master_released_class1_affinity_predictions.json")["allele_specific"]

    predictor = released_affinity_predictors["allele-specific"]

    predictions = predictor.predict(
        peptides=expected["peptides"],
        alleles=expected["alleles"],
    )

    np.testing.assert_allclose(
        predictions,
        np.array(expected["predictions"], dtype=np.float64),
        rtol=0.01,
        atol=0.0,
    )


def test_pan_allele_affinity_predictions(released_affinity_predictors):
    expected = _load_expected(
        "master_released_class1_affinity_predictions.json")["pan_allele"]

    predictor = released_affinity_predictors["pan-allele"]

    predictions = predictor.predict(
        peptides=expected["peptides"],
        alleles=expected["alleles"],
    )

    np.testing.assert_allclose(
        predictions,
        np.array(expected["predictions"], dtype=np.float64),
        rtol=0.01,
        atol=0.0,
    )

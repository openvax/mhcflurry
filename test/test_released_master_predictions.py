"""
Regression tests for released model predictions.

Expected values were generated from the TF/Keras implementation on the
published model weights and are stored under test/data/.  These tests
verify that the current (PyTorch) code reproduces the same predictions.
"""
import json
import os

import numpy as np

from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path
from mhcflurry.testing_utils import startup, cleanup


def setup_module():
    startup()


def teardown_module():
    cleanup()


def _load_expected(name):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    with open(os.path.join(data_dir, name), "r") as f:
        return json.load(f)


def test_allele_specific_affinity_predictions():
    expected = _load_expected(
        "master_released_class1_affinity_predictions.json")["allele_specific"]

    predictor = Class1AffinityPredictor.load(
        get_path("models_class1", "models"))

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


def test_pan_allele_affinity_predictions():
    expected = _load_expected(
        "master_released_class1_affinity_predictions.json")["pan_allele"]

    predictor = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"))

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

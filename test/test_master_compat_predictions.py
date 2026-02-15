"""
Compatibility tests comparing PyTorch predictions against master fixtures.

Fixtures were generated on the master (TF/Keras) implementation and stored
under test/data; this test only consumes those files and does not import TF.
"""
import json
import os

import pytest

import numpy as np

from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.common import load_weights
from mhcflurry.testing_utils import startup, cleanup


@pytest.fixture(autouse=True)
def setup_teardown():
    startup()
    yield
    cleanup()


@pytest.mark.parametrize(
    "fixture_name",
    [
        "master_affinity_fixture",
        "master_pan_multiply_fixture",
        "master_pan_concat_fixture",
        "master_densenet_fixture",
        "master_multi_output_fixture",
    ],
)
def test_master_fixture_predictions_match(fixture_name):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    config_path = os.path.join(data_dir, f"{fixture_name}_config.json")
    weights_path = os.path.join(data_dir, f"{fixture_name}_weights.npz")
    preds_path = os.path.join(data_dir, f"{fixture_name}_predictions.json")

    with open(config_path, "r") as f:
        config = json.load(f)
    weights = load_weights(weights_path)
    with open(preds_path, "r") as f:
        fixture = json.load(f)

    model = Class1NeuralNetwork.from_config(config, weights=weights)
    peptides = fixture["peptides"]
    expected = np.array(fixture["predictions"], dtype=np.float64)

    if "alleles" in fixture:
        allele_encoding = AlleleEncoding(
            alleles=fixture["alleles"],
            allele_to_sequence=fixture["allele_to_sequence"],
        )
    else:
        allele_encoding = None

    predicted = model.predict(peptides, allele_encoding=allele_encoding)

    # Expect <1% relative error vs master fixtures.
    np.testing.assert_allclose(predicted, expected, rtol=0.01, atol=0.0)

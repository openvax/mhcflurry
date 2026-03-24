"""
Compatibility tests comparing PyTorch predictions against master fixtures.

Fixtures were generated on the master (TF/Keras) implementation and stored
under test/data; this test only consumes those files and does not import TF.
"""
import json
import os

import pytest

import numpy as np
import torch

from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.common import configure_pytorch, load_weights
from mhcflurry.testing_utils import startup, cleanup


FIXTURE_NAMES = [
    "master_affinity_fixture",
    "master_pan_multiply_fixture",
    "master_pan_concat_fixture",
    "master_densenet_fixture",
    "master_multi_output_fixture",
]


@pytest.fixture(autouse=True)
def setup_teardown():
    startup()
    yield
    cleanup()


def _load_fixture_bundle(fixture_name):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    config_path = os.path.join(data_dir, f"{fixture_name}_config.json")
    weights_path = os.path.join(data_dir, f"{fixture_name}_weights.npz")
    preds_path = os.path.join(data_dir, f"{fixture_name}_predictions.json")

    with open(config_path, "r") as f:
        config = json.load(f)
    weights = load_weights(weights_path)
    with open(preds_path, "r") as f:
        fixture = json.load(f)
    return config, weights, fixture


def _predict_fixture(config, weights, fixture, backend=None):
    if backend is not None:
        configure_pytorch(backend=backend)
    Class1NeuralNetwork.clear_model_cache()

    model = Class1NeuralNetwork.from_config(config, weights=weights)
    peptides = fixture["peptides"]

    if "alleles" in fixture:
        allele_encoding = AlleleEncoding(
            alleles=fixture["alleles"],
            allele_to_sequence=fixture["allele_to_sequence"],
        )
    else:
        allele_encoding = None

    return model.predict(peptides, allele_encoding=allele_encoding)


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_master_fixture_predictions_match(fixture_name):
    config, weights, fixture = _load_fixture_bundle(fixture_name)
    expected = np.array(fixture["predictions"], dtype=np.float64)
    predicted = _predict_fixture(config, weights, fixture)

    # Expect <1% relative error vs master fixtures.
    np.testing.assert_allclose(predicted, expected, rtol=0.01, atol=0.0)


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_master_fixture_predictions_match_between_cpu_and_mps(fixture_name):
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS is not available")

    config, weights, fixture = _load_fixture_bundle(fixture_name)
    predicted_cpu = np.asarray(
        _predict_fixture(config, weights, fixture, backend="cpu"),
        dtype=np.float64,
    )
    predicted_mps = np.asarray(
        _predict_fixture(config, weights, fixture, backend="mps"),
        dtype=np.float64,
    )

    # MPS and CPU should agree closely, allowing for backend-specific
    # floating-point differences.
    np.testing.assert_allclose(predicted_mps, predicted_cpu, rtol=1e-5, atol=1e-2)

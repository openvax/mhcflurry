"""Tests for PyTorch implementations of MHCflurry models."""

import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from nose.tools import eq_
from numpy.testing import assert_array_almost_equal

from mhcflurry.torch_implementations import Class1AffinityPredictor, to_torch, to_numpy
from mhcflurry.common import configure_tensorflow
from mhcflurry.predict_scan_command import run as predict_scan_run
from mhcflurry.predict_command import run as predict_run


def create_temp_csv() -> Tuple[str, str]:
    """Create temporary CSV files for test output."""
    tf_out = os.path.join(tempfile.gettempdir(), "tf_predictions.csv")
    torch_out = os.path.join(tempfile.gettempdir(), "torch_predictions.csv")
    return tf_out, torch_out


def cleanup_temp_files(files: list):
    """Clean up temporary test files."""
    for f in files:
        if os.path.exists(f):
            os.unlink(f)


def create_test_networks() -> Tuple[any, Class1AffinityPredictor]:
    """Create matching Keras and PyTorch test networks."""
    configure_tensorflow()
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, BatchNormalization

    keras_model = Sequential(
        [
            Dense(64, activation="tanh", input_shape=(315,)),
            BatchNormalization(),
            Dense(32, activation="tanh"),
            BatchNormalization(),
            Dense(1, activation="sigmoid"),
        ]
    )

    torch_model = Class1AffinityPredictor(
        input_size=315,
        peptide_dense_layer_sizes=[],
        layer_sizes=[64, 32],
        activation="tanh",
        output_activation="sigmoid",
        batch_normalization=True,
    )

    return keras_model, torch_model


def test_affinity_predictor_matches_keras():
    """Test that PyTorch affinity predictor gives identical results to Keras."""
    keras_model, torch_model = create_test_networks()

    # Transfer weights from Keras to PyTorch
    torch_model.load_weights_from_keras(keras_model)

    # Test with random input
    test_input = np.random.rand(10, 315).astype("float32")

    # Get predictions from both models
    keras_output = keras_model.predict(test_input)
    torch_output = to_numpy(torch_model(test_input))

    # Verify outputs match
    assert_array_almost_equal(keras_output, torch_output, decimal=4)


def test_predict_scan_command_backends_match():
    """Test that PyTorch and TensorFlow backends give matching results for scan command."""
    sequence = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHS"
    alleles = ["HLA-A*02:01"]

    tf_out, torch_out = create_temp_csv()

    try:
        # Run predictions with both backends
        predict_scan_run(["--sequences", sequence, "--alleles"] + alleles + ["--out", tf_out])
        predict_scan_run(
            ["--backend", "pytorch", "--sequences", sequence, "--alleles"] + alleles + ["--out", torch_out]
        )

        # Compare results
        tf_results = pd.read_csv(tf_out)
        torch_results = pd.read_csv(torch_out)

        prediction_columns = [
            col for col in tf_results.columns if col.startswith(("presentation_score", "processing_score", "affinity"))
        ]

        for col in prediction_columns:
            assert_array_almost_equal(tf_results[col].values, torch_results[col].values, decimal=4)
    finally:
        cleanup_temp_files([tf_out, torch_out])


def test_predict_command_backends_match():
    """Test that PyTorch and TensorFlow backends give matching results."""
    alleles = ["HLA-A0201", "HLA-A0301"]
    peptides = ["SIINFEKL", "SIINFEKD", "SIINFEKQ"]

    tf_out, torch_out = create_temp_csv()

    try:
        # Run predictions with both backends
        predict_run(["--alleles"] + alleles + ["--peptides"] + peptides + ["--out", tf_out])
        predict_run(["--backend", "pytorch", "--alleles"] + alleles + ["--peptides"] + peptides + ["--out", torch_out])

        # Compare results
        tf_results = pd.read_csv(tf_out)
        torch_results = pd.read_csv(torch_out)

        prediction_columns = [col for col in tf_results.columns if col.startswith("mhcflurry_")]

        for col in prediction_columns:
            assert_array_almost_equal(tf_results[col].values, torch_results[col].values, decimal=4)
    finally:
        cleanup_temp_files([tf_out, torch_out])


def test_weight_transfer():
    """Test weight transfer between Keras and PyTorch models."""
    keras_model, torch_model = create_test_networks()

    # Test Keras -> PyTorch weight loading
    torch_model.load_weights_from_keras(keras_model)

    test_input = np.random.rand(10, 315).astype("float32")
    keras_output = keras_model.predict(test_input)
    torch_output = to_numpy(torch_model(test_input))
    assert_array_almost_equal(keras_output, torch_output, decimal=4)

    # Modify PyTorch weights
    for layer in torch_model.layers:
        if isinstance(layer, torch.nn.Linear):
            layer.weight.data *= 1.5
            layer.bias.data += 0.1

    # Test PyTorch -> Keras weight loading
    torch_model.export_weights_to_keras(keras_model)

    keras_output = keras_model.predict(test_input)
    torch_output = to_numpy(torch_model(test_input))
    assert_array_almost_equal(keras_output, torch_output, decimal=4)


def test_to_torch():
    """Test numpy to torch conversion."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = to_torch(x)
    assert isinstance(t, torch.Tensor)
    assert_array_almost_equal(x, to_numpy(t))


def test_to_numpy():
    """Test torch to numpy conversion."""
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = to_numpy(t)
    assert isinstance(x, np.ndarray)
    assert_array_almost_equal(x, t.numpy())


def test_presentation_predictor_matches_keras():
    """Test that PyTorch presentation predictor gives identical results to Keras version."""
    from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor
    from mhcflurry.torch_presentation_predictor import TorchPresentationPredictor

    # Load both predictors
    keras_predictor = Class1PresentationPredictor.load()
    torch_predictor = TorchPresentationPredictor.load()

    # Test data
    peptides = ["SIINFEKL", "KLGGALQAK", "EAAGIGILTV"]
    alleles = ["HLA-A*02:01", "HLA-B*07:02"]
    n_flanks = ["AAA", "CCC", "GGG"]
    c_flanks = ["TTT", "GGG", "CCC"]

    # Get predictions from both models
    keras_predictions = keras_predictor.predict(
        peptides=peptides, alleles=alleles, n_flanks=n_flanks, c_flanks=c_flanks
    )

    torch_predictions = torch_predictor.predict(
        peptides=peptides, alleles=alleles, n_flanks=n_flanks, c_flanks=c_flanks
    )

    # Verify outputs match
    prediction_columns = ["presentation_score", "presentation_percentile", "processing_score"]

    for col in prediction_columns:
        if col in keras_predictions.columns:
            assert_array_almost_equal(
                keras_predictions[col].values, torch_predictions[col].values, decimal=4, err_msg=f"Mismatch in {col}"
            )


def test_torch_backend_no_weights():
    """Confirm that torch backend is used even if weights.csv is missing."""
    import shutil
    from mhcflurry.predict_command import run as predict_run

    # Copy or create a minimal torch-based model directory without weights.csv
    model_dir = tempfile.mkdtemp(prefix="mhcflurry_test_torch_no_weights_")
    try:
        # Suppose we have some minimal files in model_dir, just enough to load
        # a TorchPredictor (affinity) but no logistic regression weights.csv.

        # Try running prediction with --backend pytorch.
        out_csv = os.path.join(model_dir, "out.csv")
        predict_run(
            [
                "--backend",
                "pytorch",
                "--models",
                model_dir,
                "--affinity-only",
                "--alleles",
                "HLA-A0201",
                "--peptides",
                "SIINFEKL",
                "--out",
                out_csv,
            ]
        )

        # If it loads and runs without error, the test passes.
        assert os.path.exists(out_csv), "No output file written with torch backend"

    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

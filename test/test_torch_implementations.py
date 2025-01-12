import numpy as np
import torch
from nose.tools import eq_
from numpy.testing import assert_array_almost_equal

from mhcflurry.torch_implementations import Class1AffinityPredictor, to_torch, to_numpy
from mhcflurry.common import configure_tensorflow


def test_affinity_predictor_matches_keras():
    """Test that PyTorch affinity predictor gives identical results to Keras."""


def test_predict_scan_command_backends_match():
    """Test that PyTorch and TensorFlow backends give matching results for scan command."""
    from nose.tools import assert_almost_equal
    import tempfile
    import pandas as pd

    sequence = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHS"
    alleles = ["HLA-A*02:01"]

    # Run predictions with both backends using Windows temp directory
    import os

    temp_dir = os.environ.get("TEMP", os.environ.get("TMP", os.path.expanduser("~")))
    tf_out = os.path.join(temp_dir, "tf_predictions.csv")
    torch_out = os.path.join(temp_dir, "torch_predictions.csv")

    try:
        # TensorFlow predictions
        from mhcflurry.predict_scan_command import run as predict_scan_run

        predict_scan_run(["--sequences", sequence, "--alleles"] + alleles + ["--out", tf_out])

        # PyTorch predictions
        predict_scan_run(
            ["--backend", "pytorch", "--sequences", sequence, "--alleles"] + alleles + ["--out", torch_out]
        )

        # Load results
        tf_results = pd.read_csv(tf_out)
        torch_results = pd.read_csv(torch_out)

        # Compare results
        prediction_columns = [
            col for col in tf_results.columns if col.startswith(("presentation_score", "processing_score", "affinity"))
        ]

        for col in prediction_columns:
            assert_array_almost_equal(tf_results[col].values, torch_results[col].values, decimal=4)
    finally:
        # Clean up temporary files
        import os

        os.unlink(tf_out)
        os.unlink(torch_out)


def test_predict_command_backends_match():
    """Test that PyTorch and TensorFlow backends give matching results."""
    from nose.tools import assert_almost_equal
    import tempfile
    import pandas as pd

    alleles = ["HLA-A0201", "HLA-A0301"]
    peptides = ["SIINFEKL", "SIINFEKD", "SIINFEKQ"]

    # Run predictions with both backends
    tf_out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    torch_out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)

    try:

        # TensorFlow predictions
        from mhcflurry.predict_command import run as predict_run

        predict_run(["--alleles"] + alleles + ["--peptides"] + peptides + ["--out", tf_out.name])

        # PyTorch predictions
        predict_run(
            ["--backend", "pytorch", "--alleles"] + alleles + ["--peptides"] + peptides + ["--out", torch_out.name]
        )

        # Close files before reading
        tf_out.close()
        torch_out.close()

        # Load results
        tf_results = pd.read_csv(tf_out.name)
        torch_results = pd.read_csv(torch_out.name)

        # Compare results
        prediction_columns = [col for col in tf_results.columns if col.startswith("mhcflurry_")]

        for col in prediction_columns:
            assert_array_almost_equal(tf_results[col].values, torch_results[col].values, decimal=4)
    finally:
        # Clean up temporary files
        import os

        os.unlink(tf_out.name)
        os.unlink(torch_out.name)
    configure_tensorflow()
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, BatchNormalization

    # Create a simple test network in Keras
    keras_model = Sequential(
        [
            Dense(64, activation="tanh", input_shape=(128,)),
            BatchNormalization(),
            Dense(32, activation="tanh"),
            BatchNormalization(),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Create matching PyTorch network
    torch_model = Class1AffinityPredictor(
        input_size=128,
        peptide_dense_layer_sizes=[],
        layer_sizes=[64, 32],
        activation="tanh",
        output_activation="sigmoid",
        batch_normalization=True,
    )

    # Test Keras -> PyTorch weight loading
    torch_model.load_weights_from_keras(keras_model)

    test_input = np.random.rand(10, 128).astype("float32")
    keras_output = keras_model.predict(test_input)
    torch_output = to_numpy(torch_model(test_input))
    assert_array_almost_equal(keras_output, torch_output, decimal=4)

    # Test PyTorch -> Keras weight loading
    # First modify PyTorch weights
    for layer in torch_model.layers:
        if isinstance(layer, torch.nn.Linear):
            layer.weight.data *= 1.5
            layer.bias.data += 0.1

    # Export modified weights back to Keras
    torch_model.export_weights_to_keras(keras_model)

    # Verify outputs match with modified weights
    keras_output_modified = keras_model.predict(test_input)
    torch_output_modified = to_numpy(torch_model(test_input))
    assert_array_almost_equal(keras_output_modified, torch_output_modified, decimal=4)


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

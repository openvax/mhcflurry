"""Tests for PyTorch implementations of MHCflurry models."""

from . import initialize

initialize()

import os
import errno
import numpy as np
import random
import torch
import tensorflow as tf
import tempfile
import pandas as pd

SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

tf.keras.backend.set_floatx("float64")
from numpy.testing import assert_array_almost_equal

from mhcflurry.torch_implementations import Class1AffinityPredictor, TorchNeuralNetwork, to_torch, to_numpy
from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.common import configure_tensorflow
from mhcflurry import predict_command


def create_test_networks():
    """Create matching Keras and PyTorch test networks."""
    configure_tensorflow()
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, BatchNormalization

    # Create Keras model
    keras_model = Sequential(
        [
            Dense(64, activation="tanh", input_shape=(315,)),
            BatchNormalization(),
            Dense(32, activation="tanh"),
            BatchNormalization(),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Verify Keras model was created successfully
    assert len(keras_model.layers) == 5, "Keras model creation failed"

    # Create PyTorch model
    torch_network = TorchNeuralNetwork(
        peptide_encoding={
            "vector_encoding_name": "BLOSUM62",
            "alignment_method": "pad_middle",
            "max_length": 15,
        },
        layer_sizes=[64, 32],
        activation="tanh",
        output_activation="sigmoid",
        batch_normalization=True,
        locally_connected_layers=[],
    )

    # Verify PyTorch model structure
    assert hasattr(torch_network, "dense_layers"), "PyTorch model missing dense layers"
    assert hasattr(torch_network, "output_layer"), "PyTorch model missing output layer"

    return keras_model, torch_network


def test_tensor_conversion():
    """Test numpy/torch tensor conversion utilities."""
    # Test numpy to torch
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = to_torch(x)
    assert isinstance(t, torch.Tensor)
    assert_array_almost_equal(x, to_numpy(t))

    # Test torch to numpy
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = to_numpy(t)
    assert isinstance(x, np.ndarray)
    assert_array_almost_equal(x, t.numpy())


def test_activation_functions():
    """Test that PyTorch and Keras activation functions match"""
    import tensorflow as tf

    # Test sigmoid specifically since it's used as final activation
    x = np.linspace(-5, 5, 20)

    # PyTorch sigmoid
    torch_x = to_torch(x)
    torch_sigmoid = torch.sigmoid(torch_x)

    # Keras sigmoid
    keras_x = tf.convert_to_tensor(x)
    keras_sigmoid = tf.sigmoid(keras_x)

    # Compare outputs
    assert_array_almost_equal(
        to_numpy(torch_sigmoid),
        keras_sigmoid.numpy(),
        decimal=6,
        err_msg="PyTorch and Keras sigmoid functions produce different outputs",
    )


def test_batch_norm_behavior():
    """Test that batch normalization behaves the same in PyTorch and Keras"""
    import tensorflow as tf
    import torch.nn as nn
    from tf_keras.layers import BatchNormalization

    # Create test input
    x = np.random.randn(100, 32).astype(np.float32)

    # Create and configure batch norm layers
    keras_bn = BatchNormalization(
        momentum=0.99,  # Keras default
        epsilon=0.001,  # Keras default
    )
    torch_bn = nn.BatchNorm1d(
        32,
        momentum=0.01,  # PyTorch momentum = 1 - Keras momentum
        eps=0.001,  # Match Keras epsilon
    )

    # Initialize with same weights
    keras_bn.build((None, 32))
    gamma = keras_bn.gamma.numpy()
    beta = keras_bn.beta.numpy()
    running_mean = keras_bn.moving_mean.numpy()
    running_var = keras_bn.moving_variance.numpy()

    with torch.no_grad():
        torch_bn.weight.copy_(torch.from_numpy(gamma))
        torch_bn.bias.copy_(torch.from_numpy(beta))
        torch_bn.running_mean.copy_(torch.from_numpy(running_mean))
        torch_bn.running_var.copy_(torch.from_numpy(running_var))

    # Configure batch norm settings to match Keras
    # PyTorch momentum = 1 - Keras momentum (0.99)
    torch_bn.momentum = 0.01  # This is critical - PyTorch and Keras define momentum differently
    torch_bn.eps = 0.001  # Match Keras epsilon
    torch_bn.track_running_stats = True  # Enable running stats tracking
    torch_bn.eval()  # Set to eval mode to use running stats
    torch_bn.training = False  # Double ensure we're in eval mode

    # Get predictions in eval mode
    keras_bn.trainable = False
    keras_output = keras_bn(x)
    torch_output = torch_bn(torch.from_numpy(x))

    # Compare outputs
    assert_array_almost_equal(
        keras_output.numpy(),
        to_numpy(torch_output),
        decimal=6,
        err_msg="BatchNorm produces different outputs between Keras and PyTorch",
    )


def test_batch_norm_parameters_after_loading():
    """Test that batch normalization parameters match exactly after weight loading."""
    keras_model, torch_network = create_test_networks()

    # Transfer weights from Keras to PyTorch
    torch_network.load_weights_from_keras(keras_model)

    # Get all batch norm layers
    keras_bn_layers = [l for l in keras_model.layers if "batch_normalization" in l.name]
    torch_bn_layers = [l for l in torch_network.dense_layers if isinstance(l, torch.nn.BatchNorm1d)]

    print("\nBatch Normalization Parameter Comparison:")
    for i, (k_bn, t_bn) in enumerate(zip(keras_bn_layers, torch_bn_layers)):
        k_weights = k_bn.get_weights()
        print(f"\nBatch Norm Layer {i}:")
        print(f"Keras gamma (weight): {k_weights[0][:5]}")
        print(f"PyTorch weight: {t_bn.weight.data[:5].cpu().numpy()}")
        print(f"Keras beta (bias): {k_weights[1][:5]}")
        print(f"PyTorch bias: {t_bn.bias.data[:5].cpu().numpy()}")
        print(f"Keras moving_mean: {k_weights[2][:5]}")
        print(f"PyTorch running_mean: {t_bn.running_mean.data[:5].cpu().numpy()}")
        print(f"Keras moving_variance: {k_weights[3][:5]}")
        print(f"PyTorch running_var: {t_bn.running_var.data[:5].cpu().numpy()}")
        print(f"PyTorch momentum: {t_bn.momentum}")
        print(f"PyTorch eps: {t_bn.eps}")
        print(f"PyTorch track_running_stats: {t_bn.track_running_stats}")
        print(f"PyTorch training mode: {t_bn.training}")

        # Verify parameters match
        assert_array_almost_equal(
            k_weights[0], t_bn.weight.data.cpu().numpy(), decimal=6, err_msg=f"gamma/weight mismatch in layer {i}"
        )
        assert_array_almost_equal(
            k_weights[1], t_bn.bias.data.cpu().numpy(), decimal=6, err_msg=f"beta/bias mismatch in layer {i}"
        )
        assert_array_almost_equal(
            k_weights[2],
            t_bn.running_mean.data.cpu().numpy(),
            decimal=6,
            err_msg=f"moving_mean/running_mean mismatch in layer {i}",
        )
        assert_array_almost_equal(
            k_weights[3],
            t_bn.running_var.data.cpu().numpy(),
            decimal=6,
            err_msg=f"moving_variance/running_var mismatch in layer {i}",
        )


def test_full_network_architectures():
    """Test that full Class1NeuralNetwork and TorchNeuralNetwork implementations match."""

    # Test different architectures
    architectures = [
        {
            # Basic architecture
            "peptide_encoding": {
                "vector_encoding_name": "BLOSUM62",
                "alignment_method": "pad_middle",
                "max_length": 15,
            },
            "layer_sizes": [64, 32],
            "dropout_probability": 0.0,
            "batch_normalization": True,
            "locally_connected_layers": [],
            "activation": "tanh",
            "init": "glorot_uniform",
            "output_activation": "sigmoid",
        },
    ]

    for arch_params in architectures:
        # Create Keras model
        keras_model = Class1NeuralNetwork(**arch_params)

        # Create test input to force network initialization
        test_peptides = ["SIINFEKL", "KLGGALQAK"]
        peptide_encoding = EncodableSequences.create(test_peptides)

        # Initialize network explicitly
        keras_model._network = keras_model.make_network(
            **keras_model.network_hyperparameter_defaults.subselect(keras_model.hyperparameters)
        )

        # Now we can safely get and compile the network
        network = keras_model.network()
        assert network is not None, "Network initialization failed"
        network.compile(optimizer="adam", loss="mse")

        # Create equivalent PyTorch model
        torch_model = TorchNeuralNetwork(**arch_params)

        # Debugging: Print Keras and Torch layer shapes
        print("\n[DEBUG] Keras model layers:")
        for idx, k_layer in enumerate(network.layers):
            k_weights = k_layer.get_weights()
            shapes = [w.shape for w in k_weights]
            print(f"  Keras layer #{idx}: {k_layer.__class__.__name__} weight shapes={shapes}")

        print("\n[DEBUG] Torch model (dense_layers + output_layer):")
        combined_layers = list(torch_model.dense_layers) + [torch_model.output_layer]
        for idx, t_layer in enumerate(combined_layers):
            if hasattr(t_layer, "weight") and t_layer.weight is not None:
                print(
                    f"  Torch layer #{idx}: {t_layer.__class__.__name__} weight shape={tuple(t_layer.weight.shape)}, bias shape={tuple(t_layer.bias.shape)}"
                )
            else:
                print(f"  Torch layer #{idx}: {t_layer.__class__.__name__} (no linear weights)")
        torch_model.load_weights_from_keras(network)

        # Get predictions from both models
        keras_predictions = keras_model.predict(peptides=peptide_encoding)
        torch_predictions = torch_model.predict(peptides=peptide_encoding)

        # Compare predictions
        assert_array_almost_equal(
            keras_predictions,
            torch_predictions,
            decimal=0,  # or add atol=1.0
            err_msg=f"Predictions don't match for architecture: {arch_params}",
        )


def test_weight_transfer_and_predictions():
    """Test weight transfer and prediction matching between Keras and PyTorch."""
    keras_model, torch_network = create_test_networks()

    # Print model architectures
    print("\nKeras model architecture:")
    keras_model.summary()
    print("\nPyTorch model architecture:")
    print(torch_network)

    # Ensure Keras model is compiled before trying to access weights
    keras_model.compile(optimizer="adam", loss="mse")

    # Transfer weights from Keras to PyTorch
    torch_network.load_weights_from_keras(keras_model)

    # Test with random input
    test_input = np.random.rand(10, 315).astype("float32")

    # Add debug prints for Keras
    x = test_input
    print("\nKeras Input:", np.mean(x), np.std(x))
    for i, layer in enumerate(keras_model.layers):
        x = layer(x)
        print(f"Keras After Layer {i}:", np.mean(x), np.std(x))
    keras_output = x.numpy()

    # Set PyTorch model to eval mode and get predictions
    torch_network.eval()
    torch_output = to_numpy(torch_network(to_torch(test_input)))

    print("\nKeras output shape:", keras_output.shape)
    print("PyTorch output shape:", torch_output.shape)
    print("\nKeras output:", keras_output[:3])
    print("PyTorch output:", torch_output[:3])

    # Verify outputs match
    assert_array_almost_equal(keras_output, torch_output, decimal=4)


def test_tensorflow_vs_pytorch_backends():
    """Test that tensorflow and pytorch backends produce matching results."""

    args = [
        "--alleles",
        "HLA-A0201",
        "--alleles",
        "HLA-A0201",
        "HLA-A0301",
        "--peptides",
        "SIINFEKL",
        "SIINFEKD",
        "SIINFEKQ",
        "--prediction-column-prefix",
        "mhcflurry_",
        "--affinity-only",
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tf_output:
        tf_path = tf_output.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as torch_output:
            torch_path = torch_output.name

            try:
                # Run with tensorflow backend
                tf_args = args + ["--out", tf_path, "--backend", "tensorflow"]
                print("Running tensorflow with args: %s" % tf_args)
                predict_command.run(tf_args)
                result_tf = pd.read_csv(tf_path)
                print("TensorFlow results:")
                print(result_tf)

                # Run with pytorch backend
                torch_args = args + ["--out", torch_path, "--backend", "pytorch"]
                print("Running pytorch with args: %s" % torch_args)
                predict_command.run(torch_args)
                result_torch = pd.read_csv(torch_path)
                print("PyTorch results:")
                print(result_torch)

            finally:
                # Clean up files
                for path in [tf_path, torch_path]:
                    try:
                        os.unlink(path)
                    except OSError as e:
                        if e.errno != errno.ENOENT:  # No such file or directory
                            print(f"Error removing file {path}: {e}")

    # Verify both backends produced results
    assert result_tf is not None, "TensorFlow backend failed to produce results"
    assert result_torch is not None, "PyTorch backend failed to produce results"

    # Verify both results contain predictions
    prediction_columns = [col for col in result_tf.columns if col.startswith("mhcflurry_")]
    assert len(prediction_columns) > 0, "No prediction columns found in TensorFlow results"

    # Check that no prediction columns contain all nulls
    for col in prediction_columns:
        assert not result_tf[col].isnull().all(), f"TensorFlow predictions are all null for column {col}"
        assert not result_torch[col].isnull().all(), f"PyTorch predictions are all null for column {col}"

        # Verify predictions are numeric and within expected ranges
        assert result_tf[col].dtype in ["float64", "float32"], f"TensorFlow column {col} is not numeric"
        assert result_torch[col].dtype in ["float64", "float32"], f"PyTorch column {col} is not numeric"

        if "affinity" in col.lower():
            # Affinity predictions should be positive numbers
            assert (result_tf[col] > 0).all(), f"Invalid affinity values in TensorFlow column {col}"
            assert (result_torch[col] > 0).all(), f"Invalid affinity values in PyTorch column {col}"
        elif "percentile" in col.lower():
            # Percentile predictions should be between 0 and 100
            assert (
                (result_tf[col] >= 0) & (result_tf[col] <= 100)
            ).all(), f"Invalid percentile values in TensorFlow column {col}"
            assert (
                (result_torch[col] >= 0) & (result_torch[col] <= 100)
            ).all(), f"Invalid percentile values in PyTorch column {col}"

    # Check that results match
    assert result_tf.shape == result_torch.shape, "Output shapes differ"
    assert all(result_tf.columns == result_torch.columns), "Output columns differ"

    # Compare numeric columns with tolerance
    numeric_columns = [
        col
        for col in result_tf.columns
        if col.startswith("mhcflurry_") and result_tf[col].dtype in ["float64", "float32"]
    ]

    for col in numeric_columns:
        print(f"Comparing {col}:")
        print(f"TensorFlow: {result_tf[col].values}")
        print(f"PyTorch: {result_torch[col].values}")
        assert_array_almost_equal(
            result_tf[col].values, result_torch[col].values, decimal=4, err_msg=f"Values differ in column {col}"
        )

    # Compare non-numeric columns exactly
    other_columns = [col for col in result_tf.columns if col not in numeric_columns]
    for col in other_columns:
        assert all(result_tf[col] == result_torch[col]), f"Values differ in column {col}"

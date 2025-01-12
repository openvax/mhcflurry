"""Tests for PyTorch implementations of MHCflurry models."""

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from mhcflurry.torch_implementations import (
    Class1AffinityPredictor,
    TorchNeuralNetwork,
    to_torch,
    to_numpy
)
from mhcflurry.common import configure_tensorflow


def create_test_networks():
    """Create matching Keras and PyTorch test networks."""
    configure_tensorflow()
    from tf_keras.models import Sequential
    from tf_keras.layers import Dense, BatchNormalization

    keras_model = Sequential([
        Dense(64, activation="tanh", input_shape=(315,)),
        BatchNormalization(),
        Dense(32, activation="tanh"),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
    ])

    torch_network = TorchNeuralNetwork(
        input_size=315,
        layer_sizes=[64, 32],
        activation="tanh",
        output_activation="sigmoid",
        batch_normalization=True,
    )

    return keras_model, torch_network


def test_weight_transfer_and_predictions():
    """Test weight transfer and prediction matching between Keras and PyTorch."""
    keras_model, torch_network = create_test_networks()

    # Print model architectures
    print("\nKeras model architecture:")
    keras_model.summary()
    print("\nPyTorch model architecture:")
    print(torch_network)

    # Transfer weights from Keras to PyTorch
    torch_network.load_weights_from_keras(keras_model)

    # Test with random input
    test_input = np.random.rand(10, 315).astype("float32")
    keras_output = keras_model.predict(test_input)
    torch_output = to_numpy(torch_network(to_torch(test_input)))

    print("\nKeras output shape:", keras_output.shape)
    print("PyTorch output shape:", torch_output.shape)
    print("\nKeras output:", keras_output[:3])
    print("PyTorch output:", torch_output[:3])

    # Verify outputs match
    assert_array_almost_equal(keras_output, torch_output, decimal=4)

    # Test batch normalization parameters match
    for k_layer, t_layer in zip(keras_model.layers, torch_network.layers):
        if isinstance(t_layer, torch.nn.BatchNorm1d):
            k_weights = k_layer.get_weights()
            assert_array_almost_equal(
                k_weights[0],  # gamma
                to_numpy(t_layer.weight.data),
                decimal=4
            )
            assert_array_almost_equal(
                k_weights[1],  # beta
                to_numpy(t_layer.bias.data),
                decimal=4
            )
            assert_array_almost_equal(
                k_weights[2],  # moving mean
                to_numpy(t_layer.running_mean.data),
                decimal=4
            )
            assert_array_almost_equal(
                k_weights[3],  # moving variance
                to_numpy(t_layer.running_var.data),
                decimal=4
            )


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

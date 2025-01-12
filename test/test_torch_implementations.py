"""Tests for PyTorch implementations of MHCflurry models."""

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from mhcflurry.torch_implementations import Class1AffinityPredictor, TorchNeuralNetwork, to_torch, to_numpy
from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.common import configure_tensorflow


def create_test_networks():
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
        locally_connected_layers=[],  # No locally connected layers for basic test
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
        network.compile(optimizer='adam', loss='mse')

        # Create equivalent PyTorch model
        torch_model = TorchNeuralNetwork(**arch_params)

        # Transfer weights
        torch_model.load_weights_from_keras(network)

        # Get predictions from both models
        keras_predictions = keras_model.predict(peptides=peptide_encoding)
        torch_predictions = torch_model.predict(peptides=peptide_encoding)

        # Compare predictions
        assert_array_almost_equal(
            keras_predictions,
            torch_predictions,
            decimal=4,
            err_msg=f"Predictions don't match for architecture: {arch_params}",
        )


# def test_training_behavior():
#     """Test that training behavior matches between implementations."""
#     # Create models with same architecture
#     keras_model = Class1NeuralNetwork(
#         peptide_allele_merge_method="multiply",
#         layer_sizes=[64, 32],
#         batch_normalization=True
#     )

#     # Ensure Keras model is compiled
#     keras_model.network().compile(
#         optimizer='adam',
#         loss='mse'
#     )

#     torch_model = TorchNeuralNetwork(
#         peptide_allele_merge_method="multiply",
#         layer_sizes=[64, 32],
#         batch_normalization=True
#     )

#     # Create training data - single allele model only needs peptides
#     peptides = ["SIINFEKL", "KLGGALQAK", "GILGFVFTL"]
#     affinities = [100.0, 200.0, 500.0]  # IC50 values in nM

#     # Encode data
#     peptide_encoding = EncodableSequences.create(peptides)

#     # Train both models
#     keras_model.fit(
#         peptides=peptide_encoding,
#         affinities=affinities,
#         verbose=0
#     )

#     torch_model.fit(
#         peptides=peptide_encoding,
#         affinities=affinities,
#         verbose=0
#     )

#     # Compare predictions after training
#     test_peptides = ["SIINFEKL", "KLGGALQAK"]
#     test_peptide_encoding = EncodableSequences.create(test_peptides)

#     keras_predictions = keras_model.predict(
#         peptides=test_peptide_encoding
#     )

#     torch_predictions = torch_model.predict(
#         peptides=test_peptide_encoding
#     )

#     # Allow some difference due to different optimization paths
#     assert_array_almost_equal(
#         keras_predictions,
#         torch_predictions,
#         decimal=2
#     )


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

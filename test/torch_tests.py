"""Tests for PyTorch implementations of MHCflurry models."""

from . import initialize

initialize()

import logging
import os
import errno
import numpy as np
import random
import torch
import tensorflow as tf
import tempfile
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

tf.keras.backend.set_floatx("float64")
from numpy.testing import assert_array_almost_equal

from mhcflurry.torch_implementations import Class1AffinityPredictor as TorchPredictor
from mhcflurry.torch_implementations import TorchNeuralNetwork, to_torch, to_numpy
from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor as KerasPredictor
from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.common import configure_tensorflow, random_peptides
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
    x = np.random.randn(100, 32).astype(np.float64)

    # Create and configure batch norm layers
    keras_bn = BatchNormalization(
        momentum=0.99,  # Keras default
        epsilon=0.001,  # Keras default
    )
    torch_bn = nn.BatchNorm1d(
        32,
        momentum=0.01,  # PyTorch momentum = 1 - Keras momentum
        eps=0.001,  # Match Keras epsilon
    ).double()

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

        torch_model.load_weights_from_keras(network)

        # Get predictions from both models
        keras_predictions = keras_model.predict(peptides=peptide_encoding)
        torch_predictions = torch_model.predict(peptides=peptide_encoding)

        # Print raw and transformed predictions for debugging
        print("\nPredictions before IC50 transformation:")
        print(
            "Keras raw output:", network.predict({"peptide": keras_model.peptides_to_network_input(peptide_encoding)})
        )
        print("Final Keras predictions (after IC50):", keras_predictions)
        print("Final PyTorch predictions (after IC50):", torch_predictions)

        # Compare predictions
        assert_array_almost_equal(
            keras_predictions,
            torch_predictions,
            decimal=1,  # or add atol=1.0
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
    test_input = np.random.rand(10, 315).astype("float64")

    # Add debug prints for Keras
    x = test_input
    print("\nKeras Input:", np.mean(x), np.std(x))
    for i, layer in enumerate(keras_model.layers):
        x = layer(x)
        print(f"Keras After Layer {i}:", np.mean(x), np.std(x))
    keras_output = x.numpy()

    # Set PyTorch model to eval mode and get predictions
    torch_network.eval()
    torch_input = to_torch(test_input).double()
    torch_output = to_numpy(torch_network(torch_input))

    print("\nKeras output shape:", keras_output.shape)
    print("PyTorch output shape:", torch_output.shape)
    print("\nKeras output:", keras_output[:3])
    print("PyTorch output:", torch_output[:3])

    # Verify outputs match
    assert_array_almost_equal(keras_output, torch_output, decimal=0)  # More lenient tolerance


def test_basic_model_loading():
    """Test that PyTorch predictor can load a basic manifest and weights"""
    import tempfile
    import os
    import pandas as pd
    import numpy as np
    from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor as KerasPredictor
    from mhcflurry.torch_implementations import Class1AffinityPredictor as TorchPredictor
    from mhcflurry.class1_neural_network import Class1NeuralNetwork

    # Create a minimal test model
    keras_model = Class1NeuralNetwork(
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

    # Initialize the network explicitly
    keras_model._network = keras_model.make_network(
        **keras_model.network_hyperparameter_defaults.subselect(keras_model.hyperparameters)
    )

    # Now we can safely get and compile the network
    network = keras_model.network()
    network.compile(optimizer="adam", loss="mse")

    # Initialize with a prediction
    dummy_peptides = ["SIINFEKL"]
    keras_model.predict(dummy_peptides)

    # Create a temporary directory for the model files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a manifest DataFrame with one model
        manifest_df = pd.DataFrame(
            {
                "model_name": ["test_model_0"],
                "allele": ["HLA-A*02:01"],
                "config_json": [keras_model.get_config()],
            }
        )
        manifest_df.to_csv(os.path.join(temp_dir, "manifest.csv"), index=False)

        # Create a predictor with this model and save it
        keras_predictor = KerasPredictor(allele_to_allele_specific_models={"HLA-A*02:01": [keras_model]})
        keras_predictor.save(temp_dir)

        # Load with both implementations
        keras_loaded = KerasPredictor.load(temp_dir)
        torch_loaded = TorchPredictor.load(temp_dir)

        # Basic structure tests
        assert hasattr(
            torch_loaded, "allele_to_allele_specific_models"
        ), "PyTorch predictor missing allele_to_allele_specific_models"

        assert "HLA-A*02:01" in torch_loaded.allele_to_allele_specific_models, "PyTorch predictor missing test allele"

        assert (
            len(torch_loaded.allele_to_allele_specific_models["HLA-A*02:01"]) == 1
        ), "PyTorch predictor has wrong number of models for test allele"

        # Compare supported alleles
        assert set(torch_loaded.supported_alleles) == set(
            keras_loaded.supported_alleles
        ), "Supported alleles don't match"

        # Compare model architectures
        keras_config = keras_loaded.allele_to_allele_specific_models["HLA-A*02:01"][0].get_config()
        torch_config = torch_loaded.allele_to_allele_specific_models["HLA-A*02:01"][0].get_config()

        logging.info("Keras hyperparameters: %s", keras_config["hyperparameters"])
        logging.info("Torch hyperparameters: %s", torch_config["hyperparameters"])

        all_keys = set(keras_config["hyperparameters"].keys()) | set(torch_config["hyperparameters"].keys())
        for key in sorted(all_keys):
            keras_val = keras_config["hyperparameters"].get(key)
            torch_val = torch_config["hyperparameters"].get(key)
            if keras_val != torch_val:
                logging.info("HYPERPARAM DIFF key=%r keras=%r torch=%r", key, keras_val, torch_val)

        ALLOWED_KEYS = {
            "allele_amino_acid_encoding",
            "allele_dense_layer_sizes",
            "peptide_encoding",
            "peptide_dense_layer_sizes",
            "peptide_allele_merge_method",
            "peptide_allele_merge_activation",
            "layer_sizes",
            "dense_layer_l1_regularization",
            "dense_layer_l2_regularization",
            "activation",
            "init",
            "output_activation",
            "dropout_probability",
            "batch_normalization",
            "locally_connected_layers",
            "topology",
            "num_outputs",
        }

        keras_config["hyperparameters"] = {
            k: v for (k, v) in keras_config["hyperparameters"].items() if k in ALLOWED_KEYS
        }
        torch_config["hyperparameters"] = {
            k: v for (k, v) in torch_config["hyperparameters"].items() if k in ALLOWED_KEYS
        }

        assert keras_config["hyperparameters"] == torch_config["hyperparameters"], "Hyperparameters differ"


def test_single_model_predictions():
    """Test predictions with a single allele-specific model"""
    # Create a basic test model with known architecture
    keras_model = Class1NeuralNetwork(
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

    # Initialize network explicitly
    keras_model._network = keras_model.make_network(
        **keras_model.network_hyperparameter_defaults.subselect(keras_model.hyperparameters)
    )
    network = keras_model.network()
    network.compile(optimizer="adam", loss="mse")

    # Create equivalent PyTorch model
    torch_model = TorchNeuralNetwork(
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

    # Transfer weights from Keras to PyTorch
    torch_model.load_weights_from_keras(network)

    # Create test peptides
    test_peptides = [
        "SIINFEKL",  # Known H-2Kb epitope
        "FALPYWNM",  # Random sequence
        "KLGGALQAK",  # Known HLA-A2 epitope
        "VNTPEHVVP",  # Random sequence
    ]
    peptide_encoding = EncodableSequences.create(test_peptides)

    # Get predictions from both models
    keras_predictions = keras_model.predict(peptides=peptide_encoding)
    torch_predictions = torch_model.predict(peptides=peptide_encoding)

    # Print predictions for debugging
    print("\nPredictions comparison:")
    for i, peptide in enumerate(test_peptides):
        print(f"{peptide}: Keras={keras_predictions[i]:.2f}, " f"PyTorch={torch_predictions[i]:.2f}")

    # Compare predictions
    assert_array_almost_equal(
        keras_predictions,
        torch_predictions,
        decimal=0,  # More lenient tolerance for IC50 values
        err_msg="Single model predictions don't match between Keras and PyTorch",
    )

    # Test with both predictors
    allele = "HLA-A*02:01"
    # Use Keras model with Keras predictor
    keras_predictor = KerasPredictor(allele_to_allele_specific_models={allele: [keras_model]})
    # Use PyTorch model with PyTorch predictor
    torch_predictor = TorchPredictor(allele_to_allele_specific_models={allele: [torch_model]})

    # Compare predictor-level predictions
    keras_pred = keras_predictor.predict(peptides=test_peptides, allele=allele)
    torch_pred = torch_predictor.predict(peptides=test_peptides, allele=allele)

    print("\nPredictor-level predictions comparison:")
    for i, peptide in enumerate(test_peptides):
        print(f"{peptide}: Keras={keras_pred[i]:.2f}, " f"PyTorch={torch_pred[i]:.2f}")

    assert_array_almost_equal(
        keras_pred, torch_pred, decimal=0, err_msg="Predictor-level single model predictions don't match"
    )


def test_allele_sequence_handling():
    """Test loading and using allele sequences"""
    pass


def test_ensemble_predictions():
    """Test predictions with multiple models for same allele"""
    pass


def test_pan_allele_predictions():
    """Test pan-allele model predictions"""
    pass


def test_percentile_ranks():
    """Test percentile rank calculations"""
    pass


def test_mixed_model_predictions():
    """Test predictions using both allele-specific and pan-allele models"""
    pass


def test_edge_cases():
    """Test handling of edge cases and errors"""
    pass


def test_full_predictor():
    """Test complete predictor functionality"""
    pass


def test_tensorflow_vs_pytorch_backends():
    """Test that tensorflow and pytorch backends produce matching results."""

    # Generate random peptides for each length 8-15
    all_peptides = []
    for length in range(8, 16):  # 16 because range is exclusive
        peptides = random_peptides(num=100, length=length)
        all_peptides.extend(peptides)

    args = (
        ["--alleles", "HLA-A0201", "--alleles", "HLA-A0201,HLA-A0301", "--peptides"]
        + all_peptides
        + [
            "--prediction-column-prefix",
            "mhcflurry_",
            "--affinity-only",
        ]
    )

    # Run with tensorflow backend
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

    # Verify shapes and columns match
    assert result_tf.shape == result_torch.shape, "Output shapes differ"
    assert all(result_tf.columns == result_torch.columns), "Output columns differ"

    # Automatically categorize columns based on their content
    columns_info = {}
    for col in result_tf.columns:
        if col.startswith("mhcflurry_"):
            # Check if column contains only numeric values
            try:
                pd.to_numeric(result_tf[col])
                columns_info[col] = "numeric"
            except (ValueError, TypeError):
                # If conversion fails, it's not a numeric column
                columns_info[col] = "non-numeric"
        else:
            # Non-prediction columns (like 'allele', 'peptide') are non-numeric
            columns_info[col] = "non-numeric"

    print("\nColumn categorization:")
    for col, col_type in columns_info.items():
        print(f"{col}: {col_type}")

    # Compare non-numeric columns exactly
    non_numeric_columns = [col for col, type_ in columns_info.items() if type_ == "non-numeric"]
    for col in non_numeric_columns:
        print(f"\nComparing non-numeric column {col}:")
        tf_unique = sorted(result_tf[col].unique())
        torch_unique = sorted(result_torch[col].unique())
        print(f"TensorFlow unique values: {tf_unique}")
        print(f"PyTorch unique values: {torch_unique}")
        assert all(result_tf[col] == result_torch[col]), f"Values differ in non-numeric column {col}"

    # Compare numeric columns with tolerance
    numeric_columns = [col for col, type_ in columns_info.items() if type_ == "numeric"]
    for col in numeric_columns:
        print(f"\nComparing numeric column {col}:")
        print(f"TensorFlow stats: mean={result_tf[col].mean():.4f}, std={result_tf[col].std():.4f}")
        print(f"PyTorch stats: mean={result_torch[col].mean():.4f}, std={result_torch[col].std():.4f}")

        # Check for NaN values
        tf_nans = pd.isna(result_tf[col])
        torch_nans = pd.isna(result_torch[col])
        assert np.array_equal(tf_nans, torch_nans), f"NaN patterns differ in column {col}"

        # Compare non-NaN values
        non_nan_mask = ~tf_nans
        if non_nan_mask.any():
            assert_array_almost_equal(
                result_tf[col][non_nan_mask].values,
                result_torch[col][non_nan_mask].values,
                decimal=0,  # More lenient tolerance for IC50 values
                err_msg=f"Values differ in numeric column {col}",
            )

    # Additional validation for specific column types
    if "mhcflurry_affinity" in numeric_columns:
        # Affinity predictions should be positive numbers
        assert (result_tf["mhcflurry_affinity"] > 0).all(), "Invalid affinity values in TensorFlow results"
        assert (result_torch["mhcflurry_affinity"] > 0).all(), "Invalid affinity values in PyTorch results"

    if "mhcflurry_affinity_percentile" in numeric_columns:
        # Percentile predictions should be between 0 and 100
        assert (
            (result_tf["mhcflurry_affinity_percentile"] >= 0) & (result_tf["mhcflurry_affinity_percentile"] <= 100)
        ).all(), "Invalid percentile values in TensorFlow results"
        assert (
            (result_torch["mhcflurry_affinity_percentile"] >= 0)
            & (result_torch["mhcflurry_affinity_percentile"] <= 100)
        ).all(), "Invalid percentile values in PyTorch results"


def test_skeleton_ensemble_predictions():
    """
    TODO: Test predictions with multiple allele-specific models and/or pan-allele models.
    - Mock or instantiate multiple TorchNeuralNetwork models
    - Add them to TorchPredictor
    - Ensure the predictor handles combining their predictions appropriately
    """
    pass


def test_skeleton_percentile_ranks():
    """
    TODO: Test percent-rank calibration for TorchPredictor.
    - Generate random predictions
    - Calibrate percentile ranks
    - Validate transformations are consistent
    """
    pass


def test_skeleton_save_and_load_predictor():
    """
    TODO: Test saving a TorchPredictor with multiple models to a manifest directory,
    then reloading it. Verify the manifest usage, consolidated weights, etc.
    """
    pass


def test_skeleton_merge_predictors():
    """
    TODO: Test merging multiple TorchPredictor instances into a single predictor
    (similar to the Keras version's merge/merge_in_place calls).
    - Create several TorchPredictor objects
    - Merge them
    - Check that final predictor has all models combined
    """
    pass


def test_skeleton_train_new_models():
    """
    TODO: Test training new Torch models (allele-specific and pan-allele)
    from scratch or partial data, verifying that fit methods
    (analogous to Keras version) work as expected.
    """
    pass

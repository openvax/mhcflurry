"""
Tests for Class1NeuralNetwork.
"""
import pytest

import numpy
from numpy import testing


import pandas

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.downloads import get_path
from mhcflurry.common import random_peptides

from mhcflurry.testing_utils import cleanup, startup


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    startup()
    yield
    cleanup()


@pytest.mark.slow
def test_class1_neural_network_a0205_training_accuracy():
    """Test that the network can memorize a small dataset."""
    # Memorize the dataset.
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=500,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[
            {"filters": 8, "activation": "tanh", "kernel_size": 3}
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
    )

    # First test a Class1NeuralNetwork, then a Class1AffinityPredictor.
    allele = "HLA-A*02:05"

    df = pandas.read_csv(
        get_path("data_curated", "curated_training_data.affinity.csv.bz2")
    )
    df = df.loc[df.allele == allele]
    df = df.loc[df.peptide.str.len() == 9]
    df = df.loc[df.measurement_type == "quantitative"]
    df = df.loc[df.measurement_source == "kim2014"]

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(df.peptide.values, df.measurement_value.values)
    ic50_pred = predictor.predict(df.peptide.values)
    ic50_true = df.measurement_value.values
    assert len(ic50_pred) == len(ic50_true)
    testing.assert_allclose(
        numpy.log(ic50_pred), numpy.log(ic50_true), rtol=0.2, atol=0.2
    )

    # Test that a second predictor has the same architecture json.
    # This is important for an optimization we use to re-use predictors of the
    # same architecture at prediction time.
    hyperparameters2 = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=1,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[
            {"filters": 8, "activation": "tanh", "kernel_size": 3}
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
    )
    predictor2 = Class1NeuralNetwork(**hyperparameters2)
    predictor2.fit(df.peptide.values, df.measurement_value.values, verbose=0)
    assert predictor.network().to_json() == predictor2.network().to_json()


def test_inequalities():
    """Test that inequality constraints are properly handled."""
    # Memorize the dataset.
    hyperparameters = dict(
        peptide_amino_acid_encoding="one-hot",
        activation="tanh",
        layer_sizes=[4],
        max_epochs=200,
        minibatch_size=32,
        random_negative_rate=0.0,
        random_negative_constant=0,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
        loss="custom:mse_with_inequalities_and_multiple_outputs",
    )

    dfs = []

    # Weak binders
    df = pandas.DataFrame()
    df["peptide"] = random_peptides(500, length=9)
    df["value"] = 400.0
    df["inequality1"] = "="
    df["inequality2"] = "<"
    dfs.append(df)

    # Strong binders - same peptides as above but more measurement values
    df = pandas.DataFrame()
    df["peptide"] = dfs[-1].peptide.values
    df["value"] = 1.0
    df["inequality1"] = "="
    df["inequality2"] = "="
    dfs.append(df)

    # Non-binders
    df = pandas.DataFrame()
    df["peptide"] = random_peptides(500, length=10)
    df["value"] = 1000
    df["inequality1"] = ">"
    df["inequality2"] = ">"
    dfs.append(df)

    df = pandas.concat(dfs, ignore_index=True)

    fit_kwargs = {"verbose": 0}

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(
        df.peptide.values,
        df.value.values,
        inequalities=df.inequality1.values,
        **fit_kwargs
    )
    df["prediction1"] = predictor.predict(df.peptide.values)

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(
        df.peptide.values,
        df.value.values,
        inequalities=df.inequality2.values,
        **fit_kwargs
    )
    df["prediction2"] = predictor.predict(df.peptide.values)

    # Binders should be stronger
    for pred in ["prediction1", "prediction2"]:
        assert df.loc[df.value < 1000, pred].mean() < 500
        assert df.loc[df.value >= 1000, pred].mean() > 500

    # For the binders, the (=) on the weak-binding measurement (100) in
    # inequality1 should make the prediction weaker, whereas for inequality2
    # this measurement is a "<" so it should allow the strong-binder measurement
    # to dominate.
    numpy.testing.assert_array_less(5.0, df.loc[df.value == 1].prediction1.values)
    numpy.testing.assert_array_less(df.loc[df.value == 1].prediction2.values, 2.0)
    numpy.testing.assert_allclose(df.loc[df.value == 1].prediction2.values, 1.0, atol=0.5)
    print(df.groupby("value")[["prediction1", "prediction2"]].mean())


def test_basic_training():
    """Test basic network training with synthetic data."""
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=50,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[
            {"filters": 8, "activation": "tanh", "kernel_size": 3}
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
    )

    # Generate synthetic data
    peptides = random_peptides(100, length=9)
    affinities = numpy.random.uniform(10, 50000, 100)

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(peptides, affinities, verbose=0)

    predictions = predictor.predict(peptides)
    assert len(predictions) == len(peptides)
    assert predictions.min() > 0
    assert predictions.max() < 100000


def test_split_forward_matches_full_forward():
    """forward_peptide_stage + forward_from_peptide_stage = forward (bit-identical).

    The calibration fast path (#272) precomputes peptide-side activations
    once and reuses them across many alleles. For it to be a valid
    speedup (not a silent behavior change) the split must compose to
    the same numerical output as the monolithic ``forward``.
    """
    import torch

    base_hparams = dict(
        activation="tanh",
        layer_sizes=[16, 8],
        validation_split=0.0,
        early_stopping=False,
        locally_connected_layers=[
            {"filters": 4, "activation": "tanh", "kernel_size": 3}
        ],
        peptide_allele_merge_method="concatenate",
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
    )
    predictor = Class1NeuralNetwork(**base_hparams)
    # Fake allele_representations so has_allele=True
    alle_reps = numpy.random.rand(3, 37, 21).astype(numpy.float32)
    predictor._network = predictor.make_network(
        allele_representations=alle_reps,
        **predictor.network_hyperparameter_defaults.subselect(
            predictor.hyperparameters),
    )
    predictor._network.eval()

    peptides = random_peptides(12, length=9)
    peptide_encoded = predictor.peptides_to_network_input(peptides)
    peptide_tensor = torch.from_numpy(peptide_encoded.astype(numpy.float32))
    allele_idx = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        full_out = predictor._network({
            "peptide": peptide_tensor,
            "allele": allele_idx,
        })
        stage = predictor._network.forward_peptide_stage(peptide_tensor)
        split_out = predictor._network.forward_from_peptide_stage(
            stage, allele_idx,
        )

    testing.assert_allclose(
        full_out.numpy(), split_out.numpy(), rtol=0, atol=1e-6,
        err_msg="split-forward must match monolithic forward bit-identically",
    )


@pytest.mark.parametrize(
    "encoding_name",
    [
        "BLOSUM62",
        "one-hot",
        "physchem",
        "PMBEC",
        "contact",
        "BLOSUM62+physchem",
        "PMBEC+contact",
        "PMBEC:minmax+contact:minmax",
    ],
)
def test_peptide_amino_acid_encoding_torch_forward_parity(encoding_name):
    """Torch-side fixed peptide encoding must match CPU-side encoding.

    With weights fixed and identical between a legacy (BLOSUM-encoded
    input) model and an index-encoded model, both forward passes must
    produce bit-identical outputs — the torch embedding lookup is
    mathematically the same op as the CPU-side fixed-vector table lookup.
    """
    import torch
    from mhcflurry.amino_acid import get_vector_encoding_df

    base_hparams = dict(
        activation="tanh",
        layer_sizes=[8],
        peptide_encoding={
            "vector_encoding_name": encoding_name,
            "alignment_method": "pad_middle",
            "left_edge": 4,
            "right_edge": 4,
            "max_length": 15,
        },
        validation_split=0.0,
        early_stopping=False,
        locally_connected_layers=[
            {"filters": 4, "activation": "tanh", "kernel_size": 3}
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
    )

    peptides = random_peptides(16, length=9)

    # Legacy path (flag off) — peptides encoded as (N, L, V) numpy vectors.
    legacy = Class1NeuralNetwork(
        peptide_amino_acid_encoding_torch=False,
        **base_hparams
    )
    legacy._network = legacy.make_network(
        allele_representations=None,
        **legacy.network_hyperparameter_defaults.subselect(legacy.hyperparameters),
    )
    legacy_input = legacy.peptides_to_network_input(peptides)
    assert legacy_input.ndim == 3
    assert legacy_input.shape[-1] == get_vector_encoding_df(encoding_name).shape[1]

    # Torch path (default) — peptides encoded as (N, L) int indices.
    onpath = Class1NeuralNetwork(**base_hparams)
    onpath._network = onpath.make_network(
        allele_representations=None,
        **onpath.network_hyperparameter_defaults.subselect(onpath.hyperparameters),
    )
    assert onpath._network.peptide_input_vector_encoding_name == encoding_name
    onpath_input = onpath.peptides_to_network_input(peptides)
    assert onpath_input.ndim == 2
    assert numpy.issubdtype(onpath_input.dtype, numpy.integer)
    assert "peptide_embedding_table" not in onpath._network.state_dict()
    assert len(onpath._network.get_weights_list()) == len(
        legacy._network.get_weights_list()
    )

    # Copy legacy weights into onpath so the only difference is the
    # embedding expansion — forward outputs must match bit-identically.
    onpath._network.load_state_dict(legacy._network.state_dict(), strict=False)

    with torch.no_grad():
        legacy_out = legacy._network({
            "peptide": torch.from_numpy(legacy_input.astype(numpy.float32))
        })
        onpath_out = onpath._network({
            "peptide": torch.from_numpy(onpath_input)
        })

    testing.assert_allclose(
        legacy_out.numpy(), onpath_out.numpy(), rtol=0, atol=1e-6,
        err_msg="Torch-side fixed peptide encoding must match the "
                "legacy CPU-side vector-encoded forward bit-identically",
    )


def test_peptide_torch_encoding_preserves_index_encoding_options():
    """Index path must preserve trim / unsupported-amino-acid semantics."""
    from mhcflurry.amino_acid import get_vector_encoding_df

    base_encoding = {
        "vector_encoding_name": "BLOSUM62",
        "alignment_method": "right_pad",
        "max_length": 5,
        "trim": True,
        "allow_unsupported_amino_acids": True,
    }
    peptides = ["ACDZFGH"]
    legacy = Class1NeuralNetwork(
        peptide_encoding=base_encoding,
        peptide_amino_acid_encoding_torch=False,
    )
    onpath = Class1NeuralNetwork(
        peptide_encoding=base_encoding,
    )

    legacy_input = legacy.peptides_to_network_input(peptides)
    onpath_indices = onpath.peptides_to_network_input(peptides)
    table = get_vector_encoding_df("BLOSUM62").to_numpy()
    expanded = table[onpath_indices]

    testing.assert_allclose(legacy_input, expanded)


def test_peptide_amino_acid_encoding_torch_default_and_legacy_alias():
    encoding = {
        "vector_encoding_name": "BLOSUM62",
        "alignment_method": "pad_middle",
        "left_edge": 4,
        "right_edge": 4,
        "max_length": 15,
    }
    default = Class1NeuralNetwork(peptide_encoding=encoding)
    assert default.uses_peptide_torch_encoding()
    assert default.peptides_to_network_input(["SIINFEKL"]).shape == (1, 15)

    explicit_cpu_torch = Class1NeuralNetwork(
        peptide_encoding=encoding,
        peptide_amino_acid_encoding_torch="cpu",
    )
    assert explicit_cpu_torch.uses_peptide_torch_encoding()
    assert explicit_cpu_torch.peptides_to_network_input(["SIINFEKL"]).shape == (
        1, 15,
    )

    explicit_numpy = Class1NeuralNetwork(
        peptide_encoding=encoding,
        peptide_amino_acid_encoding_torch=False,
    )
    assert not explicit_numpy.uses_peptide_torch_encoding()
    assert explicit_numpy.peptides_to_network_input(["SIINFEKL"]).shape == (
        1, 15, 21,
    )

    legacy_alias = Class1NeuralNetwork(
        peptide_encoding=encoding,
        peptide_amino_acid_encoding_gpu=False,
    )
    assert not legacy_alias.uses_peptide_torch_encoding()
    assert "peptide_amino_acid_encoding_gpu" not in legacy_alias.hyperparameters
    assert legacy_alias.hyperparameters["peptide_amino_acid_encoding_torch"] is False


def test_vector_encoded_random_negatives_use_shape_compatible_pool():
    peptides = random_peptides(12, length=9)
    affinities = numpy.random.uniform(10, 50000, len(peptides))
    predictor = Class1NeuralNetwork(
        peptide_amino_acid_encoding_torch=False,
        activation="tanh",
        layer_sizes=[4],
        locally_connected_layers=[],
        peptide_dense_layer_sizes=[],
        allele_dense_layer_sizes=[],
        dropout_probability=0.0,
        batch_normalization=False,
        dense_layer_l1_regularization=0.0,
        dense_layer_l2_regularization=0.0,
        max_epochs=1,
        early_stopping=False,
        validation_split=0.0,
        minibatch_size=4,
        random_negative_rate=1.0,
        random_negative_constant=0,
        random_negative_pool_epochs=1,
    )

    predictor.fit(peptides, affinities, verbose=0)

    fit_info = predictor.fit_info[-1]
    assert fit_info["random_negative_pool_residency"] == "host"
    assert fit_info["fit_tensor_residency"] == "device"


def test_serialization():
    """Test that network weights can be serialized and deserialized."""
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=10,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[
            {"filters": 8, "activation": "tanh", "kernel_size": 3}
        ],
    )

    peptides = random_peptides(50, length=9)
    affinities = numpy.random.uniform(10, 50000, 50)

    # Train a network
    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(peptides, affinities, verbose=0)

    # Get predictions before serialization
    preds_before = predictor.predict(peptides)

    # Serialize and deserialize
    config = predictor.get_config()
    weights = predictor.get_weights()

    predictor2 = Class1NeuralNetwork.from_config(config, weights=weights)
    preds_after = predictor2.predict(peptides)

    # Predictions should be identical
    numpy.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


def test_different_peptide_lengths():
    """Test that the network handles different peptide lengths correctly."""
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=20,
        validation_split=0.0,
    )

    # Mix of different length peptides
    peptides = (
        random_peptides(30, length=8) +
        random_peptides(30, length=9) +
        random_peptides(30, length=10) +
        random_peptides(10, length=11)
    )
    affinities = numpy.random.uniform(10, 50000, 100)

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(peptides, affinities, verbose=0)

    predictions = predictor.predict(peptides)
    assert len(predictions) == len(peptides)


def test_early_stopping():
    """Test that early stopping works correctly."""
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=1000,
        early_stopping=True,
        patience=5,
        validation_split=0.2,
    )

    peptides = random_peptides(200, length=9)
    affinities = numpy.random.uniform(10, 50000, 200)

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(peptides, affinities, verbose=0)

    # Should stop well before 1000 epochs
    # (We can't easily check this without modifying the class to expose the final epoch)
    predictions = predictor.predict(peptides)
    assert len(predictions) == len(peptides)


def test_batch_normalization():
    """Test training with batch normalization."""
    hyperparameters = dict(
        activation="relu",
        layer_sizes=[16],
        max_epochs=20,
        validation_split=0.0,
        batch_normalization=True,
    )

    peptides = random_peptides(100, length=9)
    affinities = numpy.random.uniform(10, 50000, 100)

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(peptides, affinities, verbose=0)

    predictions = predictor.predict(peptides)
    assert len(predictions) == len(peptides)


def test_dropout():
    """Test training with dropout."""
    hyperparameters = dict(
        activation="relu",
        layer_sizes=[32, 16],
        max_epochs=20,
        validation_split=0.0,
        dropout_probability=0.5,
    )

    peptides = random_peptides(100, length=9)
    affinities = numpy.random.uniform(10, 50000, 100)

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(peptides, affinities, verbose=0)

    predictions = predictor.predict(peptides)
    assert len(predictions) == len(peptides)


def test_multiple_outputs():
    """Test network with multiple outputs."""
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=50,
        validation_split=0.0,
        num_outputs=2,
        loss="custom:mse_with_inequalities_and_multiple_outputs",
        locally_connected_layers=[],
    )

    peptides = random_peptides(100, length=9)
    affinities = numpy.random.uniform(0.0, 1.0, 100)
    output_indices = numpy.random.choice([0, 1], 100)

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(
        peptides, affinities, output_indices=output_indices, verbose=0
    )

    # Predict for each output
    predictions0 = predictor.predict(peptides, output_index=0)
    predictions1 = predictor.predict(peptides, output_index=1)

    assert len(predictions0) == len(peptides)
    assert len(predictions1) == len(peptides)

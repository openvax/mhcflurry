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


def test_iterate_tensor_slice_batches_matches_dataloader_semantics():
    """Phase 4 of openvax/mhcflurry#268: tensor-slice iterator primitive.

    The iterator yields batches by slicing pre-placed tensors under a
    shuffle permutation. For a given permutation it must produce the
    same (peptide, y, weights) content as the equivalent numpy-fancy-
    indexing path that the legacy DataLoader + _FitBatchDataset takes.
    """
    import torch
    from mhcflurry.class1_neural_network import _iterate_tensor_slice_batches

    n = 23
    batch_size = 8
    peptide = torch.arange(n * 15 * 21, dtype=torch.float32).reshape(n, 15, 21)
    allele = torch.arange(n * 7, dtype=torch.float32).reshape(n, 7)
    y = torch.arange(n, dtype=torch.float32)
    weights = (torch.arange(n, dtype=torch.float32) + 1.0) * 0.5

    permutation = torch.tensor(
        [3, 19, 7, 15, 0, 21, 11, 5, 14, 2, 17, 9, 22, 1, 6, 18, 4, 13, 10, 20, 8, 12, 16],
        dtype=torch.long,
    )
    assert int(permutation.shape[0]) == n

    # drop_last=True yields 2 full batches (16 rows); the last 7 are dropped.
    batches = list(_iterate_tensor_slice_batches(
        peptide_device=peptide,
        allele_device=allele,
        y_device=y,
        weights_device=weights,
        batch_size=batch_size,
        shuffle_permutation=permutation,
        drop_last=True,
    ))
    assert len(batches) == n // batch_size
    for step, (inputs, y_batch, w_batch) in enumerate(batches):
        idx = permutation[step * batch_size : (step + 1) * batch_size]
        testing.assert_array_equal(
            inputs["peptide"].numpy(), peptide[idx].numpy()
        )
        testing.assert_array_equal(inputs["allele"].numpy(), allele[idx].numpy())
        testing.assert_array_equal(y_batch.numpy(), y[idx].numpy())
        testing.assert_array_equal(w_batch.numpy(), weights[idx].numpy())

    # drop_last=False yields an additional tail batch of size n % batch_size.
    batches_with_tail = list(_iterate_tensor_slice_batches(
        peptide_device=peptide,
        allele_device=allele,
        y_device=y,
        weights_device=weights,
        batch_size=batch_size,
        shuffle_permutation=permutation,
        drop_last=False,
    ))
    assert len(batches_with_tail) == (n // batch_size) + 1
    tail_inputs, tail_y, tail_weights = batches_with_tail[-1]
    assert int(tail_y.shape[0]) == n % batch_size
    tail_idx = permutation[(n // batch_size) * batch_size :]
    testing.assert_array_equal(
        tail_inputs["peptide"].numpy(), peptide[tail_idx].numpy()
    )
    testing.assert_array_equal(tail_y.numpy(), y[tail_idx].numpy())


def test_iterate_tensor_slice_batches_handles_missing_optional_tensors():
    """Allele and weights are optional — iterator must not emit them
    when they're None."""
    import torch
    from mhcflurry.class1_neural_network import _iterate_tensor_slice_batches

    n = 8
    peptide = torch.arange(n * 5, dtype=torch.float32).reshape(n, 5)
    y = torch.arange(n, dtype=torch.float32)
    permutation = torch.arange(n, dtype=torch.long)

    batches = list(_iterate_tensor_slice_batches(
        peptide_device=peptide,
        allele_device=None,
        y_device=y,
        weights_device=None,
        batch_size=4,
        shuffle_permutation=permutation,
        drop_last=True,
    ))
    assert len(batches) == 2
    inputs, y_batch, w_batch = batches[0]
    assert "allele" not in inputs
    assert w_batch is None
    assert int(y_batch.shape[0]) == 4


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


def test_peptide_amino_acid_encoding_gpu_forward_parity():
    """Phase 2 (openvax/mhcflurry#268): on-device BLOSUM62 forward parity.

    With weights fixed and identical between a legacy (BLOSUM-encoded
    input) model and an index-encoded model, both forward passes must
    produce bit-identical outputs — the on-device embedding lookup is
    mathematically the same op as the CPU-side BLOSUM table multiply.
    """
    import torch
    from mhcflurry.amino_acid import BLOSUM62_MATRIX

    base_hparams = dict(
        activation="tanh",
        layer_sizes=[8],
        validation_split=0.0,
        early_stopping=False,
        locally_connected_layers=[
            {"filters": 4, "activation": "tanh", "kernel_size": 3}
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
    )

    peptides = random_peptides(16, length=9)

    # Legacy path (flag off) — peptides encoded as (N, L, 21) BLOSUM62.
    legacy = Class1NeuralNetwork(**base_hparams)
    legacy._network = legacy.make_network(
        allele_representations=None,
        **legacy.network_hyperparameter_defaults.subselect(legacy.hyperparameters),
    )
    legacy_input = legacy.peptides_to_network_input(peptides)
    assert legacy_input.ndim == 3
    assert legacy_input.shape[-1] == len(BLOSUM62_MATRIX)

    # On-device path (flag on) — peptides encoded as (N, L) int indices.
    onpath = Class1NeuralNetwork(peptide_amino_acid_encoding_gpu=True, **base_hparams)
    onpath._network = onpath.make_network(
        allele_representations=None,
        **onpath.network_hyperparameter_defaults.subselect(onpath.hyperparameters),
    )
    onpath_input = onpath.peptides_to_network_input(peptides)
    assert onpath_input.ndim == 2
    assert numpy.issubdtype(onpath_input.dtype, numpy.integer)

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
        err_msg="Phase 2 on-device BLOSUM62 forward must match the "
                "legacy CPU-side BLOSUM62-encoded forward bit-identically",
    )


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

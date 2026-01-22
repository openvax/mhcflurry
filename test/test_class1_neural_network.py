"""
Tests for Class1NeuralNetwork.
"""
import pytest
from . import initialize
initialize()

import numpy
from numpy import testing

from .pytest_helpers import eq_, assert_less, assert_greater, assert_almost_equal

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

import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import numpy
from numpy import testing
numpy.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(2)

from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal

import pandas

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.downloads import get_path
from mhcflurry.common import random_peptides

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup


def test_class1_neural_network_a0205_training_accuracy():
    # Memorize the dataset.
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[16],
        max_epochs=500,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[
            {
                "filters": 8,
                "activation": "tanh",
                "kernel_size": 3
            }
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0)

    # First test a Class1NeuralNetwork, then a Class1AffinityPredictor.
    allele = "HLA-A*02:05"

    df = pandas.read_csv(
        get_path(
            "data_curated", "curated_training_data.affinity.csv.bz2"))
    df = df.loc[
        df.allele == allele
    ]
    df = df.loc[
        df.peptide.str.len() == 9
    ]
    df = df.loc[
        df.measurement_type == "quantitative"
    ]
    df = df.loc[
        df.measurement_source == "kim2014"
    ]

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(df.peptide.values, df.measurement_value.values)
    ic50_pred = predictor.predict(df.peptide.values)
    ic50_true = df.measurement_value.values
    eq_(len(ic50_pred), len(ic50_true))
    testing.assert_allclose(
        numpy.log(ic50_pred),
        numpy.log(ic50_true),
        rtol=0.2,
        atol=0.2)

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
            {
                "filters": 8,
                "activation": "tanh",
                "kernel_size": 3
            }
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0)
    predictor2 = Class1NeuralNetwork(**hyperparameters2)
    predictor2.fit(df.peptide.values, df.measurement_value.values, verbose=0)
    eq_(predictor.network().to_json(), predictor2.network().to_json())


def test_inequalities():
    # Memorize the dataset.
    hyperparameters = dict(
        peptide_amino_acid_encoding="one-hot",
        activation="tanh",
        layer_sizes=[64],
        max_epochs=200,
        minibatch_size=32,
        random_negative_rate=0.0,
        random_negative_constant=0,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[
            {
                "filters": 8,
                "activation": "tanh",
                "kernel_size": 3
            }
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
        loss="custom:mse_with_inequalities_and_multiple_outputs")

    dfs = []

    # Weak binders
    df = pandas.DataFrame()
    df["peptide"] = random_peptides(100, length=9)
    df["value"] = 100
    df["inequality1"] = "="
    df["inequality2"] = "<"
    dfs.append(df)

    # Strong binders - same peptides as above but more measurement values
    df = pandas.DataFrame()
    df["peptide"] = dfs[-1].peptide.values
    df["value"] = 1
    df["inequality1"] = "="
    df["inequality2"] = "="
    dfs.append(df)

    # Non-binders
    df = pandas.DataFrame()
    df["peptide"] = random_peptides(100, length=10)
    df["value"] = 1000
    df["inequality1"] = ">"
    df["inequality2"] = ">"
    dfs.append(df)

    df = pandas.concat(dfs, ignore_index=True)

    fit_kwargs = {'verbose': 0}

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(
        df.peptide.values,
        df.value.values,
        inequalities=df.inequality1.values,
        **fit_kwargs)
    df["prediction1"] = predictor.predict(df.peptide.values)

    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(
        df.peptide.values,
        df.value.values,
        inequalities=df.inequality2.values,
        **fit_kwargs)
    df["prediction2"] = predictor.predict(df.peptide.values)

    # Binders should be stronger
    for pred in ["prediction1", "prediction2"]:
        assert_less(df.loc[df.value < 1000, pred].mean(), 500)
        assert_greater(df.loc[df.value >= 1000, pred].mean(), 500)

    # For the binders, the (=) on the weak-binding measurement (100) in
    # inequality1 should make the prediction weaker, whereas for inequality2
    # this measurement is a "<" so it should allow the strong-binder measurement
    # to dominate.
    numpy.testing.assert_allclose(
        df.loc[df.value == 1].prediction2.values,
        1.0,
        atol=0.5)
    numpy.testing.assert_array_less(
        5.0, df.loc[df.value == 1].prediction1.values)
    print(df.groupby("value")[["prediction1", "prediction2"]].mean())

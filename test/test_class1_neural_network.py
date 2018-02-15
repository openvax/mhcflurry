from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal

import numpy
import pandas
from numpy import testing

numpy.random.seed(0)

import logging
logging.getLogger('tensorflow').disabled = True

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.downloads import get_path
from mhcflurry.common import random_peptides


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
            "data_curated", "curated_training_data.no_mass_spec.csv.bz2"))
    df = df.ix[
        df.allele == allele
    ]
    df = df.ix[
        df.peptide.str.len() == 9
    ]
    df = df.ix[
        df.measurement_type == "quantitative"
    ]
    df = df.ix[
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
        loss="custom:mse_with_inequalities",
        peptide_amino_acid_encoding="one-hot",
        activation="tanh",
        layer_sizes=[16],
        max_epochs=50,
        minibatch_size=32,
        random_negative_rate=0.0,
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

    df = pandas.DataFrame()
    df["peptide"] = random_peptides(1000, length=9)

    # First half are binders
    df["binder"] = df.index < len(df) / 2
    df["value"] = df.binder.map({True: 100, False: 5000})
    df.loc[:10, "value"] = 1.0  # some strong binders
    df["inequality1"] = "="
    df["inequality2"] = df.binder.map({True: "<", False: "="})
    df["inequality3"] = df.binder.map({True: "=", False: ">"})

    # "A" at start of peptide indicates strong binder
    df["peptide"] = [
        ("C" if not row.binder else "A") + row.peptide[1:]
        for _, row in df.iterrows()
    ]

    fit_kwargs = {'verbose': 0}

    # Prediction1 uses no inequalities (i.e. all are (=))
    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(
        df.peptide.values,
        df.value.values,
        inequalities=df.inequality1.values,
        **fit_kwargs)
    df["prediction1"] = predictor.predict(df.peptide.values)

    # Prediction2 has a (<) inequality on binders and an (=) on non-binders
    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(
        df.peptide.values,
        df.value.values,
        inequalities=df.inequality2.values,
        **fit_kwargs)
    df["prediction2"] = predictor.predict(df.peptide.values)

    # Prediction3 has a (=) inequality on binders and an (>) on non-binders
    predictor = Class1NeuralNetwork(**hyperparameters)
    predictor.fit(
        df.peptide.values,
        df.value.values,
        inequalities=df.inequality3.values,
        **fit_kwargs)
    df["prediction3"] = predictor.predict(df.peptide.values)

    df_binders = df.loc[df.binder]
    df_nonbinders = df.loc[~df.binder]

    print("***** Binders: *****")
    print(df_binders.head(5))

    print("***** Non-binders: *****")
    print(df_nonbinders.head(5))

    # Binders should always be given tighter predicted affinity than non-binders
    assert_less(df_binders.prediction1.mean(), df_nonbinders.prediction1.mean())
    assert_less(df_binders.prediction2.mean(), df_nonbinders.prediction2.mean())
    assert_less(df_binders.prediction3.mean(), df_nonbinders.prediction3.mean())

    # prediction2 binders should be tighter on average than prediction1
    # binders, since prediction2 has a (<) inequality for binders.
    # Non-binders should be about the same between prediction2 and prediction1
    assert_less(df_binders.prediction2.mean(), df_binders.prediction1.mean())
    assert_almost_equal(
        df_nonbinders.prediction2.mean(),
        df_nonbinders.prediction1.mean(),
        delta=3000)

    # prediction3 non-binders should be weaker on average than prediction2 (or 1)
    # non-binders, since prediction3 has a (>) inequality for these peptides.
    # Binders should be about the same.
    assert_greater(
        df_nonbinders.prediction3.mean(),
        df_nonbinders.prediction2.mean())
    assert_greater(
        df_nonbinders.prediction3.mean(),
        df_nonbinders.prediction1.mean())
    assert_almost_equal(
        df_binders.prediction3.mean(),
        df_binders.prediction1.mean(),
        delta=3000)


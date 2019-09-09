from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal

import numpy
import pandas
from numpy import testing

numpy.random.seed(0)

import logging
logging.getLogger('tensorflow').disabled = True

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.common import random_peptides

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup


def test_multi_output():
    hyperparameters = dict(
        loss="custom:mse_with_inequalities_and_multiple_outputs",
        activation="tanh",
        layer_sizes=[16],
        max_epochs=50,
        minibatch_size=250,
        random_negative_rate=0.0,
        random_negative_constant=0.0,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[
        ],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
        optimizer="adam",
        num_outputs=3)

    df = pandas.DataFrame()
    df["peptide"] = random_peptides(10000, length=9)
    df["output1"] = df.peptide.map(lambda s: s[4] == 'K').astype(int) * 49000 + 1
    df["output2"] = df.peptide.map(lambda s: s[3] == 'Q').astype(int) * 49000 + 1
    df["output3"] = df.peptide.map(lambda s: s[4] == 'K' or s[3] == 'Q').astype(int) * 49000 + 1

    print("output1 mean", df.output1.mean())
    print("output2 mean", df.output2.mean())

    stacked = df.set_index("peptide").stack().reset_index()
    stacked.columns = ['peptide', 'output_name', 'value']
    stacked["output_index"] = stacked.output_name.map({
        "output1": 0,
        "output2": 1,
        "output3": 2,
    })
    assert not stacked.output_index.isnull().any(), stacked

    fit_kwargs = {
        'verbose': 1,
    }

    predictor = Class1NeuralNetwork(**hyperparameters)
    stacked_train = stacked
    predictor.fit(
        stacked_train.peptide.values,
        stacked_train.value.values,
        output_indices=stacked_train.output_index.values,
        **fit_kwargs)

    result = predictor.predict(df.peptide.values, output_index=None)
    print(df.shape, result.shape)
    print(result)

    df["prediction1"] = result[:,0]
    df["prediction2"] = result[:,1]
    df["prediction3"] = result[:,2]

    df_by_peptide = df.set_index("peptide")

    correlation = pandas.DataFrame(
        numpy.corrcoef(df_by_peptide.T),
        columns=df_by_peptide.columns,
        index=df_by_peptide.columns)
    print(correlation)

    sub_correlation = correlation.loc[
        ["output1", "output2", "output3"],
        ["prediction1", "prediction2", "prediction3"],
    ]
    assert sub_correlation.iloc[0, 0] > 0.99, correlation
    assert sub_correlation.iloc[1, 1] > 0.99, correlation
    assert sub_correlation.iloc[2, 2] > 0.99, correlation


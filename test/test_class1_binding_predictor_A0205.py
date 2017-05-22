import numpy
import pandas
numpy.random.seed(0)

from mhcflurry import Class1NeuralNetwork, Class1AffinityPredictor

from nose.tools import eq_
from numpy import testing

from mhcflurry.downloads import get_path

allele = "HLA-A*02:05"

df = pandas.read_csv(
        get_path(
            "data_curated", "curated_training_data.csv.bz2"))
df = df.ix[df.allele == allele]
df = df.ix[
    df.peptide.str.len() == 9
]
df = df.ix[
    df.measurement_type == "quantitative"
]
df = df.ix[
    df.measurement_source == "kim2014"
]

hyperparameters = dict(
    activation="tanh",
    layer_sizes=[64],
    max_epochs=1000,  # Memorize the dataset.
    early_stopping=False,
    dropout_probability=0.0)


def test_class1_neural_network_A0205_training_accuracy():
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


def test_class1_neural_network_A0205_training_accuracy():
    predictor = Class1AffinityPredictor()
    predictor.fit_allele_specific_predictors(
        n_models=2,
        architecture_hyperparameters=hyperparameters,
        allele=allele,
        peptides=df.peptide.values,
        affinities=df.measurement_value.values,
    )
    ic50_pred = predictor.predict(df.peptide.values, allele=allele)
    ic50_true = df.measurement_value.values
    eq_(len(ic50_pred), len(ic50_true))
    testing.assert_allclose(
        numpy.log(ic50_pred),
        numpy.log(ic50_true),
        rtol=0.2,
        atol=0.2)

    ic50_pred_df = predictor.predict_to_dataframe(
        df.peptide.values, allele=allele)
    print(ic50_pred_df)

    ic50_pred_df2 = predictor.predict_to_dataframe(
        df.peptide.values,
        allele=allele,
        include_individual_model_predictions=True)
    print(ic50_pred_df2)


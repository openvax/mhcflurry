import numpy
import pandas
numpy.random.seed(0)

from mhcflurry import Class1BindingPredictor

from nose.tools import eq_
from numpy import testing

from mhcflurry.downloads import get_path


def test_class1_binding_predictor_A0205_training_accuracy():
    df = pandas.read_csv(
        get_path(
            "data_curated", "curated_training_data.csv.bz2"))
    df = df.ix[df.allele == "HLA-A*02:05"]
    df = df.ix[
        df.peptide.str.len() == 9
    ]
    df = df.ix[
        df.measurement_type == "quantitative"
    ]
    df = df.ix[
        df.measurement_source == "kim2014"
    ]

    predictor = Class1BindingPredictor(
        activation="tanh",
        layer_sizes=[64],
        max_epochs=1000,  # Memorize the dataset.
        early_stopping=False,
        dropout_probability=0.0)
    predictor.fit(df.peptide.values, df.measurement_value.values)
    ic50_pred = predictor.predict(df.peptide.values)
    ic50_true = df.measurement_value.values
    eq_(len(ic50_pred), len(ic50_true))
    testing.assert_allclose(
        numpy.log(ic50_pred),
        numpy.log(ic50_true),
        rtol=0.2,
        atol=0.2)

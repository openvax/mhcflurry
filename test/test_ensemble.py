import pickle

from nose.tools import eq_, assert_less

import numpy
from numpy.testing import assert_allclose
from mhcflurry.class1_allele_specific_ensemble.measurement_collection import (
    MeasurementCollection)
from mhcflurry.dataset import Dataset
from mhcflurry.downloads import get_path


from mhcflurry \
    .class1_allele_specific_ensemble \
    .class1_ensemble_multi_allele_predictor import (
        Class1EnsembleMultiAllelePredictor,
        HYPERPARAMETER_DEFAULTS)


def test_basic():
    model_hyperparameters = HYPERPARAMETER_DEFAULTS.models_grid(
        impute=[False, True],
        activation=["tanh"],
        layer_sizes=[[4], [8]],
        embedding_output_dim=[16],
        dropout_probability=[.25],
        n_training_epochs=[20])
    model = Class1EnsembleMultiAllelePredictor(
        ensemble_size=10,
        model_hyperparameters_to_search=model_hyperparameters)
    print(model)

    dataset = Dataset.from_csv(get_path(
        "data_combined_iedb_kim2014", "combined_human_class1_dataset.csv"))
    dataset_a0205_all_lengths = dataset.get_allele("HLA-A0205")
    dataset_a0205 = Dataset(
        dataset_a0205_all_lengths._df.ix[
            dataset_a0205_all_lengths._df.peptide.str.len() == 9])

    mc = MeasurementCollection.from_dataset(dataset_a0205)
    model.fit(mc)
    ic50_pred = model.predict(mc)
    ic50_true = mc.df.measurement_value
    eq_(len(ic50_pred), len(ic50_true))
    assert_allclose(
        numpy.log(ic50_pred),
        numpy.log(ic50_true),
        rtol=0.2,
        atol=0.2)

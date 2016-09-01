import numpy as np
np.random.seed(0)

from mhcflurry.dataset import Dataset
from mhcflurry import Class1BindingPredictor

from nose.tools import eq_
from numpy import testing

from mhcflurry.downloads import get_path


def test_class1_binding_predictor_A0205_training_accuracy():
    dataset = Dataset.from_csv(get_path(
        "data_combined_iedb_kim2014", "combined_human_class1_dataset.csv"))
    dataset_a0205_all_lengths = dataset.get_allele("HLA-A0205")
    dataset_a0205 = Dataset(
        dataset_a0205_all_lengths._df.ix[
            dataset_a0205_all_lengths._df.peptide.str.len() == 9])

    predictor = Class1BindingPredictor(
        name="A0205",
        embedding_output_dim=32,
        activation="tanh",
        layer_sizes=[64],
        optimizer="adam",
        dropout_probability=0.0)
    predictor.fit_dataset(dataset_a0205, n_training_epochs=1000)
    peptides = dataset_a0205.peptides
    ic50_pred = predictor.predict(peptides)
    ic50_true = dataset_a0205.affinities
    eq_(len(ic50_pred), len(ic50_true))
    testing.assert_allclose(
        np.log(ic50_pred),
        np.log(ic50_true),
        rtol=0.2,
        atol=0.2)

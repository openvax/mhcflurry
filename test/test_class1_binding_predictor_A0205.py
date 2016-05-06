from mhcflurry.dataset import Dataset
from mhcflurry.paths import CLASS1_DATA_CSV_PATH
from mhcflurry import Class1BindingPredictor

from nose.tools import eq_
import numpy as np


def class1_binding_predictor_A0205_training_accuracy():
    dataset = Dataset.from_csv(CLASS1_DATA_CSV_PATH)
    dataset_a0205 = dataset.get_allele("HLA-A0205")

    predictor = Class1BindingPredictor.from_hyperparameters(name="A0205")
    predictor.fit_dataset(dataset_a0205)
    peptides = dataset_a0205.peptides
    ic50_pred = predictor.predict(peptides)
    ic50_true = dataset_a0205.affinities
    eq_(len(ic50_pred), len(ic50_true))
    assert np.allclose(ic50_pred, ic50_true)

if __name__ == "__main__":
    class1_binding_predictor_A0205_training_accuracy()


from mhcflurry.imputation_helpers import imputer_from_name
from mhcflurry.dataset import Dataset
from mhcflurry.paths import CLASS1_DATA_CSV_PATH
from mhcflurry import Class1BindingPredictor

from fancyimpute import MICE, KNN, SoftImpute, IterativeSVD
from nose.tools import eq_
import numpy as np


def test_create_imputed_datasets_empty():
    empty_dataset = Dataset.create_empty()
    result = empty_dataset.impute_missing_values(MICE(n_imputations=25))
    eq_(result, empty_dataset)

def test_create_imputed_datasets_two_alleles():
    dataset = Dataset.from_nested_dictionary({
        "HLA-A*02:01": {
            "A" * 9: 20.0,
            "C" * 9: 40000.0,
        },
        "HLA-A*02:05": {
            "S" * 9: 500.0,
            "A" * 9: 25.0,
        },
    })
    imputed_dataset = dataset.impute_missing_values(MICE(n_imputations=25))
    eq_(imputed_dataset.unique_alleles(), {"HLA-A*02:01", "HLA-A*02:05"})
    expected_peptides = {"A" * 9, "C" * 9, "S" * 9}
    for allele_name, allele_data in imputed_dataset.groupby_allele():
        eq_(set(allele_data.peptides), expected_peptides)

def test_performance_improves_for_A0205_with_pretraining():
    # test to make sure that imputation improves predictive accuracy after a
    # small number of training iterations (5 epochs)
    dataset = Dataset.from_csv(CLASS1_DATA_CSV_PATH)
    print("Full dataset: %d pMHC entries" % len(dataset))

    limited_alleles = ["HLA-A0205", "HLA-A0201", "HLA-A0101", "HLA-B0702"]

    # restrict to just five alleles
    dataset = dataset.get_alleles(limited_alleles)
    print("After filtering to %s, # entries: %d" % (limited_alleles, len(dataset)))

    a0205_data_without_imputation = dataset.get_allele("HLA-A0205")

    print("Dataset with only A0205, # entries: %d" % len(a0205_data_without_imputation))

    predictor_without_imputation = \
        Class1BindingPredictor.from_hyperparameters(name="A0205-no-impute")

    X_index, ic50_true, sample_weights, _ = \
        a0205_data_without_imputation.kmer_index_encoding()

    assert sample_weights.min() >= 0, sample_weights.min()
    assert sample_weights.max() <= 1, sample_weights.max()
    assert ic50_true.min() >= 0, ic50_true.min()

    predictor_without_imputation.fit_kmer_encoded_arrays(
        X=X_index,
        ic50=ic50_true,
        sample_weights=sample_weights,
        n_training_epochs=10)

    ic50_pred_without_imputation = \
        predictor_without_imputation.predict_ic50_for_kmer_encoded_array(X_index)
    diff_squared = (ic50_true - ic50_pred_without_imputation) ** 2

    ic50_true_label = ic50_true <= 500
    ic50_pred_label_without_imputation = ic50_pred_without_imputation <= 500
    ic50_label_same_without_imputation = (
        ic50_true_label == ic50_pred_label_without_imputation)
    mse_without_imputation = (diff_squared * sample_weights).sum() / sample_weights.sum()
    accuracy_without_imputation = (
        ic50_label_same_without_imputation * sample_weights).sum() / sample_weights.sum()
    imputed_datset = dataset.impute_missing_values(MICE(n_imputations=25))
    print("After imputation, dataset for %s has %d entries" % (
        limited_alleles, len(imputed_datset)))
    a0205_data_with_imputation = imputed_datset.get_allele("HLA-A0205")
    print("Limited to just A0205, # entries: %d" % (len(a0205_data_with_imputation)))

    X_index_imputed, ic50_imputed, sample_weights_imputed, _ = \
        a0205_data_with_imputation.kmer_index_encoding()
    assert sample_weights_imputed.min() >= 0, sample_weights_imputed.min()
    assert sample_weights_imputed.max() <= 1, sample_weights_imputed.max()
    assert ic50_imputed.min() >= 0, ic50_imputed.min()

    predictor_with_imputation = \
        Class1BindingPredictor.from_hyperparameters(name="A0205-impute")

    predictor_with_imputation.fit_kmer_encoded_arrays(
        X=X_index,
        ic50=ic50_true,
        sample_weights=sample_weights,
        X_pretrain=X_index_imputed,
        ic50_pretrain=ic50_imputed,
        sample_weights_pretrain=sample_weights_imputed,
        n_training_epochs=500)

    ic50_pred_with_imputation = \
        predictor_with_imputation.predict_ic50_for_kmer_encoded_array(X_index)
    diff_squared = (ic50_true - ic50_pred_with_imputation) ** 2
    mse_with_imputation = (diff_squared * sample_weights).sum() / sample_weights.sum()

    ic50_pred_label_with_imputation = ic50_pred_with_imputation <= 500
    ic50_label_same_with_imputation = (
        ic50_true_label == ic50_pred_label_with_imputation)
    accuracy_with_imputation = (
        ic50_label_same_with_imputation * sample_weights).sum() / sample_weights.sum()
    print("RMS w/out imputation: %f" % (np.sqrt(mse_without_imputation),))
    print("RMS w/ imputation: %f" % (np.sqrt(mse_with_imputation),))

    assert mse_with_imputation < mse_without_imputation, \
        "Expected MSE with imputation (%f) to be less than (%f) without imputation" % (
            mse_with_imputation, mse_without_imputation)

    print("IC50 <= 500nM accuracy w/out imputation: %f" % (
        accuracy_without_imputation,))
    print("IC50 <= 500nM accuracy w/ imputation: %f" % (
        accuracy_with_imputation,))
    assert accuracy_with_imputation > accuracy_without_imputation


def test_imputer_from_name():
    mice = imputer_from_name("mice")
    assert isinstance(mice, MICE)
    softimpute = imputer_from_name("softimpute")
    assert isinstance(softimpute, SoftImpute)
    svdimpute = imputer_from_name("svd")
    assert isinstance(svdimpute, IterativeSVD)
    knnimpute = imputer_from_name("knn")
    assert isinstance(knnimpute, KNN)


if __name__ == "__main__":
    test_create_imputed_datasets_empty()
    test_create_imputed_datasets_two_alleles()
    test_performance_improves_for_A0205_with_pretraining()
    test_imputer_from_name()

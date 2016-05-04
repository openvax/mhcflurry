from mhcflurry.imputation import (
    create_imputed_datasets,
    imputer_from_name,
)
from mhcflurry.data import (
    create_allele_data_from_peptide_to_ic50_dict,
    load_allele_datasets
)
from mhcflurry.paths import CLASS1_DATA_CSV_PATH
from mhcflurry import Class1BindingPredictor

from fancyimpute import MICE, KNN, SoftImpute, IterativeSVD
from nose.tools import eq_
import numpy as np
from sklearn.cross_validation import KFold

def test_create_imputed_datasets_empty():
    result = create_imputed_datasets({}, imputer=MICE(n_imputations=25))
    eq_(result, {})

def test_create_imputed_datasets_two_alleles():
    allele_data_dict = {
        "HLA-A*02:01": create_allele_data_from_peptide_to_ic50_dict({
            "A" * 9: 20.0,
            "C" * 9: 40000.0,
        }),
        "HLA-A*02:05": create_allele_data_from_peptide_to_ic50_dict({
            "S" * 9: 500.0,
            "A" * 9: 25.0,
        }),
    }
    result = create_imputed_datasets(allele_data_dict, imputer=MICE(n_imputations=25))
    eq_(set(result.keys()), {"HLA-A*02:01", "HLA-A*02:05"})
    expected_peptides = {"A" * 9, "C" * 9, "S" * 9}
    for allele_name, allele_data in result.items():
        print(allele_name, allele_data.peptides)
        eq_(set(allele_data.peptides), expected_peptides)

def test_performance_improves_for_A0205_with_pretraining():
    # test to make sure that imputation improves predictive accuracy after a
    # small number of training iterations (5 epochs)
    allele_data_dict = load_allele_datasets(CLASS1_DATA_CSV_PATH)

    print("Available alleles: %s" % (set(allele_data_dict.keys()),))

    # restrict to just three alleles
    allele_data_dict = {
        key: allele_data_dict[key]
        for key in ["HLA-A0205", "HLA-A0201", "HLA-A0101"]
    }

    a0205_data_without_imputation = allele_data_dict["HLA-A0205"]
    predictor_without_imputation = \
        Class1BindingPredictor.from_hyperparameters(name="A0205-no-impute")

    print("Without imputation, # samples = %d, # original peptides = %d" % (
        len(a0205_data_without_imputation.peptides),
        len(set(a0205_data_without_imputation.original_peptides))))

    print(set(a0205_data_without_imputation.original_peptides))
    X_index = a0205_data_without_imputation.X_index
    Y_true = a0205_data_without_imputation.Y
    sample_weights = a0205_data_without_imputation.weights

    predictor_without_imputation.fit(
        X=X_index,
        Y=Y_true,
        sample_weights=sample_weights,
        n_training_epochs=10)

    Y_pred_without_imputation = predictor_without_imputation.predict(X_index)
    print("Y_pred without imputation: %s" % (Y_pred_without_imputation,))
    mse_without_imputation = np.mean((Y_true - Y_pred_without_imputation) ** 2)
    print("MSE w/out imputation: %f" % mse_without_imputation)

    imputed_data_dict = create_imputed_datasets(
        allele_data_dict, MICE(n_imputations=25))
    a0205_data_with_imputation = imputed_data_dict["HLA-A0205"]
    print("Imputed data, # samples = %d, # original peptides = %d" % (
        len(a0205_data_with_imputation.peptides),
        len(set(a0205_data_with_imputation.original_peptides))))

    X_index_imputed = a0205_data_with_imputation.X_index
    Y_imputed = a0205_data_with_imputation.Y
    sample_weights_imputed = a0205_data_with_imputation.weights

    predictor_with_imputation = \
        Class1BindingPredictor.from_hyperparameters(name="A0205-impute")

    predictor_with_imputation.fit(
        X=X_index,
        Y=Y_true,
        sample_weights=sample_weights,
        X_pretrain=X_index_imputed,
        Y_pretrain=Y_imputed,
        sample_weights_pretrain=sample_weights_imputed,
        n_training_epochs=10)

    Y_pred_with_imputation = predictor_with_imputation.predict(X_index)
    mse_with_imputation = np.mean((Y_true - Y_pred_with_imputation) ** 2)
    print("MSE w/ imputation: %f" % (mse_with_imputation,))

    assert mse_with_imputation < mse_without_imputation, \
        "Expected MSE with imputation (%f) to be less than (%f) without imputation" % (
            mse_with_imputation, mse_without_imputation)

def slice_of_allele_data(allele_data, indices):
    """
    Slice through all the array fields of an AlleleData object using an array
    of indices. Eventually this should just be a member function of AlleleData
    or whatever Dataset object replaces it.
    """
    kwargs = {}
    for field_name in allele_data._fields:
        x = getattr(allele_data, field_name)
        if isinstance(x, (list, np.ndarray)):
            x = x[indices]
        kwargs[field_name] = x
    return allele_data.__class__(**kwargs)


def test_MICE_accuracy_on_A0205_imputing_from_A0201_and_A0101():
    # test to make sure that imputation improves predictive accuracy after a
    # small number of training iterations (5 epochs)
    allele_data_dict = load_allele_datasets(CLASS1_DATA_CSV_PATH)

    alleles = ["HLA-A0205", "HLA-A0201", "HLA-A0101", "HLA-B0702", "H-2-KD"]

    # restrict to just three alleles
    allele_data_dict = {
        key: allele_data_dict[key]
        for key in alleles
    }

    a0205_data = allele_data_dict["HLA-A0205"]
    n_a0205 = len(a0205_data.Y)
    n_overlap = 0
    n_missing = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for (train_indices, test_indices) in KFold(
            n=n_a0205, n_folds=3, shuffle=True, random_state=0):
        a0205_data_train = slice_of_allele_data(a0205_data, train_indices)
        print("Training data: %s" % (a0205_data_train,))
        a0205_data_test = slice_of_allele_data(a0205_data, test_indices)
        print("Test data: %s" % (a0205_data_test,))
        cv_dict = {
            allele: allele_data_dict[allele]
            for allele in alleles
            if allele != "HLA-A0205"
        }
        cv_dict["HLA-A0205"] = a0205_data_train

        imputed = create_imputed_datasets(cv_dict, MICE(n_imputations=20, n_burn_in=5))
        a0205_imputed = imputed["HLA-A0205"]
        a0205_imputed_peptide_to_ic50 = {
            p: ic50
            for (p, ic50)
            in zip(a0205_imputed.original_peptides, a0205_imputed.ic50)
        }
        a0205_test_peptide_to_ic50 = {
            p: ic50
            for (p, ic50)
            in zip(a0205_data_test.original_peptides, a0205_imputed.ic50)
        }

        for (p, ic50) in a0205_test_peptide_to_ic50.items():
            if p not in a0205_imputed_peptide_to_ic50:
                n_missing += 1
                print("-- missing imputed value for peptide %s" % p)
                continue
            imputed_ic50 = a0205_imputed_peptide_to_ic50[p]
            print("-- Peptide %s has test IC50 of %f and imputed %f" % (
                p, ic50, imputed_ic50))
            n_overlap += 1
            tp += (ic50 <= 500 and imputed_ic50 <= 500)
            tn += (ic50 > 500 and imputed_ic50 > 500)
            fp += (ic50 > 500 and imputed_ic50 <= 500)
            fn += (ic50 <= 500 and imputed_ic50 > 500)
    recall = tp / float(tp + fn)
    precision = tp / float(tp + fp)
    print("TP: %d TN: %d FP: %d FN: %d" % (
        tp, tn, fp, fn))
    print("Precision %f, Recall %f" % (precision, recall))
    assert precision > 0.75, precision
    assert recall > 0.75, recall

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
    test_imputer_from_name()
    test_MICE_accuracy_on_A0205_imputing_from_A0201_and_A0101()
    test_create_imputed_datasets_empty()
    test_create_imputed_datasets_two_alleles()
    test_performance_improves_for_A0205_with_pretraining()

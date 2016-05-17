from nose.tools import eq_
from mhcflurry.dataset import Dataset

def test_create_allele_data_from_single_allele_dict():
    peptide_to_ic50_dict = {
        ("A" * 10): 1.2,
        ("C" * 9): 1000,
    }
    dataset = Dataset.from_single_allele_dictionary(
        allele_name="A0201",
        peptide_to_affinity_dict=peptide_to_ic50_dict)
    assert isinstance(dataset, Dataset)

    eq_(len(peptide_to_ic50_dict), len(dataset))
    expected_peptides = set([
        "A" * 10,
        "C" * 9,
    ])
    for pi, pj in zip(sorted(expected_peptides), sorted(dataset.peptides)):
        eq_(pi, pj)
    for pi, pj in zip(sorted(expected_peptides), sorted(dataset.unique_peptides())):
        eq_(pi, pj)

def test_dataset_random_split():
    dataset = Dataset.from_nested_dictionary({
        "H-2-Kb": {
            "SIINFEKL": 10.0,
            "FEKLSIIN": 20000.0,
            "SIFEKLIN": 50000.0,
        }})
    left, right = dataset.random_split(n=2)
    assert len(left) == 2
    assert len(right) == 1

def test_dataset_difference():
    dataset1 = Dataset.from_nested_dictionary({
        "H-2-Kb": {
            "SIINFEKL": 10.0,
            "FEKLSIIN": 20000.0,
            "SIFEKLIN": 50000.0,
        }})
    dataset2 = Dataset.from_nested_dictionary({"H-2-Kb": {"SIINFEKL": 10.0}})
    dataset_diff = dataset1.difference(dataset2)
    expected_result = Dataset.from_nested_dictionary({
        "H-2-Kb": {
            "FEKLSIIN": 20000.0,
            "SIFEKLIN": 50000.0,
        }})
    eq_(dataset_diff, expected_result)

def test_dataset_cross_validation():
    dataset = Dataset.from_nested_dictionary({
        "H-2-Kb": {
            "SIINFEKL": 10.0,
            "FEKLSIIN": 20000.0,
            "SIFEKLIN": 50000.0,
        },
        "HLA-A*02:01": {"ASASAS": 1.0, "CCC": 0.0}})

    fold_count = 0
    for train_dataset, test_dataset in dataset.cross_validation_iterator(
            test_allele="HLA-A*02:01",
            n_folds=2):
        assert train_dataset.unique_alleles() == {"H-2-Kb", "HLA-A*02:01"}
        assert test_dataset.unique_alleles() == {"HLA-A*02:01"}
        assert len(test_dataset) == 1
        fold_count += 1
    assert fold_count == 2

if __name__ == "__main__":
    test_create_allele_data_from_single_allele_dict()
    test_dataset_random_split()
    test_dataset_difference()

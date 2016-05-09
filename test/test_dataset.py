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

if __name__ == "__main__":
    test_create_allele_data_from_single_allele_dict()

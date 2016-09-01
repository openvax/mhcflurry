from mhcflurry.dataset import Dataset
from nose.tools import eq_
from . import data_path


def load_csv(filename):
    return Dataset.from_csv(data_path(filename))


def test_load_allele_datasets_8mer():
    dataset = load_csv("data_8mer.csv")
    print(dataset)
    assert len(dataset) == 1
    assert set(dataset.unique_alleles()) == {"HLA-A0201"}
    dataset_a0201 = dataset.get_allele("HLA-A0201")
    eq_(dataset, dataset_a0201)

    assert len(dataset.peptides) == 1
    assert len(dataset.affinities) == 1
    assert len(dataset.to_dataframe()) == 1
    assert len(dataset.sample_weights) == 1
    assert len(dataset.peptides[0]) == 8


def test_load_allele_datasets_9mer():
    dataset = load_csv("data_9mer.csv")
    print(dataset)
    assert len(dataset) == 1
    assert set(dataset.unique_alleles()) == {"HLA-A0201"}
    dataset_a0201 = dataset.get_allele("HLA-A0201")
    eq_(dataset, dataset_a0201)
    assert len(dataset.peptides) == 1
    assert len(dataset.affinities) == 1
    assert len(dataset.to_dataframe()) == 1
    assert len(dataset.sample_weights) == 1
    assert len(dataset.peptides[0]) == 9


def test_load_allele_datasets_10mer():
    dataset = load_csv("data_10mer.csv")
    print(dataset)
    assert len(dataset) == 1
    assert set(dataset.unique_alleles()) == {"HLA-A0201"}
    dataset_a0201 = dataset.get_allele("HLA-A0201")
    eq_(dataset, dataset_a0201)
    assert len(dataset.peptides) == 1
    assert len(dataset.affinities) == 1
    assert len(dataset.to_dataframe()) == 1
    assert len(dataset.sample_weights) == 1
    assert len(dataset.peptides[0]) == 10

if __name__ == "__main__":
    test_load_allele_datasets_8mer()
    test_load_allele_datasets_9mer()
    test_load_allele_datasets_10mer()

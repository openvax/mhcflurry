from os.path import join, dirname, realpath
from mhcflurry.data import load_allele_datasets

def load_csv(filename):
    base_dir = dirname(realpath(__file__))
    data_dir = join(base_dir, "data")
    full_path = join(data_dir, filename)
    return load_allele_datasets(full_path)

def test_load_allele_datasets_8mer():
    allele_dict = load_csv("data_8mer.csv")
    assert len(allele_dict) == 1
    assert set(allele_dict.keys()) == {"HLA-A0201"}
    dataset = allele_dict["HLA-A0201"]
    print(dataset)
    assert len(set(dataset.original_peptides)) == 1
    assert len(dataset.original_peptides) == 9
    assert len(dataset.peptides) == 9
    assert len(dataset.original_peptides[0]) == 8
    assert len(dataset.peptides[0]) == 9

def test_load_allele_datasets_9mer():
    allele_dict = load_csv("data_9mer.csv")
    assert len(allele_dict) == 1
    assert set(allele_dict.keys()) == {"HLA-A0201"}
    dataset = allele_dict["HLA-A0201"]
    print(dataset)
    assert len(dataset.original_peptides) == 1
    assert len(dataset.peptides) == 1
    assert len(dataset.original_peptides[0]) == 9
    assert dataset.original_peptides[0] == dataset.peptides[0]

def test_load_allele_datasets_10mer():
    allele_dict = load_csv("data_10mer.csv")
    assert len(allele_dict) == 1
    assert set(allele_dict.keys()) == {"HLA-A0201"}
    dataset = allele_dict["HLA-A0201"]
    print(dataset)
    assert len(set(dataset.original_peptides)) == 1
    assert len(dataset.peptides) == 10, len(dataset.peptides)
    assert len(dataset.original_peptides) == 10
    assert len(dataset.original_peptides[0]) == 10
    assert len(dataset.peptides[0]) == 9

if __name__ == "__main__":
    test_load_allele_datasets_8mer()
    test_load_allele_datasets_9mer()
    test_load_allele_datasets_10mer()

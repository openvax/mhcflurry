from nose.tools import eq_
from mhcflurry.data import (
    create_allele_data_from_peptide_to_ic50_dict,
    AlleleData
)

def test_create_allele_data_from_peptide_to_ic50_dict():
    peptide_to_ic50_dict = {
        ("A" * 10): 1.2,
        ("C" * 9): 1000,
    }
    allele_data = create_allele_data_from_peptide_to_ic50_dict(
        peptide_to_ic50_dict,
        max_ic50=50000.0)
    assert isinstance(allele_data, AlleleData)
    expected_peptides = set([
        "A" * 9,
        "C" * 9,
    ])
    peptides = set(allele_data.peptides)
    eq_(expected_peptides, peptides)

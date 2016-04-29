from mhcflurry.imputation import (
    create_imputed_datasets,
)
from mhcflurry.data import create_allele_data_from_peptide_to_ic50_dict

from fancyimpute import MICE
from nose.tools import eq_

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
        print(allele_name)
        print(allele_data)
        eq_(set(allele_data.peptides), expected_peptides)

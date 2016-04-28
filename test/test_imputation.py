from mhcflurry.imputation import create_imputed_datasets
from fancyimpute import MICE

def test_create_imputed_datasets_empty():
    result = create_imputed_datasets({}, imputer=MICE(n_imputations=25))
    eq_(result, {})

from dummy_predictors import (
    always_zero_predictor_with_unknown_AAs,
    always_one_predictor_with_unknown_AAs,
)
from mhcflurry import Ensemble

def test_ensemble_of_dummy_predictors():
    ensemble = Ensemble([
        always_one_predictor_with_unknown_AAs,
        always_zero_predictor_with_unknown_AAs])
    peptides = ["SYYFFYLLY"]
    y = ensemble.predict_peptides(peptides)
    assert len(y) == len(peptides)
    assert all(yi == 0.5 for yi in y)

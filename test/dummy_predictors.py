import numpy as np
from mhcflurry import Class1BindingPredictor

class Dummy9merIndexEncodingModel(object):
    """
    Dummy molde used for testing the pMHC binding predictor.
    """
    def __init__(self, constant_output_value=0):
        self.constant_output_value = constant_output_value

    def predict(self, X, verbose=False):
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        n_rows, n_cols = X.shape
        n_cols == 9, "Expected 9mer index input input, got %d columns" % (
            n_cols,)
        return np.ones(n_rows, dtype=float) * self.constant_output_value

always_zero_predictor_with_unknown_AAs = Class1BindingPredictor(
    model=Dummy9merIndexEncodingModel(0),
    allow_unknown_amino_acids=True)

always_zero_predictor_without_unknown_AAs = Class1BindingPredictor(
    model=Dummy9merIndexEncodingModel(0),
    allow_unknown_amino_acids=False)


always_one_predictor_with_unknown_AAs = Class1BindingPredictor(
    model=Dummy9merIndexEncodingModel(1),
    allow_unknown_amino_acids=True)

always_one_predictor_without_unknown_AAs = Class1BindingPredictor(
    model=Dummy9merIndexEncodingModel(1),
    allow_unknown_amino_acids=False)

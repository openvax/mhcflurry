import numpy as np

from mhcflurry import Class1BindingPredictor


class Dummy9merIndexEncodingModel(object):
    """
    Dummy molde used for testing the pMHC binding predictor.
    """
    def predict(self, X, verbose=False):
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        n_rows, n_cols = X.shape
        n_cols == 9, "Expected 9mer index input input, got %d columns" % (
            n_cols,)
        return np.zeros(n_rows, dtype=float)


def test_always_zero_9mer_inputs():
    predictor = Class1BindingPredictor(
        model=Dummy9merIndexEncodingModel(),
        allow_unknown_amino_acids=True)
    test_9mer_peptides = [
        "SIISIISII",
        "AAAAAAAAA",
    ]

    n_expected = len(test_9mer_peptides)
    y = predictor.predict_peptides(test_9mer_peptides)
    assert len(y) == n_expected
    assert np.all(y == 0)

    # call the predict method for 9mers directly
    y = predictor.predict_9mer_peptides(test_9mer_peptides)
    assert len(y) == n_expected
    assert np.all(y == 0)

    ic50 = predictor.predict_peptides_ic50(test_9mer_peptides)
    assert len(y) == n_expected
    assert np.all(ic50 == predictor.max_ic50), ic50


def test_always_zero_8mer_inputs():
    predictor = Class1BindingPredictor(
        model=Dummy9merIndexEncodingModel(),
        allow_unknown_amino_acids=True)
    test_8mer_peptides = [
        "SIISIISI",
        "AAAAAAAA",
    ]

    n_expected = len(test_8mer_peptides)
    y = predictor.predict_peptides(test_8mer_peptides)
    assert len(y) == n_expected
    assert np.all(y == 0)

    ic50 = predictor.predict_peptides_ic50(test_8mer_peptides)
    assert len(y) == n_expected
    assert np.all(ic50 == predictor.max_ic50), ic50


def test_always_zero_10mer_inputs():
    predictor = Class1BindingPredictor(
        model=Dummy9merIndexEncodingModel(),
        allow_unknown_amino_acids=True)
    test_10mer_peptides = [
        "SIISIISIYY",
        "AAAAAAAAYY",
    ]

    n_expected = len(test_10mer_peptides)
    y = predictor.predict_peptides(test_10mer_peptides)
    assert len(y) == n_expected
    assert np.all(y == 0)

    ic50 = predictor.predict_peptides_ic50(test_10mer_peptides)
    assert len(y) == n_expected
    assert np.all(ic50 == predictor.max_ic50), ic50


def test_encode_peptides_9mer():
    predictor = Class1BindingPredictor(
        model=Dummy9merIndexEncodingModel(),
        allow_unknown_amino_acids=True)
    X = predictor.encode_peptides(["AAASSSYYY"])
    assert X.shape[0] == 1, X.shape
    assert X.shape[1] == 9, X.shape


def test_encode_peptides_8mer():
    predictor = Class1BindingPredictor(
        model=Dummy9merIndexEncodingModel(),
        allow_unknown_amino_acids=True)
    X = predictor.encode_peptides(["AAASSSYY"])
    assert X.shape[0] == 9, (X.shape, X)
    assert X.shape[1] == 9, (X.shape, X)


def test_encode_peptides_10mer():
    predictor = Class1BindingPredictor(
        model=Dummy9merIndexEncodingModel(),
        allow_unknown_amino_acids=True)
    X = predictor.encode_peptides(["AAASSSYYFF"])
    assert X.shape[0] == 10, (X.shape, X)
    assert X.shape[1] == 9, (X.shape, X)

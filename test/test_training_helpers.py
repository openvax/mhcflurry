from mhcflurry.training_helpers import (
    check_encoded_array_shapes,
    combine_training_arrays,
)
import numpy as np
from nose.tools import eq_, assert_raises

def test_check_encoded_array_shapes():
    X = np.random.randn(40, 2)
    Y_good = np.random.randn(40)
    weights_good = np.random.randn(40)

    Y_too_long = np.random.randn(41)
    weights_empty = np.array([], dtype=float)

    eq_(check_encoded_array_shapes(X, Y_good, weights_good), X.shape)
    with assert_raises(ValueError):
        check_encoded_array_shapes(X, Y_too_long, weights_good)

    with assert_raises(ValueError):
        check_encoded_array_shapes(X, Y_good, weights_empty)

def test_combine_training_arrays():
    X = [[0, 0], [0, 1]]
    Y = [0, 1]
    weights = None
    X_pretrain = [[1, 1]]
    Y_pretrain = [2]
    weights_pretrain = None
    X_combined, Y_combined, weights_combined, n_pretrain = combine_training_arrays(
        X, Y, weights, X_pretrain, Y_pretrain, weights_pretrain)
    eq_(n_pretrain, 1)
    # expect the pretraining samples to come first
    assert (X_combined == np.array([[1, 1], [0, 0], [0, 1]])).all(), X_combined
    assert (Y_combined == np.array([2, 0, 1])).all(), Y_combined
    assert (weights_combined == np.array([1, 1, 1])).all(), weights_combined

if __name__ == "__main__":
    test_check_encoded_array_shapes()
    test_combine_training_arrays()

from mhcflurry.training_helpers import check_training_data_shapes
import numpy as np
from nose.tools import eq_, assert_raises

def test_check_training_data_shapes():
    X = np.random.randn(40, 2)
    Y_good = np.random.randn(40)
    weights_good = np.random.randn(40)

    Y_too_long = np.random.randn(41)
    weights_empty = np.array([], dtype=float)

    eq_(check_training_data_shapes(X, Y_good, weights_good), X.shape)
    with assert_raises(ValueError):
        check_training_data_shapes(X, Y_too_long, weights_good)

    with assert_raises(ValueError):
        check_training_data_shapes(X, Y_good, weights_empty)


if __name__ == "__main__":
    test_check_training_data_shapes()

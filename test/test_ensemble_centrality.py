import numpy

from numpy.testing import assert_equal

from mhcflurry import ensemble_centrality


def test_robust_mean():
    arr1 = numpy.array([
        [1, 2, 3, 4, 5],
        [-10000, 2, 3, 4, 100000],
    ])

    results = ensemble_centrality.robust_mean(arr1)
    assert_equal(results, [3, 3])

    # Should ignore nans.
    arr2 = numpy.array([
        [1, 2, 3, 4, 5],
        [numpy.nan, 2, 3, 4, numpy.nan],
    ])

    results = ensemble_centrality.robust_mean(arr2)
    assert_equal(results, [3, 3])

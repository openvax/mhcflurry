
import numpy
import warnings

from numpy.testing import assert_equal

from mhcflurry import ensemble_centrality


def test_robust_mean():
    arr1 = numpy.array([
        [1, 2, 3, 4, 5],
        [-10000, 2, 3, 4, 100],
    ])

    results = ensemble_centrality.robust_mean(arr1)
    assert_equal(results, [3, 3])

    # Should ignore nans.
    arr2 = numpy.array([
        [1, 2, 3, 4, 5],
        [numpy.nan, 1, 2, 3, numpy.nan],
        [numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    ])

    results = ensemble_centrality.CENTRALITY_MEASURES["robust_mean"](arr2)
    assert_equal(results, [3, 2, numpy.nan])

    results = ensemble_centrality.CENTRALITY_MEASURES["mean"](arr2)
    assert_equal(results, [3, 2, numpy.nan])


def test_no_runtime_warnings_for_all_nan_rows():
    arr = numpy.array([
        [numpy.nan, numpy.nan, numpy.nan],
        [1.0, 2.0, numpy.nan],
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mean = ensemble_centrality.CENTRALITY_MEASURES["mean"](arr)
        median = ensemble_centrality.CENTRALITY_MEASURES["median"](arr)
        robust = ensemble_centrality.CENTRALITY_MEASURES["robust_mean"](arr)
    assert numpy.isnan(mean[0]) and mean[1] == 1.5
    assert numpy.isnan(median[0]) and median[1] == 1.5
    assert numpy.isnan(robust[0]) and robust[1] == 1.5

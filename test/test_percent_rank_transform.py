import numpy

from mhcflurry.percent_rank_transform import PercentRankTransform

from numpy.testing import assert_allclose, assert_equal


def test_percent_rank_transform():
    model = PercentRankTransform()
    model.fit(numpy.arange(1000), bins=100)
    assert_allclose(
        model.transform([-2, 0, 50, 100, 2000]),
        [0.0, 0.0, 5.0, 10.0, 100.0],
        err_msg=str(model.__dict__))

    model2 = PercentRankTransform.from_series(model.to_series())
    assert_allclose(
        model2.transform([-2, 0, 50, 100, 2000]),
        [0.0, 0.0, 5.0, 10.0, 100.0],
        err_msg=str(model.__dict__))

    assert_equal(model.cdf, model2.cdf)
    assert_equal(model.bin_edges, model2.bin_edges)


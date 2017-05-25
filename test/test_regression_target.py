from mhcflurry.regression_target import (
    from_ic50,
    to_ic50,
)
from nose.tools import eq_


def test_regression_target_to_ic50():
    eq_(to_ic50(0, max_ic50=500.0), 500)
    eq_(to_ic50(1, max_ic50=500.0), 1.0)


def test_ic50_to_regression_target():
    eq_(from_ic50(5000, max_ic50=5000.0), 0)
    eq_(from_ic50(0, max_ic50=5000.0), 1.0)

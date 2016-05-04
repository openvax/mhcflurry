from mhcflurry.regression_target import (
    ic50_to_regression_target,
    regression_target_to_ic50,
)
from nose.tools import eq_

def test_regression_target_to_ic50():
    eq_(regression_target_to_ic50(0, max_ic50=500.0), 500)
    eq_(regression_target_to_ic50(1, max_ic50=500.0), 1.0)

def test_ic50_to_regression_target():
    eq_(ic50_to_regression_target(5000, max_ic50=5000.0), 0)
    eq_(ic50_to_regression_target(0, max_ic50=5000.0), 1.0)

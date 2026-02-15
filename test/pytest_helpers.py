"""
Test helper functions providing assertion utilities.
"""


import sys


_MHCFLURRY_COMMANDS = {
    "mhcflurry-calibrate-percentile-ranks": "mhcflurry.calibrate_percentile_ranks_command",
    "mhcflurry-class1-train-allele-specific-models": "mhcflurry.train_allele_specific_models_command",
    "mhcflurry-class1-select-allele-specific-models": "mhcflurry.select_allele_specific_models_command",
    "mhcflurry-class1-train-pan-allele-models": "mhcflurry.train_pan_allele_models_command",
    "mhcflurry-class1-select-pan-allele-models": "mhcflurry.select_pan_allele_models_command",
    "mhcflurry-class1-train-processing-models": "mhcflurry.train_processing_models_command",
    "mhcflurry-class1-select-processing-models": "mhcflurry.select_processing_models_command",
}


def mhcflurry_cli(command):
    """
    Return argv prefix to run a mhcflurry command using the current interpreter.

    This avoids picking up a system-installed mhcflurry that may rely on TensorFlow.
    """
    module = _MHCFLURRY_COMMANDS.get(command)
    if module is None:
        raise ValueError(f"Unknown mhcflurry command: {command}")
    return [sys.executable, "-c", f"from {module} import run; run()"]


def eq_(a, b, msg=None):
    """Assert that a equals b."""
    if msg:
        assert a == b, msg
    else:
        assert a == b, f"{a!r} != {b!r}"


def assert_less(a, b, msg=None):
    """Assert that a < b."""
    if msg:
        assert a < b, msg
    else:
        assert a < b, f"{a!r} is not less than {b!r}"


def assert_greater(a, b, msg=None):
    """Assert that a > b."""
    if msg:
        assert a > b, msg
    else:
        assert a > b, f"{a!r} is not greater than {b!r}"


def assert_almost_equal(a, b, places=7, msg=None):
    """Assert that a and b are equal up to `places` decimal places."""
    diff = abs(a - b)
    threshold = 10 ** (-places)
    if msg:
        assert diff < threshold, msg
    else:
        assert diff < threshold, (
            f"{a!r} != {b!r} within {places} places (diff={diff})"
        )


def assert_raises(exc_class, func=None, *args, **kwargs):
    """
    Assert that calling func raises exc_class.
    Can also be used as a context manager.
    """
    import pytest
    if func is None:
        return pytest.raises(exc_class)
    with pytest.raises(exc_class):
        func(*args, **kwargs)

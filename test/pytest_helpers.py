# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test helper functions providing assertion utilities.
"""


import sys


def mhcflurry_cli(subcommand):
    """
    Return argv prefix to run ``mhcflurry <subcommand>`` using the current interpreter.

    This avoids picking up a system-installed mhcflurry that may rely on TensorFlow.
    """
    return [sys.executable, "-m", "mhcflurry.cli.main", subcommand]


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

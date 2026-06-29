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

"""Tests for regression target conversion."""

from mhcflurry.regression_target import (
    from_ic50,
    to_ic50,
)


def test_regression_target_to_ic50():
    assert to_ic50(0, max_ic50=500.0) == 500
    assert to_ic50(1, max_ic50=500.0) == 1.0


def test_ic50_to_regression_target():
    assert from_ic50(5000, max_ic50=5000.0) == 0
    assert from_ic50(0, max_ic50=5000.0) == 1.0

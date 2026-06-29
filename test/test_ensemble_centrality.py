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

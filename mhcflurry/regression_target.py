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


def from_ic50(ic50, max_ic50=50000.0):
    """
    Convert ic50s to regression targets in the range [0.0, 1.0].

    Parameters
    ----------
    ic50 : numpy.array of float
        Affinities in nM; values outside [1, max_ic50] saturate the target.
    max_ic50 : float, default 50000.0
        Upper affinity scale in nM (must be greater than 1).

    Returns
    -------
    numpy.array of float

    """
    x = 1.0 - (numpy.log(numpy.maximum(ic50, 1e-12)) / numpy.log(max_ic50))
    return numpy.minimum(
        1.0,
        numpy.maximum(0.0, x))


def to_ic50(x, max_ic50=50000.0):
    """
    Convert regression targets in the range [0.0, 1.0] to ic50s in the range
    [1, max_ic50] nM. Values outside [0, 1] are not clipped.

    Parameters
    ----------
    x : numpy.array of float
    max_ic50 : float, default 50000.0
        Upper affinity scale in nM (must be greater than 1).

    Returns
    -------
    numpy.array of float
    """
    return max_ic50 ** (1.0 - x)

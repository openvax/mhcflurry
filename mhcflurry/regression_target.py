# Copyright (c) 2016. Mount Sinai School of Medicine
#
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

import numpy as np

MAX_IC50 = 50000.0

def ic50_to_regression_target(ic50, max_ic50=MAX_IC50):
    """
    Transform IC50 inhibitory binding concentrations to affinity values between
    [0,1] where 0 means a value greater or equal to max_ic50 and 1 means very
    strong binder.

    Parameters
    ----------
    ic50 : numpy.ndarray

    max_ic50 : float
    """
    log_ic50 = np.log(ic50) / np.log(max_ic50)
    regression_target = 1.0 - log_ic50
    # clamp to values between 0, 1
    regression_target = np.maximum(regression_target, 0.0)
    regression_target = np.minimum(regression_target, 1.0)
    return regression_target

def regression_target_to_ic50(y, max_ic50=MAX_IC50):
    """
    Transform values between [0,1] to IC50 inhibitory binding concentrations
    between [1.0, infinity]

    Parameters
    ----------
    y : numpy.ndarray of float

    max_ic50 : float

    Returns numpy.ndarray
    """
    return max_ic50 ** (1.0 - y)

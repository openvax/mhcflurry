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

def check_training_data_shapes(X, Y, weights):
    """
    Check to make sure that the shapes of X, Y, and weights are all compatible.

    Returns the numbers of rows and columns in X.
    """
    if len(X.shape) != 2:
        raise ValueError("Expected X to be 2d, got shape: %s" % (X.shape,))

    if len(Y.shape) != 1:
        raise ValueError("Expected Y to be 1d, got shape: %s" % (Y.shape,))

    if len(weights.shape) != 1:
        raise ValueError("Expected weights to be 1d, got shape: %s" % (
            weights.shape,))

    n_samples, n_dims = X.shape

    if len(Y) != n_samples:
        raise ValueError("Mismatch between len(X) = %d and len(Y) = %d" % (
            n_samples, len(Y)))

    if len(weights) != n_samples:
        raise ValueError(
            "Length of sample_weights (%d) doesn't match number of samples (%d)" % (
                len(weights),
                n_samples))

    return n_samples, n_dims

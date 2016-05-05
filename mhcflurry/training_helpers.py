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

"""
Helper functions for training predictors on fixed length encoding of peptides
along with vectors representing affinity and sample weights.

Eventually we'll have to generalize or split this to work with sequence
inputs for RNN predictors.
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
)

import numpy as np

def check_encoded_array_shapes(X, Y, sample_weights):
    """
    Check to make sure that the shapes of X, Y, and weights are all compatible.
    This function differs from check_pMHC_affinity_array_lengths in that the
    peptides are assumed to be encoded into a single 2d array of features X
    and the data is either for a single allele or allele features are included
    in X.

    Returns the numbers of rows and columns in X.
    """
    if len(X.shape) != 2:
        raise ValueError("Expected X to be 2d, got shape: %s" % (X.shape,))

    if len(Y.shape) != 1:
        raise ValueError("Expected Y to be 1d, got shape: %s" % (Y.shape,))

    if len(sample_weights.shape) != 1:
        raise ValueError("Expected weights to be 1d, got shape: %s" % (
            sample_weights.shape,))

    n_samples, n_dims = X.shape

    if len(Y) != n_samples:
        raise ValueError("Mismatch between len(X) = %d and len(Y) = %d" % (
            n_samples, len(Y)))

    if len(sample_weights) != n_samples:
        raise ValueError(
            "Length of sample_weights (%d) doesn't match number of samples (%d)" % (
                len(sample_weights),
                n_samples))

    return n_samples, n_dims

def combine_training_arrays(
        X,
        Y,
        sample_weights,
        X_pretrain,
        Y_pretrain,
        sample_weights_pretrain):
    """
    Make sure the shapes of given training and pre-training data
    conform with each other. Then concatenate the pre-training and the
    training data.

    Returns (X_combined, Y_combined, weights_combined, n_pretrain_samples)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if sample_weights is None:
        sample_weights = np.ones_like(Y)
    else:
        sample_weights = np.asarray(sample_weights)

    n_samples, n_dims = check_encoded_array_shapes(X, Y, sample_weights)

    if X_pretrain is None or Y_pretrain is None:
        X_pretrain = np.empty((0, n_dims), dtype=X.dtype)
        Y_pretrain = np.empty((0,), dtype=Y.dtype)
    else:
        X_pretrain = np.asarray(X_pretrain)
        Y_pretrain = np.asarray(Y_pretrain)

    if sample_weights_pretrain is None:
        sample_weights_pretrain = np.ones_like(Y_pretrain)
    else:
        sample_weights_pretrain = np.asarray(sample_weights_pretrain)

    n_pretrain_samples, n_pretrain_dims = check_encoded_array_shapes(
        X_pretrain, Y_pretrain, sample_weights_pretrain)

    X_combined = np.vstack([X_pretrain, X])
    Y_combined = np.concatenate([Y_pretrain, Y])
    combined_weights = np.concatenate([
        sample_weights_pretrain,
        sample_weights,
    ])
    return X_combined, Y_combined, combined_weights, n_pretrain_samples


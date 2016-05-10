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
        X_pretrain = np.zeros((0, n_dims), dtype=X.dtype)
        Y_pretrain = np.zeros((0,), dtype=Y.dtype)
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


def extend_with_negative_random_samples(
        X, Y, weights, n_random_negative_samples, max_amino_acid_encoding_value):
    """
    Extend training data with randomly generated negative samples. Assumes that
    X is an integer array of amino acid indices for fixed length peptides.

    Parameters
    ----------
    X : numpy.ndarray
        2d array of integer amino acid encodings

    Y : numpy.ndarray
        1d array of regression targets

    weights : numpy.ndarray
        1d array of sample weights (must be same length as X and Y)

    n_random_negative_samples : int
        Number of random negative samplex to create

    max_amino_acid_encoding_value : int
        Typically 20 for the standard set of amino acids or 21 if we're
        including the null character "X" used to extend 8mers into 9mers

    Returns X, Y, weights (extended with random negative samples)
    """
    assert len(X) == len(Y) == len(weights)
    if n_random_negative_samples == 0:
        return X, Y, weights
    n_cols = X.shape[1]
    X_random = np.random.randint(
        low=0,
        high=max_amino_acid_encoding_value,
        size=(n_random_negative_samples, n_cols)).astype(X.dtype)
    Y_random = np.zeros(n_random_negative_samples, dtype=float)
    weights_random = np.ones(n_random_negative_samples, dtype=float)
    X_with_negative = np.vstack([X, X_random])
    Y_with_negative = np.concatenate([Y, Y_random])
    weights_with_negative = np.concatenate([
        weights,
        weights_random])
    assert len(X_with_negative) == len(X) + n_random_negative_samples
    assert len(Y_with_negative) == len(Y) + n_random_negative_samples
    assert len(weights_with_negative) == len(weights) + n_random_negative_samples
    return X_with_negative, Y_with_negative, weights_with_negative

# Copyright (c) 2015. Mount Sinai School of Medicine
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
from .amino_acid import amino_acid_letter_indices


def hotshot_encoding(peptides, peptide_length):
    """
    Encode a set of equal length peptides as a binary matrix,
    where each letter is transformed into a length 20 vector with a single
    element that is 1 (and the others are 0).
    """
    shape = (len(peptides), peptide_length, 20)
    X = np.zeros(shape, dtype=bool)
    for i, peptide in enumerate(peptides):
        for j, amino_acid in enumerate(peptide):
            k = amino_acid_letter_indices[amino_acid]
            X[i, j, k] = 1
    return X


def index_encoding(peptides, peptide_length):
    """
    Encode a set of equal length peptides as a vector of their
    amino acid indices.
    """
    X = np.zeros((len(peptides), peptide_length), dtype=int)
    for i, peptide in enumerate(peptides):
        for j, amino_acid in enumerate(peptide):
            X[i, j] = amino_acid_letter_indices[amino_acid]
    return X


def index_encoding_of_substrings(
        peptides,
        substring_length,
        delete_exclude_start=0,
        delete_exclude_end=0):
    """
    Take peptides of varying lengths, chop them into substrings of fixed
    length and apply index encoding to these substrings.

    If a string is longer than the substring length, then it's reduced to
    the desired length by deleting characters at all possible positions.
    If positions at the start or end of a string should be exempt from deletion
    then the number of exempt characters can be controlled via
    `delete_exclude_start` and `delete_exclude_end`.

    Returns feature matrix X and a vector of substring counts.
    """
    pass


def indices_to_hotshot_encoding(X, n_indices=None, first_index_value=0):
    """
    Given an (n_samples, peptide_length) integer matrix
    convert it to a binary encoding of shape:
        (n_samples, peptide_length * n_indices)
    """
    (n_samples, peptide_length) = X.shape
    if not n_indices:
        n_indices = X.max() - first_index_value + 1

    X_binary = np.zeros((n_samples, peptide_length * n_indices), dtype=bool)
    for i, row in enumerate(X):
        for j, xij in enumerate(row):
            X_binary[i, n_indices * j + xij - first_index_value] = 1
    return X_binary.astype(float)

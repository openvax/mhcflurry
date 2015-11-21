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
from .fixed_length_peptides import fixed_length_from_many_peptides


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


def fixed_length_index_encoding(
        peptides,
        desired_length,
        start_offset_shorten=0,
        end_offset_shorten=0,
        start_offset_extend=0,
        end_offset_extend=0):
    """
    Take peptides of varying lengths, chop them into substrings of fixed
    length and apply index encoding to these substrings.

    If a string is longer than the desired length, then it's reduced to
    the desired length by deleting characters at all possible positions. When
    positions at the start or end of a string should be exempt from deletion
    then the number of exempt characters can be controlled via
    `start_offset_shorten` and `end_offset_shorten`.

    If a string is shorter than the desired length then it is filled
    with all possible characters of the alphabet at all positions. The
    parameters `start_offset_extend` and `end_offset_extend` control whether
    certain positions are excluded from insertion. The positions are
    in a "inter-residue" coordinate system, where `start_offset_extend` = 0
    refers to the position *before* the start of a peptide and, similarly,
    `end_offset_extend` = 0 refers to the position *after* the peptide.

    Returns feature matrix X, a list of original peptides for each feature
    vector, and a list of integer counts indicating how many rows share a
    particular original peptide. When two rows are expanded out of a single
    original peptide, they will both have a count of 2. These counts can
    be useful for down-weighting the importance of multiple feature vectors
    which originate from the same sample.
    """
    fixed_length, original, counts = fixed_length_from_many_peptides(
        peptides=peptides,
        desired_length=desired_length,
        start_offset_shorten=start_offset_shorten,
        end_offset_shorten=end_offset_shorten,
        start_offset_extend=start_offset_extend,
        end_offset_extend=end_offset_extend)
    X = index_encoding(fixed_length, desired_length)
    return X, original, counts

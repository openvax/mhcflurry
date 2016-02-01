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

import itertools
import logging

import numpy as np

from .amino_acid import common_amino_acids, amino_acids_with_unknown

common_amino_acid_letters = common_amino_acids.letters()


def all_kmers(k, alphabet=common_amino_acid_letters):
    """
    Generates all k-mer peptide sequences

    Parameters
    ----------
    k : int

    alphabet : str | list of characters
    """
    alphabets = [alphabet] * k
    return [
        "".join(combination)
        for combination
        in itertools.product(*alphabets)
    ]


class CombinatorialExplosion(Exception):
    pass


def extend_peptide(
        peptide,
        desired_length,
        start_offset,
        end_offset,
        insert_amino_acid_letters=common_amino_acid_letters):
    """Extend peptide by inserting every possible amino acid combination
    if we're trying to e.g. turn an 8mer into 9mers.

    Parameters
    ----------
    peptide : str

    desired_length : int

    start_offset : int
        How many characters (from the position before the start of the string)
        to skip before inserting characters.


    end_offset : int
        Last character position from the end where we insert new characters,
        where 0 is the position after the last character.

    insert_alphabet : str | list of character
    """
    n = len(peptide)
    assert n < desired_length, \
        "%s (length = %d) is too long! Must be shorter than %d" % (
            peptide, n, desired_length)
    n_missing = desired_length - n
    if n_missing > 3:
        raise CombinatorialExplosion(
            "Cannot transform %s of length %d into a %d-mer peptide" % (
                peptide, n, desired_length))
    return [
        peptide[:i] + extra + peptide[i:]
        for i in range(start_offset, n - end_offset + 1)
        for extra in all_kmers(
            n_missing,
            alphabet=insert_amino_acid_letters)
    ]


def shorten_peptide(
        peptide,
        desired_length,
        start_offset,
        end_offset,
        insert_amino_acid_letters=common_amino_acid_letters):
    """Shorten peptide if trying to e.g. turn 10mer into 9mers

    Parameters
    ----------

    peptide : str

    desired_length : int

    start_offset : int

    end_offset : int

    alphabet : str | list of characters
    """
    n = len(peptide)
    assert n > desired_length, \
        "%s (length = %d) is too short! Must be longer than %d" % (
            peptide, n, desired_length)
    n_skip = n - desired_length
    assert n_skip > 0, \
        "Expected length of peptide %s %d to be greater than %d" % (
            peptide, n, desired_length)
    end_range = n - end_offset - n_skip + 1
    return [
        peptide[:i] + peptide[i + n_skip:]
        for i in range(start_offset, end_range)
    ]


def fixed_length_from_many_peptides(
        peptides,
        desired_length,
        start_offset_extend=2,
        end_offset_extend=1,
        start_offset_shorten=2,
        end_offset_shorten=0,
        insert_amino_acid_letters=common_amino_acid_letters):
    """
    Create a set of fixed-length k-mer peptides from a collection of varying
    length peptides.


    Shorter peptides are filled in using all possible amino acids at any
    insertion site between start_offset_shorten and -end_offset_shorten
    where start_offset_extend=0 represents insertions before the string
    and end_offset_extend=0 represents insertions after the string's ending.

    Longer peptides are shortened by deleting contiguous residues, starting
    from start_offset_shorten and ending with -end_offset_shorten. Unlike
    peptide extensions, the offsets for shortening a peptide range between
    the first and last positions (rather than between the positions *before*
    the string starts and the position *after*).

    We can recreate the methods from:
       Accurate approximation method for prediction of class I MHC
       affinities for peptides of length 8, 10 and 11 using prediction
       tools trained on 9mers.
    by Lundegaard et. al. (http://bioinformatics.oxfordjournals.org/content/24/11/1397)
    with the following settings:
        - desired_length = 9
        - start_offset_extend = 3
        - end_offset_extend = 2
        - start_offset_shorten = 3
        - end_offset_shorten = 1

    Returns three lists:
        - a list of fixed length peptides (all of length `desired_length`)
        - a list of the original peptides from which subsequences were
          contracted or lengthened
        - a list of integers indicating the number of fixed length peptides
         generated from each variable length peptide.

    Example:
        kmers, original, counts = fixed_length_from_many_peptides(
            peptides=["ABC", "A"]
            desired_length=2,
            start_offset_extend=0,
            end_offset_extend=0,
            start_offset_shorten=0,
            end_offset_shorten=0,
            insert_amino_acid_letters="ABC")
        kmers == ["BC", "AC", "AB", "AA", "BA", "CA", "AA", "AB", "AC"]
        original == ["ABC", "ABC", "ABC", "A", "A", "A", "A", "A", "A"]
        counts == [3, 3, 3, 6, 6, 6, 6, 6, 6]

    Parameters
    ----------
    peptide : list of str

    desired_length : int

    start_offset_extend : int

    end_offset_extend : int

    start_offset_shorten : int

    end_offset_shorten : int

    insert_amino_acid_letters : str | list of characters
    """
    all_fixed_length_peptides = []
    original_peptide_sequences = []
    counts = []
    for peptide in peptides:
        n = len(peptide)
        if n == desired_length:
            fixed_length_peptides = [peptide]
        elif n < desired_length:
            try:
                fixed_length_peptides = extend_peptide(
                    peptide=peptide,
                    desired_length=desired_length,
                    start_offset=start_offset_extend,
                    end_offset=end_offset_extend,
                    insert_amino_acid_letters=insert_amino_acid_letters)
            except CombinatorialExplosion:
                logging.warn(
                    "Peptide %s is too short to be extended to length %d" % (
                        peptide, desired_length))
                continue
        else:
            fixed_length_peptides = shorten_peptide(
                peptide=peptide,
                desired_length=desired_length,
                start_offset=start_offset_shorten,
                end_offset=end_offset_shorten,
                insert_amino_acid_letters=insert_amino_acid_letters)

        n_fixed_length = len(fixed_length_peptides)
        all_fixed_length_peptides.extend(fixed_length_peptides)
        original_peptide_sequences.extend([peptide] * n_fixed_length)
        counts.extend([n_fixed_length] * n_fixed_length)
    return all_fixed_length_peptides, original_peptide_sequences, counts


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
        end_offset_extend=0,
        allow_unknown_amino_acids=False):
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
    if allow_unknown_amino_acids:
        insert_letters = ["X"]
        index_encoding = amino_acids_with_unknown.index_encoding
    else:
        insert_letters = common_amino_acids.letters()
        index_encoding = common_amino_acids.index_encoding

    fixed_length, original, counts = fixed_length_from_many_peptides(
        peptides=peptides,
        desired_length=desired_length,
        start_offset_shorten=start_offset_shorten,
        end_offset_shorten=end_offset_shorten,
        start_offset_extend=start_offset_extend,
        end_offset_extend=end_offset_extend,
        insert_amino_acid_letters=insert_letters)
    X = index_encoding(fixed_length, desired_length)
    return X, original, counts

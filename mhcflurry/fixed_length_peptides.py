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

from .amino_acid import amino_acid_letters


def all_kmers(k, alphabet=amino_acid_letters):
    """
    Generates all k-mer peptide sequences
    """
    return [
        "".join(combination)
        for combination
        in itertools.product(list(alphabet) * k)
    ]


def extend_peptide(
        peptide,
        desired_length,
        start_offset,
        end_offset,
        alphabet=amino_acid_letters):
    """Extend peptide by inserting every possible amino acid combination
    if we're trying to e.g. turn an 8mer into 9mers
    """
    n = len(peptide)
    assert n < desired_length
    n_missing = desired_length - n
    if n_missing > 3:
        raise ValueError(
            "Cannot transform %s of length %d into a %d-mer peptide" % (
                peptide, n, desired_length))
    return [
        peptide[:i] + extra + peptide[i:]
        for i in range(start_offset, n - end_offset)
        for extra in all_kmers(n_missing, alphabet=alphabet)
    ]


def shorten_peptide(
        peptide,
        desired_length,
        start_offset,
        end_offset,
        alphabet=amino_acid_letters):
    """Shorten peptide if trying to e.g. turn 10mer into 9mers"""
    n = len(peptide)
    assert n > desired_length
    n_skip = n - desired_length
    assert n_skip > 0, \
        "Expected length of peptide %s %d to be greater than %d" % (
            peptide, n, desired_length)
    end_range = n - end_offset - n_skip + 1
    return [
        peptide[:i] + peptide[i + n_skip:]
        for i in range(start_offset, end_range)
    ]


def fixed_length_from_single_peptide(
        peptide,
        desired_length,
        start_offset_extend=2,
        end_offset_extend=1,
        start_offset_shorten=2,
        end_offset_shorten=0,
        alphabet=amino_acid_letters):
    """
    Create a set of fixed-length k-mer peptides from a single peptide of any
    length. Shorter peptides are filled in using all possible amino acids at any
    insertion site between (start_offset, -end_offset).

    We can recreate the methods from:
       Accurate approximation method for prediction of class I MHC
       affinities for peptides of length 8, 10 and 11 using prediction
       tools trained on 9mers.
    by Lundegaard et. al. (http://bioinformatics.oxfordjournals.org/content/24/11/1397)
    with the following settings:
        - desired_length = 9
        - start_offset_extend = 2
        - end_offset_extend = 1
        - start_offset_shorten = 2
        - end_offset_shorten = 0
    """
    n = len(peptide)
    if n == desired_length:
        return [peptide]
    elif n < desired_length:
        return extend_peptide(
            peptide=peptide,
            desired_length=desired_length,
            start_offset=start_offset_extend,
            end_offset=end_offset_extend,
            alphabet=alphabet)
    else:
        return shorten_peptide(
            peptide=peptide,
            desired_length=desired_length,
            start_offset=start_offset_shorten,
            end_offset=end_offset_shorten,
            alphabet=alphabet)


def fixed_length_from_many_peptides(
        peptides,
        desired_length,
        start_offset_extend=2,
        end_offset_extend=1,
        start_offset_shorten=2,
        end_offset_shorten=0,
        alphabet=amino_acid_letters):
    """
    Create a set of fixed-length k-mer peptides from a collection of varying
    length peptides. Shorter peptides are filled in using all possible amino
    acids at any insertion site between
        [start_offset_extend, length - end_offset_extend).
    Longer peptides are made smaller by deleting contiguous residues between
        [start_offset_shorten, length - end_offset_shorten)

    We can recreate the methods from:
       Accurate approximation method for prediction of class I MHC
       affinities for peptides of length 8, 10 and 11 using prediction
       tools trained on 9mers.
    by Lundegaard et. al. (http://bioinformatics.oxfordjournals.org/content/24/11/1397)
    with the following settings:
        - desired_length = 9
        - start_offset_extend = 2
        - end_offset_extend = 1
        - start_offset_shorten = 2
        - end_offset_shorten = 0

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
            alphabet="ABC")
        kmers == ["BC", "AC", "AB", "AA", "BA", "CA", "AA", "AB", "AC"]
        original == ["ABC", "ABC", "ABC", "A", "A", "A", "A", "A", "A"]
        counts == [3, 3, 3, 6, 6, 6, 6, 6, 6]
    """
    fixed_length_peptides = []
    original_peptide_sequences = []
    number_expanded = []
    for peptide in peptides:
        fixed_length_peptides = fixed_length_from_single_peptide(
            peptide,
            desired_length=desired_length,
            start_offset_extend=start_offset_extend,
            end_offset_extend=end_offset_extend,
            start_offset_shorten=start_offset_shorten,
            end_offset_shorten=end_offset_shorten,
            alphabet=alphabet)
        n_fixed_length = len(fixed_length_peptides)
        fixed_length_peptides.extend(fixed_length_peptides)
        original_peptide_sequences.extend([peptide] * n_fixed_length)
        number_expanded.extend([n_fixed_length] * n_fixed_length)
    return fixed_length_peptides, original_peptide_sequences, number_expanded

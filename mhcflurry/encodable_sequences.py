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

from __future__ import (
    print_function,
    division,
    absolute_import,
)

import math

import pandas
import numpy
from six import StringIO

import typechecks

from . import amino_acid

AMINO_ACIDS = list(amino_acid.COMMON_AMINO_ACIDS_WITH_UNKNOWN.keys())

BLOSUM62_MATRIX = pandas.read_table(StringIO("""
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  X
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3  0
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  0
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  0
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1  0
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  0
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3  0
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3  0
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1  0
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1  0
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1  0
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2  0
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0 
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3  0
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1  0
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4  0
X  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
"""), sep='\s+').loc[AMINO_ACIDS, AMINO_ACIDS]
assert (BLOSUM62_MATRIX == BLOSUM62_MATRIX.T).all().all()

ENCODING_DFS = {
    "BLOSUM62": BLOSUM62_MATRIX,
    "one-hot": pandas.DataFrame([
        [1 if i == j else 0 for i in range(len(AMINO_ACIDS))]
        for j in range(len(AMINO_ACIDS))
    ], index=AMINO_ACIDS, columns=AMINO_ACIDS)
}


def available_vector_encodings():
    """
    Return list of supported amino acid vector encodings.

    Returns
    -------
    list of string

    """
    return list(ENCODING_DFS)


def vector_encoding_length(name):
    """
    Return the length of the given vector encoding.

    Parameters
    ----------
    name : string

    Returns
    -------
    int
    """
    return ENCODING_DFS[name].shape[1]


def index_encoding(sequences, letter_to_index_dict):
    """
    Given a sequence of n strings all of length k, return a k * n array where
    the (i, j)th element is letter_to_index_dict[sequence[i][j]].
    
    Parameters
    ----------
    sequences : list of length n of strings of length k
    letter_to_index_dict : dict : string -> int

    Returns
    -------
    numpy.array of integers with shape (k, n)
    """
    df = pandas.DataFrame(iter(s) for s in sequences)
    result = df.replace(letter_to_index_dict)
    return result.values


def fixed_vectors_encoding(sequences, letter_to_vector_function):
    """
    Given a sequence of n strings all of length k, return a n * k * m array where
    the (i, j)th element is letter_to_vector_function(sequence[i][j]).

    Parameters
    ----------
    sequences : list of length n of strings of length k
    letter_to_vector_function : function : string -> vector of length m

    Returns
    -------
    numpy.array of integers with shape (n, k, m)
    """
    arr = numpy.array([list(s) for s in sequences])
    result = numpy.vectorize(
        letter_to_vector_function, signature='()->(n)')(arr)
    return result


class EncodableSequences(object):
    """
    Sequences of amino acids.
    
    This class caches various encodings of a list of sequences.
    """
    unknown_character = "X"

    @classmethod
    def create(klass, sequences):
        """
        Factory that returns an EncodableSequences given a list of
        strings. As a convenience, you can also pass it an EncodableSequences
        instance, in which case the object is returned unchanged.
        """
        if isinstance(sequences, klass):
            return sequences
        return klass(sequences)

    def __init__(self, sequences):
        typechecks.require_iterable_of(
            sequences, typechecks.string_types, "sequences")
        self.sequences = numpy.array(sequences)
        self.encoding_cache = {}
        self.fixed_sequence_length = None
        if len(self.sequences) > 0 and all(
                len(s) == len(self.sequences[0]) for s in self.sequences):
            self.fixed_sequence_length = len(self.sequences[0])

    def __len__(self):
        return len(self.sequences)

    def variable_length_to_fixed_length_categorical(
            self, left_edge=4, right_edge=4, max_length=15):
        """
        Encode variable-length sequences using a fixed-length encoding designed
        for preserving the anchor positions of class I peptides.
        
        The sequences must be of length at least left_edge + right_edge, and at
        most max_length.
        
        Parameters
        ----------
        left_edge : int, size of fixed-position left side
        right_edge : int, size of the fixed-position right side
        max_length : sequence length of the resulting encoding

        Returns
        -------
        numpy.array of integers with shape (num sequences, max_length)
        """

        cache_key = (
            "fixed_length_categorical",
            left_edge,
            right_edge,
            max_length)

        if cache_key not in self.encoding_cache:
            fixed_length_sequences = [
                self.sequence_to_fixed_length_string(
                    sequence,
                    left_edge=left_edge,
                    right_edge=right_edge,
                    max_length=max_length)
                for sequence in self.sequences
            ]
            self.encoding_cache[cache_key] = index_encoding(
                fixed_length_sequences, amino_acid.AMINO_ACID_INDEX)
        return self.encoding_cache[cache_key]

    def variable_length_to_fixed_length_vector_encoding(
            self, vector_encoding_name, left_edge=4, right_edge=4, max_length=15):
        """
        Encode variable-length sequences using a fixed-length encoding designed
        for preserving the anchor positions of class I peptides.

        The sequences must be of length at least left_edge + right_edge, and at
        most max_length.

        Parameters
        ----------
        vector_encoding_name : string
            How to represent amino acids.
            One of "BLOSUM62", "one-hot", etc. Full list of supported vector
            encodings is given by available_vector_encodings().
        left_edge : int, size of fixed-position left side
        right_edge : int, size of the fixed-position right side
        max_length : sequence length of the resulting encoding

        Returns
        -------
        numpy.array with shape (num sequences, max_length, m) where m is
        vector_encoding_length(vector_encoding_name)

        """
        cache_key = (
            "fixed_length_vector_encoding",
            vector_encoding_name,
            left_edge,
            right_edge,
            max_length)
        if cache_key not in self.encoding_cache:
            fixed_length_sequences = [
                self.sequence_to_fixed_length_string(
                    sequence,
                    left_edge=left_edge,
                    right_edge=right_edge,
                    max_length=max_length)
                for sequence in self.sequences
            ]
            result = fixed_vectors_encoding(
                fixed_length_sequences,
                ENCODING_DFS[vector_encoding_name].loc.__getitem__)
            assert result.shape[0] == len(self.sequences)
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]


    @classmethod
    def sequence_to_fixed_length_string(
            klass, sequence, left_edge=4, right_edge=4, max_length=15):
        """
        Transform a string of length at least left_edge + right_edge and at
        most max_length into a string of length max_length using a scheme
        designed to preserve the anchor positions of class I peptides.
        
        The first left_edge characters in the input always map to the first
        left_edge characters in the output. Similarly for the last right_edge
        characters. The middle characters are filled in based on the length,
        with the X character filling in the blanks.
        
        For example, using defaults:
        
        AAAACDDDD -> AAAAXXXCXXXDDDD
        
        
        Parameters
        ----------
        sequence : string
        left_edge : int
        right_edge : int
        max_length : int

        Returns
        -------
        string of length max_length

        """
        if len(sequence) < left_edge + right_edge:
            raise ValueError(
                "Sequence '%s' (length %d) unsupported: length must be at "
                "least %d" % (sequence, len(sequence), left_edge + right_edge))
        if len(sequence) > max_length:
            raise ValueError(
                "Sequence '%s' (length %d) unsupported: length must be at "
                "most %d" % (sequence, len(sequence), max_length))

        middle_length = max_length - left_edge - right_edge

        num_null = max_length - len(sequence)
        num_null_left = int(math.ceil(num_null / 2))
        num_null_right = int(math.floor(num_null / 2))
        num_not_null_middle = middle_length - num_null
        string_encoding = "".join([
            sequence[:left_edge],
            klass.unknown_character * num_null_left,
            sequence[left_edge:left_edge + num_not_null_middle],
            klass.unknown_character * num_null_right,
            sequence[-right_edge:],
        ])
        assert len(string_encoding) == max_length
        return string_encoding

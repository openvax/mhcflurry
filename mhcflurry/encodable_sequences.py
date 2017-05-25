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

import typechecks

from . import amino_acid


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


def one_hot_encoding(index_encoded, alphabet_size):
    """
    Given an n * k array of integers in the range [0, alphabet_size), return
    an n * k * alphabet_size array where element (i, k, j) is 1 if element
    (i, k) == j in the input array and zero otherwise.
    
    Parameters
    ----------
    index_encoded : numpy.array of integers with shape (n, k)
    alphabet_size : int 

    Returns
    -------
    numpy.array of integers of shape (n, k, alphabet_size)

    """
    alphabet_size = int(alphabet_size)
    (num_sequences, sequence_length) = index_encoded.shape
    result = numpy.zeros(
        (num_sequences, sequence_length, alphabet_size),
        dtype='int32')

    # Transform the index encoded array into an array of indices into the
    # flattened result, which we will set to 1.
    flattened_indices = (
        index_encoded +
        (
            sequence_length * alphabet_size * numpy.arange(num_sequences)
        ).reshape((-1, 1)) +
        numpy.tile(numpy.arange(sequence_length),
                   (num_sequences, 1)) * alphabet_size)
    result.put(flattened_indices, 1)
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

    def fixed_length_categorical(self):
        """
        Returns a categorical encoding (i.e. integers 0 <= x < 21) of the
        sequences, which must already be all the same length.
        
        Returns
        -------
        numpy.array of integers
        """
        cache_key = ("categorical",)
        if cache_key not in self.encoding_cache:
            assert self.fixed_sequence_length
            self.encoding_cache[cache_key] = index_encoding(
                self.sequences, amino_acid.AMINO_ACID_INDEX)
        return self.encoding_cache[cache_key]

    def fixed_length_one_hot(self):
        """
        Returns a binary one-hot encoding of the  sequences, which must already
        be all the same length.
        
        Returns
        -------
        numpy.array of integers
        """
        cache_key = ("one_hot",)
        if cache_key not in self.encoding_cache:
            assert self.fixed_sequence_length
            encoded = self.categorical_encoding()
            result = one_hot_encoding(
                encoded, alphabet_size=len(amino_acid.AMINO_ACID_INDEX))
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]

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

    def variable_length_to_fixed_length_one_hot(
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
        binary numpy.array with shape (num sequences, max_length, 21)
        """

        cache_key = (
            "fixed_length_one_hot",
            left_edge,
            right_edge,
            max_length)

        if cache_key not in self.encoding_cache:
            encoded = self.variable_length_to_fixed_length_categorical(
                left_edge=left_edge,
                right_edge=right_edge,
                max_length=max_length)
            result = one_hot_encoding(
                encoded, alphabet_size=len(amino_acid.AMINO_ACID_INDEX))
            assert result.shape == (
                len(self.sequences),
                encoded.shape[1],
                len(amino_acid.AMINO_ACID_INDEX))
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

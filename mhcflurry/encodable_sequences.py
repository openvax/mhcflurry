from __future__ import (
    print_function,
    division,
    absolute_import,
)

import math
from six import string_types

import numpy
import pandas

from . import amino_acid


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
        if not all(isinstance(obj, string_types) for obj in sequences):
            raise ValueError("Sequence of strings is required")
        self.sequences = numpy.array(sequences)
        lengths = pandas.Series(self.sequences).str.len()

        self.min_length = lengths.min()
        self.max_length = lengths.max()

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
            fixed_length_sequences = (
                self.sequences_to_fixed_length_index_encoded_array(
                    self.sequences,
                    left_edge=left_edge,
                    right_edge=right_edge,
                    max_length=max_length))
            self.encoding_cache[cache_key] = fixed_length_sequences
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
            fixed_length_sequences = (
                self.sequences_to_fixed_length_index_encoded_array(
                    self.sequences,
                    left_edge=left_edge,
                    right_edge=right_edge,
                    max_length=max_length))
            result = amino_acid.fixed_vectors_encoding(
                fixed_length_sequences,
                amino_acid.ENCODING_DATA_FRAMES[vector_encoding_name])
            assert result.shape[0] == len(self.sequences)
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]

    @classmethod
    def sequences_to_fixed_length_index_encoded_array(
            klass, sequences, left_edge=4, right_edge=4, max_length=15):
        """
        Transform a sequence of strings, where each string is of length at least
        left_edge + right_edge and at most max_length into strings of length
        max_length using a scheme designed to preserve the anchor positions of
        class I peptides.

        The first left_edge characters in the input always map to the first
        left_edge characters in the output. Similarly for the last right_edge
        characters. The middle characters are filled in based on the length,
        with the X character filling in the blanks.

        For example, using defaults:

        AAAACDDDD -> AAAAXXXCXXXDDDD

        The strings are also converted to int categorical amino acid indices.

        Parameters
        ----------
        sequence : string
        left_edge : int
        right_edge : int
        max_length : int

        Returns
        -------
        numpy array of shape (len(sequences), max_length) and dtype int
        """

        # Result array is int32, filled with X (null amino acid) value.
        result = numpy.full(
            fill_value=amino_acid.AMINO_ACID_INDEX['X'],
            shape=(len(sequences), max_length),
            dtype="int32")

        df = pandas.DataFrame({"peptide": sequences})
        df["length"] = df.peptide.str.len()

        middle_length = max_length - left_edge - right_edge

        # For efficiency we handle each supported peptide length using bulk
        # array operations.
        for (length, sub_df) in df.groupby("length"):
            if length < left_edge + right_edge:
                raise ValueError(
                    "Sequence '%s' (length %d) unsupported: length must be at "
                    "least %d. There are %d total peptides with this length." % (
                        sub_df.iloc[0].peptide, length, left_edge + right_edge,
                        len(sub_df)))
            if length > max_length:
                raise ValueError(
                    "Sequence '%s' (length %d) unsupported: length must be at "
                    "most %d. There are %d total peptides with this length." % (
                        sub_df.iloc[0].peptide, length, max_length,
                        len(sub_df)))

            # Array of shape (num peptides, length) giving fixed-length amino
            # acid encoding each peptide of the current length.
            fixed_length_sequences = numpy.stack(
                sub_df.peptide.map(
                    lambda s: numpy.array([
                        amino_acid.AMINO_ACID_INDEX[char] for char in s
                    ])).values)

            num_null = max_length - length
            num_null_left = int(math.ceil(num_null / 2))
            num_middle_filled = middle_length - num_null
            middle_start = left_edge + num_null_left

            # Set left edge
            result[sub_df.index, :left_edge] = fixed_length_sequences[
                :, :left_edge
            ]

            # Set middle.
            result[
                sub_df.index,
                middle_start : middle_start + num_middle_filled
            ] = fixed_length_sequences[
                :, left_edge : left_edge + num_middle_filled
            ]

            # Set right edge.
            result[
                sub_df.index,
                -right_edge:
            ] = fixed_length_sequences[:, -right_edge:]
        return result

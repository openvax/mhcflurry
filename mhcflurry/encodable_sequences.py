"""
Class for encoding variable-length peptides to fixed-size numerical matrices
"""
from __future__ import (
    print_function,
    division,
    absolute_import,
)

import math
from six import string_types
from functools import partial

import numpy
import pandas

from . import amino_acid


class EncodingError(ValueError):
    """
    Exception raised when peptides cannot be encoded
    """
    def __init__(self, message, supported_peptide_lengths):
        self.supported_peptide_lengths = supported_peptide_lengths
        ValueError.__init__(
            self,
            message + " Supported lengths: %s - %s." % supported_peptide_lengths)


class EncodableSequences(object):
    """
    Class for encoding variable-length peptides to fixed-size numerical matrices
    
    This class caches various encodings of a list of sequences.

    In practice this is used only for peptides. To encode MHC allele sequences,
    see AlleleEncoding.
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
        lengths = pandas.Series(self.sequences, dtype=numpy.object_).str.len()

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
            self,
            alignment_method="pad_middle",
            left_edge=4,
            right_edge=4,
            max_length=15):
        """
        Encode variable-length sequences to a fixed-size index-encoded (integer)
        matrix.

        See `sequences_to_fixed_length_index_encoded_array` for details.
        
        Parameters
        ----------
        alignment_method : string
            One of "pad_middle" or "left_pad_right_pad"
        left_edge : int, size of fixed-position left side
            Only relevant for pad_middle alignment method
        right_edge : int, size of the fixed-position right side
            Only relevant for pad_middle alignment method
        max_length : maximum supported peptide length

        Returns
        -------
        numpy.array of integers with shape (num sequences, encoded length)

        For pad_middle, the encoded length is max_length. For left_pad_right_pad,
        it's 3 * max_length.
        """

        cache_key = (
            "fixed_length_categorical",
            alignment_method,
            left_edge,
            right_edge,
            max_length)

        if cache_key not in self.encoding_cache:
            fixed_length_sequences = (
                self.sequences_to_fixed_length_index_encoded_array(
                    self.sequences,
                    alignment_method=alignment_method,
                    left_edge=left_edge,
                    right_edge=right_edge,
                    max_length=max_length))
            self.encoding_cache[cache_key] = fixed_length_sequences
        return self.encoding_cache[cache_key]

    def variable_length_to_fixed_length_vector_encoding(
            self,
            vector_encoding_name,
            alignment_method="pad_middle",
            left_edge=4,
            right_edge=4,
            max_length=15,
            trim=False,
            allow_unsupported_amino_acids=False):
        """
        Encode variable-length sequences to a fixed-size matrix. Amino acids
        are encoded as specified by the vector_encoding_name argument.

        See `sequences_to_fixed_length_index_encoded_array` for details.

        See also: variable_length_to_fixed_length_categorical.

        Parameters
        ----------
        vector_encoding_name : string
            How to represent amino acids.
            One of "BLOSUM62", "one-hot", etc. Full list of supported vector
            encodings is given by available_vector_encodings().
        alignment_method : string
            One of "pad_middle" or "left_pad_right_pad"
        left_edge : int
            Size of fixed-position left side.
            Only relevant for pad_middle alignment method
        right_edge : int
            Size of the fixed-position right side.
            Only relevant for pad_middle alignment method
        max_length : int
            Maximum supported peptide length
        trim : bool
            If True, longer sequences will be trimmed to fit the maximum
            supported length. Not supported for all alignment methods.
        allow_unsupported_amino_acids : bool
            If True, non-canonical amino acids will be replaced with the X
            character before encoding.

        Returns
        -------
        numpy.array with shape (num sequences, encoded length, m)

        where
            - m is the vector encoding length (usually 21).
            - encoded length is max_length if alignment_method is pad_middle;
              3 * max_length if it's left_pad_right_pad.
        """
        cache_key = (
            "fixed_length_vector_encoding",
            vector_encoding_name,
            alignment_method,
            left_edge,
            right_edge,
            max_length,
            trim,
            allow_unsupported_amino_acids)
        if cache_key not in self.encoding_cache:
            fixed_length_sequences = (
                self.sequences_to_fixed_length_index_encoded_array(
                    self.sequences,
                    alignment_method=alignment_method,
                    left_edge=left_edge,
                    right_edge=right_edge,
                    max_length=max_length,
                    trim=trim,
                    allow_unsupported_amino_acids=allow_unsupported_amino_acids))
            result = amino_acid.fixed_vectors_encoding(
                fixed_length_sequences,
                amino_acid.ENCODING_DATA_FRAMES[vector_encoding_name])
            assert result.shape[0] == len(self.sequences)
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]

    @classmethod
    def sequences_to_fixed_length_index_encoded_array(
            klass,
            sequences,
            alignment_method="pad_middle",
            left_edge=4,
            right_edge=4,
            max_length=15,
            trim=False,
            allow_unsupported_amino_acids=False):
        """
        Encode variable-length sequences to a fixed-size index-encoded (integer)
        matrix.

        How variable length sequences get mapped to fixed length is set by the
        "alignment_method" argument. Supported alignment methods are:

            pad_middle
                Encoding designed for preserving the anchor positions of class
                I peptides. This is what is used in allele-specific models.
                
                Each string must be of length at least left_edge + right_edge
                and at most max_length. The first left_edge characters in the
                input always map to the first left_edge characters in the
                output. Similarly for the last right_edge characters. The
                middle characters are filled in based on the length, with the
                X character filling in the blanks.

                Example:

                AAAACDDDD -> AAAAXXXCXXXDDDD

            left_pad_centered_right_pad
                Encoding that makes no assumptions on anchor positions but is
                3x larger than pad_middle, since it duplicates the peptide
                (left aligned + centered + right aligned). This is what is used
                for the pan-allele models.

                Example:

                AAAACDDDD -> AAAACDDDDXXXXXXXXXAAAACDDDDXXXXXXXXXAAAACDDDD

            left_pad_right_pad
                Same as left_pad_centered_right_pad but only includes left-
                and right-padded peptide.

                Example:

                AAAACDDDD -> AAAACDDDDXXXXXXXXXXXXAAAACDDDD

        Parameters
        ----------
        sequences : list of string
        alignment_method : string
            One of "pad_middle" or "left_pad_right_pad"
        left_edge : int
            Size of fixed-position left side.
            Only relevant for pad_middle alignment method
        right_edge : int
            Size of the fixed-position right side.
            Only relevant for pad_middle alignment method
        max_length : int
            maximum supported peptide length
        trim : bool
            If True, longer sequences will be trimmed to fit the maximum
            supported length. Not supported for all alignment methods.
        allow_unsupported_amino_acids : bool
            If True, non-canonical amino acids will be replaced with the X
            character before encoding.

        Returns
        -------
        numpy.array of integers with shape (num sequences, encoded length)

        For pad_middle, the encoded length is max_length. For left_pad_right_pad,
        it's 2 * max_length. For left_pad_centered_right_pad, it's
        3 * max_length.
        """
        if allow_unsupported_amino_acids:
            fill_value = amino_acid.AMINO_ACID_INDEX['X']

            def get_amino_acid_index(a):
                return amino_acid.AMINO_ACID_INDEX.get(a, fill_value)
        else:
            get_amino_acid_index = amino_acid.AMINO_ACID_INDEX.__getitem__

        result = None
        if alignment_method == 'pad_middle':
            if trim:
                raise NotImplementedError("trim not supported")

            # Result array is int32, filled with X (null amino acid) value.
            result = numpy.full(
                fill_value=amino_acid.AMINO_ACID_INDEX['X'],
                shape=(len(sequences), max_length),
                dtype="int32")

            df = pandas.DataFrame({"peptide": sequences}, dtype=numpy.object_)
            df["length"] = df.peptide.str.len()

            middle_length = max_length - left_edge - right_edge
            min_length = left_edge + right_edge

            # For efficiency we handle each supported peptide length using bulk
            # array operations.
            for (length, sub_df) in df.groupby("length"):
                if length < min_length or length > max_length:
                    raise EncodingError(
                        "Sequence '%s' (length %d) unsupported. There are %d "
                        "total peptides with this length." % (
                            sub_df.iloc[0].peptide,
                            length,
                            len(sub_df)), supported_peptide_lengths=(
                                min_length, max_length))

                # Array of shape (num peptides, length) giving fixed-length
                # amino acid encoding each peptide of the current length.
                fixed_length_sequences = numpy.stack(
                    sub_df.peptide.map(
                        lambda s: numpy.array([
                            get_amino_acid_index(char) for char in s
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
        elif alignment_method == "left_pad_right_pad":
            if trim:
                raise NotImplementedError("trim not supported")

            # We arbitrarily set a minimum length of 5, although this encoding
            # could handle smaller peptides.
            min_length = 5

            # Result array is int32, filled with X (null amino acid) value.
            result = numpy.full(
                fill_value=amino_acid.AMINO_ACID_INDEX['X'],
                shape=(len(sequences), max_length * 2),
                dtype="int32")

            df = pandas.DataFrame({"peptide": sequences}, dtype=numpy.object_)

            # For efficiency we handle each supported peptide length using bulk
            # array operations.
            for (length, sub_df) in df.groupby(df.peptide.str.len()):
                if length < min_length or length > max_length:
                    raise EncodingError(
                        "Sequence '%s' (length %d) unsupported. There are %d "
                        "total peptides with this length." % (
                            sub_df.iloc[0].peptide,
                            length,
                            len(sub_df)), supported_peptide_lengths=(
                                min_length, max_length))

                # Array of shape (num peptides, length) giving fixed-length
                # amino acid encoding each peptide of the current length.
                fixed_length_sequences = numpy.stack(sub_df.peptide.map(
                    lambda s: numpy.array([
                        get_amino_acid_index(char) for char in s
                    ])).values)

                # Set left edge
                result[sub_df.index, :length] = fixed_length_sequences

                # Set right edge.
                result[sub_df.index, -length:] = fixed_length_sequences
        elif alignment_method == "left_pad_centered_right_pad":
            if trim:
                raise NotImplementedError("trim not supported")

            # We arbitrarily set a minimum length of 5, although this encoding
            # could handle smaller peptides.
            min_length = 5

            # Result array is int32, filled with X (null amino acid) value.
            result = numpy.full(
                fill_value=amino_acid.AMINO_ACID_INDEX['X'],
                shape=(len(sequences), max_length * 3),
                dtype="int32")

            df = pandas.DataFrame({"peptide": sequences}, dtype=numpy.object_)

            # For efficiency we handle each supported peptide length using bulk
            # array operations.
            for (length, sub_df) in df.groupby(df.peptide.str.len()):
                if length < min_length or length > max_length:
                    raise EncodingError(
                        "Sequence '%s' (length %d) unsupported. There are %d "
                        "total peptides with this length." % (
                            sub_df.iloc[0].peptide,
                            length,
                            len(sub_df)), supported_peptide_lengths=(
                                min_length, max_length))

                # Array of shape (num peptides, length) giving fixed-length
                # amino acid encoding each peptide of the current length.
                fixed_length_sequences = numpy.stack(sub_df.peptide.map(
                    lambda s: numpy.array([
                        get_amino_acid_index(char) for char in s
                    ])).values)

                # Set left edge
                result[sub_df.index, :length] = fixed_length_sequences

                # Set right edge.
                result[sub_df.index, -length:] = fixed_length_sequences

                # Set center.
                center_left_padding = int(
                    math.floor((max_length - length) / 2))
                center_left_offset = max_length + center_left_padding
                result[
                    sub_df.index,
                    center_left_offset : center_left_offset + length
                ] = fixed_length_sequences
        elif alignment_method in ("right_pad", "left_pad"):
            min_length = 1

            # Result array is int32, filled with X (null amino acid) value.
            result = numpy.full(
                fill_value=amino_acid.AMINO_ACID_INDEX['X'],
                shape=(len(sequences), max_length),
                dtype="int32")

            df = pandas.DataFrame({"peptide": sequences}, dtype=numpy.object_)

            # For efficiency we handle each supported peptide length using bulk
            # array operations.
            for (length, sub_df) in df.groupby(df.peptide.str.len()):
                if length < min_length or (not trim and length > max_length):
                    raise EncodingError(
                        "Sequence '%s' (length %d) unsupported. There are %d "
                        "total peptides with this length." % (
                            sub_df.iloc[0].peptide,
                            length,
                            len(sub_df)), supported_peptide_lengths=(
                                min_length, max_length))

                peptides = sub_df.peptide
                if length > max_length:
                    # Trim.
                    if alignment_method == "right_pad":
                        peptides = peptides.str.slice(0, max_length)
                    else:
                        peptides = peptides.str.slice(length - max_length)

                # Array of shape (num peptides, length) giving fixed-length
                # amino acid encoding each peptide of the current length.
                fixed_length_sequences = numpy.stack(peptides.map(
                    lambda s: numpy.array([
                        get_amino_acid_index(char) for char in s
                    ])).values)

                if alignment_method == "right_pad":
                    # Left align (i.e. pad right): set left edge
                    result[sub_df.index, :length] = fixed_length_sequences
                else:
                    # Right align: set right edge.
                    result[sub_df.index, -length:] = fixed_length_sequences

        else:
            raise NotImplementedError(
                "Unsupported alignment method: %s" % alignment_method)


        return result

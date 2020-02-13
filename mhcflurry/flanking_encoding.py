"""
Class for encoding variable-length flanking and peptides to
fixed-size numerical matrices
"""
from __future__ import (
    print_function, division, absolute_import, )

from six import string_types
from collections import namedtuple

from .encodable_sequences import EncodingError, EncodableSequences

import numpy
import pandas


EncodingResult =  namedtuple(
    "EncodingResult", ["array", "peptide_lengths"])

class FlankingEncoding(object):
    """
    """
    unknown_character = "X"

    def __init__(self, peptides, n_flanks, c_flanks):
        self.dataframe = pandas.DataFrame({
            "peptide": peptides,
            "n_flank": n_flanks,
            "c_flank": c_flanks,
        }, dtype=str)
        self.encoding_cache = {}

    def __len__(self):
        return len(self.dataframe)

    def vector_encode(
            self,
            vector_encoding_name,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            allow_unsupported_amino_acids=True):
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
        left_edge : int, size of fixed-position left side
            Only relevant for pad_middle alignment method
        right_edge : int, size of the fixed-position right side
            Only relevant for pad_middle alignment method
        max_length : maximum supported peptide length

        Returns
        -------
        numpy.array with shape (num sequences, encoded length, m)

        where
            - m is the vector encoding length (usually 21).
            - encoded length is max_length if alignment_method is pad_middle;
              3 * max_length if it's left_pad_right_pad.
        """
        cache_key = (
            "vector_encode",
            vector_encoding_name,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            allow_unsupported_amino_acids)
        if cache_key not in self.encoding_cache:
            result = self.encode(
                vector_encoding_name=vector_encoding_name,
                df=self.dataframe,
                peptide_max_length=peptide_max_length,
                n_flank_length=n_flank_length,
                c_flank_length=c_flank_length,
                allow_unsupported_amino_acids=allow_unsupported_amino_acids)
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]

    @staticmethod
    def encode(
        vector_encoding_name,
        df,
        peptide_max_length,
        n_flank_length,
        c_flank_length,
        allow_unsupported_amino_acids=False):
        """
        """
        error_df = df.loc[
            (df.peptide.str.len() > peptide_max_length) |
            (df.peptide.str.len() < 1)
        ]
        if len(error_df) > 0:
            raise EncodingError(
                "Sequence '%s' (length %d) unsupported. There are %d "
                "total peptides with this length." % (
                    error_df.iloc[0].peptide,
                    len(error_df.iloc[0].peptide),
                    len(error_df)),
                supported_peptide_lengths=(1, peptide_max_length + 1))

        if n_flank_length > 0:
            n_flanks = df.n_flank.str.pad(
                n_flank_length,
                side="left",
                fillchar="X").str.slice(-n_flank_length).str.upper()
        else:
            n_flanks = pandas.Series([""] * len(df))

        c_flanks = df.c_flank.str.pad(
            c_flank_length,
            side="right",
            fillchar="X").str.slice(0, c_flank_length).str.upper()
        peptides = df.peptide.str.upper()

        concatenated = n_flanks + peptides + c_flanks

        encoder = EncodableSequences.create(concatenated.values)
        array = encoder.variable_length_to_fixed_length_vector_encoding(
            vector_encoding_name=vector_encoding_name,
            alignment_method="right_pad",
            max_length=n_flank_length + peptide_max_length + c_flank_length,
            allow_unsupported_amino_acids=allow_unsupported_amino_acids)

        result = EncodingResult(
            array, peptide_lengths=peptides.str.len().values)

        return result

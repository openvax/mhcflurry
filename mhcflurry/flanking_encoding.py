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
    Encode peptides and optionally their N- and C-flanking sequences into fixed
    size numerical matrices. Similar to EncodableSequences but with support
    for flanking sequences and the encoding scheme used by the processing
    predictor.

    Instances of this class have an immutable list of peptides with
    flanking sequences. Encodings are cached in the instances for faster
    performance when the same set of peptides needs to encoded more than once.
    """
    unknown_character = "X"

    def __init__(self, peptides, n_flanks, c_flanks):
        """
        Constructor. Sequences of any lengths can be passed.

        Parameters
        ----------
        peptides : list of string
            Peptide sequences
        n_flanks : list of string [same length as peptides]
            Upstream sequences
        c_flanks : list of string [same length as peptides]
            Downstream sequences
        """
        self.dataframe = pandas.DataFrame({
            "peptide": peptides,
            "n_flank": n_flanks,
            "c_flank": c_flanks,
        }, dtype=str)
        self.encoding_cache = {}

    def __len__(self):
        """
        Number of peptides.
        """
        return len(self.dataframe)

    def vector_encode(
            self,
            vector_encoding_name,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            allow_unsupported_amino_acids=True):
        """
        Encode variable-length sequences to a fixed-size matrix.

        Parameters
        ----------
        vector_encoding_name : string
            How to represent amino acids. One of "BLOSUM62", "one-hot", etc.
            See `amino_acid.available_vector_encodings()`.
        peptide_max_length : int
            Maximum supported peptide length.
        n_flank_length : int
            Maximum supported N-flank length
        c_flank_length : int
            Maximum supported C-flank length
        allow_unsupported_amino_acids : bool
            If True, non-canonical amino acids will be replaced with the X
            character before encoding.

        Returns
        -------
        numpy.array with shape (num sequences, length, m)

        where
            - num sequences is number of peptides, i.e. len(self)
            - length is peptide_max_length + n_flank_length + c_flank_length
            - m is the vector encoding length (usually 21).
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
        Encode variable-length sequences to a fixed-size matrix.

        Helper function. Users should use `vector_encode`.

        Parameters
        ----------
        vector_encoding_name : string
        df : pandas.DataFrame
        peptide_max_length : int
        n_flank_length : int
        c_flank_length : int
        allow_unsupported_amino_acids : bool

        Returns
        -------
        numpy.array
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

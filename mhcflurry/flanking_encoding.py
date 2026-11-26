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

"""
Class for encoding variable-length flanking and peptides to
fixed-size numerical matrices
"""
from collections import namedtuple
import logging

from .encodable_sequences import EncodingError, EncodableSequences

import numpy
import pandas


EncodingResult = namedtuple(
    "EncodingResult",
    ["array", "peptide_lengths", "unsupported_mask"],
    defaults=(None,))


class FlankingEncoding(object):
    """
    Encode peptides and optionally their N- and C-flanking sequences into fixed
    size numerical matrices. Similar to EncodableSequences but with support
    for flanking sequences and the encoding scheme used by the processing
    predictor.

    Instances of this class have an immutable list of peptides with
    flanking sequences. Encodings are cached in the instances for faster
    performance when the same set of peptides needs to be encoded more than once.
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
        def _as_flank_strings(values):
            return [
                "" if (value is None or pandas.isna(value)) else str(value)
                for value in values
            ]

        self.dataframe = pandas.DataFrame({
            "peptide": [str(p) for p in peptides],
            "n_flank": _as_flank_strings(n_flanks),
            "c_flank": _as_flank_strings(c_flanks),
        }, dtype=str)
        self.encoding_cache = {}
        self.tensor_cache = {}

    def __getstate__(self):
        """Drop device tensors when pickling."""
        state = self.__dict__.copy()
        state["tensor_cache"] = {}
        return state

    def __len__(self):
        """
        Number of peptides.
        """
        return len(self.dataframe)

    def clear_tensor_cache(self):
        """Release cached torch tensors held by this encoding."""
        self.tensor_cache.clear()

    def categorical_encode(
            self,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            allow_unsupported_amino_acids=True,
            throw=True):
        """
        Encode variable-length sequences to fixed-size amino-acid indices.

        The resulting ``int8`` array is suited for torch-side embedding lookup.
        """
        cache_key = (
            "categorical_encode",
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            allow_unsupported_amino_acids,
            throw)
        if cache_key not in self.encoding_cache:
            result = self.encode_indices(
                df=self.dataframe,
                peptide_max_length=peptide_max_length,
                n_flank_length=n_flank_length,
                c_flank_length=c_flank_length,
                allow_unsupported_amino_acids=allow_unsupported_amino_acids,
                throw=throw)
            self.encoding_cache[cache_key] = result
        return self.encoding_cache[cache_key]

    def categorical_encode_tensors(
            self,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            device,
            allow_unsupported_amino_acids=True,
            throw=True):
        """
        Encode variable-length sequences to cached torch tensors.

        The CPU numpy encoding remains the source of truth. This method keeps
        a device-resident view/copy keyed by encoding parameters and device so
        repeated processing models can slice the same tensors without per-batch
        host-to-device transfers.
        """
        import torch

        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        cache_key = (
            "categorical_encode_tensors",
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            allow_unsupported_amino_acids,
            throw,
            device.type,
            device.index)
        if cache_key not in self.tensor_cache:
            encoded = self.categorical_encode(
                peptide_max_length=peptide_max_length,
                n_flank_length=n_flank_length,
                c_flank_length=c_flank_length,
                allow_unsupported_amino_acids=allow_unsupported_amino_acids,
                throw=throw)

            sequence_array = encoded.array
            if not sequence_array.flags.writeable:
                sequence_array = sequence_array.copy()
            peptide_lengths = numpy.asarray(encoded.peptide_lengths)
            if not peptide_lengths.flags.writeable:
                peptide_lengths = peptide_lengths.copy()

            # These transfers are blocking: the source tensors come from
            # torch.from_numpy(...) and are not pinned, so non_blocking=True
            # would be a silent no-op. This is a one-time, cached transfer per
            # cache key (not a per-batch hot path), so pinning the host buffer
            # to enable an async copy isn't worth the extra staging copy.
            self.tensor_cache[cache_key] = EncodingResult(
                array=torch.from_numpy(sequence_array).to(device),
                peptide_lengths=torch.from_numpy(peptide_lengths).to(device),
                unsupported_mask=(
                    torch.from_numpy(numpy.asarray(
                        encoded.unsupported_mask, dtype=bool,
                    )).to(device)
                    if encoded.unsupported_mask is not None
                    else None
                ))
        return self.tensor_cache[cache_key]

    @staticmethod
    def _fixed_length_index_encoding(
            df,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            allow_unsupported_amino_acids=False,
            throw=True):
        """
        Encode variable-length sequences to a fixed-size matrix.

        Helper function. Users should use `categorical_encode`.

        Parameters
        ----------
        df : pandas.DataFrame
        peptide_max_length : int
        n_flank_length : int
        c_flank_length : int
        allow_unsupported_amino_acids : bool
        throw : bool

        Returns a tuple ``(index_array, peptide_lengths, error_df)``.
        """
        df = df.reset_index(drop=True).copy()
        error_df = df.loc[
            (df.peptide.str.len() > peptide_max_length) |
            (df.peptide.str.len() < 1)
        ]
        if len(error_df) > 0:
            message = (
                "Sequence '%s' (length %d) unsupported. %d total "
                "peptide(s) have unsupported lengths." % (
                    error_df.iloc[0].peptide,
                    len(error_df.iloc[0].peptide),
                    len(error_df)))
            if throw:
                raise EncodingError(
                    message,
                    supported_peptide_lengths=(1, peptide_max_length))
            logging.warning(message)

            # Replace invalid peptides with X's. The encoding will be set to
            # NaNs for these peptides farther below.
            df.loc[error_df.index, "peptide"] = "X" * peptide_max_length

        if n_flank_length > 0:
            n_flanks = df.n_flank.str.pad(
                n_flank_length,
                side="left",
                fillchar="X").str.slice(-n_flank_length).str.upper()
        else:
            n_flanks = pandas.Series([""] * len(df), dtype=str)

        c_flanks = df.c_flank.str.pad(
            c_flank_length,
            side="right",
            fillchar="X").str.slice(0, c_flank_length).str.upper()
        peptides = df.peptide.str.upper()

        concatenated = n_flanks + peptides + c_flanks

        encoder = EncodableSequences.create(concatenated.values)
        array = encoder.variable_length_to_fixed_length_categorical(
            alignment_method="right_pad",
            max_length=n_flank_length + peptide_max_length + c_flank_length,
            allow_unsupported_amino_acids=allow_unsupported_amino_acids)

        return (array, peptides.str.len().values, error_df)

    @staticmethod
    def encode_indices(
            df,
            peptide_max_length,
            n_flank_length,
            c_flank_length,
            allow_unsupported_amino_acids=False,
            throw=True):
        """
        Encode variable-length sequences to fixed-size amino-acid indices.
        """
        array, peptide_lengths, error_df = FlankingEncoding._fixed_length_index_encoding(
            df=df,
            peptide_max_length=peptide_max_length,
            n_flank_length=n_flank_length,
            c_flank_length=c_flank_length,
            allow_unsupported_amino_acids=allow_unsupported_amino_acids,
            throw=throw)
        unsupported_mask = numpy.zeros(len(array), dtype=bool)
        if len(error_df) > 0:
            unsupported_mask[error_df.index] = True
        return EncodingResult(
            array.astype("int8", copy=False),
            peptide_lengths=peptide_lengths,
            unsupported_mask=unsupported_mask)

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

"""Parity and edge-case tests for fast byte-lookup FlankingEncoding (#290)."""

import numpy
import pandas
import pytest

from mhcflurry import amino_acid
from mhcflurry.encodable_sequences import EncodableSequences, EncodingError
from mhcflurry.flanking_encoding import FlankingEncoding


def _reference_encode_indices(
        df,
        peptide_max_length,
        n_flank_length,
        c_flank_length,
        allow_unsupported_amino_acids=False,
        throw=True):
    """Exact pre-#290 pandas + EncodableSequences reference implementation."""
    df = df.reset_index(drop=True)
    peptide_lengths = df.peptide.str.len()
    error_df = df.loc[
        (peptide_lengths > peptide_max_length) | (peptide_lengths < 1)
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
        df = df.copy()
        df.loc[error_df.index, "peptide"] = "X" * peptide_max_length
        peptide_lengths = df.peptide.str.len()

    n_flanks = df.n_flank.str.pad(
        n_flank_length, side="left", fillchar="X"
    ).str.slice(
        -n_flank_length if n_flank_length > 0 else len(df.n_flank)
    )
    if n_flank_length == 0:
        n_flanks = pandas.Series([""] * len(df), index=df.index)
    c_flanks = df.c_flank.str.pad(
        c_flank_length, side="right", fillchar="X"
    ).str.slice(0, c_flank_length)

    concatenated = (n_flanks + df.peptide + c_flanks).str.upper()
    encodable = EncodableSequences.create(concatenated.values)
    index_array = encodable.variable_length_to_fixed_length_categorical(
        alignment_method="right_pad",
        max_length=n_flank_length + peptide_max_length + c_flank_length,
        allow_unsupported_amino_acids=allow_unsupported_amino_acids,
    )
    unsupported_mask = numpy.zeros(len(index_array), dtype=bool)
    if len(error_df) > 0:
        unsupported_mask[error_df.index] = True
    return (
        index_array.astype("int8", copy=False),
        peptide_lengths.values,
        unsupported_mask,
    )


@pytest.mark.parametrize("n_flank_length,c_flank_length", [
    (0, 0),
    (5, 5),
    (15, 15),
    (8, 3),
    (0, 10),
    (10, 0),
])
@pytest.mark.parametrize("allow_unsupported", [True, False])
def test_flanking_encoding_matches_reference_exact(
        n_flank_length, c_flank_length, allow_unsupported):
    peptides = [
        "SIINFEKL",
        "siinfekl",
        "A",
        "ACDEFGHIKLMNPQR",
        "NLVPMVATV",
        "GILGFVFTL",
        "QCV",
        "X" * 8,
    ]
    n_flanks = [
        "",
        "QWERTYIPSDFGHKLMA",
        "m",
        "acdef",
        None,
        "ACDEFGHIKLMNPQRSTVWY",
        "QWE",
        "XX",
    ]
    c_flanks = [
        "",
        "FGHKLCVNMQWERTYIP",
        "k",
        "ghikl",
        float("nan"),
        "ACDEFGHIKLMNPQRSTVWY",
        "MNV",
        "",
    ]
    if allow_unsupported:
        peptides.extend(["SIINBZOU", "siin*j-?"])
        n_flanks.extend(["BZUO123", "ac*de"])
        c_flanks.extend(["-*?BZUO", "fg!hi"])

    encoding = FlankingEncoding(peptides, n_flanks, c_flanks)
    actual = encoding.categorical_encode(
        peptide_max_length=15,
        n_flank_length=n_flank_length,
        c_flank_length=c_flank_length,
        allow_unsupported_amino_acids=allow_unsupported,
        throw=True,
    )
    ref_array, ref_lengths, ref_mask = _reference_encode_indices(
        encoding.dataframe,
        peptide_max_length=15,
        n_flank_length=n_flank_length,
        c_flank_length=c_flank_length,
        allow_unsupported_amino_acids=allow_unsupported,
        throw=True,
    )
    assert actual.array.dtype == numpy.int8
    numpy.testing.assert_array_equal(actual.array, ref_array)
    numpy.testing.assert_array_equal(actual.peptide_lengths, ref_lengths)
    numpy.testing.assert_array_equal(actual.unsupported_mask, ref_mask)

    tensors = encoding.categorical_encode_tensors(
        peptide_max_length=15,
        n_flank_length=n_flank_length,
        c_flank_length=c_flank_length,
        device="cpu",
        allow_unsupported_amino_acids=allow_unsupported,
        throw=True,
    )
    numpy.testing.assert_array_equal(tensors.array.numpy(), ref_array)
    numpy.testing.assert_array_equal(tensors.peptide_lengths.numpy(), ref_lengths)
    numpy.testing.assert_array_equal(tensors.unsupported_mask.numpy(), ref_mask)


def test_flanking_encoding_empty_input():
    encoding = FlankingEncoding([], [], [])
    result = encoding.categorical_encode(
        peptide_max_length=15,
        n_flank_length=15,
        c_flank_length=15,
    )
    assert result.array.shape == (0, 45)
    assert result.array.dtype == numpy.int8
    assert result.peptide_lengths.shape == (0,)
    assert result.unsupported_mask.shape == (0,)


def test_flanking_encoding_invalid_lengths_throw_and_no_throw():
    peptides = ["SIINFEKL", "", "A" * 16, "NLVPMVATV"]
    n_flanks = ["AAA", "CCC", "DDD", "EEE"]
    c_flanks = ["GGG", "HHH", "III", "KKK"]
    encoding = FlankingEncoding(peptides, n_flanks, c_flanks)

    with pytest.raises(EncodingError) as exc_info:
        encoding.categorical_encode(
            peptide_max_length=15, n_flank_length=5, c_flank_length=5, throw=True)
    assert exc_info.value.supported_peptide_lengths == (1, 15)

    actual = encoding.categorical_encode(
        peptide_max_length=15, n_flank_length=5, c_flank_length=5, throw=False)
    ref_array, ref_lengths, ref_mask = _reference_encode_indices(
        encoding.dataframe,
        peptide_max_length=15,
        n_flank_length=5,
        c_flank_length=5,
        allow_unsupported_amino_acids=True,
        throw=False,
    )
    numpy.testing.assert_array_equal(actual.array, ref_array)
    numpy.testing.assert_array_equal(actual.peptide_lengths, ref_lengths)
    numpy.testing.assert_array_equal(actual.unsupported_mask, ref_mask)
    assert actual.unsupported_mask.tolist() == [False, True, True, False]


def test_flanking_encoding_strict_rejects_unsupported_characters():
    encoding = FlankingEncoding(["SIINbEKL"], ["AAAAA"], ["CCCCC"])
    with pytest.raises(KeyError) as exc_info:
        encoding.categorical_encode(
            peptide_max_length=15,
            n_flank_length=5,
            c_flank_length=5,
            allow_unsupported_amino_acids=False,
        )
    assert exc_info.value.args == ("B",)


def test_flanking_encoding_non_ascii_fallback():
    encoding = FlankingEncoding(["SIINFEKL", "SIIN\u00e9EKL"], ["AA", "CC"], ["GG", "TT"])
    actual = encoding.categorical_encode(
        peptide_max_length=15,
        n_flank_length=5,
        c_flank_length=5,
        allow_unsupported_amino_acids=True,
    )
    x_idx = amino_acid.AMINO_ACID_INDEX["X"]
    assert int(actual.array[1, 5 + 4]) == x_idx

    with pytest.raises(KeyError):
        encoding.categorical_encode(
            peptide_max_length=15,
            n_flank_length=5,
            c_flank_length=5,
            allow_unsupported_amino_acids=False,
        )

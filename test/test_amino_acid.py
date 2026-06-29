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

"""Tests for amino acid encoding."""

from mhcflurry import amino_acid
from numpy.testing import assert_equal
import numpy
import pandas
import warnings

letter_to_index_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
}


def test_index_and_one_hot_encoding():
    letter_to_vector_df = pandas.DataFrame(
        [
            [1, 0, 0,],
            [0, 1, 0,],
            [0, 0, 1,]
        ], columns=[0, 1, 2]
    )

    index_encoding = amino_acid.index_encoding(
        ["AAAA", "ABCA"], letter_to_index_dict)
    assert_equal(
        index_encoding,
        [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
        ])
    one_hot = amino_acid.fixed_vectors_encoding(
        index_encoding,
        letter_to_vector_df)
    assert one_hot.shape == (2, 4, 3)
    assert_equal(
        one_hot[0],
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])
    assert_equal(
        one_hot[1],
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ])


def test_fixed_vectors_encoding_allows_non_square_vector_tables():
    letter_to_vector_df = pandas.DataFrame(
        [
            [1, 10],
            [2, 20],
            [3, 30],
        ],
        columns=["small", "large"],
    )
    index_encoding = amino_acid.index_encoding(
        ["ABCA"], letter_to_index_dict)
    encoded = amino_acid.fixed_vectors_encoding(
        index_encoding,
        letter_to_vector_df)
    assert encoded.shape == (1, 4, 2)
    assert_equal(encoded[0], [[1, 10], [2, 20], [3, 30], [1, 10]])


def test_physchem_and_composite_vector_encodings():
    physchem = amino_acid.get_vector_encoding_df("physchem")
    assert physchem.shape == (21, 14)
    assert physchem.loc["X"].tolist() == [0.0] * 14
    assert list(physchem.columns[:4]) == [
        "z_kd_hydropathy",
        "z_grantham_composition",
        "z_grantham_polarity",
        "z_grantham_volume",
    ]
    assert physchem.loc["D", "side_chain_charge"] == -1.0
    assert physchem.loc["K", "side_chain_charge"] == 1.0
    assert physchem.loc["Y", "aromatic"] == 1.0
    assert physchem.loc["S", "hydroxyl"] == 1.0

    atchley = amino_acid.get_vector_encoding_df("atchley")
    assert atchley.shape == (21, 5)
    assert list(atchley.columns) != list(physchem.columns)

    composite = amino_acid.get_vector_encoding_df("BLOSUM62+physchem")
    assert composite.shape == (21, 35)
    assert amino_acid.vector_encoding_length("BLOSUM62+physchem") == 35
    assert_equal(
        composite.loc["A"].values[:21],
        amino_acid.get_vector_encoding_df("BLOSUM62").loc["A"].values,
    )
    assert_equal(
        composite.loc["A"].values[21:],
        physchem.loc["A"].values,
    )


def test_substitution_matrix_encodings_extend_unknown():
    pmbec = amino_acid.get_vector_encoding_df("PMBEC")
    assert pmbec.shape == (21, 21)
    assert_equal(pmbec.loc["X"].values, [0.0] * 21)
    assert_equal(pmbec["X"].values, [0.0] * 21)
    assert_equal(pmbec.values, pmbec.values.T)
    assert pmbec.loc["A", "A"] == numpy.float32(0.322860152036)
    assert pmbec.loc["E", "R"] == numpy.float32(-0.0697402405064)
    assert amino_acid.get_vector_encoding_df("pmbec") is pmbec

    contact = amino_acid.get_vector_encoding_df("contact")
    assert contact.shape == (21, 21)
    assert_equal(contact.loc["X"].values, [0.0] * 21)
    assert_equal(contact["X"].values, [0.0] * 21)
    assert_equal(contact.values, contact.values.T)
    assert contact.loc["A", "A"] == numpy.float32(-0.06711)
    assert contact.loc["V", "Y"] == numpy.float32(0.03319)
    assert amino_acid.get_vector_encoding_df("simons1999-contact") is contact
    assert amino_acid.get_vector_encoding_df("SIMK990103") is contact

    composite = amino_acid.get_vector_encoding_df("PMBEC+contact")
    assert composite.shape == (21, 42)
    assert amino_acid.vector_encoding_length("PMBEC+contact") == 42

    pmbec_minmax = amino_acid.get_vector_encoding_df("PMBEC:minmax")
    common = list(amino_acid.COMMON_AMINO_ACIDS)
    pmbec_minmax_common = pmbec_minmax.loc[common, common].values
    assert pmbec_minmax.shape == pmbec.shape
    assert pmbec_minmax_common.min() == numpy.float32(-1.0)
    assert pmbec_minmax_common.max() == numpy.float32(1.0)
    assert_equal(pmbec_minmax.loc["X"].values, [0.0] * 21)
    assert_equal(pmbec_minmax["X"].values, [0.0] * 21)

    normalized_composite = amino_acid.get_vector_encoding_df(
        "PMBEC:minmax+contact:minmax")
    assert normalized_composite.shape == (21, 42)
    assert amino_acid.vector_encoding_length(
        "PMBEC:minmax+contact:minmax") == 42


def test_index_encoding_no_downcast_futurewarning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        index_encoding = amino_acid.index_encoding(
            ["AAAA", "ABCA"], letter_to_index_dict)
    assert index_encoding.dtype.kind in ("i", "u")


def test_peptide_index_round_trip_preserves_string():
    """Single-source-of-truth round-trip: peptide → indices → peptide."""
    for peptide in ["SIINFEKL", "ACDEFGHIKLMNPQRSTVWY", "XAXAAA", ""]:
        indices = amino_acid.peptide_to_indices(peptide)
        assert indices.dtype == numpy.dtype("int8")
        recovered = amino_acid.indices_to_peptide(indices)
        assert recovered == peptide.upper()


def test_blosum62_reference_values():
    """Pin canonical NCBI BLOSUM62 entries so silent matrix drift is caught."""
    m = amino_acid.BLOSUM62_MATRIX
    expected_diagonal = {
        "A": 4, "R": 5, "N": 6, "D": 6, "C": 9, "Q": 5, "E": 5, "G": 6,
        "H": 8, "I": 4, "L": 4, "K": 5, "M": 5, "F": 6, "P": 7, "S": 4,
        "T": 5, "W": 11, "Y": 7, "V": 4,
    }
    for aa, value in expected_diagonal.items():
        assert m.loc[aa, aa] == value, (aa, m.loc[aa, aa])
    # A few well-known off-diagonals.
    assert m.loc["W", "Y"] == 2
    assert m.loc["F", "Y"] == 3
    assert m.loc["I", "L"] == 2
    assert m.loc["K", "R"] == 2
    # Local X convention (self-similarity 1, all others 0).
    assert m.loc["X", "X"] == 1
    assert (m.loc["X"].drop("X") == 0).all()
    # Symmetry.
    assert (m.values == m.values.T).all()


def test_atchley_reference_rows():
    """Pin Atchley et al. (2005, PNAS) factor-1..5 values for spot rows."""
    a = amino_acid.ATCHLEY_FACTORS
    references = {
        "A": [-0.591, -1.302, -0.733, 1.570, -0.146],
        "R": [1.538, -0.055, 1.502, 0.440, 2.897],
        "W": [-0.595, 0.009, 0.672, -2.128, -0.184],
        "V": [-1.337, -0.279, -0.544, 1.242, -1.262],
    }
    for aa, ref in references.items():
        numpy.testing.assert_allclose(a.loc[aa].values, ref, atol=1e-6)
    # X is the neutral row.
    assert (a.loc["X"].values == 0.0).all()


def test_vector_encoding_index_table_matches_fixed_vectors_encoding():
    """int8 indices + embedding-table lookup must equal the legacy widening."""
    table = amino_acid.vector_encoding_index_table("BLOSUM62")
    assert table.dtype == numpy.float32
    indices = amino_acid.peptide_to_indices("SIINFEKL")
    assert indices.dtype == numpy.dtype("int8")
    via_embedding = table[indices.astype(numpy.int64)]
    via_legacy = amino_acid.fixed_vectors_encoding(
        indices.reshape(1, -1), amino_acid.BLOSUM62_MATRIX)[0].astype(numpy.float32)
    numpy.testing.assert_array_equal(via_embedding, via_legacy)


def test_peptide_index_alphabet_anchors():
    """Position of letters in AMINO_ACIDS is the contract device-resident
    code relies on. Pin the well-known anchors so a future reorder breaks
    the test, not silently the encoded tensor layout."""
    assert amino_acid.AMINO_ACIDS[0] == "A"
    assert amino_acid.AMINO_ACIDS[amino_acid.X_INDEX] == "X"
    assert amino_acid.X_INDEX == 20
    assert amino_acid.NUM_COMMON_AMINO_ACIDS == 20
    assert amino_acid.AMINO_ACID_INDEX["A"] == 0
    assert amino_acid.AMINO_ACID_INDEX["X"] == 20
    # Sorted alphabetically over the 20 common AAs.
    common = sorted(amino_acid.COMMON_AMINO_ACIDS)
    for i, letter in enumerate(common):
        assert amino_acid.AMINO_ACID_INDEX[letter] == i

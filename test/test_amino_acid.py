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

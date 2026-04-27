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

    simk = amino_acid.get_vector_encoding_df("SIMK990103")
    assert simk.shape == (21, 21)
    assert_equal(simk.loc["X"].values, [0.0] * 21)
    assert_equal(simk["X"].values, [0.0] * 21)
    assert_equal(simk.values, simk.values.T)
    assert simk.loc["A", "A"] == numpy.float32(-0.06711)
    assert simk.loc["V", "Y"] == numpy.float32(0.03319)

    composite = amino_acid.get_vector_encoding_df("PMBEC+SIMK990103")
    assert composite.shape == (21, 42)
    assert amino_acid.vector_encoding_length("PMBEC+SIMK990103") == 42


def test_index_encoding_no_downcast_futurewarning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        index_encoding = amino_acid.index_encoding(
            ["AAAA", "ABCA"], letter_to_index_dict)
    assert index_encoding.dtype.kind in ("i", "u")

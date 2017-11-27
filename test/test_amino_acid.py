from mhcflurry import amino_acid
from nose.tools import eq_
from numpy.testing import assert_equal
import numpy
import pandas

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
    eq_(one_hot.shape, (2, 4, 3))
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


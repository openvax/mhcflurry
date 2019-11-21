from mhcflurry import auxiliary_input
from nose.tools import eq_
from numpy.testing import assert_equal
import numpy
import pandas


def test_gene():
    alleles1 = [
        "HLA-A*02:01",
        "HLA-A*02:01",
        "HLA-B*07:02",
        "HLA-B*07:02",
        "HLA-C*03:01",
        "HLA-C*02:01",
    ]
    alleles2 = [
        "HLA-A*03:01",
        "HLA-A*20:01",
        "HLA-B*03:01",
        "HLA-C*03:01",
        "HLA-C*07:01",
        "HLA-C*02:01",
    ]

    encoder = auxiliary_input.AuxiliaryInputEncoder(
        alleles=[alleles1, alleles2])
    result = encoder.get_array(features=["gene"])
    print(result)
    assert_equal(
        result,
        [
            [
                [1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0],
            ],
            [
                [1, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0],
            ],
        ])
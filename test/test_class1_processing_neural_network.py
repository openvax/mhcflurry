import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True
import re
import numpy
from numpy import testing
numpy.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(2)

from sklearn.metrics import roc_auc_score

from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal

import pandas
import pprint

from mhcflurry.class1_processing_neural_network import Class1ProcessingNeuralNetwork
from mhcflurry.common import random_peptides
from mhcflurry.amino_acid import BLOSUM62_MATRIX
from mhcflurry.flanking_encoding import FlankingEncoding

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup


table = BLOSUM62_MATRIX.apply(
    tuple).reset_index().set_index(0).to_dict()['index']


def decode_matrix(array):
    """
    Convert BLOSSUM62-encoded sequences to amino acid strings.

    Parameters
    ----------
    array : shape (num, length, dim) where num is number of sequences,
    length is the length of the sequences, and dim is the BLOSUM62 dimensions
    (i.e. 21).

    Returns
    -------
    list of strings
    """
    (num, length, dim) = array.shape
    assert dim == 21

    results = []
    for row in array:
        item = "".join([
            table[tuple(x)] for x in row
        ])
        results.append(item)
    return results


def test_neural_network_input():
    model = Class1ProcessingNeuralNetwork(
        peptide_max_length=12,
        n_flank_length=8,
        c_flank_length=5)

    tests = [
        {
            # Input
            "peptide": "SIINFEKL",
            "n_flank": "QWERTYIPSDFG",
            "c_flank": "FGHKLCVNMQWE",

            # Expected results
            "sequence": "TYIPSDFGSIINFEKLFGHKLXXXX",
        },
        {
            # Input
            "peptide": "QCV",
            "n_flank": "QWERTYIPSDFG",
            "c_flank": "FGHKLCVNMQWE",

            # Expected results
            "sequence": "TYIPSDFGQCVFGHKLXXXXXXXXX",
        },
        {
            # Input
            "peptide": "QCVSQCVSQCVS",
            "n_flank": "QWE",
            "c_flank": "MNV",

            # Expected results
            "sequence": "XXXXXQWEQCVSQCVSQCVSMNVXX",
        },
        {
            # Input
            "peptide": "QCVSQCVSQCVS",
            "n_flank": "",
            "c_flank": "MNV",

            # Expected results
            "sequence": "XXXXXXXXQCVSQCVSQCVSMNVXX",
        },
        {
            # Input
            "peptide": "QCVSQCVSQCVS",
            "n_flank": "",
            "c_flank": "",

            # Expected results
            "sequence": "XXXXXXXXQCVSQCVSQCVSXXXXX",
        },
    ]

    for (i, d) in enumerate(tests):
        encoding = FlankingEncoding(
            peptides=[d['peptide']],
            n_flanks=[d['n_flank']],
            c_flanks=[d['c_flank']])

        results = model.network_input(encoding)
        (decoded,) = decode_matrix(results['sequence'])

        numpy.testing.assert_equal(decoded, d['sequence'])
        numpy.testing.assert_equal(results['peptide_length'], len(d['peptide']))

    # Test all at once
    df = pandas.DataFrame(tests)
    encoding = FlankingEncoding(df.peptide, df.n_flank, df.c_flank)
    results = model.network_input(encoding)
    df["decoded"] = decode_matrix(results['sequence'])
    numpy.testing.assert_array_equal(df.decoded.values, df.sequence.values)
    numpy.testing.assert_equal(
        results['peptide_length'], df.peptide.str.len().values)


def test_small():
    train_basic_network(num=10000)


def test_more():
    train_basic_network(
        num=10000,
        flanking_averages=False,
        convolutional_kernel_size=3,
        c_flank_length=0,
        n_flank_length=3,
        post_convolutional_dense_layer_sizes=[8])


def test_basic_indexing(num=10000, do_assertions=True, **hyperparameters):
    def is_hit(n_flank, c_flank, peptide):
        return peptide[0] in "SIQVL" and peptide[-1] in "YIPASD"

    def is_hit1(n_flank, c_flank, peptide):
        return peptide[0] in "SIQVL"

    def is_hit2(n_flank, c_flank, peptide):
        return peptide[-1] in "YIPASD"

    hypers = {
        "convolutional_kernel_size": 1,
        "flanking_averages": False,
    }

    train_basic_network(num=10000, is_hit=is_hit1, **hypers)
    train_basic_network(num=10000, is_hit=is_hit2, **hypers)
    train_basic_network(num=10000, is_hit=is_hit, **hypers)


def train_basic_network(num, do_assertions=True, is_hit=None, **hyperparameters):
    use_hyperparameters = {
        "max_epochs": 100,
        "peptide_max_length": 12,
        "n_flank_length": 8,
        "c_flank_length": 8,
        "convolutional_kernel_size": 3,
        "flanking_averages": True,
        "min_delta": 0.01,
    }
    use_hyperparameters.update(hyperparameters)

    df = pandas.DataFrame({
        "n_flank": random_peptides(num / 2, 10) + random_peptides(num / 2, 1),
        "c_flank": random_peptides(num, 10),
        "peptide": random_peptides(num / 2, 11) + random_peptides(num / 2, 8),
    }).sample(frac=1.0)

    if is_hit is None:
        n_cleavage_regex = "[AILQSV][SINFEKLH][MNPQYK]"

        def is_hit(n_flank, c_flank, peptide):
            if re.search(n_cleavage_regex, peptide):
                return False  # peptide is cleaved
            return bool(re.match(n_cleavage_regex, n_flank[-1:] + peptide))

    df["hit"] = [
        is_hit(row.n_flank, row.c_flank, row.peptide)
        for (_, row) in df.iterrows()
    ]

    train_df = df.sample(frac=0.9)
    test_df = df.loc[~df.index.isin(train_df.index)]

    print(
        "Generated dataset",
        len(df),
        "hits: ",
        df.hit.sum(),
        "frac:",
        df.hit.mean())

    network = Class1ProcessingNeuralNetwork(**use_hyperparameters)
    network.fit(
        sequences=FlankingEncoding(
            peptides=train_df.peptide.values,
            n_flanks=train_df.n_flank.values,
            c_flanks=train_df.c_flank.values),
        targets=train_df.hit.values,
        verbose=0)

    network.network().summary()

    for df in [train_df, test_df]:
        df["predictions"] = network.predict(
            df.peptide.values,
            df.n_flank.values,
            df.c_flank.values)

    train_auc = roc_auc_score(train_df.hit.values, train_df.predictions.values)
    test_auc = roc_auc_score(test_df.hit.values, test_df.predictions.values)

    print("Train auc", train_auc)
    print("Test auc", test_auc)

    if do_assertions:
        assert_greater(train_auc, 0.9)
        assert_greater(test_auc, 0.85)

    return network


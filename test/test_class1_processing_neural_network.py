"""
Tests for Class1ProcessingNeuralNetwork.
"""
import pytest

import re
import numpy
from sklearn.metrics import roc_auc_score
import pandas

from mhcflurry.class1_processing_neural_network import Class1ProcessingNeuralNetwork
from mhcflurry.common import random_peptides
from mhcflurry.amino_acid import BLOSUM62_MATRIX
from mhcflurry.flanking_encoding import FlankingEncoding

from mhcflurry.testing_utils import cleanup, startup

numpy.random.seed(0)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    startup()
    yield
    cleanup()


table = dict([
    (tuple(encoding), amino_acid)
    for amino_acid, encoding in BLOSUM62_MATRIX.iterrows()
])


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
    """Test that input encoding produces expected sequences."""
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
    numpy.testing.assert_array_equal(df.decoded.to_numpy(), df.sequence.to_numpy())
    numpy.testing.assert_equal(
        results['peptide_length'], df.peptide.str.len().values)


def test_small():
    """Test basic network training with small dataset."""
    train_basic_network(num=10000)


@pytest.mark.slow
def test_more():
    """Test network with different hyperparameters."""
    train_basic_network(
        num=10000,
        flanking_averages=False,
        convolutional_kernel_size=3,
        c_flank_length=0,
        n_flank_length=3,
        post_convolutional_dense_layer_sizes=[8])


@pytest.mark.slow
def test_basic_indexing(num=10000, do_assertions=True, **hyperparameters):
    """Test that basic indexing patterns are learned."""
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
    """Train a processing network and check performance."""
    use_hyperparameters = {
        "max_epochs": 100,
        "peptide_max_length": 12,
        "n_flank_length": 8,
        "c_flank_length": 8,
        "convolutional_kernel_size": 3,
        "flanking_averages": False,  # Use False for reliable convergence
        "min_delta": 0.01,
    }
    use_hyperparameters.update(hyperparameters)

    df = pandas.DataFrame({
        "n_flank": random_peptides(int(num / 2), 10) + random_peptides(int(num / 2), 1),
        "c_flank": random_peptides(num, 10),
        "peptide": random_peptides(int(num / 2), 11) + random_peptides(int(num / 2), 8),
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

    train_df = df.sample(frac=0.9).copy()
    test_df = df.loc[~df.index.isin(train_df.index)].copy()

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

    print(network.network())

    for df_subset in [train_df, test_df]:
        df_subset["predictions"] = network.predict(
            df_subset.peptide.values,
            df_subset.n_flank.values,
            df_subset.c_flank.values)

    train_auc = roc_auc_score(train_df.hit.values, train_df.predictions.values)
    test_auc = roc_auc_score(test_df.hit.values, test_df.predictions.values)

    print("Train auc", train_auc)
    print("Test auc", test_auc)

    if do_assertions:
        assert train_auc > 0.9
        assert test_auc > 0.85

    return network


def test_serialization():
    """Test that network weights can be serialized and deserialized."""
    hyperparameters = {
        "max_epochs": 10,
        "peptide_max_length": 12,
        "n_flank_length": 5,
        "c_flank_length": 5,
    }

    # Generate training data
    peptides = random_peptides(100, length=9)
    n_flanks = random_peptides(100, length=10)
    c_flanks = random_peptides(100, length=10)
    targets = numpy.random.choice([0.0, 1.0], 100)

    # Train a network
    network = Class1ProcessingNeuralNetwork(**hyperparameters)
    flanking = FlankingEncoding(peptides, n_flanks, c_flanks)
    network.fit(flanking, targets, verbose=0)

    # Get predictions before serialization
    preds_before = network.predict_encoded(flanking)

    # Serialize and deserialize
    config = network.get_config()
    weights = network.get_weights()

    network2 = Class1ProcessingNeuralNetwork.from_config(config, weights=weights)
    preds_after = network2.predict_encoded(flanking)

    # Predictions should be close (some small differences may occur due to dropout eval mode)
    numpy.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)


def test_different_peptide_lengths():
    """Test that different peptide lengths are handled correctly."""
    hyperparameters = {
        "max_epochs": 10,
        "peptide_max_length": 15,
        "n_flank_length": 5,
        "c_flank_length": 5,
    }

    # Mix of peptide lengths
    peptides = (
        random_peptides(30, length=8) +
        random_peptides(30, length=9) +
        random_peptides(30, length=10) +
        random_peptides(10, length=11)
    )
    n_flanks = random_peptides(100, length=10)
    c_flanks = random_peptides(100, length=10)
    targets = numpy.random.choice([0.0, 1.0], 100)

    network = Class1ProcessingNeuralNetwork(**hyperparameters)
    flanking = FlankingEncoding(peptides, n_flanks, c_flanks)
    network.fit(flanking, targets, verbose=0)

    predictions = network.predict_encoded(flanking)
    assert len(predictions) == len(peptides)


def test_empty_flanks():
    """Test that empty flanking sequences are handled correctly."""
    hyperparameters = {
        "max_epochs": 10,
        "peptide_max_length": 12,
        "n_flank_length": 5,
        "c_flank_length": 5,
    }

    peptides = random_peptides(50, length=9)
    n_flanks = [""] * 50
    c_flanks = [""] * 50
    targets = numpy.random.choice([0.0, 1.0], 50)

    network = Class1ProcessingNeuralNetwork(**hyperparameters)
    flanking = FlankingEncoding(peptides, n_flanks, c_flanks)
    network.fit(flanking, targets, verbose=0)

    predictions = network.predict_encoded(flanking)
    assert len(predictions) == len(peptides)
    assert numpy.isfinite(predictions).all()


def test_prediction_range():
    """Test that predictions are in the expected range [0, 1]."""
    hyperparameters = {
        "max_epochs": 20,
        "peptide_max_length": 12,
        "n_flank_length": 5,
        "c_flank_length": 5,
    }

    peptides = random_peptides(100, length=9)
    n_flanks = random_peptides(100, length=10)
    c_flanks = random_peptides(100, length=10)
    targets = numpy.random.choice([0.0, 1.0], 100)

    network = Class1ProcessingNeuralNetwork(**hyperparameters)
    flanking = FlankingEncoding(peptides, n_flanks, c_flanks)
    network.fit(flanking, targets, verbose=0)

    predictions = network.predict_encoded(flanking)

    # Predictions should be between 0 and 1 (sigmoid output)
    assert predictions.min() >= 0
    assert predictions.max() <= 1

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

N_CLEAVAGE_REGEX = "[AILQSV][SINFEKLH][MNPQYK]"
N_CLEAVAGE_FIRST = "AILQSV"
N_CLEAVAGE_SECOND = "SINFEKLH"
N_CLEAVAGE_THIRD = "MNPQYK"
INDEX_FIRST_HITS = "SIQVL"
INDEX_LAST_HITS = "YIPASD"


def _cycle(letters, offset):
    """Return a deterministic residue from ``letters``."""
    return letters[offset % len(letters)]


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


@pytest.mark.slow
@pytest.mark.integration
def test_small():
    """Test basic network training with small dataset."""
    train_basic_network(
        num=512,
        dataset_factory=make_cleavage_motif_dataset,
        max_epochs=12,
        minibatch_size=128,
        validation_split=0.0,
        dropout_rate=0.0,
        convolutional_kernel_l1_l2=[0.0, 0.0],
        learning_rate=0.01)


@pytest.mark.slow
@pytest.mark.integration
def test_more():
    """Test network with different hyperparameters."""
    train_basic_network(
        num=512,
        dataset_factory=make_cleavage_motif_dataset,
        max_epochs=12,
        minibatch_size=128,
        validation_split=0.0,
        dropout_rate=0.0,
        convolutional_kernel_l1_l2=[0.0, 0.0],
        learning_rate=0.01,
        flanking_averages=False,
        convolutional_kernel_size=3,
        c_flank_length=0,
        n_flank_length=3,
        post_convolutional_dense_layer_sizes=[8])


@pytest.mark.slow
@pytest.mark.integration
def test_basic_indexing():
    """Test that basic indexing patterns are learned."""
    def is_hit(n_flank, c_flank, peptide):
        return peptide[0] in INDEX_FIRST_HITS and peptide[-1] in INDEX_LAST_HITS

    def is_hit1(n_flank, c_flank, peptide):
        return peptide[0] in INDEX_FIRST_HITS

    def is_hit2(n_flank, c_flank, peptide):
        return peptide[-1] in INDEX_LAST_HITS

    hypers = {
        "convolutional_kernel_size": 1,
        "flanking_averages": False,
        "max_epochs": 12,
        "minibatch_size": 128,
        "validation_split": 0.0,
        "dropout_rate": 0.0,
        "convolutional_kernel_l1_l2": [0.0, 0.0],
        "learning_rate": 0.01,
    }

    train_basic_network(
        num=512,
        is_hit=is_hit1,
        dataset_factory=make_indexing_dataset_factory(first=True, last=False),
        **hypers)
    train_basic_network(
        num=512,
        is_hit=is_hit2,
        dataset_factory=make_indexing_dataset_factory(first=False, last=True),
        **hypers)
    train_basic_network(
        num=512,
        is_hit=is_hit,
        dataset_factory=make_indexing_dataset_factory(first=True, last=True),
        **hypers)


def make_cleavage_motif_dataset(num, hyperparameters):
    """Return balanced examples for the N-flank cleavage motif task."""
    peptide_length = min(9, hyperparameters["peptide_max_length"])
    n_flank_length = max(1, hyperparameters["n_flank_length"])
    c_flank_length = hyperparameters["c_flank_length"]
    rows = []

    for i in range(num):
        hit = i < num // 2
        peptide = list("G" * peptide_length)
        n_flank = list("G" * n_flank_length)

        if hit:
            n_flank[-1] = _cycle(N_CLEAVAGE_FIRST, i)
            peptide[0] = _cycle(N_CLEAVAGE_SECOND, i)
            peptide[1] = _cycle(N_CLEAVAGE_THIRD, i)
        else:
            negative_case = i % 3
            n_flank[-1] = (
                "D" if negative_case == 0 else _cycle(N_CLEAVAGE_FIRST, i)
            )
            peptide[0] = (
                "D" if negative_case == 1 else _cycle(N_CLEAVAGE_SECOND, i)
            )
            peptide[1] = (
                "D" if negative_case == 2 else _cycle(N_CLEAVAGE_THIRD, i)
            )

        rows.append({
            "n_flank": "".join(n_flank),
            "c_flank": "G" * c_flank_length,
            "peptide": "".join(peptide),
            "hit": hit,
        })

    return pandas.DataFrame(rows).sample(frac=1.0, random_state=0)


def make_indexing_dataset_factory(first, last):
    """Return a dataset factory for first/last peptide indexing tasks."""

    def make_indexing_dataset(num, hyperparameters):
        peptide_length = min(9, hyperparameters["peptide_max_length"])
        rows = []

        for i in range(num):
            hit = i < num // 2
            peptide = list("G" * peptide_length)

            if hit or first:
                peptide[0] = _cycle(INDEX_FIRST_HITS, i)
            if hit or last:
                peptide[-1] = _cycle(INDEX_LAST_HITS, i)

            if not hit:
                if first and last:
                    negative_case = i % 3
                    if negative_case in (0, 2):
                        peptide[0] = "G"
                    if negative_case in (1, 2):
                        peptide[-1] = "G"
                elif first:
                    peptide[0] = "G"
                elif last:
                    peptide[-1] = "G"

            rows.append({
                "n_flank": "G" * hyperparameters["n_flank_length"],
                "c_flank": "G" * hyperparameters["c_flank_length"],
                "peptide": "".join(peptide),
                "hit": hit,
            })

        return pandas.DataFrame(rows).sample(frac=1.0, random_state=0)

    return make_indexing_dataset


def train_basic_network(
        num, do_assertions=True, is_hit=None, dataset_factory=None,
        **hyperparameters):
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

    if dataset_factory is None:
        df = pandas.DataFrame({
            "n_flank": (
                random_peptides(int(num / 2), 10) +
                random_peptides(int(num / 2), 1)
            ),
            "c_flank": random_peptides(num, 10),
            "peptide": (
                random_peptides(int(num / 2), 11) +
                random_peptides(int(num / 2), 8)
            ),
        }).sample(frac=1.0)

        if is_hit is None:
            def is_hit(n_flank, c_flank, peptide):
                if re.search(N_CLEAVAGE_REGEX, peptide):
                    return False  # peptide is cleaved
                return bool(re.match(N_CLEAVAGE_REGEX, n_flank[-1:] + peptide))

        df["hit"] = [
            is_hit(row.n_flank, row.c_flank, row.peptide)
            for (_, row) in df.iterrows()
        ]
    else:
        df = dataset_factory(num, use_hyperparameters)
        if is_hit is not None:
            numpy.testing.assert_array_equal(
                df.hit.values,
                [
                    is_hit(row.n_flank, row.c_flank, row.peptide)
                    for (_, row) in df.iterrows()
                ])

    train_df = df.sample(frac=0.9, random_state=1).copy()
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

    if do_assertions:
        train_auc = roc_auc_score(train_df.hit.values, train_df.predictions.values)
        test_auc = roc_auc_score(test_df.hit.values, test_df.predictions.values)

        print("Train auc", train_auc)
        print("Test auc", test_auc)

        assert train_auc > 0.9
        assert test_auc > 0.85

    return network


def test_serialization():
    """Test that network weights can be serialized and deserialized."""
    hyperparameters = {
        "max_epochs": 1,
        "minibatch_size": 100000,
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
        "max_epochs": 1,
        "minibatch_size": 100000,
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
        "max_epochs": 1,
        "minibatch_size": 100000,
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
        "max_epochs": 1,
        "minibatch_size": 100000,
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

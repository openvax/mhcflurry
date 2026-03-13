"""
Training variant tests for PyTorch migration.

Tests training with different hyperparameter combinations that are valid
but not exercised by the existing test suite, plus a functional test that
trains a single network on synthetic A*02:01-motif data and verifies that
a known epitope is predicted as a strong binder.
"""
import random

import numpy as np
import pytest
import torch

from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.common import random_peptides
from mhcflurry.testing_utils import startup, cleanup


@pytest.fixture(autouse=True)
def setup_teardown():
    startup()
    yield
    cleanup()


def _seed(s=42):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def _make_model(**overrides):
    defaults = dict(
        activation="tanh",
        layer_sizes=[16],
        locally_connected_layers=[],
        peptide_dense_layer_sizes=[],
        allele_dense_layer_sizes=[],
        dropout_probability=0.0,
        batch_normalization=False,
        dense_layer_l1_regularization=0.0,
        dense_layer_l2_regularization=0.0,
        max_epochs=30,
        early_stopping=False,
        validation_split=0.0,
        minibatch_size=32,
        random_negative_rate=0.0,
        random_negative_constant=0,
    )
    defaults.update(overrides)
    return Class1NeuralNetwork(**defaults)


# ---------------------------------------------------------------------------
# Training with locally connected layers
# ---------------------------------------------------------------------------

def test_train_with_locally_connected():
    _seed(1)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        locally_connected_layers=[
            {"filters": 4, "activation": "tanh", "kernel_size": 3},
        ],
        layer_sizes=[8],
        max_epochs=10,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80
    assert preds.min() > 0


# ---------------------------------------------------------------------------
# Training with dropout
# ---------------------------------------------------------------------------

def test_train_with_dropout():
    _seed(2)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        dropout_probability=0.5,  # keep probability
        layer_sizes=[16, 8],
        max_epochs=10,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80


# ---------------------------------------------------------------------------
# Training with batch normalization
# ---------------------------------------------------------------------------

def test_train_with_batch_normalization():
    _seed(3)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        batch_normalization=True,
        activation="relu",
        layer_sizes=[16],
        max_epochs=10,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80


# ---------------------------------------------------------------------------
# Training with combined options: LC + dropout + batch norm
# ---------------------------------------------------------------------------

def test_train_lc_dropout_batchnorm():
    _seed(4)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        locally_connected_layers=[
            {"filters": 4, "activation": "tanh", "kernel_size": 3},
        ],
        dropout_probability=0.8,
        batch_normalization=True,
        layer_sizes=[16, 8],
        max_epochs=10,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80


# ---------------------------------------------------------------------------
# Training with skip-connections (DenseNet) topology
# ---------------------------------------------------------------------------

def test_train_with_skip_connections():
    _seed(5)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        topology="with-skip-connections",
        layer_sizes=[8, 8, 4],
        max_epochs=10,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80


# ---------------------------------------------------------------------------
# Training with peptide dense layers
# ---------------------------------------------------------------------------

def test_train_with_peptide_dense_layers():
    _seed(6)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        peptide_dense_layer_sizes=[16],
        layer_sizes=[8],
        max_epochs=10,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80


# ---------------------------------------------------------------------------
# Training with different optimizers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("optimizer", ["adam", "sgd", "rmsprop"])
def test_train_with_optimizer(optimizer):
    _seed(7)
    peptides = random_peptides(60, length=9)
    affinities = np.random.uniform(10, 50000, 60)

    model = _make_model(
        optimizer=optimizer,
        learning_rate=0.01,
        layer_sizes=[8],
        max_epochs=5,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 60


# ---------------------------------------------------------------------------
# Training with L2 regularization
# ---------------------------------------------------------------------------

def test_train_with_l2_regularization():
    _seed(8)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        dense_layer_l2_regularization=0.01,
        layer_sizes=[8],
        max_epochs=10,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80


# ---------------------------------------------------------------------------
# Training with L1 + L2 regularization combined
# ---------------------------------------------------------------------------

def test_train_with_l1_l2_regularization():
    _seed(9)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        dense_layer_l1_regularization=0.01,
        dense_layer_l2_regularization=0.01,
        layer_sizes=[8],
        max_epochs=10,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80


# ---------------------------------------------------------------------------
# Training with random negatives
# ---------------------------------------------------------------------------

def test_train_with_random_negatives():
    _seed(10)
    peptides = random_peptides(80, length=9)
    affinities = np.random.uniform(10, 50000, 80)

    model = _make_model(
        random_negative_rate=1.0,
        random_negative_constant=10,
        layer_sizes=[8],
        max_epochs=5,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 80


# ---------------------------------------------------------------------------
# Training with validation split + early stopping
# ---------------------------------------------------------------------------

def test_train_with_early_stopping():
    _seed(11)
    peptides = random_peptides(100, length=9)
    affinities = np.random.uniform(10, 50000, 100)

    model = _make_model(
        validation_split=0.2,
        early_stopping=True,
        patience=3,
        max_epochs=200,
        layer_sizes=[8],
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 100
    # Should have stopped early (well before 200)
    n_epochs = len(model.fit_info[-1]["loss"])
    assert n_epochs < 200


# ---------------------------------------------------------------------------
# Serialization round-trip preserves predictions after combined-option training
# ---------------------------------------------------------------------------

def test_serialization_with_lc_dropout_batchnorm():
    _seed(12)
    peptides = random_peptides(60, length=9)
    affinities = np.random.uniform(10, 50000, 60)

    model = _make_model(
        locally_connected_layers=[
            {"filters": 4, "activation": "tanh", "kernel_size": 3},
        ],
        dropout_probability=0.8,
        batch_normalization=True,
        layer_sizes=[8],
        max_epochs=5,
    )
    model.fit(peptides, affinities, verbose=0)
    preds_before = model.predict(peptides)

    config = model.get_config()
    weights = model.get_weights()
    restored = Class1NeuralNetwork.from_config(config, weights=weights)
    preds_after = restored.predict(peptides)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


# ---------------------------------------------------------------------------
# Training with mixed-length peptides
# ---------------------------------------------------------------------------

def test_train_mixed_lengths_with_lc():
    _seed(13)
    peptides = (
        random_peptides(30, length=8) +
        random_peptides(30, length=9) +
        random_peptides(20, length=10) +
        random_peptides(10, length=11)
    )
    affinities = np.random.uniform(10, 50000, 90)

    model = _make_model(
        locally_connected_layers=[
            {"filters": 4, "activation": "tanh", "kernel_size": 3},
        ],
        layer_sizes=[8],
        max_epochs=5,
    )
    model.fit(peptides, affinities, verbose=0)
    preds = model.predict(peptides)
    assert len(preds) == 90


# ---------------------------------------------------------------------------
# Functional test: learn A*02:01 motif from synthetic data
# ---------------------------------------------------------------------------

_A0201_P2 = list("LM")        # anchor at position 2
_A0201_P9 = list("LVI")       # anchor at C-terminal position
_OTHER_AA = list("ACDEFGHIKNPQRSTVWY")  # non-anchor residues


def _random_aa(choices, rng):
    return choices[rng.randint(0, len(choices) - 1)]


def _generate_a0201_binder(rng, length=9):
    """Generate a peptide with canonical A*02:01 P2+P9 motifs."""
    pep = [_random_aa(_OTHER_AA, rng) for _ in range(length)]
    pep[1] = _random_aa(_A0201_P2, rng)          # P2 anchor
    pep[length - 1] = _random_aa(_A0201_P9, rng)  # Pend anchor
    return "".join(pep)


def _generate_non_binder(rng, length=9):
    """Generate a peptide that avoids A*02:01 anchors at P2 and Pend."""
    non_p2 = [aa for aa in _OTHER_AA if aa not in _A0201_P2]
    non_p9 = [aa for aa in _OTHER_AA if aa not in _A0201_P9]
    pep = [_random_aa(_OTHER_AA, rng) for _ in range(length)]
    pep[1] = _random_aa(non_p2, rng)
    pep[length - 1] = _random_aa(non_p9, rng)
    return "".join(pep)


def test_learn_a0201_motif():
    """
    Train a single Class1NeuralNetwork on 200 synthetic peptides
    (100 binders with A*02:01-like P2/Pend motifs at 1 nM,
     100 non-binders at 50000 nM) and verify that SLLQHLIGL
    (a canonical A*02:01 epitope with P2=L, P9=L) is predicted
    as a strong binder (<= 500 nM).
    """
    rng = np.random.RandomState(314)

    binders = [_generate_a0201_binder(rng) for _ in range(100)]
    non_binders = [_generate_non_binder(rng) for _ in range(100)]

    peptides = binders + non_binders
    affinities = np.concatenate([
        np.full(100, 1.0),      # strong binders
        np.full(100, 50000.0),  # non-binders
    ])

    _seed(271)
    model = _make_model(
        locally_connected_layers=[
            {"filters": 8, "activation": "tanh", "kernel_size": 3},
        ],
        layer_sizes=[32],
        max_epochs=200,
        early_stopping=False,
        validation_split=0.0,
        learning_rate=0.001,
        optimizer="adam",
        minibatch_size=32,
        dense_layer_l1_regularization=0.0,
    )
    model.fit(peptides, affinities, verbose=0)

    # SLLQHLIGL — HLA-A*02:01 Tax epitope
    # P2 = L (canonical A*02:01 anchor), P9 = L (canonical A*02:01 anchor)
    test_pred = model.predict(["SLLQHLIGL"])[0]
    print(f"SLLQHLIGL predicted affinity: {test_pred:.1f} nM")
    assert test_pred <= 500, (
        f"SLLQHLIGL should be predicted as strong binder, got {test_pred:.1f} nM"
    )

    # A peptide with wrong anchors should be predicted as weak binder
    weak_pred = model.predict(["SAAQHQIGA"])[0]
    print(f"SAAQHQIGA predicted affinity: {weak_pred:.1f} nM")
    assert weak_pred > 1000, (
        f"Non-motif peptide should be predicted weak, got {weak_pred:.1f} nM"
    )

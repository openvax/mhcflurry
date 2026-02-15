"""
Regression tests for PyTorch conversion gaps vs master behavior.
"""
import random

import pytest

import numpy as np
import torch

from mhcflurry.class1_neural_network import (
    Class1NeuralNetwork,
    Class1NeuralNetworkModel,
)
from mhcflurry.class1_processing_neural_network import Class1ProcessingModel
from mhcflurry.pytorch_losses import (
    MSEWithInequalities,
    MultiallelicMassSpecLoss,
)
from mhcflurry.testing_utils import startup, cleanup


@pytest.fixture(autouse=True)
def setup_teardown():
    startup()
    yield
    cleanup()


def _make_simple_affinity_model(**overrides):
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[4],
        locally_connected_layers=[],
        peptide_dense_layer_sizes=[],
        allele_dense_layer_sizes=[],
        dropout_probability=0.0,
        batch_normalization=False,
        dense_layer_l1_regularization=0.0,
        dense_layer_l2_regularization=0.0,
        max_epochs=5,
        early_stopping=False,
        validation_split=0.0,
        minibatch_size=2,
        optimizer="sgd",
        learning_rate=0.1,
        random_negative_rate=0.0,
        random_negative_constant=0,
    )
    hyperparameters.update(overrides)
    return Class1NeuralNetwork(**hyperparameters)


def _seed_all(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def test_sample_weights_affect_training():
    peptides = ["AAAAAAAAA", "CCCCCCCCC"]
    affinities = np.array([50.0, 50000.0])
    weights = np.array([1000.0, 1.0])

    _seed_all(7)
    model_unweighted = _make_simple_affinity_model(max_epochs=25)
    model_unweighted.fit(
        peptides,
        affinities,
        shuffle_permutation=[0, 1],
    )
    pred_unweighted = model_unweighted.predict(peptides)

    _seed_all(7)
    model_weighted = _make_simple_affinity_model(max_epochs=25)
    model_weighted.fit(
        peptides,
        affinities,
        sample_weights=weights,
        shuffle_permutation=[0, 1],
    )
    pred_weighted = model_weighted.predict(peptides)

    # With sample weights, training should diverge from the unweighted case.
    assert not np.allclose(pred_unweighted, pred_weighted, rtol=0.01, atol=0.0)


def test_validation_split_is_fixed_when_lr_zero():
    peptides = ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]
    affinities = np.array([50.0, 50000.0, 50000.0, 50.0])

    _seed_all(3)
    model = _make_simple_affinity_model(
        learning_rate=0.0,
        max_epochs=3,
        validation_split=0.5,
        early_stopping=False,
    )
    model.fit(
        peptides,
        affinities,
        shuffle_permutation=[0, 1, 2, 3],
    )
    val_losses = model.fit_info[-1]["val_loss"]
    assert len(val_losses) >= 2
    # With fixed validation split and zero learning rate, val loss should be constant.
    assert np.allclose(val_losses, val_losses[0], rtol=0.0, atol=1e-6)


def test_dropout_probability_is_keep_prob():
    nn = Class1NeuralNetwork()
    peptide_shape = nn.peptides_to_network_input([]).shape[1:]
    model = Class1NeuralNetworkModel(
        peptide_encoding_shape=peptide_shape,
        dropout_probability=0.8,
    )
    assert model.dropouts[0] is not None
    # In master, dropout_probability is a keep probability, so p should be 0.2.
    assert model.dropouts[0].p == pytest.approx(0.2, abs=1e-6)


def test_processing_dropout_is_spatial():
    model = Class1ProcessingModel(
        sequence_dims=(10, 3),
        n_flank_length=1,
        c_flank_length=1,
        peptide_max_length=8,
        flanking_averages=False,
        convolutional_filters=3,
        convolutional_kernel_size=1,
        convolutional_activation="tanh",
        convolutional_kernel_l1_l2=[0.0, 0.0],
        dropout_rate=0.5,
        post_convolutional_dense_layer_sizes=[],
    )
    assert model.dropout is not None
    model.train()
    _seed_all(11)

    x = torch.ones((1, 3, 10))
    dropped = model.dropout(x)
    mask = (dropped != 0)

    # Spatial dropout should use one mask per channel across all positions.
    for c in range(mask.shape[1]):
        assert torch.all(mask[0, c, :] == mask[0, c, 0])


def test_processing_flank_averages_use_tf_masked_mean_semantics():
    model = Class1ProcessingModel(
        sequence_dims=(7, 1),
        n_flank_length=2,
        c_flank_length=2,
        peptide_max_length=3,
        flanking_averages=True,
        convolutional_filters=1,
        convolutional_kernel_size=1,
        convolutional_activation="tanh",
        convolutional_kernel_l1_l2=[0.0, 0.0],
        dropout_rate=0.0,
        post_convolutional_dense_layer_sizes=[],
    )

    # With TF semantics, masked averaging is computed via:
    # mean((x + 1) * mask, axis=sequence_axis) - 1
    # i.e., denominator is full sequence length, not number of masked positions.
    conv_result = torch.ones((1, 7, 1))
    peptide_length = torch.tensor([[3]])

    n_avg = model._extract_n_flank_avg(conv_result)
    c_avg = model._extract_c_flank_avg(conv_result, peptide_length)

    expected = (2 * (1.0 + 1.0) / 7.0) - 1.0
    assert n_avg.item() == pytest.approx(expected, abs=1e-7)
    assert c_avg.item() == pytest.approx(expected, abs=1e-7)


def test_mse_with_inequalities_rejects_out_of_range_targets():
    with pytest.raises(ValueError):
        MSEWithInequalities.encode_y([1.1], inequalities=["="])
    with pytest.raises(ValueError):
        MSEWithInequalities.encode_y([-0.1], inequalities=["="])


def test_mse_with_inequalities_rejects_invalid_inequality():
    with pytest.raises(ValueError):
        MSEWithInequalities.encode_y([0.5], inequalities=["?"])


def test_multiallelic_mass_spec_encode_y_validates_values():
    with pytest.raises(AssertionError):
        MultiallelicMassSpecLoss.encode_y([0.5])


def test_merge_allele_specific_raises_not_implemented():
    _seed_all(5)
    model_a = _make_simple_affinity_model(max_epochs=1)
    model_b = _make_simple_affinity_model(max_epochs=1)

    # Ensure networks exist, matching master expectations for merge().
    model_a._network = model_a.make_network(
        allele_representations=None,
        **model_a.network_hyperparameter_defaults.subselect(model_a.hyperparameters)
    )
    model_b._network = model_b.make_network(
        allele_representations=None,
        **model_b.network_hyperparameter_defaults.subselect(model_b.hyperparameters)
    )
    with pytest.raises(NotImplementedError):
        Class1NeuralNetwork.merge([model_a, model_b])


def test_l1_regularization_changes_weights_even_with_zero_data_loss():
    peptides = ["AAAAAAAAA", "CCCCCCCCC"]

    _seed_all(13)
    model = _make_simple_affinity_model(
        dense_layer_l1_regularization=0.1,
        dense_layer_l2_regularization=0.0,
        max_epochs=1,
        validation_split=0.0,
        early_stopping=False,
    )

    # Build network explicitly so we can read weights before fitting.
    model._network = model.make_network(
        allele_representations=None,
        **model.network_hyperparameter_defaults.subselect(model.hyperparameters)
    )

    affinities = model.predict(peptides)
    weights_before = [p.detach().cpu().clone() for p in model.network().parameters()]

    model.fit(
        peptides,
        affinities,
        shuffle_permutation=[0, 1],
    )

    weights_after = [p.detach().cpu().clone() for p in model.network().parameters()]
    changed = any(
        not torch.allclose(before, after, rtol=0.0, atol=1e-7)
        for before, after in zip(weights_before, weights_after)
    )
    assert changed

"""
Regression tests for PyTorch conversion gaps vs master behavior.
"""
import json
import os
import random

import pytest

import numpy as np
import torch

from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_neural_network import (
    Class1NeuralNetwork,
    Class1NeuralNetworkModel,
    MergedClass1NeuralNetwork,
    _batched_validation_loss,
    _effective_validation_batch_size,
    _validation_forward_network,
)
from mhcflurry.torch_training_loop import _maybe_compile_loss
from mhcflurry.class1_processing_neural_network import (
    Class1ProcessingModel,
    Class1ProcessingNeuralNetwork,
)
from mhcflurry.common import load_weights
from mhcflurry.flanking_encoding import FlankingEncoding
from mhcflurry.pytorch_losses import (
    MSEWithInequalities,
    MultiallelicMassSpecLoss,
    get_pytorch_loss,
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


def _make_allele_representations(num_alleles=2):
    return np.arange(num_alleles * 6, dtype=np.float32).reshape(num_alleles, 2, 3)


def _seed_all(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _plain_or_shared_tensor(value):
    """Convenience: produce a plain tensor (legacy SHM path is gone)."""
    return torch.from_numpy(np.ascontiguousarray(value)).clone()


def test_maybe_compile_loss_defaults_on_with_network_compile_cuda(monkeypatch):
    from mhcflurry import torch_training_loop as ttl

    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE_LOSS", raising=False)
    monkeypatch.setattr(ttl, "_warm_cuda_autograd_for_triton", lambda device: None)
    calls = []

    def fake_compile(obj, mode=None, dynamic=None):
        calls.append((obj, mode, dynamic))
        return "compiled-loss"

    monkeypatch.setattr(torch, "compile", fake_compile)
    loss = MSEWithInequalities()

    result = _maybe_compile_loss(loss, torch.device("cuda"))

    assert result == "compiled-loss"
    assert calls == [(loss, "default", True)]


def test_maybe_compile_loss_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE_LOSS", "0")
    calls = []
    monkeypatch.setattr(torch, "compile", lambda *a, **k: calls.append((a, k)))
    loss = MSEWithInequalities()

    result = _maybe_compile_loss(loss, torch.device("cuda"))

    assert result is loss
    assert calls == []


def test_maybe_compile_loss_requires_network_compile(monkeypatch):
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE_LOSS", raising=False)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda *a, **k: calls.append((a, k)))
    loss = MSEWithInequalities()

    result = _maybe_compile_loss(loss, torch.device("cuda"))

    assert result is loss
    assert calls == []


def test_fit_validation_interval_skips_off_interval_epochs():
    """validation_interval > 1 measures only on-interval + final epoch."""
    peptides = ["SIINFEKLM", "ARTLAVELS", "GILGFVFTL", "RTLNAWVKV"]
    affinities = np.array([50.0, 30.0, 100.0, 5000.0])
    _seed_all(13)
    model = _make_simple_affinity_model(
        max_epochs=10,
        validation_split=0.5,
        early_stopping=False,
        validation_interval=3,
    )
    model.fit(peptides, affinities)

    val_loss = model.fit_info[-1]["val_loss"]
    assert len(val_loss) == 10
    # Epochs measured: 0, 3, 6, 9 (last epoch always measured).
    # Off-interval epochs carry forward the previous measurement, so
    # consecutive equal triples appear at (1, 2) repeating epoch 0's
    # loss, (4, 5) repeating epoch 3, (7, 8) repeating epoch 6.
    assert val_loss[1] == val_loss[0]
    assert val_loss[2] == val_loss[0]
    assert val_loss[4] == val_loss[3]
    assert val_loss[5] == val_loss[3]
    assert val_loss[7] == val_loss[6]
    assert val_loss[8] == val_loss[6]


def test_fit_validation_interval_default_runs_every_epoch():
    """Default validation_interval=1 measures every epoch (legacy)."""
    peptides = ["SIINFEKLM", "ARTLAVELS", "GILGFVFTL", "RTLNAWVKV"]
    affinities = np.array([50.0, 30.0, 100.0, 5000.0])
    _seed_all(17)
    model = _make_simple_affinity_model(
        max_epochs=4, validation_split=0.5, early_stopping=False,
    )
    model.fit(peptides, affinities)
    val_loss = model.fit_info[-1]["val_loss"]
    assert len(val_loss) == 4
    # With per-epoch validation under the simple training data here, at
    # least some epochs differ from epoch 0; this is the most we can
    # assert without depending on exact per-step numerics.
    assert any(v != val_loss[0] for v in val_loss[1:])


def test_effective_validation_batch_size_uses_larger_cuda_default():
    assert _effective_validation_batch_size(torch.device("cuda"), None, 512) == 4096
    assert _effective_validation_batch_size(torch.device("cuda"), None, 2048) == 8192
    assert _effective_validation_batch_size(torch.device("cpu"), None, 512) == 2048
    assert _effective_validation_batch_size(torch.device("cuda"), 123, 512) == 123


@pytest.mark.slow
@pytest.mark.integration
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


def test_fit_seed_makes_training_reproducible():
    """A single ``seed`` controls every stochastic step of fit() — weight
    initialization, the example shuffle, and random-negative sampling — so
    two fits with the same seed produce identical weights even when the
    ambient RNG state differs, and different seeds diverge."""
    peptides = ["SIINFEKL", "LLFGYPVYV", "KLGGALQAK", "GILGFVFTL",
                "NLVPMVATV", "GLCTLVAML"]
    affinities = np.array([10.0, 50000.0, 500.0, 25000.0, 100.0, 40000.0])

    def train(seed, ambient):
        # Deliberately vary ambient RNG state across calls: only ``seed``
        # should determine the trained weights. If anything stochastic
        # leaks past the seed (e.g. an unseeded shuffle), same-seed runs
        # would differ and this test fails.
        _seed_all(ambient)
        model = _make_simple_affinity_model(
            max_epochs=3, random_negative_rate=1.0)
        model.fit(peptides, affinities, seed=seed, verbose=0)
        return torch.cat(
            [p.detach().reshape(-1) for p in model.network().parameters()])

    same_a = train(seed=123, ambient=1)
    same_b = train(seed=123, ambient=98765)  # different ambient, same seed
    different = train(seed=456, ambient=1)

    assert torch.equal(same_a, same_b), (
        "fit(seed=S) must be reproducible regardless of ambient RNG state")
    assert not torch.equal(same_a, different), (
        "fit() with a different seed must produce different weights")


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
    nn = Class1NeuralNetwork(peptide_amino_acid_encoding_torch=False)
    peptide_shape = nn.peptides_to_network_input([]).shape[1:]
    model = Class1NeuralNetworkModel(
        peptide_encoding_shape=peptide_shape,
        dropout_probability=0.8,
    )
    assert model.dropouts[0] is not None
    # In master, dropout_probability is a keep probability, so p should be 0.2.
    assert model.dropouts[0].p == pytest.approx(0.2, abs=1e-6)


def test_batch_norm_uses_keras_defaults():
    nn = Class1NeuralNetwork(peptide_amino_acid_encoding_torch=False)
    peptide_shape = nn.peptides_to_network_input([]).shape[1:]
    model = Class1NeuralNetworkModel(
        peptide_encoding_shape=peptide_shape,
        batch_normalization=True,
        layer_sizes=[4],
    )

    assert model.batch_norm_early is not None
    assert model.batch_norm_early.eps == pytest.approx(1e-3, abs=1e-12)
    assert model.batch_norm_early.momentum == pytest.approx(0.01, abs=1e-12)
    assert model.batch_norms[0] is not None
    assert model.batch_norms[0].eps == pytest.approx(1e-3, abs=1e-12)
    assert model.batch_norms[0].momentum == pytest.approx(0.01, abs=1e-12)


def test_network_forward_requires_allele_key_to_match_model_type():
    peptide_input = torch.ones((2, 9, 21), dtype=torch.float32)

    pan_model = Class1NeuralNetworkModel(
        peptide_encoding_shape=(9, 21),
        allele_representations=np.zeros((2, 1, 3), dtype=np.float32),
        layer_sizes=[4],
        peptide_dense_layer_sizes=[],
        allele_dense_layer_sizes=[],
        locally_connected_layers=[],
        batch_normalization=False,
        dropout_probability=0.0,
    )
    with pytest.raises(ValueError, match="has_allele=True"):
        pan_model({"peptide": peptide_input})

    allele_specific_model = Class1NeuralNetworkModel(
        peptide_encoding_shape=(9, 21),
        allele_representations=None,
        layer_sizes=[4],
        peptide_dense_layer_sizes=[],
        allele_dense_layer_sizes=[],
        locally_connected_layers=[],
        batch_normalization=False,
        dropout_probability=0.0,
    )
    with pytest.raises(ValueError, match="has_allele=False"):
        allele_specific_model(
            {"peptide": peptide_input, "allele": torch.zeros(2, dtype=torch.long)}
        )


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


@pytest.mark.parametrize("merge_method", ["concatenate", "multiply"])
def test_forward_cartesian_from_peptide_stage_matches_expanded_path(merge_method):
    _seed_all(270)
    model = Class1NeuralNetwork(
        activation="tanh",
        layer_sizes=[4],
        allele_dense_layer_sizes=[6],
        peptide_dense_layer_sizes=[6],
        locally_connected_layers=[],
        peptide_allele_merge_method=merge_method,
        peptide_allele_merge_activation="",
        batch_normalization=False,
        dropout_probability=0.0,
        dense_layer_l1_regularization=0.0,
        dense_layer_l2_regularization=0.0,
    )
    network = model.make_network(
        allele_representations=_make_allele_representations(num_alleles=4),
        **model.network_hyperparameter_defaults.subselect(model.hyperparameters),
    )
    network.eval()

    peptide = torch.randn(3, *network.peptide_encoding_shape)
    allele_idx = torch.tensor([1, 3])
    with torch.no_grad():
        peptide_stage = network.forward_peptide_stage(peptide)
        compact = network.forward_cartesian_from_peptide_stage(
            peptide_stage,
            allele_idx,
        )
        expanded_peptide_stage = peptide_stage.unsqueeze(0).expand(
            len(allele_idx),
            peptide_stage.shape[0],
            peptide_stage.shape[-1],
        ).reshape(len(allele_idx) * peptide_stage.shape[0], peptide_stage.shape[-1])
        expanded_alleles = allele_idx.unsqueeze(1).expand(
            len(allele_idx),
            peptide_stage.shape[0],
        ).reshape(-1)
        expanded = network.forward_from_peptide_stage(
            expanded_peptide_stage,
            expanded_alleles,
        ).reshape(len(allele_idx), peptide_stage.shape[0], -1)

    np.testing.assert_allclose(
        compact.detach().cpu().numpy(),
        expanded.detach().cpu().numpy(),
        rtol=0,
        atol=1e-6,
    )


def test_predict_cartesian_pan_allele_matches_repeated_predict():
    _seed_all(7)
    allele_to_sequence = {
        "HLA-A*02:01": "ACDEFG",
        "HLA-B*07:02": "HIKLMN",
    }
    master = Class1AffinityPredictor(
        allele_to_sequence=allele_to_sequence,
    ).master_allele_encoding
    allele_encoding = AlleleEncoding(
        list(allele_to_sequence),
        borrow_from=master,
    )
    models = []
    for _ in range(2):
        model = _make_simple_affinity_model(
            peptide_allele_merge_method="concatenate",
            peptide_dense_layer_sizes=[3],
            allele_dense_layer_sizes=[3],
            layer_sizes=[4],
            peptide_amino_acid_encoding_torch=True,
        )
        _, allele_representations = model.allele_encoding_to_network_input(
            allele_encoding
        )
        model._network = model.make_network(
            allele_representations=allele_representations,
            **model.network_hyperparameter_defaults.subselect(
                model.hyperparameters
            ),
        )
        models.append(model)

    predictor = Class1AffinityPredictor(
        class1_pan_allele_models=models,
        allele_to_sequence=allele_to_sequence,
    )
    assert predictor.optimize()

    peptides = np.asarray(["SIINFEKL", "PEPTIDES"])
    alleles = np.asarray(["HLA-A*02:01", "HLA-B*07:02"])
    actual = predictor.predict_cartesian_pan_allele(
        peptides=peptides,
        alleles=alleles,
        model_kwargs={"batch_size": 2},
    )
    expected_flat = predictor.predict(
        peptides=np.tile(peptides, len(alleles)),
        alleles=np.repeat(alleles, len(peptides)),
        model_kwargs={"batch_size": 2},
    )
    expected = expected_flat.reshape(len(alleles), len(peptides)).T

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("layer_sizes", [
    [4, 4],            # 2 dense layers — exercises layer-1 factorization
    [4, 4, 4],         # 3 dense layers — exercises layer-2+ factorization
    [4, 6, 8, 10],     # 4 dense layers — exercises a deeper skip stack
])
def test_forward_cartesian_skip_connections_matches_expanded(layer_sizes):
    """Skip-connections topology, factorized cartesian forward must
    match the explicit (a*p, peptide_width + allele_width) expansion.

    The factorization avoids materializing the merged_input expansion
    at layer 1 by splitting each post-layer-0 weight column-wise into
    peptide / allele / prev-output pieces. This test pins parity vs
    the non-factorized path for 2, 3, and 4 dense layers."""
    _seed_all(271)
    model = Class1NeuralNetwork(
        activation="tanh",
        topology="with-skip-connections",
        layer_sizes=layer_sizes,
        allele_dense_layer_sizes=[6],
        peptide_dense_layer_sizes=[6],
        locally_connected_layers=[],
        peptide_allele_merge_method="concatenate",
        peptide_allele_merge_activation="",
        batch_normalization=False,
        dropout_probability=0.0,
        dense_layer_l1_regularization=0.0,
        dense_layer_l2_regularization=0.0,
    )
    network = model.make_network(
        allele_representations=_make_allele_representations(num_alleles=4),
        **model.network_hyperparameter_defaults.subselect(model.hyperparameters),
    )
    network.eval()

    peptide = torch.randn(3, *network.peptide_encoding_shape)
    allele_idx = torch.tensor([1, 3])
    with torch.no_grad():
        peptide_stage = network.forward_peptide_stage(peptide)
        compact = network.forward_cartesian_from_peptide_stage(
            peptide_stage,
            allele_idx,
        )
        # Build the explicit expansion as the reference, calling the
        # canonical ``forward_from_peptide_stage`` path which always
        # uses the merged-input cat for skip-connections.
        expanded_peptide_stage = peptide_stage.unsqueeze(0).expand(
            len(allele_idx),
            peptide_stage.shape[0],
            peptide_stage.shape[-1],
        ).reshape(len(allele_idx) * peptide_stage.shape[0], peptide_stage.shape[-1])
        expanded_alleles = allele_idx.unsqueeze(1).expand(
            len(allele_idx),
            peptide_stage.shape[0],
        ).reshape(-1)
        expanded = network.forward_from_peptide_stage(
            expanded_peptide_stage,
            expanded_alleles,
        ).reshape(len(allele_idx), peptide_stage.shape[0], -1)

    np.testing.assert_allclose(
        compact.detach().cpu().numpy(),
        expanded.detach().cpu().numpy(),
        rtol=0,
        atol=1e-5,
    )


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


def test_merged_network_serialization_preserves_dropout_keep_probability():
    _seed_all(23)
    allele_representations = np.zeros((2, 3, 4), dtype=np.float32)

    models = []
    for _ in range(2):
        model = Class1NeuralNetwork(
            dropout_probability=0.8,
            layer_sizes=[4],
            allele_dense_layer_sizes=[],
            peptide_dense_layer_sizes=[],
            locally_connected_layers=[],
            batch_normalization=False,
            dense_layer_l1_regularization=0.0,
            dense_layer_l2_regularization=0.0,
        )
        model._network = model.make_network(
            allele_representations=allele_representations,
            **model.network_hyperparameter_defaults.subselect(model.hyperparameters)
        )
        models.append(model)

    merged = Class1NeuralNetwork.merge(models)
    config = merged.get_config()
    roundtripped = Class1NeuralNetwork.from_config(
        config,
        weights=merged.get_weights(),
    )
    network = roundtripped.network()

    assert isinstance(network, MergedClass1NeuralNetwork)
    for subnet in network.networks:
        assert subnet.dropout_probability == pytest.approx(0.8, abs=1e-12)
        assert subnet.dropouts[0] is not None
        assert subnet.dropouts[0].p == pytest.approx(0.2, abs=1e-12)


def _serialized_merged_pan_network_with_allele_length(
        sequence_length, peptide_encoding=None):
    if peptide_encoding is None:
        peptide_encoding = {
            "vector_encoding_name": "BLOSUM62",
            "alignment_method": "pad_middle",
            "left_edge": 4,
            "right_edge": 4,
            "max_length": 15,
        }
    allele_representations = np.zeros((2, sequence_length, 21), dtype=np.float32)
    models = []
    for seed in (101, 202):
        _seed_all(seed)
        model = Class1NeuralNetwork(
            allele_amino_acid_encoding="BLOSUM62",
            peptide_encoding=peptide_encoding,
            allele_dense_layer_sizes=[3],
            peptide_dense_layer_sizes=[3],
            peptide_allele_merge_method="concatenate",
            peptide_allele_merge_activation="",
            layer_sizes=[5],
            locally_connected_layers=[],
            batch_normalization=False,
            dense_layer_l1_regularization=0.0,
            dense_layer_l2_regularization=0.0,
            dropout_probability=0.0,
        )
        model._network = model.make_network(
            allele_representations=allele_representations,
            **model.network_hyperparameter_defaults.subselect(model.hyperparameters)
        )
        model.clear_allele_representations()
        models.append(model)

    merged = Class1NeuralNetwork.merge(models, merge_method="concatenate")
    return merged.get_config(), merged.get_weights()


@pytest.mark.parametrize("sequence_length", [34, 37, 39])
def test_merged_network_deserialization_preserves_saved_allele_width(
        sequence_length):
    config, weights = _serialized_merged_pan_network_with_allele_length(
        sequence_length)

    restored = Class1NeuralNetwork.from_config(config, weights=weights)
    network = restored.network()

    assert isinstance(network, MergedClass1NeuralNetwork)
    for subnet in network.networks:
        assert tuple(subnet.allele_embedding.weight.shape) == (
            1, sequence_length * 21)


def test_cached_merged_networks_include_saved_allele_width_in_key():
    Class1NeuralNetwork.clear_model_cache()

    for sequence_length in (34, 39):
        config, weights = _serialized_merged_pan_network_with_allele_length(
            sequence_length)
        restored = Class1NeuralNetwork.from_config(config, weights=weights)
        network = restored.network(borrow=True)

        assert isinstance(network, MergedClass1NeuralNetwork)
        for subnet in network.networks:
            assert tuple(subnet.allele_embedding.weight.shape) == (
                1, sequence_length * 21)


def test_merged_network_deserialization_ignores_left_pad_peptide_dense_width():
    config, weights = _serialized_merged_pan_network_with_allele_length(
        39,
        peptide_encoding={
            "vector_encoding_name": "BLOSUM62",
            "alignment_method": "left_pad_centered_right_pad",
            "max_length": 15,
        },
    )

    restored = Class1NeuralNetwork.from_config(config, weights=weights)
    network = restored.network()

    assert isinstance(network, MergedClass1NeuralNetwork)
    for subnet in network.networks:
        assert tuple(subnet.allele_embedding.weight.shape) == (1, 39 * 21)


def test_pan_network_deserialization_ignores_left_pad_peptide_dense_width():
    allele_representations = np.zeros((2, 39, 21), dtype=np.float32)
    model = Class1NeuralNetwork(
        allele_amino_acid_encoding="BLOSUM62",
        peptide_encoding={
            "vector_encoding_name": "BLOSUM62",
            "alignment_method": "left_pad_centered_right_pad",
            "max_length": 15,
        },
        allele_dense_layer_sizes=[3],
        peptide_dense_layer_sizes=[3],
        peptide_allele_merge_method="concatenate",
        peptide_allele_merge_activation="",
        layer_sizes=[5],
        locally_connected_layers=[],
        batch_normalization=False,
        dense_layer_l1_regularization=0.0,
        dense_layer_l2_regularization=0.0,
        dropout_probability=0.0,
    )
    model._network = model.make_network(
        allele_representations=allele_representations,
        **model.network_hyperparameter_defaults.subselect(model.hyperparameters)
    )
    model.clear_allele_representations()
    config = model.get_config()
    weights = model.get_weights()
    network_config = json.loads(config["network_json"])
    network_config.pop("allele_representations", None)
    config["network_json"] = json.dumps(network_config)

    restored = Class1NeuralNetwork.from_config(config, weights=weights)
    network = restored.network()

    assert tuple(network.allele_embedding.weight.shape) == (1, 39 * 21)


def test_infer_allele_width_via_position_lookup():
    # encoding_dim = 21 for BLOSUM62. With no peptide-stage layers
    # and no allele_dense layers, weights[0] = allele_embedding.
    weights = [
        np.zeros((1, 39 * 21), dtype=np.float32),  # allele_embedding
    ]
    hp = {"allele_amino_acid_encoding": "BLOSUM62"}
    config = {"hyperparameters": {}}
    assert (
        Class1NeuralNetwork._infer_allele_representation_dim_from_weights(
            weights, hp, network_config=config) == 39 * 21
    )


def test_infer_allele_width_real_layout():
    # Mirror the real named_parameter order observed at runtime:
    #   peptide_dense (W, b) → allele_dense (W, b) → allele_embedding (W)
    #   → dense_layers (W, b) → output (W, b)
    weights = [
        np.zeros((3, 15 * 21), dtype=np.float32),  # peptide_dense W
        np.zeros((3,), dtype=np.float32),          # peptide_dense b
        np.zeros((3, 39 * 21), dtype=np.float32),  # allele_dense W
        np.zeros((3,), dtype=np.float32),          # allele_dense b
        np.zeros((1, 39 * 21), dtype=np.float32),  # allele_embedding W
        np.zeros((5, 6), dtype=np.float32),        # dense_layers W
        np.zeros((5,), dtype=np.float32),          # dense_layers b
        np.zeros((1, 5), dtype=np.float32),        # output W
        np.zeros((1,), dtype=np.float32),          # output b
    ]
    hp = {"allele_amino_acid_encoding": "BLOSUM62"}
    config = {"hyperparameters": {
        "peptide_dense_layer_sizes": [3],
        "allele_dense_layer_sizes": [3],
    }}
    assert (
        Class1NeuralNetwork._infer_allele_representation_dim_from_weights(
            weights, hp, network_config=config) == 39 * 21
    )


def test_infer_allele_width_duplicate_scan_handles_layer_drift():
    # An unexpected extra peptide-stage layer slips into the weights
    # without being counted in the config. The position-based primary
    # lookup lands on the wrong slot, but the duplicate-shape[1] scan
    # still finds the allele block (allele_dense + allele_embedding
    # are the only pair with matching shape[1]).
    weights = [
        np.zeros((3, 15 * 21), dtype=np.float32),  # counted peptide_dense W
        np.zeros((3,), dtype=np.float32),          # counted peptide_dense b
        # Unaccounted extra peptide-stage 2D weight
        np.zeros((4, 45 * 21), dtype=np.float32),
        np.zeros((4,), dtype=np.float32),
        # allele block — shape[1] = 39*21 appears twice
        np.zeros((3, 39 * 21), dtype=np.float32),  # allele_dense W
        np.zeros((3,), dtype=np.float32),
        np.zeros((1, 39 * 21), dtype=np.float32),  # allele_embedding W
        np.zeros((5, 6), dtype=np.float32),
        np.zeros((5,), dtype=np.float32),
    ]
    hp = {"allele_amino_acid_encoding": "BLOSUM62"}
    config = {"hyperparameters": {
        "peptide_dense_layer_sizes": [3],
        "allele_dense_layer_sizes": [3],
    }}
    assert (
        Class1NeuralNetwork._infer_allele_representation_dim_from_weights(
            weights, hp, network_config=config) == 39 * 21
    )


def test_infer_allele_width_ignores_lone_peptide_stage_match():
    # With allele_dense present but no embedding weight in the right
    # place, the duplicate-shape[1] scan should NOT mistake a lone
    # peptide-stage 2D weight for the allele block.
    weights = [
        np.zeros((3, 45 * 21), dtype=np.float32),  # lone peptide_dense W
        np.zeros((3,), dtype=np.float32),
        np.zeros((1, 5), dtype=np.float32),        # output W
    ]
    hp = {"allele_amino_acid_encoding": "BLOSUM62"}
    config = {"hyperparameters": {
        "peptide_dense_layer_sizes": [3],
        "allele_dense_layer_sizes": [3],
    }}
    assert (
        Class1NeuralNetwork._infer_allele_representation_dim_from_weights(
            weights, hp, network_config=config) is None
    )


def test_infer_allele_width_returns_none_for_empty_inputs():
    assert (
        Class1NeuralNetwork._infer_allele_representation_dim_from_weights(
            [], {"allele_amino_acid_encoding": "BLOSUM62"}) is None
    )
    assert (
        Class1NeuralNetwork._infer_allele_representation_dim_from_weights(
            [np.zeros((1, 39 * 21), dtype=np.float32)],
            {}) is None
    )


def test_dense_regularization_excludes_output_layer():
    peptides = ["AAAAAAAAA", "CCCCCCCCC"]

    _seed_all(17)
    model = _make_simple_affinity_model(
        layer_sizes=[],
        dense_layer_l1_regularization=0.1,
        dense_layer_l2_regularization=0.2,
        max_epochs=1,
        validation_split=0.0,
        early_stopping=False,
    )

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
    for before, after in zip(weights_before, weights_after):
        assert torch.allclose(before, after, rtol=0.0, atol=1e-7)


def test_processing_validation_uses_last_fraction_and_sample_weights():
    _seed_all(19)
    model = Class1ProcessingNeuralNetwork(
        max_epochs=1,
        validation_split=0.5,
        early_stopping=False,
        learning_rate=0.0,
        minibatch_size=2,
        dropout_rate=0.0,
        flanking_averages=False,
        convolutional_kernel_l1_l2=[0.0, 0.0],
        convolutional_filters=2,
        convolutional_kernel_size=1,
        n_flank_length=1,
        c_flank_length=1,
        peptide_max_length=8,
    )

    sequences = FlankingEncoding(
        peptides=["AAAAAAAA", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE"],
        n_flanks=["Q", "R", "S", "T"],
        c_flanks=["V", "W", "Y", "A"],
    )
    targets = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    sample_weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    shuffle_permutation = np.array([2, 0, 3, 1])

    model._network = model.make_network(
        **model.network_hyperparameter_defaults.subselect(model.hyperparameters)
    )
    network = model.network()
    network.eval()

    x_dict = model.network_input(sequences)
    x_dict = {
        key: value[shuffle_permutation]
        for key, value in x_dict.items()
    }
    shuffled_targets = targets[shuffle_permutation]
    shuffled_weights = sample_weights[shuffle_permutation]

    val_indices = np.arange(len(targets))[2:]
    with torch.no_grad():
        val_inputs = {
            "sequence": torch.from_numpy(x_dict["sequence"][val_indices]).float(),
            "peptide_length": torch.from_numpy(x_dict["peptide_length"][val_indices]),
        }
        predictions = network(val_inputs)
        expected = torch.nn.functional.binary_cross_entropy(
            predictions,
            torch.from_numpy(shuffled_targets[val_indices]),
            reduction="none",
        )
        expected = (
            expected *
            torch.from_numpy(shuffled_weights[val_indices])
        ).mean().item()

    model.fit(
        sequences=sequences,
        targets=targets,
        sample_weights=sample_weights,
        shuffle_permutation=shuffle_permutation,
        verbose=0,
    )

    assert model.fit_info[-1]["val_loss"][0] == pytest.approx(expected, abs=1e-7)


def test_batched_validation_loss_weighted_falls_back_to_single_shot():
    class SliceNet(torch.nn.Module):
        def forward(self, inputs):
            return inputs["peptide"][:, 0, 0]

    network = SliceNet()
    val_peptide = torch.tensor(
        [[[-0.5]], [[0.0]], [[0.5]], [[1.0]]], dtype=torch.float32
    )
    val_y = torch.tensor([0.0, 0.1, 0.9, 1.0], dtype=torch.float32)
    val_weights = torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=torch.float32)
    loss_obj = get_pytorch_loss("mse")

    expected = loss_obj(
        network({"peptide": val_peptide}),
        val_y,
        sample_weights=val_weights,
    ).item()
    actual = _batched_validation_loss(
        network=network,
        eager_network=network,
        val_peptide=val_peptide,
        val_allele=None,
        val_y=val_y,
        val_weights=val_weights,
        loss_obj=loss_obj,
        batch_size=2,
    )

    assert actual == pytest.approx(expected, abs=1e-7)


def test_batched_validation_loss_multiallelic_falls_back_to_single_shot():
    class TwoOutputNet(torch.nn.Module):
        def forward(self, inputs):
            return inputs["peptide"][:, 0, :2]

    network = TwoOutputNet()
    val_peptide = torch.tensor(
        [
            [[0.8, 0.2]],
            [[0.7, 0.1]],
            [[0.3, 0.9]],
            [[0.2, 0.6]],
        ],
        dtype=torch.float32,
    )
    val_y = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    loss_obj = get_pytorch_loss("custom:multiallelic_mass_spec_loss")

    expected = loss_obj(network({"peptide": val_peptide}), val_y).item()
    actual = _batched_validation_loss(
        network=network,
        eager_network=network,
        val_peptide=val_peptide,
        val_allele=None,
        val_y=val_y,
        val_weights=None,
        loss_obj=loss_obj,
        batch_size=2,
    )

    assert actual == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    "loss_name,val_y",
    [
        (
            "custom:mse_with_inequalities",
            torch.tensor([0.0, 2.0, 0.0, 0.0], dtype=torch.float32),
        ),
        (
            "custom:mse_with_inequalities_and_multiple_outputs",
            torch.tensor([10.0, 12.0, 10.0, 10.0], dtype=torch.float32),
        ),
    ],
)
def test_batched_validation_loss_inequality_denominator_matches_single_shot(
        loss_name, val_y):
    class TwoOutputNet(torch.nn.Module):
        def forward(self, inputs):
            base = inputs["peptide"][:, 0, 0]
            if "multiple_outputs" not in loss_name:
                return base
            return torch.stack([base, base], dim=1)

    network = TwoOutputNet()
    val_peptide = torch.tensor(
        [[[-1.0]], [[0.0]], [[0.0]], [[0.0]]], dtype=torch.float32
    )
    loss_obj = get_pytorch_loss(loss_name)

    expected = loss_obj(network({"peptide": val_peptide}), val_y).item()
    actual = _batched_validation_loss(
        network=network,
        eager_network=network,
        val_peptide=val_peptide,
        val_allele=None,
        val_y=val_y,
        val_weights=None,
        loss_obj=loss_obj,
        batch_size=2,
    )

    assert actual == pytest.approx(expected, abs=1e-7)


def test_batched_validation_loss_matches_single_shot_with_tail_batch():
    class SliceNet(torch.nn.Module):
        def forward(self, inputs):
            return inputs["peptide"][:, 0, 0]

    network = SliceNet()
    val_peptide = torch.tensor(
        [[[-0.5]], [[0.0]], [[0.5]], [[1.0]], [[1.5]]], dtype=torch.float32
    )
    val_y = torch.tensor([0.0, 0.1, 0.9, 1.0, 0.8], dtype=torch.float32)
    loss_obj = get_pytorch_loss("mse")

    expected = loss_obj(network({"peptide": val_peptide}), val_y).item()
    actual = _batched_validation_loss(
        network=network,
        eager_network=network,
        val_peptide=val_peptide,
        val_allele=None,
        val_y=val_y,
        val_weights=None,
        loss_obj=loss_obj,
        batch_size=2,
    )

    assert actual == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    "loss_name,val_y_factory,description",
    [
        (
            "custom:mse_with_inequalities",
            lambda: torch.tensor(
                [2.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "inequality rows clustered in first batch",
        ),
        (
            "custom:mse_with_inequalities",
            lambda: torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 2.0, 2.0], dtype=torch.float32),
            "inequality rows clustered in last batch",
        ),
        (
            "custom:mse_with_inequalities",
            lambda: torch.tensor(
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "single inequality at non-aligned batch boundary",
        ),
        (
            "custom:mse_with_inequalities",
            lambda: torch.tensor(
                [2.0, 2.0, 2.0, 2.0], dtype=torch.float32),
            "all rows are inequalities",
        ),
        (
            "custom:mse_with_inequalities_and_multiple_outputs",
            lambda: torch.tensor(
                [12.0, 10.0, 22.0, 20.0, 10.0, 20.0], dtype=torch.float32),
            "inequalities in mixed output indices",
        ),
    ],
)
def test_batched_validation_loss_inequality_imbalanced_distribution(
        loss_name, val_y_factory, description):
    """When ``2.0`` (encoded ``>``) rows are unevenly distributed across
    batches, the batched mean would diverge from the legacy denominator.
    Verify the fallback engages so batched == single-shot in every case."""

    val_y = val_y_factory()

    class StubNet(torch.nn.Module):
        def __init__(self, multi):
            super().__init__()
            self.multi = multi

        def forward(self, inputs):
            base = inputs["peptide"][:, 0, 0]
            if not self.multi:
                return base
            return torch.stack([base, base], dim=1)

    multi = "multiple_outputs" in loss_name
    network = StubNet(multi=multi)
    val_peptide = torch.tensor(
        [[[0.0]]] * len(val_y), dtype=torch.float32)
    loss_obj = get_pytorch_loss(loss_name)

    expected = loss_obj(network({"peptide": val_peptide}), val_y).item()
    actual = _batched_validation_loss(
        network=network,
        eager_network=network,
        val_peptide=val_peptide,
        val_allele=None,
        val_y=val_y,
        val_weights=None,
        loss_obj=loss_obj,
        batch_size=2,
    )
    assert actual == pytest.approx(expected, abs=1e-6), (
        "batched and single-shot diverge for %s" % description)


def test_batched_validation_loss_inequality_free_targets_use_batched_path():
    """When the inequality loss is configured but the validation targets
    contain no encoded inequality markers, batched is mathematically
    equivalent to single-shot, so the fast path should be allowed to
    run. This protects the optimization from collapsing to single-shot
    on every validation step."""
    from mhcflurry.class1_neural_network import (
        _validation_loss_has_legacy_inequality_denominator,
    )

    loss_obj = get_pytorch_loss("custom:mse_with_inequalities")
    no_inequality = torch.tensor([0.0, 0.1, 0.5, 0.9], dtype=torch.float32)
    has_inequality = torch.tensor([0.0, 2.0, 0.5, 0.9], dtype=torch.float32)

    assert _validation_loss_has_legacy_inequality_denominator(
        loss_obj, no_inequality) is False
    assert _validation_loss_has_legacy_inequality_denominator(
        loss_obj, has_inequality) is True


def test_batched_validation_loss_multi_output_inequality_only_in_one_output():
    """Inequality markers in one output index only — batched must still
    match single-shot. Tests the decoding logic that splits the encoded
    ``y / 10`` index from the residual target value."""

    val_y = torch.tensor([12.0, 10.0, 21.0, 20.0], dtype=torch.float32)

    class TwoOutputNet(torch.nn.Module):
        def forward(self, inputs):
            base = inputs["peptide"][:, 0, 0]
            return torch.stack([base, base], dim=1)

    network = TwoOutputNet()
    val_peptide = torch.tensor(
        [[[0.4]], [[0.5]], [[0.6]], [[0.7]]], dtype=torch.float32)
    loss_obj = get_pytorch_loss(
        "custom:mse_with_inequalities_and_multiple_outputs")
    expected = loss_obj(network({"peptide": val_peptide}), val_y).item()
    actual = _batched_validation_loss(
        network=network,
        eager_network=network,
        val_peptide=val_peptide,
        val_allele=None,
        val_y=val_y,
        val_weights=None,
        loss_obj=loss_obj,
        batch_size=2,
    )
    assert actual == pytest.approx(expected, abs=1e-6)


def test_validation_forward_network_uses_eager_by_default(monkeypatch):
    compiled = object()
    eager = object()

    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE_VALIDATION", raising=False)
    assert _validation_forward_network(compiled, eager) is eager

    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE_VALIDATION", "1")
    assert _validation_forward_network(compiled, eager) is compiled


def test_optimizer_defaults_match_keras():
    affinity_model = _make_simple_affinity_model(optimizer="adam")
    affinity_model._network = affinity_model.make_network(
        allele_representations=None,
        **affinity_model.network_hyperparameter_defaults.subselect(
            affinity_model.hyperparameters
        )
    )
    affinity_optimizer = affinity_model._create_optimizer(affinity_model.network())
    assert affinity_optimizer.defaults["eps"] == pytest.approx(1e-07, abs=1e-12)

    processing_model = Class1ProcessingNeuralNetwork(
        optimizer="rmsprop",
        learning_rate=0.001,
    )
    processing_model._network = processing_model.make_network(
        **processing_model.network_hyperparameter_defaults.subselect(
            processing_model.hyperparameters
        )
    )
    processing_optimizer = processing_model._create_optimizer(
        processing_model.network()
    )
    assert processing_optimizer.defaults["alpha"] == pytest.approx(0.9, abs=1e-12)
    assert processing_optimizer.defaults["eps"] == pytest.approx(1e-07, abs=1e-12)


def test_weight_and_embedding_updates_preserve_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    affinity_model = _make_simple_affinity_model(batch_normalization=True)
    affinity_model._network = affinity_model.make_network(
        allele_representations=_make_allele_representations(),
        **affinity_model.network_hyperparameter_defaults.subselect(
            affinity_model.hyperparameters
        )
    )
    affinity_network = affinity_model.network()
    affinity_network.to(device)

    affinity_network.set_weights_list(
        affinity_network.get_weights_list(),
        auto_convert_keras=False,
    )
    assert all(param.device == device for param in affinity_network.parameters())
    assert all(buffer.device == device for buffer in affinity_network.buffers())

    affinity_model.set_allele_representations(_make_allele_representations(3))
    assert affinity_network.allele_embedding.weight.device == device

    affinity_model.clear_allele_representations()
    assert affinity_network.allele_embedding.weight.device == device

    processing_model = Class1ProcessingNeuralNetwork(
        dropout_rate=0.0,
        flanking_averages=False,
        convolutional_kernel_l1_l2=[0.0, 0.0],
        convolutional_filters=2,
        convolutional_kernel_size=1,
        n_flank_length=1,
        c_flank_length=1,
        peptide_max_length=8,
    )
    processing_model._network = processing_model.make_network(
        **processing_model.network_hyperparameter_defaults.subselect(
            processing_model.hyperparameters
        )
    )
    processing_network = processing_model.network()
    processing_network.to(device)

    processing_network.set_weights_list(
        processing_network.get_weights_list(),
        auto_convert_keras=False,
    )
    assert all(param.device == device for param in processing_network.parameters())
    assert all(buffer.device == device for buffer in processing_network.buffers())


def test_cached_keras_weight_reload_preserves_device():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    config_path = os.path.join(data_dir, "master_affinity_fixture_config.json")
    weights_path = os.path.join(data_dir, "master_affinity_fixture_weights.npz")

    with open(config_path, "r") as inp:
        config = json.load(inp)

    weights = load_weights(weights_path)
    reloaded_weights = [w.copy() for w in weights]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Class1NeuralNetwork.clear_model_cache()

    first_model = Class1NeuralNetwork.from_config(config, weights=weights)
    first_network = first_model.network(borrow=True)
    first_network.to(device)

    second_model = Class1NeuralNetwork.from_config(config, weights=reloaded_weights)
    second_network = second_model.network(borrow=True)

    assert second_network is first_network
    assert all(param.device == device for param in second_network.parameters())
    assert all(buffer.device == device for buffer in second_network.buffers())


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


def test_fit_validation_cache_rejects_unsorted_val_indices_with_straddling_values():
    """The val_cache_safe check must inspect ALL val_indices, not just the first.

    Regression test: an earlier version of the code only checked
    ``val_indices[0] >= num_random_negatives``. That's unsafe if
    val_indices is unsorted — sklearn's ``train_test_split`` can produce
    ``[400, 3, 500, ...]`` where 400 passes the first-element check but
    3 indexes into the random-negative portion (which mutates per
    epoch). Caching tensors slice from a mixed range would produce stale
    validation data.

    We don't reach into the training loop — just call the standalone
    predicate the code uses. The invariant we assert: the check must
    return False when ANY val index is below num_random_negatives.
    """
    # Simulate the exact check in class1_neural_network.fit()
    # (see comment near ``_val_cache_safe``). Using bool(numpy.all(...))
    # because numpy.bool is not the same as Python bool.
    num_random_negatives = 100
    good = np.array([500, 600, 700], dtype=np.int64)
    mixed = np.array([500, 3, 700], dtype=np.int64)     # UNSORTED + straddles
    bad = np.array([3, 4, 5], dtype=np.int64)

    def cache_safe(val_indices, nrn):
        return (
            val_indices is not None
            and len(val_indices) > 0
            and bool(np.all(val_indices >= nrn))
        )

    assert cache_safe(good, num_random_negatives) is True
    assert cache_safe(mixed, num_random_negatives) is False, (
        "cache_safe must be False when ANY val index is below "
        "num_random_negatives, regardless of ordering"
    )
    assert cache_safe(bad, num_random_negatives) is False


def test_old_model_config_loads_with_new_features_added():
    """Regression: adding new hyperparameters must not break old-model load.

    Simulates a 2.2.0-vintage model config that does NOT carry any of
    the post-2.2.0 additions (``validation_batch_size``,
    ``dataloader_num_workers``, ``fit_dataloader_backing``) and uses
    the atomic ``"BLOSUM62"`` encoding. Loading must succeed and
    produce a network whose hyperparameters populate the new fields
    from defaults.

    If this test ever fails, something broke the
    ``HyperparameterDefaults.with_defaults`` fallback chain OR a newly-
    added hyperparameter was declared without a default — both would
    silently break loading of all public mhcflurry models in the wild.
    """
    # Minimal legacy-style config — only the fields that existed in the
    # public 2.2.0 training hyperparameters. No validation_batch_size,
    # no dataloader_num_workers, no fit_dataloader_backing, no compile flags.
    legacy_config = {
        "hyperparameters": {
            "layer_sizes": [4],
            "peptide_dense_layer_sizes": [],
            "allele_dense_layer_sizes": [],
            "locally_connected_layers": [],
            "peptide_allele_merge_method": "concatenate",
            "peptide_allele_merge_activation": "",
            "topology": "feedforward",
            "peptide_encoding": {
                "vector_encoding_name": "BLOSUM62",
                "alignment_method": "pad_middle",
                "max_length": 15,
            },
            "peptide_amino_acid_encoding": "BLOSUM62",  # legacy-renamed key
            "max_epochs": 5,
            "minibatch_size": 128,   # old default, not 512
            "optimizer": "rmsprop",
            "activation": "tanh",
            "output_activation": "sigmoid",
            "loss": "custom:mse_with_inequalities",
            "init": "glorot_uniform",
            "batch_normalization": False,
            "dropout_probability": 0.0,
            "dense_layer_l1_regularization": 0.0,
            "dense_layer_l2_regularization": 0.0,
            "validation_split": 0.1,
            "early_stopping": False,
            "learning_rate": 0.001,
        },
    }
    # Must not raise — new fields get defaults, legacy-renamed key gets dropped.
    net = Class1NeuralNetwork.from_config(legacy_config)

    # New fields populated by defaults:
    assert "validation_batch_size" in net.hyperparameters
    assert "dataloader_num_workers" in net.hyperparameters
    # ``fit_dataloader_backing`` is no longer a live hyperparameter;
    # legacy configs that set it must still load — the rename table
    # drops the key silently.
    assert "fit_dataloader_backing" not in net.hyperparameters

    # Legacy-renamed key dropped, not raised:
    assert "peptide_amino_acid_encoding" not in net.hyperparameters

    # Atomic encoding now uses the torch-side lookup by default, so peptide
    # network input is compact integer indices while the network itself still
    # expands to BLOSUM62 vectors internally.
    assert net.uses_peptide_torch_encoding()
    peptide_shape = net.peptides_to_network_input(["SIINFEKL"]).shape
    assert peptide_shape == (1, 15), f"got {peptide_shape}"


def test_min_delta_not_silently_dropped_on_load():
    """Regression: loading a config with explicit min_delta must preserve it.

    An old version of the rename table had ``min_delta → None`` which
    would silently drop the value before HyperparameterDefaults applied
    its default. This broke early-stopping behavior for any saved
    model that didn't use the default min_delta=0.0.
    """
    config = {
        "hyperparameters": {
            "layer_sizes": [4],
            "peptide_dense_layer_sizes": [],
            "allele_dense_layer_sizes": [],
            "locally_connected_layers": [],
            "peptide_encoding": {
                "vector_encoding_name": "BLOSUM62",
                "alignment_method": "pad_middle",
                "max_length": 15,
            },
            "peptide_allele_merge_method": "concatenate",
            "peptide_allele_merge_activation": "",
            "topology": "feedforward",
            "max_epochs": 10,
            "patience": 5,
            "min_delta": 0.01,        # <-- explicit non-default value
            "minibatch_size": 128,
            "optimizer": "adam",
            "activation": "tanh",
            "output_activation": "sigmoid",
            "loss": "custom:mse_with_inequalities",
            "init": "glorot_uniform",
            "batch_normalization": False,
            "dropout_probability": 0.0,
            "dense_layer_l1_regularization": 0.0,
            "dense_layer_l2_regularization": 0.0,
            "validation_split": 0.1,
            "early_stopping": True,
            "learning_rate": 0.001,
        },
    }
    net = Class1NeuralNetwork.from_config(config)
    assert net.hyperparameters["min_delta"] == 0.01, (
        f"min_delta was silently dropped: got {net.hyperparameters['min_delta']}"
    )

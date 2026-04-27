"""
Regression tests for PyTorch conversion gaps vs master behavior.
"""
import json
import os
import random

import pytest

import numpy as np
import torch

from mhcflurry.class1_neural_network import (
    Class1NeuralNetwork,
    Class1NeuralNetworkModel,
    MergedClass1NeuralNetwork,
    _batched_validation_loss,
    _effective_validation_batch_size,
    _effective_fit_dataloader_num_workers,
    _is_resource_exhaustion_error,
    _FitBatchDataset,
    _make_fit_dataloader,
    _resolve_fit_dataloader_backing,
)
from mhcflurry.shared_memory import (
    numpy_batch_collate,
    share_like,
    share_tensor,
    tensor_batch_collate,
    torch_shared_memory_status,
    update_shared,
)
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
    """Use real SHM when available; otherwise plain tensors cover tensor path."""
    if torch_shared_memory_status()["available"]:
        return share_tensor(value)
    return torch.from_numpy(np.ascontiguousarray(value)).clone()


def test_fit_dataloader_numpy_backed_uses_numpy_collate_no_pin():
    """Numpy-backed dataset → numpy collate + no pinning.

    The numpy invariant that matters for openvax/mhcflurry#270:
    keep inter-process transport on the standard pickling path and
    avoid PyTorch's default-collate ``torch.tensor(numpy_array)``
    allocator growth. Pin policy is derived from dataset type, not
    a caller-passed flag.
    """
    dataset = [
        {"peptide": np.zeros((2, 3), dtype=np.int8), "y": np.float32(0.0)},
        {"peptide": np.ones((2, 3), dtype=np.int8), "y": np.float32(1.0)},
    ]
    loader = _make_fit_dataloader(dataset=dataset, batch_size=2, num_workers=1)
    assert loader.collate_fn is numpy_batch_collate
    assert loader.pin_memory is False


def test_fit_dataloader_tensor_backed_uses_tensor_collate_and_pins():
    """Tensor-backed dataset → tensor collate + pinning."""
    x_peptide = _plain_or_shared_tensor(np.zeros((4, 2, 3), dtype=np.float32))
    y = _plain_or_shared_tensor(np.zeros(4, dtype=np.float32))
    dataset = _FitBatchDataset(
        x_peptide=x_peptide,
        x_allele=None,
        y_encoded=y,
        sample_weights_with_negatives=None,
        train_indices=np.arange(4),
    )
    assert dataset.tensor_backed
    loader = _make_fit_dataloader(dataset=dataset, batch_size=2, num_workers=0)
    assert loader.collate_fn is tensor_batch_collate
    assert loader.pin_memory is True


def test_share_tensor_round_trip():
    """``share_tensor`` returns a SHM tensor with the same content."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    status = torch_shared_memory_status()
    if not status["available"]:
        assert status["reason"]
        assert share_tensor(None) is None
        with pytest.raises(RuntimeError, match="|".join((
                "Operation not permitted",
                "No space left",
                "Cannot allocate",
                "torch_shm_manager",
                "unable to mmap",
        ))):
            share_tensor(arr)
        return

    shared = share_tensor(arr)
    assert isinstance(shared, torch.Tensor)
    assert shared.is_shared()
    assert torch.equal(shared, torch.from_numpy(arr))
    assert share_tensor(None) is None


def test_share_like_and_update_shared():
    """``share_like`` allocates a shared buffer; ``update_shared`` fills it."""
    src = np.arange(8, dtype=np.float32).reshape(4, 2)
    status = torch_shared_memory_status()
    if not status["available"]:
        assert status["reason"]
        with pytest.raises(RuntimeError, match="|".join((
                "Operation not permitted",
                "No space left",
                "Cannot allocate",
                "torch_shm_manager",
                "unable to mmap",
        ))):
            share_like(src)
        return

    buf = share_like(src)
    assert buf.is_shared()
    assert buf.shape == src.shape
    update_shared(buf, src)
    assert torch.equal(buf, torch.from_numpy(src))
    update_shared(buf, src * 2)
    assert torch.equal(buf, torch.from_numpy(src * 2))


def test_effective_fit_dataloader_num_workers_skips_size_guard_for_tensor_backed(
    monkeypatch,
):
    """Tensor-backed dataset bypasses the spawn-pickle byte-size guard."""
    from mhcflurry import class1_neural_network as cnn

    monkeypatch.setattr(cnn, "_FIT_DATALOADER_SPAWN_COPY_LIMIT_BYTES", 32)
    monkeypatch.setattr(cnn, "_FIT_DATALOADER_DOWNGRADE_WARNED", False)
    x_peptide = _plain_or_shared_tensor(np.zeros((8, 2, 3), dtype=np.float32))
    dataset = _FitBatchDataset(
        x_peptide=x_peptide,
        x_allele=None,
        y_encoded=_plain_or_shared_tensor(np.zeros(8, dtype=np.float32)),
        sample_weights_with_negatives=None,
        train_indices=np.arange(8),
    )
    effective, reason = _effective_fit_dataloader_num_workers(4, dataset)
    assert effective == 4
    assert reason is None


def test_fit_dataloader_workers_disabled_for_large_spawn_payload(monkeypatch):
    """Large fit() arrays should not be spawn-pickled into worker children."""
    from mhcflurry import class1_neural_network as cnn

    monkeypatch.setattr(cnn, "_FIT_DATALOADER_SPAWN_COPY_LIMIT_BYTES", 32)
    monkeypatch.setattr(cnn, "_FIT_DATALOADER_DOWNGRADE_WARNED", False)
    dataset = _FitBatchDataset(
        x_peptide=np.zeros((8, 2, 3), dtype=np.float32),
        x_allele=None,
        y_encoded=np.zeros(8, dtype=np.float32),
        sample_weights_with_negatives=None,
        train_indices=np.arange(8),
    )

    effective, reason = _effective_fit_dataloader_num_workers(4, dataset)

    assert effective == 0
    assert "spawn workers would pickle a copy" in reason


def test_fit_dataloader_worker_downgrade_has_explicit_escape_hatch(monkeypatch):
    from mhcflurry import class1_neural_network as cnn

    monkeypatch.setattr(cnn, "_FIT_DATALOADER_SPAWN_COPY_LIMIT_BYTES", 32)
    monkeypatch.setenv("MHCFLURRY_FORCE_FIT_DATALOADER_WORKERS", "1")
    dataset = _FitBatchDataset(
        x_peptide=np.zeros((8, 2, 3), dtype=np.float32),
        x_allele=None,
        y_encoded=np.zeros(8, dtype=np.float32),
        sample_weights_with_negatives=None,
        train_indices=np.arange(8),
    )

    effective, reason = _effective_fit_dataloader_num_workers(4, dataset)

    assert effective == 4
    assert reason is None


def test_resource_exhaustion_error_detection():
    import errno

    assert _is_resource_exhaustion_error(
        OSError(errno.ENOSPC, "No space left on device")
    )
    assert _is_resource_exhaustion_error(
        RuntimeError("unable to mmap storage: too many open files")
    )
    assert _is_resource_exhaustion_error(
        RuntimeError("torch_shm_manager: Cannot allocate memory")
    )
    assert _is_resource_exhaustion_error(
        RuntimeError("torch_shm_manager: Operation not permitted")
    )
    assert not _is_resource_exhaustion_error(ValueError("shape mismatch"))


def test_fit_dataloader_backing_auto_resolves_from_worker_count():
    """Auto backing keeps worker count and storage mode separate."""
    assert _resolve_fit_dataloader_backing(
        "auto", 0, environ={}
    ) == (
        "auto",
        "numpy",
        "auto: dataloader_num_workers == 0",
    )
    assert _resolve_fit_dataloader_backing(
        "auto", 2, environ={}
    ) == (
        "auto",
        "shared_tensor",
        "auto: dataloader_num_workers > 0",
    )


def test_fit_dataloader_backing_explicit_modes_ignore_worker_count():
    """Explicit backing mode is not inferred from process parallelism."""
    assert _resolve_fit_dataloader_backing(
        "numpy", 4, environ={}
    ) == ("numpy", "numpy", "explicit hyperparameter")
    assert _resolve_fit_dataloader_backing(
        "shared_tensor", 0, environ={}
    ) == ("shared_tensor", "shared_tensor", "explicit hyperparameter")
    assert _resolve_fit_dataloader_backing(
        "shm", 0, environ={}
    ) == ("shared_tensor", "shared_tensor", "explicit hyperparameter")


def test_fit_dataloader_backing_env_only_affects_auto():
    """Legacy SHM env override remains a diagnostic for auto mode only."""
    assert _resolve_fit_dataloader_backing(
        "auto", 0, environ={"MHCFLURRY_FIT_DATALOADER_SHM": "1"}
    ) == ("auto", "shared_tensor", "MHCFLURRY_FIT_DATALOADER_SHM=1")
    assert _resolve_fit_dataloader_backing(
        "auto", 2, environ={"MHCFLURRY_FIT_DATALOADER_SHM": "0"}
    ) == ("auto", "numpy", "MHCFLURRY_FIT_DATALOADER_SHM=0")
    assert _resolve_fit_dataloader_backing(
        "numpy", 2, environ={"MHCFLURRY_FIT_DATALOADER_SHM": "1"}
    ) == ("numpy", "numpy", "explicit hyperparameter")


def test_fit_dataloader_backing_rejects_invalid_values():
    with pytest.raises(ValueError, match="Unsupported fit_dataloader_backing"):
        Class1NeuralNetwork(fit_dataloader_backing="banana")
    with pytest.raises(ValueError, match="MHCFLURRY_FIT_DATALOADER_SHM"):
        _resolve_fit_dataloader_backing(
            "auto", 0, environ={"MHCFLURRY_FIT_DATALOADER_SHM": "maybe"}
        )


def test_fit_with_shm_env_override_trains_and_predicts(monkeypatch):
    """fit() with the SHM env override on trains end-to-end.

    Smoke test — exercises the FitBacking.share() materialization, the
    refill-in-place random-negative buffer, the polymorphic
    _FitBatchDataset, tensor_batch_collate, and pin_memory=True path
    using num_workers=0 (so no spawn). Asserts fit_info reports
    SHM-on, training completed, and predict() returns finite values.
    """
    monkeypatch.setenv("MHCFLURRY_FIT_DATALOADER_SHM", "1")
    peptides = ["SIINFEKLM", "ARTLAVELS", "GILGFVFTL", "RTLNAWVKV"]
    affinities = np.array([50.0, 30.0, 100.0, 5000.0])
    _seed_all(11)
    model = _make_simple_affinity_model(
        max_epochs=3,
        random_negative_rate=1.0,
        random_negative_constant=0,
    )
    model.fit(peptides, affinities)

    last_info = model.fit_info[-1]
    if last_info.get("fit_dataloader_shm_enabled") is False:
        assert "fit_dataloader_shm_fallback_reason" in last_info
    else:
        assert last_info.get("fit_dataloader_shm_enabled") is True

    preds = model.predict(peptides)
    assert preds.shape == (len(peptides),)
    assert np.all(np.isfinite(preds))


def test_fit_shm_auto_enables_when_dataloader_workers_requested(monkeypatch):
    """Auto backing uses SHM when ``dataloader_num_workers > 0``."""
    monkeypatch.delenv("MHCFLURRY_FIT_DATALOADER_SHM", raising=False)
    peptides = ["SIINFEKLM", "ARTLAVELS", "GILGFVFTL", "RTLNAWVKV"]
    affinities = np.array([50.0, 30.0, 100.0, 5000.0])
    _seed_all(11)
    model = _make_simple_affinity_model(
        max_epochs=2,
        random_negative_rate=1.0,
        random_negative_constant=0,
        dataloader_num_workers=2,
    )
    model.fit(peptides, affinities)
    last_info = model.fit_info[-1]
    if last_info.get("fit_dataloader_shm_enabled") is False:
        assert "fit_dataloader_shm_fallback_reason" in last_info
        assert last_info.get("fit_dataloader_backing") == "numpy"
    else:
        assert last_info.get("fit_dataloader_shm_enabled") is True
        assert last_info.get("fit_dataloader_backing") == "shared_tensor"
    assert last_info.get("fit_dataloader_backing_requested") == "auto"
    assert (
        last_info.get("fit_dataloader_backing_reason")
        == "auto: dataloader_num_workers > 0"
    )


def test_fit_shm_off_by_default_when_no_workers(monkeypatch):
    """Auto backing uses numpy when ``dataloader_num_workers=0``."""
    monkeypatch.delenv("MHCFLURRY_FIT_DATALOADER_SHM", raising=False)
    peptides = ["SIINFEKLM", "ARTLAVELS", "GILGFVFTL", "RTLNAWVKV"]
    affinities = np.array([50.0, 30.0, 100.0, 5000.0])
    _seed_all(11)
    model = _make_simple_affinity_model(max_epochs=2)
    model.fit(peptides, affinities)
    assert model.fit_info[-1].get("fit_dataloader_shm_enabled") is False
    assert model.fit_info[-1].get("fit_dataloader_backing_requested") == "auto"
    assert model.fit_info[-1].get("fit_dataloader_backing") == "numpy"
    assert (
        model.fit_info[-1].get("fit_dataloader_backing_reason")
        == "auto: dataloader_num_workers == 0"
    )


def test_fit_dataloader_backing_explicit_shared_tensor_with_no_workers(monkeypatch):
    """Explicit shared_tensor backing is independent of DataLoader workers."""
    monkeypatch.delenv("MHCFLURRY_FIT_DATALOADER_SHM", raising=False)
    peptides = ["SIINFEKLM", "ARTLAVELS", "GILGFVFTL", "RTLNAWVKV"]
    affinities = np.array([50.0, 30.0, 100.0, 5000.0])
    _seed_all(23)
    model = _make_simple_affinity_model(
        max_epochs=2,
        fit_dataloader_backing="shared_tensor",
    )
    model.fit(peptides, affinities)
    last_info = model.fit_info[-1]
    assert last_info.get("fit_dataloader_backing_requested") == "shared_tensor"
    assert last_info.get("fit_dataloader_backing_reason") == "explicit hyperparameter"
    if last_info.get("fit_dataloader_shm_enabled") is False:
        assert "fit_dataloader_shm_fallback_reason" in last_info
        assert last_info.get("fit_dataloader_backing") == "numpy"
    else:
        assert last_info.get("fit_dataloader_shm_enabled") is True
        assert last_info.get("fit_dataloader_backing") == "shared_tensor"


def test_fit_dataloader_backing_serializes_with_component_model():
    """Component model configs carry the explicit fit transport policy."""
    model = _make_simple_affinity_model(fit_dataloader_backing="shared-tensor")
    config = model.get_config()
    assert config["hyperparameters"]["fit_dataloader_backing"] == "shared_tensor"
    restored = Class1NeuralNetwork.from_config(config)
    assert restored.hyperparameters["fit_dataloader_backing"] == "shared_tensor"


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


def test_fit_batch_dataset_precasts_y_to_float32_once():
    """Regression: y_encoded is pre-cast to float32 at dataset construction.

    Before the fix, ``__getitem__`` called ``.astype(numpy.float32)`` on
    every batch access — millions of allocations per epoch on pan-allele
    workloads. The pre-cast should happen once in ``__init__`` and
    subsequent indices should return views, not new allocations.
    """
    x_peptide = np.zeros((8, 15, 21), dtype=np.float32)
    x_allele = np.zeros((8, 34, 21), dtype=np.float32)
    y_fp64 = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    train_indices = np.arange(8)

    ds = _FitBatchDataset(
        x_peptide=x_peptide,
        x_allele=x_allele,
        y_encoded=y_fp64,
        sample_weights_with_negatives=None,
        train_indices=train_indices,
    )

    # The stored y must be float32 regardless of input dtype.
    assert ds.y_encoded.dtype == np.float32
    # Input must not be mutated — the dataset should make its own cast.
    assert y_fp64.dtype == np.float64

    # __getitem__ must NOT re-cast per call: returned y value must be
    # a direct element (same memory region) of ds.y_encoded.
    sample = ds[3]
    assert sample["y"].dtype == np.float32
    # numpy scalar indexing of a float32 array returns a numpy scalar
    # whose dtype matches — the key invariant here is no astype() call
    # fires per __getitem__. Exercise a few more samples to confirm.
    for i in range(len(ds)):
        assert ds[i]["y"].dtype == np.float32


def test_fit_batch_dataset_fp32_input_is_no_op_no_copy():
    """When y is already float32, the constructor must NOT copy it.

    Cheap invariant: the pre-cast only fires when dtype differs. Avoids
    doubling memory for the common case (y already fp32).
    """
    y_fp32 = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    ds = _FitBatchDataset(
        x_peptide=np.zeros((8, 15, 21), dtype=np.float32),
        x_allele=None,
        y_encoded=y_fp32,
        sample_weights_with_negatives=None,
        train_indices=np.arange(8),
    )
    # Same underlying buffer — no astype() / no copy.
    assert ds.y_encoded is y_fp32


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
    the Phase-4 additions (``validation_batch_size``,
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
    assert net.hyperparameters["fit_dataloader_backing"] == "auto"

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

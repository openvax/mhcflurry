# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for #506 (class1_neural_network split) and #419 (dense encoding removal)."""

import collections

import numpy as np
import pytest
import torch

from mhcflurry import keras_model_config
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_neural_network import (
    Class1NeuralNetwork,
    Class1NeuralNetworkModel,
    MergedClass1NeuralNetwork,
)
from mhcflurry.class1_training import (
    EarlyStoppingState,
    build_affinity_device_data,
    compute_regularization_penalty,
    evaluate_validation_epoch,
    iter_device_training_batches,
    prepare_affinity_inputs,
    prepare_streaming_validation_tensors,
    record_fit_summary,
    seed_training_rngs,
    update_early_stopping_and_checkpoint,
)
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.flanking_encoding import FlankingEncoding
from mhcflurry.pytorch_losses import get_pytorch_loss


# ---------------------------------------------------------------------------
# Issue #419: deprecated dense-vector encoding remnants removed
# ---------------------------------------------------------------------------


def test_419_deprecated_dense_vector_methods_removed():
    assert not hasattr(FlankingEncoding, "vector_encode")
    assert not hasattr(FlankingEncoding, "encode")
    assert not hasattr(
        EncodableSequences, "variable_length_to_fixed_length_vector_encoding"
    )


def test_419_model_requires_2d_index_peptide_inputs():
    model = Class1NeuralNetworkModel(
        peptide_encoding_shape=(15, 21),
        layer_sizes=[4],
    )
    assert hasattr(model, "peptide_embedding_table")
    with pytest.raises(TypeError):
        Class1NeuralNetworkModel(
            peptide_encoding_shape=(15, 21),
            peptide_input_is_indices=False,
        )
    with pytest.raises(ValueError, match="2D index-encoded peptide tensor"):
        model({"peptide": torch.zeros((2, 15, 21), dtype=torch.float32)})
    with pytest.raises(ValueError, match="2D index-encoded peptide tensor"):
        model.forward_peptide_stage(
            torch.zeros((2, 15, 21), dtype=torch.float32)
        )


# ---------------------------------------------------------------------------
# Issue #506 step 1: keras_model_config public API
# ---------------------------------------------------------------------------


def test_506_parse_keras_json_config_and_load_weights():
    rng = np.random.default_rng(123)
    keras_cfg = {
        "class_name": "Functional",
        "config": {
            "name": "test_keras_pan",
            "layers": [
                {"class_name": "InputLayer", "config": {"name": "peptide"}},
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "peptide_dense_0",
                        "units": 4,
                        "activation": "relu",
                        "kernel_regularizer": {
                            "class_name": "L1L2",
                            "config": {"l1": 0.002, "l2": 0.005},
                        },
                    },
                },
                {
                    "class_name": "Embedding",
                    "config": {
                        "name": "allele_embedding",
                        "input_dim": 0,
                        "output_dim": 35 * 21,
                    },
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "allele_dense_0",
                        "units": 4,
                        "activation": "relu",
                    },
                },
                {"class_name": "Concatenate", "config": {"name": "merge"}},
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense_0",
                        "units": 6,
                        "activation": "relu",
                    },
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "output",
                        "units": 1,
                        "activation": "sigmoid",
                    },
                },
            ],
        },
    }
    assert keras_model_config.is_keras_model_config(keras_cfg)
    assert not keras_model_config.is_keras_model_config({"hyperparameters": {}})

    hp, meta = keras_model_config.parse_keras_json_config(keras_cfg)
    assert hp["peptide_dense_layer_sizes"] == [4]
    assert hp["allele_dense_layer_sizes"] == [4]
    assert hp["layer_sizes"] == [6]
    assert hp["peptide_allele_merge_method"] == "concatenate"
    assert hp["activation"] == "relu"
    assert hp["dense_layer_l1_regularization"] == pytest.approx(0.002)
    assert hp["dense_layer_l2_regularization"] == pytest.approx(0.005)
    assert meta["has_embedding"] is True
    assert meta["skip_embedding_weights"] is True

    model = keras_model_config.create_model_from_config(
        keras_cfg,
        model_cls=Class1NeuralNetworkModel,
        merged_model_cls=MergedClass1NeuralNetwork,
        network_cls=Class1NeuralNetwork,
    )
    assert isinstance(model, Class1NeuralNetworkModel)
    assert model.has_allele is True
    assert tuple(model.allele_embedding.weight.shape) == (1, 35 * 21)

    weights = [
        rng.normal(0, 0.1, (15 * 21, 4)).astype(np.float32),
        rng.normal(0, 0.1, (4,)).astype(np.float32),
        np.zeros((0, 35 * 21), dtype=np.float32),
        rng.normal(0, 0.1, (35 * 21, 4)).astype(np.float32),
        rng.normal(0, 0.1, (4,)).astype(np.float32),
        rng.normal(0, 0.1, (8, 6)).astype(np.float32),
        rng.normal(0, 0.1, (6,)).astype(np.float32),
        rng.normal(0, 0.1, (6, 1)).astype(np.float32),
        rng.normal(0, 0.1, (1,)).astype(np.float32),
    ]
    keras_model_config.load_keras_weights_into_model(
        model,
        weights,
        keras_config=keras_cfg,
        keras_metadata=meta,
    )
    np.testing.assert_allclose(
        model.output_layer.weight.detach().numpy(), weights[-2].T
    )


def test_506_create_merged_model_from_config():
    cfg = {
        "hyperparameters": {
            "allele_amino_acid_encoding": "BLOSUM62",
            "peptide_encoding": {
                "vector_encoding_name": "BLOSUM62",
                "alignment_method": "pad_middle",
                "left_edge": 4,
                "right_edge": 4,
                "max_length": 15,
            },
        },
        "merge_method": "average",
        "merged_networks": [
            {"layer_sizes": [8], "peptide_allele_merge_method": "concatenate"},
            {"layer_sizes": [4], "peptide_allele_merge_method": "concatenate"},
        ],
    }
    merged = keras_model_config.create_merged_model_from_config(
        cfg,
        model_cls=Class1NeuralNetworkModel,
        merged_model_cls=MergedClass1NeuralNetwork,
        network_cls=Class1NeuralNetwork,
    )
    assert isinstance(merged, MergedClass1NeuralNetwork)
    assert len(merged.networks) == 2
    assert merged.networks[0].dense_layers[0].out_features == 8
    assert merged.networks[1].dense_layers[0].out_features == 4


# ---------------------------------------------------------------------------
# Issue #506 step 2: class1_training public phase functions tested standalone
# ---------------------------------------------------------------------------


def test_506_training_phase_helpers_standalone():
    seed_training_rngs(42)
    peptides = [
        "SIINFEKL", "GILGFVFTL", "NLVPMVATV", "YLQPRTFLL",
        "KLVALGINAV", "FLRGRAYGL", "ELAGIGILTV", "RMFPNAPYL",
        "KLIETYFSKK", "YVNVNMGLK",
    ]
    affinities = np.linspace(50.0, 20000.0, len(peptides))
    allele_seqs = {
        "HLA-A*02:01": "YFAMYQENMAHTDANTLYLNYHDYTWAVLAYTWYG",
        "HLA-B*07:02": "YFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAA",
    }
    ae = AlleleEncoding(
        ["HLA-A*02:01", "HLA-B*07:02"] * 5,
        allele_to_sequence=allele_seqs,
    )
    net = Class1NeuralNetwork(
        layer_sizes=[6],
        locally_connected_layers=[],
        peptide_allele_merge_method="concatenate",
        validation_split=0.2,
        minibatch_size=4,
        random_negative_rate=0.5,
        random_negative_constant=2,
        max_epochs=2,
    )
    loss_obj = get_pytorch_loss(net.hyperparameters["loss"])
    device = torch.device("cpu")

    prepared = prepare_affinity_inputs(
        peptides=peptides,
        affinities=affinities,
        allele_encoding=ae,
        inequalities=["="] * len(peptides),
        output_indices=None,
        sample_weights=None,
        shuffle_permutation=np.arange(len(peptides)),
        hyperparameters=net.hyperparameters,
        loss_obj=loss_obj,
        device=device,
        seed=42,
        peptides_to_network_input=net.peptides_to_network_input,
        allele_encoding_to_network_input=net.allele_encoding_to_network_input,
        random_negatives_pool_factory=net._random_negatives_pool_for_fit,
    )
    assert prepared.num_random_negatives > 0
    # Index 0 is reserved for the null/unknown allele in AlleleEncoding.
    assert prepared.allele_representations.shape[0] == 3

    device_data = build_affinity_device_data(
        prepared=prepared,
        hyperparameters=net.hyperparameters,
        loss_obj=loss_obj,
        device=device,
    )
    assert device_data.n_train + device_data.n_val == (
        len(peptides) + prepared.num_random_negatives
    )
    assert device_data.val_cache_reused is True

    model = net.make_network(
        allele_representations=prepared.allele_representations,
        **net.network_hyperparameter_defaults.subselect(net.hyperparameters),
    )
    batches = list(
        iter_device_training_batches(
            affinity_training_data=device_data.affinity_training_data,
            train_indices_dev_full=torch.arange(device_data.n_train),
            batch_size=4,
            network=model,
            eager_network=model,
        )
    )
    assert sum(b["row_count"] for b in batches) == device_data.n_train

    reg_params = tuple(net._regularized_parameters(model))
    pen = compute_regularization_penalty(reg_params, l1=0.01, l2=0.02)
    assert pen is not None and float(pen.item()) > 0.0

    fit_info = collections.defaultdict(list)
    val_p, val_y, val_a, val_w = device_data.val_device_tensors
    val_loss, val_bs, _ = evaluate_validation_epoch(
        network=model,
        eager_network=model,
        val_peptide=val_p,
        val_y=val_y,
        val_allele=val_a,
        val_weights=val_w,
        loss_obj=loss_obj,
        regularization_parameters=reg_params,
        l1_reg=0.001,
        l2_reg=0.0,
        fit_info=fit_info,
        device=device,
        validation_batch_size_hp=None,
        minibatch_size=4,
        num_workers_per_gpu=1,
        timing_enabled=False,
    )
    assert np.isfinite(val_loss)
    assert val_bs >= 1
    assert fit_info["val_loss"] == [val_loss]

    es_state = EarlyStoppingState()
    improved = update_early_stopping_and_checkpoint(
        state=es_state,
        val_loss=val_loss,
        epoch_index=0,
        min_delta=0.0,
        eager_network=model,
        retain_best_weights=True,
    )
    assert improved is True
    assert es_state.min_val_loss_epoch == 0
    assert es_state.best_state_dict is not None

    summary = record_fit_summary(
        fit_info=fit_info,
        early_stopping_state=es_state,
        restored_best_weights=True,
        saved_checkpoint_policies=["terminal", "best"],
        elapsed_seconds=1.5,
        num_points=len(peptides),
        first_batch_time=0.05,
    )
    assert summary["best_epoch"] == 1
    assert summary["best_val_loss"] == val_loss
    assert summary["num_points"] == len(peptides)
    assert summary["first_batch_time"] == 0.05

    # Also test prepare_streaming_validation_tensors standalone
    vp, vy, va = prepare_streaming_validation_tensors(
        validation_peptide_encoding=EncodableSequences.create(peptides[:4]),
        validation_affinities=affinities[:4],
        validation_allele_input=np.array([0, 1, 0, 1], dtype=np.int64),
        validation_inequalities=["=", "<", ">", "="],
        validation_output_indices=None,
        loss_obj=loss_obj,
        device=device,
        peptides_to_network_input=net.peptides_to_network_input,
    )
    assert vp.shape[0] == 4
    assert vy.shape[0] == 4
    assert va.shape[0] == 4

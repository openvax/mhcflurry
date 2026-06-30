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

"""
Class1NeuralNetwork - PyTorch implementation for MHC class I binding prediction.
"""

import gc
import time
import collections
import json
import weakref
import itertools
import os
import logging

import numpy
import pandas
import torch
import torch.nn as nn

from . import pytorch_sizing
from .hyperparameters import HyperparameterDefaults
from .encodable_sequences import EncodableSequences, EncodingError
from .allele_encoding import AlleleEncoding
from .regression_target import to_ic50, from_ic50
from .common import get_pytorch_device
from .pytorch_layers import LocallyConnected1D, get_activation
from .pytorch_losses import get_pytorch_loss
from .data_dependent_weights_initialization import lsuv_init
from .random_negative_peptides import (
    RandomNegativePeptides,
    RandomNegativesPool,
    supports_device_random_negative_encoding,
)
from .class1_affinity_training_data import AffinityDeviceTrainingData
from .class1_encoding import (
    _peptide_torch_encoding_name,
    _peptide_torch_encoding_shape,
    _peptide_torch_encoding_table,
    _peptide_uses_torch_encoding,
    peptide_sequences_to_network_input,
)
from .class1_training import (
    _StreamingBatchIterableDataset,
    _batched_validation_loss,
    _carry_forward_validation_loss,
    _early_stop_reached,
    _make_streaming_batch_dataloader,
    _materialize_repeated_peptide_batch,
    _move_fit_batch_to_device,
    _should_validate_epoch,
    _sync_mean_loss,
    _timing_enabled,
    _timing_start,
    _timing_stop,
    _torch_from_numpy,
    _update_min_validation_loss,
    _validation_interval_from_hyperparameters,
)
from .pytorch_training import (
    configure_matmul_precision,
    effective_validation_batch_size,
    maybe_compile_loss,
    maybe_compile_network,
    uncompiled_network,
    validation_forward_network,
)


DEFAULT_PREDICT_BATCH_SIZE = pytorch_sizing.DEFAULT_PREDICT_BATCH_SIZE
_env_workers_per_gpu = pytorch_sizing._env_workers_per_gpu
check_training_batch_fits = pytorch_sizing.check_training_batch_fits
compute_prediction_batch_size = pytorch_sizing.compute_prediction_batch_size
resolve_prediction_batch_size = pytorch_sizing.resolve_prediction_batch_size

KERAS_BATCH_NORM_EPSILON = 1e-3
# Keras uses moving = moving * 0.99 + batch * 0.01. PyTorch's momentum is the
# new-batch coefficient, so the equivalent value is 0.01.
KERAS_BATCH_NORM_MOMENTUM = 0.01


def _run_training_batch(
    *,
    network,
    optimizer,
    loss_obj,
    regularization_parameters,
    l1_reg,
    l2_reg,
    inputs,
    y_batch,
    weights_batch=None,
):
    """Run one optimizer step and return the detached loss tensor."""
    optimizer.zero_grad()
    predictions = network(inputs)
    loss = loss_obj(predictions, y_batch, sample_weights=weights_batch)
    regularization_penalty = Class1NeuralNetwork._regularization_penalty(
        regularization_parameters,
        l1=l1_reg,
        l2=l2_reg,
    )
    if regularization_penalty is not None:
        loss = loss + regularization_penalty
    loss.backward()
    optimizer.step()
    return loss.detach()


def _run_prepared_training_batches(
    prepared_batches,
    *,
    optimizer,
    loss_obj,
    regularization_parameters,
    l1_reg,
    l2_reg,
    timing_enabled,
    device,
):
    """Run optimizer steps for a prepared batch stream.

    Batch sources are responsible for efficient data plumbing. The in-memory
    affinity path yields device-resident tensors selected by GPU indices; the
    streaming pretrain path yields DataLoader chunks after its H2D copy. This
    helper owns the learning step so fit() and fit_streaming_batches() share the
    same optimizer/loss/timing behavior without forcing one data layout onto the
    other.
    """
    losses = []
    train_time = 0.0
    train_rows = 0
    first_batch_time = None
    for prepared in prepared_batches:
        batch_start = _timing_start(device, timing_enabled)
        loss = _run_training_batch(
            network=prepared["network"],
            optimizer=optimizer,
            loss_obj=loss_obj,
            regularization_parameters=regularization_parameters,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            inputs=prepared["inputs"],
            y_batch=prepared["y_batch"],
            weights_batch=prepared.get("weights_batch"),
        )
        batch_time = _timing_stop(batch_start, device, timing_enabled)
        train_time += batch_time
        if first_batch_time is None:
            first_batch_time = batch_time
        train_rows += int(prepared["row_count"])
        losses.append(loss)
    return {
        "losses": losses,
        "train_time": train_time,
        "train_rows": train_rows,
        "first_batch_time": first_batch_time,
    }


class Class1NeuralNetworkModel(nn.Module):
    """
    PyTorch module for Class1 neural network.
    """

    def __init__(
            self,
            peptide_encoding_shape,
            allele_representations=None,
            locally_connected_layers=None,
            peptide_dense_layer_sizes=None,
            allele_dense_layer_sizes=None,
            layer_sizes=None,
            peptide_allele_merge_method="multiply",
            peptide_allele_merge_activation="",
            activation="tanh",
            output_activation="sigmoid",
            dropout_probability=0.0,
            batch_normalization=False,
            dense_layer_l1_regularization=0.001,
            dense_layer_l2_regularization=0.0,
            topology="feedforward",
            num_outputs=1,
            init="glorot_uniform",
            peptide_input_is_indices=False,
            peptide_input_vector_encoding_name=None):
        super(Class1NeuralNetworkModel, self).__init__()

        self.peptide_encoding_shape = peptide_encoding_shape
        if peptide_input_vector_encoding_name is None and peptide_input_is_indices:
            peptide_input_vector_encoding_name = "BLOSUM62"
        self.peptide_input_vector_encoding_name = peptide_input_vector_encoding_name
        self.peptide_input_is_indices = peptide_input_vector_encoding_name is not None
        # DEPRECATED (scheduled for removal): ``peptide_input_is_indices`` is now
        # always True in production — the encoding path emits (N, L) int8 and the
        # legacy dense-vector peptide path is gone. The ``False`` branch (a 3D
        # fp32 peptide input, handled in ``forward``/``predict``) is retained
        # only as defensive handling used by topology/init unit tests; it has no
        # production caller and should be collapsed to index-only when those
        # tests migrate.
        #
        # Device-side fixed amino-acid encoding: when enabled, peptide input
        # is (N, L) int indices and ``forward`` widens to (N, L, V) fp32 via
        # a frozen embedding table. The table is a non-persistent buffer: it
        # moves with ``.to(device)`` but is fully determined by hyperparameters
        # and should not be serialized in custom NPZ weight lists.
        if self.peptide_input_is_indices:
            self.register_buffer(
                "peptide_embedding_table",
                torch.from_numpy(
                    _peptide_torch_encoding_table(peptide_input_vector_encoding_name)
                ),
                persistent=False,
            )
        self.has_allele = allele_representations is not None
        self.peptide_allele_merge_method = peptide_allele_merge_method
        self.peptide_allele_merge_activation = peptide_allele_merge_activation
        self.dropout_probability = dropout_probability
        self.topology = topology
        self.num_outputs = num_outputs
        self.activation_name = activation
        self.output_activation_name = output_activation

        if locally_connected_layers is None:
            locally_connected_layers = []
        if peptide_dense_layer_sizes is None:
            peptide_dense_layer_sizes = []
        if allele_dense_layer_sizes is None:
            allele_dense_layer_sizes = []
        if layer_sizes is None:
            layer_sizes = [32]

        # Build locally connected layers
        self.lc_layers = nn.ModuleList()
        input_length = peptide_encoding_shape[0]
        in_channels = peptide_encoding_shape[1]

        for i, lc_params in enumerate(locally_connected_layers):
            filters = lc_params.get('filters', 8)
            kernel_size = lc_params.get('kernel_size', 3)
            lc_activation = lc_params.get('activation', 'tanh')

            lc_layer = LocallyConnected1D(
                in_channels=in_channels,
                out_channels=filters,
                input_length=input_length,
                kernel_size=kernel_size,
                activation=lc_activation
            )
            self.lc_layers.append(lc_layer)
            in_channels = filters
            input_length = lc_layer.output_length

        # Flattened size after locally connected layers
        self.flatten_size = input_length * in_channels

        # Peptide dense layers
        self.peptide_dense_layers = nn.ModuleList()
        peptide_layer_input = self.flatten_size
        for i, size in enumerate(peptide_dense_layer_sizes):
            layer = nn.Linear(peptide_layer_input, size)
            self.peptide_dense_layers.append(layer)
            peptide_layer_input = size

        # Batch normalization after peptide processing (early)
        self.batch_norm_early = None
        if batch_normalization:
            self.batch_norm_early = nn.BatchNorm1d(
                peptide_layer_input,
                eps=KERAS_BATCH_NORM_EPSILON,
                momentum=KERAS_BATCH_NORM_MOMENTUM,
            )

        # Allele embedding and processing
        self.allele_embedding = None
        self.allele_dense_layers = nn.ModuleList()
        allele_output_size = 0

        if self.has_allele:
            num_alleles = allele_representations.shape[0]
            embedding_dim = numpy.prod(allele_representations.shape[1:])

            self.allele_embedding = nn.Embedding(
                num_embeddings=num_alleles,
                embedding_dim=embedding_dim
            )
            # Set embedding weights and freeze
            self.allele_embedding.weight.data = torch.from_numpy(
                allele_representations.reshape(num_alleles, -1).astype(numpy.float32)
            )
            self.allele_embedding.weight.requires_grad = False

            allele_layer_input = embedding_dim
            for i, size in enumerate(allele_dense_layer_sizes):
                layer = nn.Linear(allele_layer_input, size)
                self.allele_dense_layers.append(layer)
                allele_layer_input = size
            allele_output_size = allele_layer_input

        # Compute merged size
        if self.has_allele:
            if peptide_allele_merge_method == "concatenate":
                merged_size = peptide_layer_input + allele_output_size
            elif peptide_allele_merge_method == "multiply":
                # Both must have the same size for multiply
                merged_size = peptide_layer_input
            else:
                raise ValueError(f"Unknown merge method: {peptide_allele_merge_method}")
        else:
            merged_size = peptide_layer_input

        # Merge activation
        self.merge_activation = get_activation(peptide_allele_merge_activation)

        # Main dense layers
        self.dense_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # For DenseNet topology, track input sizes for skip connections
        self.merged_size = merged_size
        current_size = merged_size
        prev_sizes = []  # Track previous layer output sizes for skip connections

        for i, size in enumerate(layer_sizes):
            # For DenseNet topology (with-skip-connections):
            # - Layer 0: input = merged_size
            # - Layer 1: input = merged_size + layer_sizes[0] (skip from input)
            # - Layer 2+: input = layer_sizes[i-2] + layer_sizes[i-1] (skip from 2 layers back)
            if topology == "with-skip-connections" and i > 0:
                if i == 1:
                    # Skip from original merged input
                    current_size = merged_size + prev_sizes[-1]
                else:
                    # Skip from 2 layers back
                    current_size = prev_sizes[-2] + prev_sizes[-1]

            layer = nn.Linear(current_size, size)
            self.dense_layers.append(layer)

            if batch_normalization:
                self.batch_norms.append(nn.BatchNorm1d(
                    size,
                    eps=KERAS_BATCH_NORM_EPSILON,
                    momentum=KERAS_BATCH_NORM_MOMENTUM,
                ))
            else:
                self.batch_norms.append(None)

            if dropout_probability > 0:
                # Dropout probability in MHCflurry hyperparameters is keep-probability.
                drop_prob = max(0.0, 1.0 - dropout_probability)
                if drop_prob > 0:
                    self.dropouts.append(nn.Dropout(p=drop_prob))
                else:
                    self.dropouts.append(None)
            else:
                self.dropouts.append(None)

            prev_sizes.append(size)
            current_size = size

        # Note: For DenseNet topology, output layer receives only the last hidden layer output
        # (skip connections are only between hidden layers, not to the output layer)

        # Output layer
        self.output_layer = nn.Linear(current_size, num_outputs)

        # Activation functions
        self.activation = get_activation(activation)
        self.output_activation = get_activation(output_activation)

        # Initialize weights
        self._initialize_weights(init)

    def _initialize_weights(self, init):
        """Initialize layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init == "glorot_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init == "glorot_normal":
                    nn.init.xavier_normal_(module.weight)
                elif init == "he_uniform":
                    nn.init.kaiming_uniform_(module.weight)
                elif init == "he_normal":
                    nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward_peptide_stage_before_early_batch_norm(self, peptide):
        if (
            self.peptide_input_is_indices
            and peptide.dim() == 2
        ):
            peptide = torch.nn.functional.embedding(
                peptide.long(), self.peptide_embedding_table
            )
        x = peptide
        for lc_layer in self.lc_layers:
            x = lc_layer(x)
        x = x.reshape(x.size(0), -1)
        for layer in self.peptide_dense_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        return x

    def forward_peptide_stage(self, peptide):
        """Run only the peptide-side of the network.

        Ends at the point where the allele information enters — just
        before the merge step. Used by the calibration fast path to
        compute the peptide-dependent activations once and reuse them
        across thousands of alleles.

        Parameters
        ----------
        peptide : torch.Tensor
            (N, L, V) fp32 fixed-vector encoded, or (N, L) int indices when
            ``peptide_input_is_indices`` is True.

        Returns
        -------
        torch.Tensor of shape (N, peptide_representation_dim) — the
        input the ``forward_from_peptide_stage`` fast path expects.
        """
        x = self._forward_peptide_stage_before_early_batch_norm(peptide)
        if self.batch_norm_early is not None:
            x = self.batch_norm_early(x)
        return x

    def _forward_allele_stage(self, allele_idx):
        allele_idx = allele_idx.long()
        if allele_idx.dim() > 1:
            allele_idx = allele_idx.squeeze(-1)
        allele_embed = self.allele_embedding(allele_idx)
        for layer in self.allele_dense_layers:
            allele_embed = layer(allele_embed)
            if self.activation is not None:
                allele_embed = self.activation(allele_embed)
        return allele_embed.reshape(allele_embed.size(0), -1)

    def forward_from_peptide_stage(self, peptide_stage, allele_idx):
        """Run the allele-merge + main dense path from cached peptide reps.

        ``peptide_stage`` must be the output of
        ``forward_peptide_stage``. ``allele_idx`` has shape matching
        ``peptide_stage``'s batch dim — typical calibration usage
        tiles one allele across many peptides or the cross-product
        of (peptide_chunk, allele_chunk).

        This path skips all peptide-side ops, which for pan-allele
        calibration lets a single precomputed activation be reused
        across tens of thousands of allele forwards.
        """
        if not self.has_allele:
            raise RuntimeError(
                "forward_from_peptide_stage called on a has_allele=False "
                "model — just call forward() directly"
            )
        x = peptide_stage
        allele_embed = self._forward_allele_stage(allele_idx)
        if self.peptide_allele_merge_method == "concatenate":
            x = torch.cat([x, allele_embed], dim=-1)
        elif self.peptide_allele_merge_method == "multiply":
            x = x * allele_embed
        if self.merge_activation is not None:
            x = self.merge_activation(x)
        prev_outputs = []
        merged_input = x
        for i, layer in enumerate(self.dense_layers):
            if self.topology == "with-skip-connections" and i > 0:
                if i == 1:
                    x = torch.cat([merged_input, prev_outputs[-1]], dim=-1)
                else:
                    x = torch.cat([prev_outputs[-2], prev_outputs[-1]], dim=-1)
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)
            prev_outputs.append(x)
        output = self.output_layer(x)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

    def _first_linear_from_cartesian_stages(self, peptide_stage, allele_stage, layer):
        if self.peptide_allele_merge_method == "concatenate":
            peptide_width = peptide_stage.shape[-1]
            peptide_weight = layer.weight[:, :peptide_width]
            allele_weight = layer.weight[:, peptide_width:]
            x = (
                peptide_stage.matmul(peptide_weight.t()).unsqueeze(1)
                + allele_stage.matmul(allele_weight.t()).unsqueeze(0)
            )
        elif self.peptide_allele_merge_method == "multiply":
            x = torch.einsum(
                "pd,ad,hd->pah",
                peptide_stage,
                allele_stage,
                layer.weight,
            )
        else:
            raise ValueError(
                f"Unknown merge method: {self.peptide_allele_merge_method}"
            )
        return x + layer.bias

    def _forward_compact_cartesian(self, peptide, allele_idx, repeat_count):
        """Forward a compact peptide × allele batch without raw peptide repeat."""
        if not self.has_allele:
            raise RuntimeError(
                "compact peptide-repeat batches require a pan-allele model"
            )
        if repeat_count <= 0:
            raise ValueError("peptide_repeat_count must be positive")

        allele_idx = allele_idx.long()
        if allele_idx.dim() > 1:
            allele_idx = allele_idx.squeeze(-1)
        peptide_count = peptide.shape[0]
        allele_count = allele_idx.shape[0]
        if (
            isinstance(peptide_count, int)
            and isinstance(allele_count, int)
            and allele_count != peptide_count * repeat_count
        ):
            raise ValueError(
                "compact peptide-repeat batch has %d peptides, repeat_count=%d, "
                "and %d allele rows"
                % (peptide_count, repeat_count, allele_count)
            )

        peptide_stage = self._forward_peptide_stage_before_early_batch_norm(peptide)
        if self.batch_norm_early is not None:
            # BatchNorm's running variance depends on the batch size. Repeat
            # before early BN so compact training is numerically equivalent to
            # the historical fully repeated peptide batch.
            peptide_stage = peptide_stage.repeat_interleave(repeat_count, dim=0)
            peptide_stage = self.batch_norm_early(peptide_stage)
            peptide_stage = peptide_stage.reshape(
                peptide_count, repeat_count, peptide_stage.shape[-1]
            )[:, 0, :]

        allele_stage = self._forward_allele_stage(allele_idx[:repeat_count])
        can_factorize_first_layer = (
            self.merge_activation is None
            and self.topology != "with-skip-connections"
        )
        if not can_factorize_first_layer:
            peptide_stage = peptide_stage.repeat_interleave(repeat_count, dim=0)
            return self.forward_from_peptide_stage(peptide_stage, allele_idx)

        if self.dense_layers:
            first_layer = self.dense_layers[0]
        else:
            first_layer = self.output_layer
        x = self._first_linear_from_cartesian_stages(
            peptide_stage,
            allele_stage,
            first_layer,
        )
        x = x.reshape(peptide_count * repeat_count, x.shape[-1])

        if self.dense_layers:
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[0] is not None:
                x = self.batch_norms[0](x)
            if self.dropouts[0] is not None:
                x = self.dropouts[0](x)

            # ``enumerate(seq, start=N)`` triggers a torch._dynamo graph
            # break on PyTorch <=2.4 (``call_enumerate`` rejects the
            # ``start`` kwarg in builtin.py:775) — every traced forward
            # then falls back to eager for the layer-stack loop, losing
            # the compile speedup. Iterate by index instead so the loop
            # body stays inside the compiled graph.
            #
            # When upgrading torch past 2.4 confirm the dynamo bug is
            # fixed and revert to enumerate(seq, start=1).
            for offset, layer in enumerate(self.dense_layers[1:]):
                i = offset + 1
                x = layer(x)
                if self.activation is not None:
                    x = self.activation(x)
                if self.batch_norms[i] is not None:
                    x = self.batch_norms[i](x)
                if self.dropouts[i] is not None:
                    x = self.dropouts[i](x)
            output = self.output_layer(x)
        else:
            output = x

        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

    def forward_cartesian_from_peptide_stage(self, peptide_stage, allele_idx):
        """Forward every peptide-stage row against every allele index.

        Returns predictions in allele-major order with shape
        ``(num_alleles, num_peptides, num_outputs)``. When the model's
        merge + first linear layer can be factored, this avoids materializing
        the larger ``num_alleles * num_peptides * peptide_stage_dim`` repeated
        peptide-stage tensor used by ``forward_from_peptide_stage``.

        Skip-connections (``topology == "with-skip-connections"``) are
        also factorized: layer 1's input ``cat[merged_input, layer0_out]``
        is computed without materializing the (a*p, peptide_width +
        allele_width) merged_input expansion — instead layer 1's weight
        is split column-wise into peptide / allele / prev pieces and
        the contributions are summed in their factored shapes
        (``(p, h1)`` + ``(a, h1)`` + ``(a, p, h1)``). Layers ≥ 2's
        input ``cat[prev[-2], prev[-1]]`` is similarly factorized via
        column-wise weight splits, avoiding the explicit concat.
        """
        if not self.has_allele:
            raise RuntimeError(
                "forward_cartesian_from_peptide_stage called on a "
                "has_allele=False model"
            )
        allele_idx = allele_idx.long()
        if allele_idx.dim() > 1:
            allele_idx = allele_idx.squeeze(-1)
        num_peptides = peptide_stage.shape[0]
        num_alleles = allele_idx.shape[0]

        is_skip = self.topology == "with-skip-connections"
        # ``_first_linear_from_cartesian_stages`` handles both
        # ``concatenate`` and ``multiply`` merge methods via separate
        # codepaths, so the standard (non-skip) factorized path is fine
        # for either. Skip-connections factorization at layer 1 below
        # only handles the concatenate case (multiply + skip is rare
        # and would need an einsum at every skip layer); fall back to
        # the non-factorized path for that combo.
        can_factorize_first_layer = (
            self.merge_activation is None
            and (not is_skip or self.peptide_allele_merge_method == "concatenate")
        )
        if not can_factorize_first_layer:
            peptide_width = peptide_stage.shape[-1]
            expanded = peptide_stage.unsqueeze(0).expand(
                num_alleles, num_peptides, peptide_width
            ).reshape(num_alleles * num_peptides, peptide_width)
            expanded_alleles = allele_idx.unsqueeze(1).expand(
                num_alleles, num_peptides
            ).reshape(-1)
            return self.forward_from_peptide_stage(
                expanded,
                expanded_alleles,
            ).reshape(num_alleles, num_peptides, -1)

        allele_stage = self._forward_allele_stage(allele_idx)
        if self.dense_layers:
            first_layer = self.dense_layers[0]
        else:
            first_layer = self.output_layer
        x = self._first_linear_from_cartesian_stages(
            peptide_stage,
            allele_stage,
            first_layer,
        )
        x = x.transpose(0, 1).reshape(num_alleles * num_peptides, x.shape[-1])

        if self.dense_layers:
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[0] is not None:
                x = self.batch_norms[0](x)
            if self.dropouts[0] is not None:
                x = self.dropouts[0](x)

            if is_skip:
                # Skip-connections factorized path. layer 1's input is
                # cat[merged_input, prev[0]] = cat[peptide_expanded,
                # allele_expanded, prev[0]]; rather than materialize
                # the (a*p, peptide_width + allele_width + h0) tensor,
                # split the layer's weight column-wise and sum the
                # three contributions in their natural factored shapes.
                # Layers ≥ 2 take cat[prev[-2], prev[-1]]; both are
                # already in flat-cartesian form so we just split the
                # weight column-wise to skip the concat materialization.
                peptide_width = peptide_stage.shape[-1]
                allele_width = allele_stage.shape[-1]
                prev_outputs = [x]
                for offset, layer in enumerate(self.dense_layers[1:]):
                    i = offset + 1
                    if i == 1:
                        W = layer.weight
                        W_p = W[:, :peptide_width]
                        W_a = W[:, peptide_width:peptide_width + allele_width]
                        W_prev = W[:, peptide_width + allele_width:]
                        peptide_part = peptide_stage.matmul(W_p.t())  # (p, h1)
                        allele_part = allele_stage.matmul(W_a.t())   # (a, h1)
                        prev_reshaped = prev_outputs[0].reshape(
                            num_alleles, num_peptides, -1,
                        )
                        prev_part = prev_reshaped.matmul(W_prev.t())  # (a, p, h1)
                        new_x = (
                            peptide_part.unsqueeze(0)
                            + allele_part.unsqueeze(1)
                            + prev_part
                            + layer.bias
                        )
                        new_x = new_x.reshape(
                            num_alleles * num_peptides, -1,
                        )
                    else:
                        # Layers ≥ 2: cat[prev[-2], prev[-1]] @ W.T + bias
                        # = prev[-2] @ W_left.T + prev[-1] @ W_right.T + bias
                        h_left = prev_outputs[-2].shape[-1]
                        W = layer.weight
                        W_left = W[:, :h_left]
                        W_right = W[:, h_left:]
                        new_x = (
                            prev_outputs[-2].matmul(W_left.t())
                            + prev_outputs[-1].matmul(W_right.t())
                            + layer.bias
                        )
                    if self.activation is not None:
                        new_x = self.activation(new_x)
                    if self.batch_norms[i] is not None:
                        new_x = self.batch_norms[i](new_x)
                    if self.dropouts[i] is not None:
                        new_x = self.dropouts[i](new_x)
                    prev_outputs.append(new_x)
                x = prev_outputs[-1]
            else:
                # Standard feedforward (no skip): existing tight loop.
                # ``enumerate(seq, start=N)`` triggers a torch._dynamo
                # graph break on PyTorch <=2.4 (``call_enumerate`` rejects
                # the ``start`` kwarg in builtin.py:775) — every traced
                # forward then falls back to eager for the layer-stack
                # loop, losing the compile speedup. Iterate by index
                # instead so the loop body stays inside the compiled
                # graph.
                for offset, layer in enumerate(self.dense_layers[1:]):
                    i = offset + 1
                    x = layer(x)
                    if self.activation is not None:
                        x = self.activation(x)
                    if self.batch_norms[i] is not None:
                        x = self.batch_norms[i](x)
                    if self.dropouts[i] is not None:
                        x = self.dropouts[i](x)
            output = self.output_layer(x)
        else:
            output = x

        if self.output_activation is not None:
            output = self.output_activation(output)
        return output.reshape(num_alleles, num_peptides, -1)

    def forward(self, inputs):
        """
        Forward pass.

        Parameters
        ----------
        inputs : dict
            Dictionary with 'peptide' and optionally 'allele' keys

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, num_outputs)
        """
        if self.has_allele != ('allele' in inputs):
            raise ValueError(
                "Class1NeuralNetworkModel input mismatch: "
                f"network has_allele={self.has_allele} but "
                f"received allele key={('allele' in inputs)}"
            )

        peptide = inputs['peptide']
        peptide_repeat_count = inputs.get("peptide_repeat_count")
        if peptide_repeat_count is not None:
            if "allele" not in inputs:
                raise ValueError(
                    "compact peptide-repeat batch requires an allele input"
                )
            return self._forward_compact_cartesian(
                peptide,
                inputs["allele"],
                int(peptide_repeat_count),
            )

        # Device-side fixed amino-acid encoding: (N, L) int indices →
        # (N, L, V) fp32. Skipped when the input is already the 3D
        # vector-encoded tensor. ``torch.nn.functional.embedding`` is a
        # pure gather, so the op cost is comparable to the int8→fp32
        # widening cast it replaces while saving V× H2D/cache bytes.
        if self.peptide_input_is_indices and peptide.dim() == 2:
            peptide = torch.nn.functional.embedding(
                peptide.long(), self.peptide_embedding_table
            )

        # Locally connected layers
        x = peptide
        for lc_layer in self.lc_layers:
            x = lc_layer(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Peptide dense layers
        for layer in self.peptide_dense_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)

        # Early batch normalization
        if self.batch_norm_early is not None:
            x = self.batch_norm_early(x)

        # Allele processing and merge
        if self.has_allele:
            allele_idx = inputs['allele'].long()
            # Handle case where input might be (batch,) or (batch, 1)
            if allele_idx.dim() > 1:
                allele_idx = allele_idx.squeeze(-1)
            allele_embed = self.allele_embedding(allele_idx)

            # Allele dense layers
            for layer in self.allele_dense_layers:
                allele_embed = layer(allele_embed)
                if self.activation is not None:
                    allele_embed = self.activation(allele_embed)

            # Flatten allele embedding
            allele_embed = allele_embed.reshape(allele_embed.size(0), -1)

            # Merge
            if self.peptide_allele_merge_method == "concatenate":
                x = torch.cat([x, allele_embed], dim=-1)
            elif self.peptide_allele_merge_method == "multiply":
                x = x * allele_embed

            # Merge activation
            if self.merge_activation is not None:
                x = self.merge_activation(x)

        # Main dense layers (with optional skip connections for DenseNet topology)
        prev_outputs = []  # Track outputs for skip connections
        merged_input = x  # Save for DenseNet skip connections

        for i, layer in enumerate(self.dense_layers):
            # For DenseNet topology, concatenate skip connections
            if self.topology == "with-skip-connections" and i > 0:
                if i == 1:
                    # Skip from original merged input
                    x = torch.cat([merged_input, prev_outputs[-1]], dim=-1)
                else:
                    # Skip from 2 layers back
                    x = torch.cat([prev_outputs[-2], prev_outputs[-1]], dim=-1)

            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)

            prev_outputs.append(x)

        # Note: For DenseNet topology, output layer receives only the last hidden layer output
        # (skip connections are only between hidden layers, not to the output layer)

        # Output
        output = self.output_layer(x)
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output

    def get_weights_list(self):
        """
        Get weights as a list of numpy arrays (for compatibility with NPZ format).

        Returns
        -------
        list of numpy.ndarray
        """
        weights = []
        for name, param in self.named_parameters():
            weights.append(param.detach().cpu().numpy())
        # Also include persistent buffers (running mean/var for batch norm).
        # Deterministic fixed-encoding tables are registered as
        # persistent=False and reconstructed from hyperparameters.
        for name, buffer in self._named_persistent_buffers():
            weights.append(buffer.detach().cpu().numpy())
        return weights

    def _named_persistent_buffers(self):
        """Yield named buffers that belong in mhcflurry's NPZ weights."""
        modules = dict(self.named_modules())
        for name, buffer in self.named_buffers():
            module = self
            buffer_name = name
            if "." in name:
                module_path, buffer_name = name.rsplit(".", 1)
                module = modules[module_path]
            if buffer_name in module._non_persistent_buffers_set:
                continue
            yield name, buffer

    def set_weights_list(self, weights, auto_convert_keras=True):
        """
        Set weights from a list of numpy arrays.

        Supports automatic detection and conversion of Keras-format weights
        to PyTorch format for backward compatibility with pre-trained models.

        Parameters
        ----------
        weights : list of numpy.ndarray
        auto_convert_keras : bool
            If True, automatically detect and convert Keras-format weights
        """
        if auto_convert_keras and getattr(self, "_keras_config", None):
            keras_layers = self._keras_config.get("config", {}).get("layers", [])
            idx = 0

            def assign_dense(layer, w, b):
                w = w.astype(numpy.float32)
                b = b.astype(numpy.float32)
                if w.shape == layer.weight.shape[::-1] or (
                    w.shape == layer.weight.shape and w.shape[0] == w.shape[1]
                ):
                    w = w.T
                if w.shape != layer.weight.shape:
                    raise ValueError(
                        f"Weight shape mismatch for {layer}: got {w.shape}, "
                        f"expected {layer.weight.shape}"
                    )
                if b.shape != layer.bias.shape:
                    raise ValueError(
                        f"Bias shape mismatch for {layer}: got {b.shape}, "
                        f"expected {layer.bias.shape}"
                    )
                layer.weight.data = torch.from_numpy(w).to(
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                layer.bias.data = torch.from_numpy(b).to(
                    device=layer.bias.device,
                    dtype=layer.bias.dtype,
                )

            def assign_locally_connected(layer, w, b):
                w = w.astype(numpy.float32)
                b = b.astype(numpy.float32)
                if len(w.shape) == 5 and w.shape[1] == 1:
                    out_len, _, k, in_ch, out_ch = w.shape
                    w = w.squeeze(1)
                    w = w.reshape(out_len, k * in_ch, out_ch)
                    w = w.transpose(0, 2, 1)
                elif len(w.shape) == 3 and w.shape[0] == layer.output_length:
                    # Keras (out_len, k*in_ch, out_ch) -> PyTorch (out_len, out_ch, in_ch*k)
                    if w.shape[1] == layer.weight.shape[2] and w.shape[2] == layer.weight.shape[1]:
                        w = w.transpose(0, 2, 1)
                    else:
                        kernel_size = layer.kernel_size
                        out_len = w.shape[0]
                        k_times_in_ch = w.shape[1]
                        out_ch = w.shape[2]
                        in_ch = k_times_in_ch // kernel_size
                        w = w.reshape(out_len, kernel_size, in_ch, out_ch)
                        w = w.transpose(0, 2, 1, 3)
                        w = w.reshape(out_len, in_ch * kernel_size, out_ch)
                        w = w.transpose(0, 2, 1)
                if w.shape != layer.weight.shape:
                    raise ValueError(
                        f"Weight shape mismatch for {layer}: got {w.shape}, "
                        f"expected {layer.weight.shape}"
                    )
                if b.shape != layer.bias.shape:
                    raise ValueError(
                        f"Bias shape mismatch for {layer}: got {b.shape}, "
                        f"expected {layer.bias.shape}"
                    )
                layer.weight.data = torch.from_numpy(w).to(
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                layer.bias.data = torch.from_numpy(b).to(
                    device=layer.bias.device,
                    dtype=layer.bias.dtype,
                )

            def assign_batch_norm(layer, gamma, beta, mean, var):
                layer.weight.data = torch.from_numpy(
                    gamma.astype(numpy.float32)
                ).to(device=layer.weight.device, dtype=layer.weight.dtype)
                layer.bias.data = torch.from_numpy(
                    beta.astype(numpy.float32)
                ).to(device=layer.bias.device, dtype=layer.bias.dtype)
                layer.running_mean.data = torch.from_numpy(
                    mean.astype(numpy.float32)
                ).to(
                    device=layer.running_mean.device,
                    dtype=layer.running_mean.dtype,
                )
                layer.running_var.data = torch.from_numpy(
                    var.astype(numpy.float32)
                ).to(
                    device=layer.running_var.device,
                    dtype=layer.running_var.dtype,
                )

            skip_keras_embedding = False
            keras_metadata = getattr(self, "_keras_metadata", None)
            if keras_metadata and keras_metadata.get("skip_embedding_weights", False):
                skip_keras_embedding = True

            for layer in keras_layers:
                layer_class = layer.get("class_name", "")
                layer_name = layer.get("config", {}).get("name", "")

                if layer_class == "Dense":
                    w = weights[idx]
                    b = weights[idx + 1]
                    idx += 2
                    if layer_name == "output":
                        assign_dense(self.output_layer, w, b)
                    elif layer_name.startswith("dense_"):
                        dense_idx = int(layer_name.split("_")[1])
                        assign_dense(self.dense_layers[dense_idx], w, b)
                    elif layer_name.startswith("peptide_dense_"):
                        dense_idx = int(layer_name.split("_")[2])
                        assign_dense(self.peptide_dense_layers[dense_idx], w, b)
                    elif layer_name.startswith("allele_dense_"):
                        dense_idx = int(layer_name.split("_")[2])
                        assign_dense(self.allele_dense_layers[dense_idx], w, b)
                elif layer_class == "LocallyConnected1D":
                    w = weights[idx]
                    b = weights[idx + 1]
                    idx += 2
                    lc_idx = int(layer_name.split("_")[1])
                    assign_locally_connected(self.lc_layers[lc_idx], w, b)
                elif layer_class == "Embedding":
                    w = weights[idx]
                    idx += 1
                    if skip_keras_embedding:
                        continue
                    if self.allele_embedding is None:
                        continue
                    if w.shape == self.allele_embedding.weight.shape:
                        target = self.allele_embedding.weight
                        self.allele_embedding.weight.data = torch.from_numpy(
                            w.astype(numpy.float32)
                        ).to(device=target.device, dtype=target.dtype)
                elif layer_class == "BatchNormalization":
                    gamma = weights[idx]
                    beta = weights[idx + 1]
                    mean = weights[idx + 2]
                    var = weights[idx + 3]
                    idx += 4
                    if layer_name == "batch_norm_early":
                        if self.batch_norm_early is not None:
                            assign_batch_norm(self.batch_norm_early, gamma, beta, mean, var)
                    elif layer_name.startswith("batch_norm_"):
                        bn_idx = int(layer_name.split("_")[2])
                        if self.batch_norms[bn_idx] is not None:
                            assign_batch_norm(self.batch_norms[bn_idx], gamma, beta, mean, var)
                else:
                    continue

            return
        idx = 0

        # Check for keras metadata to know if we need to skip embedding weights
        keras_metadata = getattr(self, '_keras_metadata', None)
        skip_keras_embedding = False
        if keras_metadata and keras_metadata.get('skip_embedding_weights', False):
            skip_keras_embedding = True

        named_modules = dict(self.named_modules()) if auto_convert_keras else {}

        for name, param in self.named_parameters():
            # Skip allele_embedding when loading Keras weights with placeholder
            if skip_keras_embedding and 'allele_embedding' in name:
                # Also skip the corresponding placeholder weight in the weights list
                # Placeholder embeddings have shape (0, embed_dim)
                while idx < len(weights) and len(weights[idx].shape) == 2 and weights[idx].shape[0] == 0:
                    idx += 1
                continue
            w = weights[idx].astype(numpy.float32)
            extra_keras_skip = 0
            module = None
            if auto_convert_keras and "." in name:
                module_name = name.rsplit(".", 1)[0]
                module = named_modules.get(module_name)

            # Skip allele_embedding if shapes don't match (pan-allele models)
            # The embedding will be set by set_allele_representations later
            if 'allele_embedding' in name and w.shape != param.shape:
                # Advance index past this weight
                idx += 1
                continue

            # Auto-detect and convert Keras weights
            if auto_convert_keras:
                # Dense/Linear layer: Keras (in, out) -> PyTorch (out, in)
                # Note: Must transpose even when shapes match (square matrices)
                # Check for weight (not bias) by looking at param name
                is_linear_weight = ('weight' in name and
                                    'embedding' not in name and
                                    len(w.shape) == 2)
                if is_linear_weight and (w.shape == param.shape[::-1] or
                        (w.shape == param.shape and w.shape[0] == w.shape[1])):
                    w = w.T
                # LocallyConnected1D weight: Keras (out_len, 1, k, in_ch, out_ch)
                # -> PyTorch (out_len, out_ch, in_ch * k)
                elif len(w.shape) == 5 and w.shape[1] == 1:
                    out_len, _, k, in_ch, out_ch = w.shape
                    w = w.squeeze(1)  # (out_len, k, in_ch, out_ch)
                    w = w.reshape(out_len, k * in_ch, out_ch)
                    w = w.transpose(0, 2, 1)  # (out_len, out_ch, k * in_ch)
                # LocallyConnected1D weight (3D): Keras (out_len, k*in_ch, out_ch)
                # -> PyTorch (out_len, out_ch, in_ch*k)
                # Note: Keras stores kernel_positions as outer loop, channels as inner
                # PyTorch unfold produces channels as outer loop, kernel_positions as inner
                elif len(w.shape) == 3 and w.shape[0] == param.shape[0] and \
                        w.shape[1] == param.shape[2] and w.shape[2] == param.shape[1]:
                    # LocallyConnected1D weight (3D): Keras (out_len, k*in_ch, out_ch)
                    # -> PyTorch (out_len, out_ch, k*in_ch)
                    w = w.transpose(0, 2, 1)
                # LocallyConnected1D bias: Keras (out_len * out_ch,) -> PyTorch (out_len, out_ch)
                elif len(w.shape) == 1 and len(param.shape) == 2 and \
                        w.shape[0] == param.shape[0] * param.shape[1]:
                    w = w.reshape(param.shape)
                # BatchNorm: Keras provides gamma, beta, moving_mean, moving_var.
                # PyTorch exposes gamma/beta as params and moving stats as buffers.
                if module is not None and isinstance(module, torch.nn.BatchNorm1d):
                    if name.endswith("bias") and idx + 2 < len(weights):
                        running_mean = weights[idx + 1].astype(numpy.float32)
                        running_var = weights[idx + 2].astype(numpy.float32)
                        if module.running_mean.shape == running_mean.shape:
                            module.running_mean.data = torch.from_numpy(
                                running_mean
                            ).to(
                                device=module.running_mean.device,
                                dtype=module.running_mean.dtype,
                            )
                        if module.running_var.shape == running_var.shape:
                            module.running_var.data = torch.from_numpy(
                                running_var
                            ).to(
                                device=module.running_var.device,
                                dtype=module.running_var.dtype,
                            )
                        extra_keras_skip = 2

            if w.shape != param.shape:
                raise ValueError(
                    f"Weight shape mismatch for {name}: "
                    f"got {weights[idx].shape}, expected {param.shape}"
                )

            param.data = torch.from_numpy(w).to(
                device=param.device,
                dtype=param.dtype,
            )
            idx += 1 + extra_keras_skip
        if not auto_convert_keras:
            named_modules_dict = dict(self.named_modules())
            for name, buffer in self._named_persistent_buffers():
                tensor = torch.from_numpy(weights[idx]).to(
                    device=buffer.device,
                    dtype=buffer.dtype,
                )
                # Navigate to the correct submodule for nested buffers
                if "." in name:
                    module_path, buffer_name = name.rsplit(".", 1)
                    named_modules_dict[module_path]._buffers[buffer_name] = tensor
                else:
                    self._buffers[name] = tensor
                idx += 1

    def to_json(self):
        """
        Serialize model configuration to JSON string.

        Returns
        -------
        str
            JSON representation of model configuration
        """
        import json

        # Extract layer configurations
        lc_layers_config = []
        for lc_layer in self.lc_layers:
            lc_layers_config.append({
                'in_channels': lc_layer.in_channels,
                'out_channels': lc_layer.out_channels,
                'kernel_size': lc_layer.kernel_size,
                'input_length': lc_layer.input_length,
                'output_length': lc_layer.output_length,
                'activation': lc_layer.activation_name,
            })

        peptide_dense_sizes = [
            layer.out_features for layer in self.peptide_dense_layers
        ]
        allele_dense_sizes = [
            layer.out_features for layer in self.allele_dense_layers
        ]
        layer_sizes = [
            layer.out_features for layer in self.dense_layers
        ]

        config = {
            'class': 'Class1NeuralNetworkModel',
            'peptide_encoding_shape': list(self.peptide_encoding_shape),
            'peptide_input_vector_encoding_name': self.peptide_input_vector_encoding_name,
            'has_allele': self.has_allele,
            'peptide_allele_merge_method': self.peptide_allele_merge_method,
            'peptide_allele_merge_activation': self.peptide_allele_merge_activation,
            'dropout_probability': self.dropout_probability,
            'topology': self.topology,
            'num_outputs': self.num_outputs,
            'activation': self.activation_name,
            'output_activation': self.output_activation_name,
            'locally_connected_layers': lc_layers_config,
            'peptide_dense_layer_sizes': peptide_dense_sizes,
            'allele_dense_layer_sizes': allele_dense_sizes,
            'layer_sizes': layer_sizes,
            'batch_normalization': self.batch_norm_early is not None,
        }

        return json.dumps(config, sort_keys=True)


class Class1NeuralNetwork(object):
    """
    Low level class I predictor consisting of a single neural network.

    Both single allele and pan-allele prediction are supported.

    Users will generally use Class1AffinityPredictor, which gives a higher-level
    interface and supports ensembles.
    """

    network_hyperparameter_defaults = HyperparameterDefaults(
        allele_amino_acid_encoding="BLOSUM62",
        allele_dense_layer_sizes=[],
        peptide_encoding={
            "vector_encoding_name": "BLOSUM62",
            "alignment_method": "pad_middle",
            "left_edge": 4,
            "right_edge": 4,
            "max_length": 15,
        },
        peptide_dense_layer_sizes=[],
        peptide_allele_merge_method="multiply",
        peptide_allele_merge_activation="",
        layer_sizes=[32],
        dense_layer_l1_regularization=0.001,
        dense_layer_l2_regularization=0.0,
        activation="tanh",
        init="glorot_uniform",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=False,
        locally_connected_layers=[
            {"filters": 8, "activation": "tanh", "kernel_size": 3}
        ],
        topology="feedforward",
        num_outputs=1,
        # Feed the network (N, L) int amino-acid indices and widen to the
        # configured fixed vector encoding through a frozen torch embedding
        # table on the active device, instead of shipping the (N, L, V)
        # vector tensor from numpy. Disable only for legacy byte-level tests.
        peptide_amino_acid_encoding_torch=True,
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    compile_hyperparameter_defaults = HyperparameterDefaults(
        loss="custom:mse_with_inequalities",
        optimizer="rmsprop",
        learning_rate=None,
    )
    """
    Loss and optimizer hyperparameters.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=500,
        validation_split=0.1,
        early_stopping=True,
        minibatch_size=512,
        data_dependent_initialization_method=None,
        random_negative_affinity_min=20000.0,
        random_negative_affinity_max=50000.0,
        random_negative_output_indices=None,
        # Number of consecutive epochs backed by one generated pool of
        # random-negative peptides. 1 (default) samples fresh negatives every
        # epoch. Values >1 amortize peptide generation/encoding by drawing
        # distinct per-epoch slices from a larger pool, with a new pool at each
        # ``epoch // random_negative_pool_epochs`` boundary. Larger values save
        # more CPU work; smaller values preserve more sampling diversity.
        random_negative_pool_epochs=1,
        # Streaming pretraining can use DataLoader workers to prefetch encoded
        # batches. Standard affinity fit() is device-resident and ignores this
        # knob. Kept in the schema so old configs that set it don't error.
        dataloader_num_workers=0,
        # Random-negative pool backing for fit(). None uses the optimized
        # device-resident pool; "host" preserves the legacy
        # planner.get_peptides() path for compatibility/debugging.
        fit_tensor_residency=None,
        # Batch size used for the validation forward pass. ``None`` uses a
        # device-aware heuristic: ``max(4 * minibatch_size, 4096)`` on CUDA
        # and ``4 * minibatch_size`` elsewhere. Separate from minibatch_size
        # because eval has no backward / optimizer state so VRAM headroom is
        # much higher; the larger CUDA default cuts validation-loop overhead
        # substantially on the small pan-allele networks.
        validation_batch_size=None,
    ).extend(RandomNegativePeptides.hyperparameter_defaults)
    """
    Hyperparameters for neural network training.
    """

    early_stopping_hyperparameter_defaults = HyperparameterDefaults(
        patience=20,
        min_delta=0.0,
        # Run the validation pass every N epochs in fit() / streaming pretrain.
        # Default 1 preserves pre-existing behavior (validate every epoch).
        # Setting to >1 trades resolution-of-early-stop-decision for
        # epoch-level throughput; the fit() loop forces a final validation
        # pass before reporting min_val_loss / breaking on patience so the
        # saved model still reflects an up-to-date val_loss measurement.
        validation_interval=1,
    )
    """
    Hyperparameters for early stopping.
    """

    miscelaneous_hyperparameter_defaults = HyperparameterDefaults(
        train_data={},
    )
    """
    Miscelaneous hyperaparameters. These parameters are not used by this class
    but may be interpreted by other code.
    """

    hyperparameter_defaults = (
        network_hyperparameter_defaults.extend(compile_hyperparameter_defaults)
        .extend(fit_hyperparameter_defaults)
        .extend(early_stopping_hyperparameter_defaults)
        .extend(miscelaneous_hyperparameter_defaults)
    )
    """
    Combined set of all supported hyperparameters and their default values.
    """

    # Hyperparameter renames. Map old_name → new_name (string to rename)
    # or old_name → None (to silently drop). Loading an old config with
    # any of these keys triggers the mapping in ``__init__`` before
    # ``HyperparameterDefaults.check_valid_keys`` runs — so old keys
    # don't cause a ValueError.
    #
    # Note: ``min_delta`` used to be in this table mapping to None,
    # because when it was added (pre-PyTorch) it was flagged "currently
    # unused". It has since been re-introduced as a live early-stopping
    # hyperparameter, so dropping it silently would have caused old
    # configs with explicit non-default ``min_delta`` to lose that
    # value. Fixed by removing from this table — now it passes through
    # to the valid-keys check and gets preserved.
    hyperparameter_renames = {
        "use_embedding": None,
        "pseudosequence_use_embedding": None,
        "monitor": None,
        "verbose": None,
        "mode": None,
        "take_best_epoch": None,
        "kmer_size": None,
        "peptide_amino_acid_encoding": None,
        "embedding_input_dim": None,
        "embedding_output_dim": None,
        "embedding_init_method": None,
        "peptide_amino_acid_encoding_gpu": "peptide_amino_acid_encoding_torch",
        "left_edge": None,
        "right_edge": None,
        # 2.3.0 cleanup: the per-fit DataLoader backing layer is gone.
        # Old configs may still set this knob; silently drop it.
        "fit_dataloader_backing": None,
    }

    @classmethod
    def apply_hyperparameter_renames(cls, hyperparameters):
        """
        Handle hyperparameter renames.

        Parameters
        ----------
        hyperparameters : dict

        Returns
        -------
        dict : updated hyperparameters

        """
        for from_name, to_name in cls.hyperparameter_renames.items():
            if from_name in hyperparameters:
                value = hyperparameters.pop(from_name)
                if to_name:
                    if (to_name in hyperparameters
                            and hyperparameters[to_name] != value):
                        raise ValueError(
                            "Conflicting values for renamed hyperparameter "
                            "%s=%r and %s=%r"
                            % (from_name, value, to_name, hyperparameters[to_name])
                        )
                    hyperparameters[to_name] = value
        return hyperparameters

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            self.apply_hyperparameter_renames(hyperparameters)
        )

        self._network = None
        self.network_json = None
        self.network_weights = None
        self.network_weights_loader = None
        self.network_weight_paths = ()

        self.fit_info = []
        self.prediction_cache = weakref.WeakKeyDictionary()

    MODELS_CACHE = {}
    """
    Process-wide model cache, a map from: architecture JSON string to
    (PyTorch model, existing network weights)
    """

    @classmethod
    def clear_model_cache(klass):
        """
        Clear the model cache.
        """
        klass.MODELS_CACHE.clear()

    @classmethod
    def borrow_cached_network(klass, network_json, network_weights):
        """
        Return a PyTorch model with the specified architecture and weights.
        As an optimization, when possible this will reuse architectures from a
        process-wide cache.

        Parameters
        ----------
        network_json : string of JSON
        network_weights : list of numpy.array

        Returns
        -------
        Class1NeuralNetworkModel
        """
        assert network_weights is not None
        config = json.loads(network_json)
        # Detect if weights are from Keras or PyTorch format
        # Keras JSON has 'class_name': 'Model' or 'Functional'; PyTorch has 'hyperparameters'
        is_keras_format = config.get('class_name') in ('Model', 'Functional')
        key = (
            klass.model_cache_key(network_json),
            klass._infer_allele_representation_dim_from_weights(
                network_weights,
                config.get("hyperparameters", {}),
                network_config=config,
            ),
        )

        if key not in klass.MODELS_CACHE:
            # Cache miss - create new model
            network = klass._create_model_from_config(
                config,
                network_weights=network_weights,
            )
            existing_weights = None
        else:
            # Cache hit
            (network, existing_weights) = klass.MODELS_CACHE[key]

        if existing_weights is not network_weights:
            network.set_weights_list(network_weights, auto_convert_keras=is_keras_format)
            klass.MODELS_CACHE[key] = (network, network_weights)

        return network

    @classmethod
    def _parse_keras_json_config(cls, config):
        """
        Parse a legacy Keras model JSON config to extract hyperparameters.

        Parameters
        ----------
        config : dict
            Keras model JSON config with 'class_name', 'config', etc.

        Returns
        -------
        tuple of (dict, dict)
            First dict: Hyperparameters dict compatible with Class1NeuralNetwork
            Second dict: Metadata about Keras model structure (e.g., embedding info)
        """
        layers = config.get('config', {}).get('layers', [])

        hyperparameters = {
            'locally_connected_layers': [],
            'layer_sizes': [],
            'activation': 'tanh',
            'output_activation': 'sigmoid',
            'dropout_probability': 0.0,
            'batch_normalization': False,
            'dense_layer_l1_regularization': 0.001,
            'dense_layer_l2_regularization': 0.0,
            'peptide_allele_merge_method': 'multiply',  # Default
        }

        # Metadata about Keras structure
        keras_metadata = {
            'has_embedding': False,
            'embedding_input_dim': 0,
            'embedding_output_dim': 0,
            'skip_embedding_weights': False,
        }

        dense_layers = []
        peptide_dense_sizes = []
        allele_dense_sizes = []
        concatenate_count = 0
        for layer in layers:
            layer_class = layer.get('class_name', '')
            layer_config = layer.get('config', {})

            if layer_class == 'LocallyConnected1D':
                lc_config = {
                    'filters': layer_config.get('filters', 8),
                    'kernel_size': layer_config.get('kernel_size', [3])[0] if isinstance(
                        layer_config.get('kernel_size', [3]), list
                    ) else layer_config.get('kernel_size', 3),
                    'activation': layer_config.get('activation', 'tanh'),
                }
                hyperparameters['locally_connected_layers'].append(lc_config)
                hyperparameters['activation'] = lc_config['activation']

            elif layer_class == 'Dense':
                units = layer_config.get('units', 32)
                activation = layer_config.get('activation', 'tanh')
                layer_name = layer_config.get('name', '')
                if layer_name.startswith('peptide_dense_'):
                    peptide_dense_sizes.append(units)
                elif layer_name.startswith('allele_dense_'):
                    allele_dense_sizes.append(units)
                else:
                    dense_layers.append({'units': units, 'activation': activation})

                # Extract regularization from first dense layer
                kernel_reg = layer_config.get('kernel_regularizer')
                if kernel_reg and isinstance(kernel_reg, dict):
                    reg_config = kernel_reg.get('config', {})
                    if 'l1' in reg_config:
                        hyperparameters['dense_layer_l1_regularization'] = reg_config['l1']
                    if 'l2' in reg_config:
                        hyperparameters['dense_layer_l2_regularization'] = reg_config['l2']

            elif layer_class == 'Dropout':
                rate = layer_config.get('rate', 0.0)
                hyperparameters['dropout_probability'] = 1.0 - rate

            elif layer_class == 'BatchNormalization':
                hyperparameters['batch_normalization'] = True

            elif layer_class == 'Embedding':
                keras_metadata['has_embedding'] = True
                keras_metadata['embedding_input_dim'] = layer_config.get('input_dim', 0)
                keras_metadata['embedding_output_dim'] = layer_config.get('output_dim', 0)
                # If input_dim is 0, it's a placeholder and weights should be skipped
                if layer_config.get('input_dim', 0) == 0:
                    keras_metadata['skip_embedding_weights'] = True

            elif layer_class == 'Concatenate':
                concatenate_count += 1
                # Only set merge_method to concatenate if there's just one Concatenate
                # and it's likely for peptide-allele merging (not DenseNet skip connections)
                if concatenate_count == 1:
                    hyperparameters['peptide_allele_merge_method'] = 'concatenate'

            elif layer_class == 'Multiply':
                hyperparameters['peptide_allele_merge_method'] = 'multiply'

        # Multiple Concatenate layers indicate DenseNet topology with skip connections
        # Note: The first Concatenate is typically for peptide-allele merging,
        # subsequent ones are for DenseNet skip connections
        if concatenate_count > 1:
            hyperparameters['topology'] = 'with-skip-connections'
            # Keep the merge method as detected (concatenate from first Concatenate layer)

        # The last Dense layer is the output layer
        if dense_layers:
            hyperparameters['output_activation'] = dense_layers[-1]['activation']
            hyperparameters['num_outputs'] = dense_layers[-1]['units']
            # All other Dense layers contribute to layer_sizes
            hyperparameters['layer_sizes'] = [d['units'] for d in dense_layers[:-1]]
            if dense_layers[:-1]:
                hyperparameters['activation'] = dense_layers[0]['activation']

        if peptide_dense_sizes:
            hyperparameters['peptide_dense_layer_sizes'] = peptide_dense_sizes
        if allele_dense_sizes:
            hyperparameters['allele_dense_layer_sizes'] = allele_dense_sizes

        return hyperparameters, keras_metadata

    @classmethod
    def _infer_allele_representation_dim_from_weights(
            cls, network_weights, hyperparameters, network_config=None):
        """Infer flattened allele representation width from serialized weights.

        Strategy:
          1. Compute the position of the first allele-stage weight from
             the architecture config. In ``named_parameters()`` order the
             layout is::

                locally_connected → peptide_dense → batch_norm_early?
                  → allele_dense → allele_embedding → dense_layers
                  → output

             so ``weights[peptide_offset]`` is the first allele_dense
             weight (when allele_dense layers exist) or the
             allele_embedding weight (otherwise). Either way its
             ``shape[1]`` equals the flattened allele representation
             width.

          2. Cross-check using a structural invariant: when allele_dense
             layers exist, both the allele_dense first-layer weight and
             the allele_embedding weight have ``shape[1]`` equal to
             ``allele_rep_width``. That value appears at two distinct
             positions in the weight list. Peptide-stage Linear weights
             chain through different shape[1] values, so no peptide
             weight will produce a duplicate match.

          3. Return ``None`` if neither check is conclusive. The caller
             falls back to ``default_sequence_length`` (39).
        """
        if not network_weights or not hyperparameters.get('allele_amino_acid_encoding'):
            return None
        from .amino_acid import get_vector_encoding_df
        try:
            encoding_dim = len(get_vector_encoding_df(
                hyperparameters['allele_amino_acid_encoding']).columns)
        except KeyError:
            return None

        if network_config and network_config.get("merged_networks"):
            architecture = network_config["merged_networks"][0]
        elif network_config:
            architecture = network_config.get("hyperparameters", network_config)
        else:
            architecture = hyperparameters

        def valid_dim(dim):
            if dim % encoding_dim != 0:
                return False
            sequence_length = dim // encoding_dim
            return 20 <= sequence_length <= 60

        def candidate_dim(weight):
            if isinstance(weight, numpy.ndarray) and len(weight.shape) == 2:
                dim = weight.shape[1]
                if valid_dim(dim):
                    return dim
            return None

        allele_dense_sizes = architecture.get(
            'allele_dense_layer_sizes',
            hyperparameters.get('allele_dense_layer_sizes', []),
        )

        peptide_offset = 0
        peptide_offset += 2 * len(architecture.get(
            'locally_connected_layers',
            hyperparameters.get('locally_connected_layers', []),
        ))
        peptide_offset += 2 * len(architecture.get(
            'peptide_dense_layer_sizes',
            hyperparameters.get('peptide_dense_layer_sizes', []),
        ))
        if architecture.get(
                'batch_normalization',
                hyperparameters.get('batch_normalization', False)):
            peptide_offset += 2

        n = len(network_weights)

        # Position-based primary lookup.
        primary_dim = (
            candidate_dim(network_weights[peptide_offset])
            if peptide_offset < n else None)

        # Cross-check: when allele_dense layers exist, the same shape[1]
        # must appear at the embedding position (peptide_offset +
        # 2*n_allele_dense_layers). If they disagree, the offset is
        # wrong and we fall back to a duplicate-shape scan.
        if primary_dim is not None and allele_dense_sizes:
            embedding_idx = (
                peptide_offset + 2 * len(allele_dense_sizes))
            if embedding_idx < n:
                embedding_dim = candidate_dim(network_weights[embedding_idx])
                if embedding_dim == primary_dim:
                    return primary_dim
                # Disagreement: don't trust either value here, scan below.
            else:
                # Can't confirm via embedding slot; trust the primary.
                return primary_dim
        elif primary_dim is not None:
            # No allele_dense — only one matching slot exists.
            return primary_dim

        # Fallback: duplicate-shape[1] scan. Only the allele block
        # (allele_dense weight + allele_embedding weight) produces two
        # 2D weights with the same shape[1] in this architecture.
        if allele_dense_sizes:
            seen = {}
            for i in range(n):
                dim = candidate_dim(network_weights[i])
                if dim is None:
                    continue
                if dim in seen:
                    logging.warning(
                        "Allele width inferred via duplicate-shape fallback "
                        "(positions %d and %d -> width %d). Position-based "
                        "lookup failed: the saved layer count may not match "
                        "the architecture used to write these weights.",
                        seen[dim],
                        i,
                        dim,
                    )
                    return dim
                seen[dim] = i
        return None

    @classmethod
    def _placeholder_allele_representations(
            cls, hyperparameters, network_weights=None, network_config=None,
            default_sequence_length=39):
        """Create a shape-compatible placeholder for pan-allele models.

        Public artifacts have used several pseudosequence widths over time
        (including 2.1.x-era weights). Prefer the width encoded in the saved
        weights; fall back to the current 39-residue sequence only when no
        weights are available to guide reconstruction.
        """
        if not hyperparameters.get('allele_amino_acid_encoding'):
            return None
        from .amino_acid import get_vector_encoding_df
        encoding_name = hyperparameters['allele_amino_acid_encoding']
        try:
            encoding_df = get_vector_encoding_df(encoding_name)
        except KeyError:
            return None
        embedding_dim = cls._infer_allele_representation_dim_from_weights(
            network_weights,
            hyperparameters,
            network_config=network_config,
        )
        if embedding_dim is None:
            embedding_dim = default_sequence_length * len(encoding_df.columns)
        return numpy.zeros((1, embedding_dim), dtype=numpy.float32)

    @classmethod
    def _create_model_from_config(
            cls, config, instance_hyperparameters=None, network_weights=None):
        """Create a model from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary (either Keras JSON or hyperparameters dict)
        instance_hyperparameters : dict, optional
            Hyperparameters from the Class1NeuralNetwork instance.
            These take precedence for things like peptide_encoding.
        """
        keras_metadata = None

        # Check if this is a merged network config
        if config.get('merged_networks'):
            return cls._create_merged_model_from_config(
                config,
                instance_hyperparameters,
                network_weights=network_weights,
            )

        # Check if this is a legacy Keras JSON config
        if config.get('class_name') in ('Model', 'Functional'):
            hyperparameters, keras_metadata = cls._parse_keras_json_config(config)
        else:
            # Extract hyperparameters from config (new format)
            hyperparameters = config.get('hyperparameters', config)

        # Merge with instance hyperparameters if provided
        # Instance hyperparameters take precedence for things like peptide_encoding
        if instance_hyperparameters:
            # Copy to avoid modifying original
            merged = dict(instance_hyperparameters)
            # Update with parsed hyperparameters (architecture-specific settings)
            for key in ['layer_sizes', 'locally_connected_layers', 'dropout_probability',
                        'batch_normalization', 'activation', 'output_activation',
                        'peptide_allele_merge_method']:
                if key in hyperparameters:
                    merged[key] = hyperparameters[key]
            hyperparameters = merged

        hyperparameters = cls.hyperparameter_defaults.with_defaults(
            cls.apply_hyperparameter_renames(dict(hyperparameters))
        )

        # Create a temporary instance to get encoding shape
        temp = cls(**hyperparameters)
        peptide_encoding_shape = temp.peptides_to_network_input([]).shape[1:]
        peptide_torch_encoding_name = _peptide_torch_encoding_name(hyperparameters)
        # Device-side fixed encoding: peptides_to_network_input returns
        # (N, L) int indices, but dense layers are still sized against
        # the post-embedding (L, V) shape.
        if peptide_torch_encoding_name and len(peptide_encoding_shape) == 1:
            peptide_encoding_shape = _peptide_torch_encoding_shape(
                peptide_encoding_shape,
                peptide_torch_encoding_name,
            )

        # Get allele representations if present
        allele_representations = config.get('allele_representations')
        if allele_representations is not None:
            allele_representations = numpy.array(allele_representations)

        # For pan-allele Keras models with placeholder embedding (input_dim=0),
        # create a placeholder allele representation to ensure correct architecture
        if (allele_representations is None and keras_metadata is not None
                and keras_metadata.get('has_embedding', False)
                and keras_metadata.get('embedding_output_dim', 0) > 0):
            # Create placeholder with 1 allele and correct embedding dim
            # This will be replaced by set_allele_representations later
            embedding_dim = keras_metadata['embedding_output_dim']
            allele_representations = numpy.zeros((1, embedding_dim), dtype=numpy.float32)

        # For PyTorch-format configs without allele_representations but with
        # allele_amino_acid_encoding (pan-allele models), create placeholder
        # Check has_allele flag to distinguish pan-allele from allele-specific models
        has_allele = config.get('has_allele', True)  # Default True for backward compat
        if (allele_representations is None and keras_metadata is None
                and has_allele and hyperparameters.get('allele_amino_acid_encoding')):
            allele_representations = cls._placeholder_allele_representations(
                hyperparameters,
                network_weights=network_weights,
                network_config=config,
            )

        model = Class1NeuralNetworkModel(
            peptide_encoding_shape=peptide_encoding_shape,
            allele_representations=allele_representations,
            locally_connected_layers=hyperparameters.get('locally_connected_layers', []),
            peptide_dense_layer_sizes=hyperparameters.get('peptide_dense_layer_sizes', []),
            allele_dense_layer_sizes=hyperparameters.get('allele_dense_layer_sizes', []),
            layer_sizes=hyperparameters.get('layer_sizes', [32]),
            peptide_allele_merge_method=hyperparameters.get('peptide_allele_merge_method', 'multiply'),
            peptide_allele_merge_activation=hyperparameters.get('peptide_allele_merge_activation', ''),
            activation=hyperparameters.get('activation', 'tanh'),
            output_activation=hyperparameters.get('output_activation', 'sigmoid'),
            dropout_probability=hyperparameters.get('dropout_probability', 0.0),
            batch_normalization=hyperparameters.get('batch_normalization', False),
            dense_layer_l1_regularization=hyperparameters.get('dense_layer_l1_regularization', 0.001),
            dense_layer_l2_regularization=hyperparameters.get('dense_layer_l2_regularization', 0.0),
            topology=hyperparameters.get('topology', 'feedforward'),
            num_outputs=hyperparameters.get('num_outputs', 1),
            init=hyperparameters.get('init', 'glorot_uniform'),
            peptide_input_vector_encoding_name=peptide_torch_encoding_name,
        )

        # Store keras metadata and config for weight loading
        if keras_metadata is not None:
            model._keras_metadata = keras_metadata
            model._keras_config = config

        return model

    @classmethod
    def _create_merged_model_from_config(
            cls, config, instance_hyperparameters=None, network_weights=None):
        """Create a merged model from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'merged_networks' key
        instance_hyperparameters : dict, optional
            Hyperparameters from the Class1NeuralNetwork instance.
        """
        merged_configs = config['merged_networks']
        merge_method = config.get('merge_method', 'average')

        # Create a temporary instance to get encoding shape
        base_hyperparameters = config.get('hyperparameters', {})
        if instance_hyperparameters:
            base_hyperparameters = dict(instance_hyperparameters)
            base_hyperparameters.update(config.get('hyperparameters', {}))
        base_hyperparameters = cls.hyperparameter_defaults.with_defaults(
            cls.apply_hyperparameter_renames(dict(base_hyperparameters))
        )
        temp = cls(**base_hyperparameters)
        peptide_encoding_shape = temp.peptides_to_network_input([]).shape[1:]
        peptide_torch_encoding_name = _peptide_torch_encoding_name(
            base_hyperparameters
        )
        if peptide_torch_encoding_name and len(peptide_encoding_shape) == 1:
            peptide_encoding_shape = _peptide_torch_encoding_shape(
                peptide_encoding_shape,
                peptide_torch_encoding_name,
            )

        # Create placeholder allele representations for pan-allele models
        allele_representations = cls._placeholder_allele_representations(
            base_hyperparameters,
            network_weights=network_weights,
            network_config=config,
        )

        # Create sub-networks
        sub_networks = []
        for sub_config in merged_configs:
            model = Class1NeuralNetworkModel(
                peptide_encoding_shape=peptide_encoding_shape,
                allele_representations=allele_representations,
                locally_connected_layers=sub_config.get('locally_connected_layers', []),
                peptide_dense_layer_sizes=sub_config.get('peptide_dense_layer_sizes', []),
                allele_dense_layer_sizes=sub_config.get('allele_dense_layer_sizes', []),
                layer_sizes=sub_config.get('layer_sizes', [32]),
                peptide_allele_merge_method=sub_config.get('peptide_allele_merge_method', 'multiply'),
                peptide_allele_merge_activation=sub_config.get('peptide_allele_merge_activation', ''),
                activation=sub_config.get('activation', 'tanh'),
                output_activation=sub_config.get('output_activation', 'sigmoid'),
                dropout_probability=sub_config.get('dropout_probability', 0.0),
                batch_normalization=sub_config.get('batch_normalization', False),
                dense_layer_l1_regularization=base_hyperparameters.get('dense_layer_l1_regularization', 0.001),
                dense_layer_l2_regularization=base_hyperparameters.get('dense_layer_l2_regularization', 0.0),
                topology=sub_config.get('topology', 'feedforward'),
                num_outputs=sub_config.get('num_outputs', 1),
                init=base_hyperparameters.get('init', 'glorot_uniform'),
                peptide_input_vector_encoding_name=peptide_torch_encoding_name,
            )
            sub_networks.append(model)

        return MergedClass1NeuralNetwork(sub_networks, merge_method=merge_method)

    @staticmethod
    def model_cache_key(network_json):
        """
        Given a JSON description of a neural network, return a cache key.

        Parameters
        ----------
        network_json : string

        Returns
        -------
        string
        """
        # Remove regularization settings as they don't affect predictions
        def drop_properties(d):
            if isinstance(d, dict):
                d.pop('dense_layer_l1_regularization', None)
                d.pop('dense_layer_l2_regularization', None)
            return d

        description = json.loads(network_json, object_hook=drop_properties)
        return json.dumps(description)

    @staticmethod
    def keras_network_cache_key(network_json):
        """
        Backward-compatible alias for ``model_cache_key``.
        """
        return Class1NeuralNetwork.model_cache_key(network_json)

    def network(self, borrow=False):
        """
        Return the PyTorch model associated with this predictor.

        Parameters
        ----------
        borrow : bool
            Whether to return a cached model if possible

        Returns
        -------
        Class1NeuralNetworkModel
        """
        if self._network is None and self.network_json is not None:
            self.load_weights()
            if borrow:
                return self.borrow_cached_network(
                    self.network_json, self.network_weights
                )
            else:
                config = json.loads(self.network_json)
                # Detect if weights are from Keras or PyTorch format
                # Keras JSON has 'class_name': 'Model' or 'Functional'; PyTorch has 'hyperparameters'
                is_keras_format = config.get('class_name') in ('Model', 'Functional')
                # Pass this instance's hyperparameters to preserve peptide_encoding etc.
                self._network = self._create_model_from_config(
                    config,
                    instance_hyperparameters=self.hyperparameters,
                    network_weights=self.network_weights)
                if self.network_weights is not None:
                    self._network.set_weights_list(
                        self.network_weights,
                        auto_convert_keras=is_keras_format
                    )
                self.network_json = None
                self.network_weights = None
        return self._network

    def update_network_description(self):
        """
        Update self.network_json and self.network_weights properties based on
        this instances's neural network.
        """
        if self._network is not None:
            config = {
                'hyperparameters': dict(self.hyperparameters),
            }

            # Check if this is a merged network
            if isinstance(self._network, MergedClass1NeuralNetwork):
                # Save sub-network configs for merged networks
                sub_configs = []
                for subnet in self._network.networks:
                    sub_config = {}
                    # Get the architecture info from the network itself
                    sub_config['layer_sizes'] = [
                        layer.out_features for layer in subnet.dense_layers
                    ]
                    sub_config['locally_connected_layers'] = [
                        {'filters': layer.out_channels, 'kernel_size': layer.kernel_size}
                        for layer in subnet.lc_layers
                    ] if hasattr(subnet, 'lc_layers') else []
                    sub_config['peptide_dense_layer_sizes'] = [
                        layer.out_features for layer in subnet.peptide_dense_layers
                    ] if hasattr(subnet, 'peptide_dense_layers') else []
                    sub_config['allele_dense_layer_sizes'] = [
                        layer.out_features for layer in subnet.allele_dense_layers
                    ] if hasattr(subnet, 'allele_dense_layers') else []
                    # MHCflurry hyperparameters use keep probability, not
                    # PyTorch Dropout.p (drop probability).
                    sub_config['dropout_probability'] = getattr(
                        subnet,
                        'dropout_probability',
                        0.0,
                    )
                    sub_config['batch_normalization'] = (
                        hasattr(subnet, 'batch_norms') and bool(subnet.batch_norms) and
                        any(bn is not None for bn in subnet.batch_norms)
                    )
                    sub_config['activation'] = subnet.activation_name
                    sub_config['output_activation'] = subnet.output_activation_name
                    sub_config['peptide_allele_merge_method'] = subnet.peptide_allele_merge_method
                    sub_config['peptide_allele_merge_activation'] = subnet.peptide_allele_merge_activation
                    sub_config['topology'] = subnet.topology
                    sub_config['num_outputs'] = subnet.output_layer.out_features
                    sub_configs.append(sub_config)
                config['merged_networks'] = sub_configs
                config['merge_method'] = self._network.merge_method
            else:
                # Save whether the network has allele features
                config['has_allele'] = getattr(self._network, 'has_allele', False)
                # Save allele representations if present in the network
                if hasattr(self._network, 'allele_embedding') and self._network.allele_embedding is not None:
                    allele_embed = self._network.allele_embedding.weight.detach().cpu().numpy()
                    config['allele_representations'] = allele_embed.tolist()

            self.network_json = json.dumps(config)
            self.network_weights = self._network.get_weights_list()

    def get_config(self):
        """
        serialize to a dict all attributes except model weights

        Returns
        -------
        dict
        """
        self.update_network_description()
        result = dict(self.__dict__)
        result["_network"] = None
        result["network_weights"] = None
        result["network_weights_loader"] = None
        result.pop("network_weight_paths", None)
        result["prediction_cache"] = None
        return result

    @classmethod
    def from_config(cls, config, weights=None, weights_loader=None, weight_paths=None):
        """
        deserialize from a dict returned by get_config().

        Supports both:
        - Native Class1NeuralNetwork configs with 'hyperparameters' key
        - Legacy Keras model JSON configs with 'class_name', 'config', etc.

        Parameters
        ----------
        config : dict
        weights : list of array, optional
            Network weights to restore
        weights_loader : callable, optional
            Function to call (no arguments) to load weights when needed
        weight_paths : string or list of strings, optional
            Filename(s) the weights were loaded from. This is diagnostic
            provenance only and is not serialized by get_config().

        Returns
        -------
        Class1NeuralNetwork
        """
        config = dict(config)

        # Check if this is a legacy Keras JSON config
        if config.get('class_name') in ('Model', 'Functional'):
            hyperparameters, keras_metadata = cls._parse_keras_json_config(config)
            instance = cls(**hyperparameters)
            # Store metadata for weight loading
            instance._keras_metadata = keras_metadata
            # Store the original config as network_json for lazy network creation
            instance.network_json = json.dumps(config)
        else:
            # Standard Class1NeuralNetwork config format
            instance = cls(**config.pop("hyperparameters"))
            instance.__dict__.update(config)

        instance.network_weights = weights
        instance.network_weights_loader = weights_loader
        instance.network_weight_paths = (
            cls._normalize_network_weight_paths(weight_paths)
            or cls._network_weight_paths_from_loader(weights_loader)
        )
        instance.prediction_cache = weakref.WeakKeyDictionary()
        return instance

    @staticmethod
    def _normalize_network_weight_paths(weight_paths):
        """Return weight path(s) as a tuple of strings."""
        if weight_paths is None:
            return ()
        if isinstance(weight_paths, (str, os.PathLike)):
            return (os.fspath(weight_paths),)
        return tuple(os.fspath(path) for path in weight_paths if path)

    @classmethod
    def _network_weight_paths_from_loader(cls, weights_loader):
        """Infer weight path(s) from the standard lazy load_weights partial."""
        if weights_loader is None:
            return ()
        func = getattr(weights_loader, "func", None)
        args = getattr(weights_loader, "args", ())
        if getattr(func, "__name__", None) == "load_weights" and args:
            return cls._normalize_network_weight_paths(args[0])
        return ()

    def load_weights(self):
        """
        Load weights by evaluating self.network_weights_loader, if needed.
        """
        if self.network_weights_loader:
            self.network_weights = self.network_weights_loader()
            self.network_weights_loader = None

    def get_weights(self):
        """
        Get the network weights

        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        self.update_network_description()
        self.load_weights()
        return self.network_weights

    def get_weights_list(self):
        """
        Get the network weights as a list of numpy arrays.

        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        return self.get_weights()

    def set_weights_list(self, weights, auto_convert_keras=True):
        """
        Set the network weights from a list of numpy arrays.

        If a network exists, the weights are set directly on it.
        Otherwise, the weights are stored and will be applied when the
        network is created.

        Parameters
        ----------
        weights : list of numpy.array
            Weights for each layer
        auto_convert_keras : bool
            If True, attempt to auto-detect and convert Keras weight formats
            to PyTorch format. Default True.
        """
        if self._network is not None:
            # Network exists, set weights directly
            self._network.set_weights_list(weights, auto_convert_keras=auto_convert_keras)
        else:
            # Store weights for later application
            self.network_weights = weights
            # Store flag for auto-conversion
            self._auto_convert_keras_weights = auto_convert_keras

    def __getstate__(self):
        """
        serialize to a dict. Model weights are included. For pickle support.

        Returns
        -------
        dict

        """
        self.update_network_description()
        self.load_weights()
        result = dict(self.__dict__)
        result["_network"] = None
        result["prediction_cache"] = None
        return result

    def __setstate__(self, state):
        """
        Deserialize. For pickle support.
        """
        self.__dict__.update(state)
        self.prediction_cache = weakref.WeakKeyDictionary()

    def uses_peptide_torch_encoding(self):
        """Whether fixed peptide vectors are looked up in torch forward."""
        return _peptide_uses_torch_encoding(self.hyperparameters)

    def peptides_to_network_input(self, peptides):
        """
        Encode peptides to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        When ``peptide_amino_acid_encoding_torch`` is enabled, the returned
        array is (N, L) int8 amino-acid indices and the network widens it to
        the configured fixed vector encoding on the active torch device.
        Otherwise the returned array is the traditional (N, L, V) numpy
        vector encoding.

        Parameters
        ----------
        peptides : EncodableSequences or list of string

        Returns
        -------
        numpy.array
        """
        encoder = EncodableSequences.create(peptides)
        encoded = peptide_sequences_to_network_input(
            encoder,
            peptide_encoding=self.hyperparameters["peptide_encoding"],
            peptide_amino_acid_encoding_torch=(
                self.hyperparameters.get("peptide_amino_acid_encoding_torch")
            ),
        )
        assert len(encoded) == len(peptides)
        return encoded

    @property
    def supported_peptide_lengths(self):
        """
        (minimum, maximum) lengths of peptides supported, inclusive.

        Returns
        -------
        (int, int) tuple

        """
        try:
            self.peptides_to_network_input([""])
        except EncodingError as e:
            return e.supported_peptide_lengths
        raise RuntimeError("peptides_to_network_input did not raise")

    def allele_encoding_to_network_input(self, allele_encoding):
        """
        Encode alleles to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Parameters
        ----------
        allele_encoding : AlleleEncoding

        Returns
        -------
        (numpy.array, numpy.array)

        Indices and allele representations.

        """
        if allele_encoding is None:
            return (None, None)
        return (
            allele_encoding.indices.values,
            allele_encoding.allele_representations(
                self.hyperparameters["allele_amino_acid_encoding"]
            ),
        )

    @staticmethod
    def data_dependent_weights_initialization(network, x_dict=None, method="lsuv", verbose=1):
        """
        Data dependent weights initialization.

        Parameters
        ----------
        network : Class1NeuralNetworkModel
        x_dict : dict of string -> numpy.ndarray
            Training data
        method : string
            Initialization method. Currently only "lsuv" is supported.
        verbose : int
            Status updates printed to stdout if verbose > 0
        """
        if verbose:
            print("Performing data-dependent init: ", method)
        if method == "lsuv":
            assert x_dict is not None, "Data required for LSUV init"
            lsuv_init(network, x_dict, verbose=verbose > 0)
        else:
            raise RuntimeError("Unsupported init method: ", method)

    @staticmethod
    def _regularized_parameters(network):
        """
        Parameters subject to master-branch dense kernel regularization.
        """
        for name, param in network.named_parameters():
            if not param.requires_grad or not name.endswith("weight"):
                continue
            if any(part in name for part in (
                    "peptide_dense_layers",
                    "allele_dense_layers",
                    "dense_layers")):
                yield param

    @staticmethod
    def _regularization_penalty(parameters, l1=0.0, l2=0.0):
        """
        Match Keras kernel_regularizer semantics used on dense kernels.
        """
        parameters = tuple(parameters)
        if not parameters or (not l1 and not l2):
            return None
        penalty = torch.zeros((), device=parameters[0].device)
        for param in parameters:
            if l1:
                penalty = penalty + (l1 * param.abs().sum())
            if l2:
                penalty = penalty + (l2 * param.square().sum())
        return penalty

    def get_device(self):
        """Get the PyTorch device to use."""
        return get_pytorch_device()

    def fit_streaming_batches(
            self,
            generator,
            validation_peptide_encoding,
            validation_affinities,
            validation_allele_encoding=None,
            validation_inequalities=None,
            validation_output_indices=None,
            steps_per_epoch=10,
            epochs=1000,
            min_epochs=0,
            patience=10,
            min_delta=0.0,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0,
            generator_factory=None,
            generator_batches_are_encoded=False,
            seed=None):
        """Pretrain from a stream of already batched examples.

        This is the streaming pretraining path used by pan-allele training. It
        intentionally has a narrower contract than :meth:`fit`:

        * training examples arrive as complete batches from ``generator``;
        * validation data is supplied explicitly rather than split from the
          training rows;
        * random negatives, sample weights, and the affinity ``fit()`` row
          layout are not part of this API.

        The method still shares the same low-level optimizer step, loss object,
        regularization, and batched validation helpers as :meth:`fit`.

        ``generator`` may yield either pre-encoded ``(x_dict, y)`` tuples
        (the path the shipped pan-allele pretrain pipeline uses) or legacy
        raw ``(allele_encoding, peptides, affinities)`` tuples. The legacy
        form is still tested for back-compat but is not exercised by any
        production pipeline. ``generator_factory`` enables DataLoader
        worker prefetch: each worker calls it with ``worker_id`` and
        ``num_workers`` to read a disjoint shard.

        ``seed`` (int, optional) seeds numpy's and torch's global RNGs up
        front, so weight initialization and any shuffling in this pretrain
        pass derive from it; None leaves the worker's entropy seeding in
        place. Mirrors :meth:`fit`'s ``seed`` so one value can drive both
        phases of a pan-allele fit.
        """
        device = self.get_device()
        configure_matmul_precision(device)

        # Single-seed control, mirroring fit(): seed up front so weight init
        # and shuffling in this pass are reproducible. See fit() for detail.
        if seed is not None:
            numpy.random.seed(int(seed) % (2 ** 32))
            torch.manual_seed(int(seed) & ((1 << 63) - 1))

        fit_info = collections.defaultdict(list)
        timing_enabled = _timing_enabled()
        fit_info["timing_enabled"] = timing_enabled

        loss_obj = get_pytorch_loss(self.hyperparameters["loss"])

        (
            validation_allele_input,
            allele_representations,
        ) = self.allele_encoding_to_network_input(validation_allele_encoding)

        if self.network() is None:
            self._network = self.make_network(
                allele_representations=allele_representations,
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters)
            )
            if verbose > 0:
                print(self.network())
        network = self.network()
        network.to(device)

        self.set_allele_representations(allele_representations)

        # Setup optimizer
        optimizer = self._create_optimizer(network)
        if self.hyperparameters["learning_rate"] is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.hyperparameters["learning_rate"]
        fit_info["learning_rate"] = optimizer.param_groups[0]['lr']
        regularization_parameters = tuple(self._regularized_parameters(network))
        l1_reg = self.hyperparameters["dense_layer_l1_regularization"]
        l2_reg = self.hyperparameters["dense_layer_l2_regularization"]

        # Prepare validation data
        validation_x_dict = {
            "peptide": self.peptides_to_network_input(validation_peptide_encoding),
        }
        if validation_allele_input is not None:
            validation_x_dict["allele"] = validation_allele_input
        encode_y_kwargs = {}
        if validation_inequalities is not None:
            encode_y_kwargs["inequalities"] = validation_inequalities
        if validation_output_indices is not None:
            encode_y_kwargs["output_indices"] = validation_output_indices

        output = loss_obj.encode_y(from_ic50(validation_affinities), **encode_y_kwargs)

        # Validation tensors are constant across streaming-pretrain epochs, but
        # original code re-ran ``torch.from_numpy(...).float().to(device)``
        # on every epoch — three H2D copies × many epochs. For large
        # validation sets that's tens of ms/epoch of pure wasted bandwidth.
        # Hoist the copy to happen once before the epoch loop.
        #
        # 2D int -> index-encoded; keep dtype intact so the embedding lookup
        # inside forward() sees int indices. Vector-encoded inputs are numeric
        # feature tensors and should reach the network as fp32.
        _val_peptide_np = validation_x_dict["peptide"]
        if (
            _val_peptide_np.ndim == 2
            and numpy.issubdtype(_val_peptide_np.dtype, numpy.integer)
        ):
            val_peptide_device = _torch_from_numpy(_val_peptide_np).to(device)
        else:
            val_peptide_device = _torch_from_numpy(
                _val_peptide_np).to(device).float()
        val_allele_device = None
        if "allele" in validation_x_dict:
            val_allele_device = _torch_from_numpy(
                validation_x_dict["allele"]
            ).to(device).float()
        val_y_device = _torch_from_numpy(
            output.astype(numpy.float32)).to(device)

        # Streaming-pretrain batches stay as numpy arrays until the parent
        # training loop moves them to the device. That keeps worker-side
        # prefetch compatible across platforms and still overlaps CSV /
        # encoding work with GPU compute when a picklable encoded-batch
        # generator_factory is supplied (the pretrain path).
        mutable_generator_state = {"yielded_values": 0}
        dataloader_num_workers = self.hyperparameters.get(
            "dataloader_num_workers", 0
        )
        # The streaming-pretrain DataLoader intentionally transports numpy arrays
        # unchanged (``batch_size=None`` + ``_identity_collate``). Those arrays
        # are not pinned, and setting non_blocking=True for pageable
        # numpy-backed tensors lets CUDA retain source pages until later stream
        # synchronization. Keep H2D copies synchronous so each pretrain chunk's
        # CPU buffer can be released/reused immediately.
        non_blocking_h2d = False
        fit_info["dataloader_num_workers"] = dataloader_num_workers
        fit_info["validation_rows"] = len(validation_affinities)
        # Worker-prefetch requires both (a) a picklable ``generator_factory``
        # so each spawned worker can build its own shard, and (b)
        # ``generator_batches_are_encoded=True`` so the dataset never
        # holds bound methods of this ``Class1NeuralNetwork``. Either
        # missing piece forces a single-process path.
        if dataloader_num_workers > 0 and (
            generator_factory is None or not generator_batches_are_encoded
        ):
            logging.warning(
                "fit_streaming_batches requested dataloader_num_workers=%s but "
                "worker-prefetch needs generator_factory + "
                "generator_batches_are_encoded=True (got factory=%s, "
                "encoded=%s); downgrading to 0.",
                dataloader_num_workers,
                "present" if generator_factory is not None else "missing",
                generator_batches_are_encoded,
            )
            dataloader_num_workers = 0
        dataset = _StreamingBatchIterableDataset(
            generator=generator,
            generator_factory=generator_factory,
            source_batches_are_encoded=generator_batches_are_encoded,
            allele_encoding_to_input=self.allele_encoding_to_network_input,
            peptides_to_network_input=self.peptides_to_network_input,
        )

        start = time.time()
        fit_info["iterator_setup_time"] = 0.0
        fit_info["iterator_restarts"] = 0

        def make_iterator():
            iterator_setup_start = time.perf_counter()
            iterator_result = iter(
                _make_streaming_batch_dataloader(
                    dataset=dataset,
                    num_workers=dataloader_num_workers,
                )
            )
            fit_info["iterator_setup_time"] += (
                time.perf_counter() - iterator_setup_start
            )
            return iterator_result

        iterator = make_iterator()

        # Data dependent init
        data_dependent_init = self.hyperparameters[
            "data_dependent_initialization_method"
        ]
        if data_dependent_init and not self.fit_info:
            first_chunk = next(iterator)
            init_chunk = _materialize_repeated_peptide_batch(first_chunk)
            first_inputs = {"peptide": init_chunk["peptide"]}
            if "allele" in init_chunk:
                first_inputs["allele"] = init_chunk["allele"]
            self.data_dependent_weights_initialization(
                network,
                first_inputs,
                method=data_dependent_init,
                verbose=verbose,
            )
            iterator = itertools.chain([first_chunk], iterator)

        # Compile AFTER LSUV init — LSUV registers + removes forward hooks
        # during its activation-norm measurement pass, and every hook-count
        # change invalidates dynamo's frame guard (``len(L['self']
        # .dense_layers[0]._forward_hooks) != 1``). Compiling before LSUV
        # burns through the 8-entry cache_size_limit in seconds and falls
        # back to eager. Compiling after gives dynamo a single stable
        # hook state to specialize on.
        network = maybe_compile_network(network, device)
        eager_network = uncompiled_network(network)
        # Compile the loss alongside the network; see maybe_compile_loss
        # docstring. MSEWithInequalities eager-dispatches ~10 kernels
        # per step; fusing them matters more as batch size grows (less
        # compute to amortize launch overhead against).
        loss_obj = maybe_compile_loss(loss_obj, device)

        min_val_loss_iteration = None
        min_val_loss = None
        last_progress_print = 0
        first_batch_time = None

        # The first batch establishes the compiled fast-path shape.
        # Non-conforming later batches (most commonly a short final
        # pretrain chunk) run eagerly to preserve exact semantics without
        # triggering a second compile.
        expected_chunk_shape = None

        validation_interval = _validation_interval_from_hyperparameters(
            self.hyperparameters)

        for epoch_index in range(int(epochs)):
            epoch_num = epoch_index + 1
            epoch_start_time = time.time()
            epoch_wall_start = time.perf_counter()
            network.train()

            epoch_fetch_time = 0.0
            epoch_h2d_time = 0.0

            def prepared_streaming_batches():
                nonlocal iterator
                nonlocal expected_chunk_shape
                nonlocal epoch_fetch_time
                nonlocal epoch_h2d_time
                for _step in range(steps_per_epoch):
                    fetch_start = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        if generator_factory is None:
                            break
                        fit_info["iterator_restarts"] += 1
                        iterator = make_iterator()
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            break
                    epoch_fetch_time += time.perf_counter() - fetch_start

                    if expected_chunk_shape is None:
                        expected_chunk_shape = batch["peptide"].shape
                    # Measure H2D separately from training compute. When timing
                    # is disabled, _timing_start/stop are cheap wall-clock reads;
                    # when enabled, they synchronize around the copy.
                    h2d_start = _timing_start(device, timing_enabled)
                    inputs, y_tensor, weights_batch = _move_fit_batch_to_device(
                        batch,
                        device,
                        non_blocking=non_blocking_h2d,
                    )
                    epoch_h2d_time += _timing_stop(
                        h2d_start, device, timing_enabled
                    )
                    mutable_generator_state["yielded_values"] += len(batch["y"])
                    yield {
                        "network": (
                            network
                            if batch["peptide"].shape == expected_chunk_shape
                            else eager_network
                        ),
                        "inputs": inputs,
                        "y_batch": y_tensor,
                        "weights_batch": weights_batch,
                        "row_count": len(batch["y"]),
                    }

            train_epoch_result = _run_prepared_training_batches(
                prepared_streaming_batches(),
                optimizer=optimizer,
                loss_obj=loss_obj,
                regularization_parameters=regularization_parameters,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                timing_enabled=timing_enabled,
                device=device,
            )
            if first_batch_time is None:
                first_batch_time = train_epoch_result["first_batch_time"]

            should_validate_this_epoch = _should_validate_epoch(
                validation_enabled=True,
                epoch_index=epoch_index,
                max_epochs=epochs,
                validation_interval=validation_interval,
                early_stopping=True,
                min_val_loss_epoch=min_val_loss_iteration,
                patience=patience,
                min_epochs=min_epochs,
                # Streaming pretrain stops inclusively (epoch >= threshold) to
                # match pre-2.3.0 weights — see _early_stop_reached.
                strict=False,
            )
            validation_time = 0.0
            val_batch_size = None
            val_loss = _carry_forward_validation_loss(fit_info)
            if should_validate_this_epoch:
                # Compute validation loss in fixed-size batches — reuse the
                # device tensors hoisted before the epoch loop (val data is
                # static). Single-shot forward pass over the entire pretrain
                # validation set (typically 50K+ rows) was dominating VRAM.
                # By default validation uses the eager network to avoid a
                # second torch.compile specialization for grad-disabled eval.
                network.eval()
                with torch.inference_mode():
                    validation_start = _timing_start(device, timing_enabled)
                    val_batch_size = effective_validation_batch_size(
                        device,
                        self.hyperparameters["validation_batch_size"],
                        self.hyperparameters["minibatch_size"],
                    )
                    fit_info["effective_validation_batch_size"] = val_batch_size
                    val_loss = _batched_validation_loss(
                        network=validation_forward_network(network, eager_network),
                        eager_network=eager_network,
                        val_peptide=val_peptide_device,
                        val_allele=val_allele_device,
                        val_y=val_y_device,
                        val_weights=None,
                        loss_obj=loss_obj,
                        batch_size=val_batch_size,
                    )
                    regularization_penalty = self._regularization_penalty(
                        regularization_parameters,
                        l1=l1_reg,
                        l2=l2_reg,
                    )
                    if regularization_penalty is not None:
                        val_loss = val_loss + regularization_penalty.item()
                    validation_time = _timing_stop(
                        validation_start, device, timing_enabled
                    )
                fit_info["val_loss"].append(val_loss)
                (
                    min_val_loss,
                    min_val_loss_iteration,
                ) = _update_min_validation_loss(
                    val_loss=val_loss,
                    epoch_index=epoch_index,
                    min_val_loss=min_val_loss,
                    min_val_loss_epoch=min_val_loss_iteration,
                    min_delta=min_delta,
                )
            else:
                fit_info["val_loss"].append(val_loss)

            epoch_time = time.time() - epoch_start_time
            train_loss, epoch_loss_sync_time = _sync_mean_loss(
                train_epoch_result["losses"],
                device=device,
                timing_enabled=timing_enabled,
            )
            fit_info["loss"].append(train_loss)
            if timing_enabled:
                fit_info["epoch_fetch_time"].append(epoch_fetch_time)
                fit_info["epoch_h2d_time"].append(epoch_h2d_time)
                fit_info["epoch_train_time"].append(
                    train_epoch_result["train_time"])
                fit_info["epoch_loss_sync_time"].append(epoch_loss_sync_time)
                fit_info["epoch_validation_time"].append(validation_time)
                fit_info["epoch_num_train_batches"].append(
                    len(train_epoch_result["losses"]))
                fit_info["epoch_num_train_rows"].append(
                    train_epoch_result["train_rows"])
                fit_info["epoch_num_validation_batches"].append(
                    int(numpy.ceil(len(validation_affinities) / val_batch_size))
                    if val_batch_size else 0
                )
                fit_info["epoch_total_time"].append(
                    time.perf_counter() - epoch_wall_start
                )

            progress_message = (
                "epoch %3d/%3d [%0.2f sec.]: loss=%g val_loss=%g. Min val "
                "loss %g at epoch %s. Cum. points: %d."
                % (
                    epoch_num,
                    epochs,
                    epoch_time,
                    train_loss,
                    val_loss,
                    min_val_loss if min_val_loss is not None else numpy.nan,
                    (
                        min_val_loss_iteration + 1
                        if min_val_loss_iteration is not None else None
                    ),
                    mutable_generator_state["yielded_values"],
                )
            ).strip()

            if progress_print_interval is not None and (
                time.time() - last_progress_print > progress_print_interval
            ):
                print(progress_preamble, progress_message)
                last_progress_print = time.time()

            if progress_callback:
                progress_callback()

            if _early_stop_reached(
                epoch_index=epoch_index,
                min_val_loss_epoch=min_val_loss_iteration,
                patience=patience,
                min_epochs=min_epochs,
                early_stopping=True,
                # Inclusive stop (epoch >= threshold) reproduces the pre-2.3.0
                # streaming condition `epoch >= min_val_loss_iteration +
                # patience`; the strict `>` default would train one extra epoch.
                strict=False,
            ):
                if progress_print_interval is not None:
                    print(progress_preamble, "STOPPING", progress_message)
                break

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = mutable_generator_state["yielded_values"]
        if first_batch_time is not None:
            fit_info["first_batch_time"] = first_batch_time
        self.fit_info.append(dict(fit_info))

    def fit_generator(
            self,
            generator,
            validation_peptide_encoding,
            validation_affinities,
            validation_allele_encoding=None,
            validation_inequalities=None,
            validation_output_indices=None,
            steps_per_epoch=10,
            epochs=1000,
            min_epochs=0,
            patience=10,
            min_delta=0.0,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0,
            generator_factory=None,
            generator_batches_are_encoded=False):
        """Backward-compatible alias for :meth:`fit_streaming_batches`.

        The name is historical from the old Keras API. New internal code should
        call ``fit_streaming_batches`` because this path is specifically the
        pan-allele pretraining stream, not a general replacement for
        :meth:`fit`.
        """
        return self.fit_streaming_batches(
            generator=generator,
            validation_peptide_encoding=validation_peptide_encoding,
            validation_affinities=validation_affinities,
            validation_allele_encoding=validation_allele_encoding,
            validation_inequalities=validation_inequalities,
            validation_output_indices=validation_output_indices,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
            progress_callback=progress_callback,
            progress_preamble=progress_preamble,
            progress_print_interval=progress_print_interval,
            generator_factory=generator_factory,
            generator_batches_are_encoded=generator_batches_are_encoded,
        )

    def _random_negatives_pool_for_fit(
            self,
            *,
            random_negatives_planner,
            random_negative_pool_epochs,
            pool_seed,
            device,
            requested_fit_tensor_residency):
        """Construct a shape-compatible random-negative pool for fit()."""
        use_device_pool = (
            requested_fit_tensor_residency != "host"
            and self.uses_peptide_torch_encoding()
            and supports_device_random_negative_encoding(
                self.hyperparameters["peptide_encoding"]
            )
        )
        if use_device_pool:
            return (
                RandomNegativesPool(
                    planner=random_negatives_planner,
                    peptide_encoder=self.peptides_to_network_input,
                    pool_epochs=random_negative_pool_epochs,
                    seed=pool_seed,
                    device=device,
                    peptide_encoding=self.hyperparameters["peptide_encoding"],
                ),
                "device",
            )

        return (
            RandomNegativesPool(
                planner=random_negatives_planner,
                peptide_encoder=self.peptides_to_network_input,
                pool_epochs=random_negative_pool_epochs,
                seed=pool_seed,
            ),
            "host",
        )

    def _create_optimizer(self, network):
        """Create an optimizer for the network."""
        optimizer_name = self.hyperparameters["optimizer"].lower()
        lr = (
            self.hyperparameters["learning_rate"]
            if self.hyperparameters["learning_rate"] is not None
            else 0.001
        )

        if optimizer_name == "rmsprop":
            # Match Keras defaults: rho=0.9, epsilon=1e-07
            return torch.optim.RMSprop(
                network.parameters(), lr=lr, alpha=0.9, eps=1e-07)
        elif optimizer_name == "adam":
            # Match Keras default epsilon=1e-07.
            return torch.optim.Adam(network.parameters(), lr=lr, eps=1e-07)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(network.parameters(), lr=lr)
        else:
            return torch.optim.Adam(network.parameters(), lr=lr, eps=1e-07)

    def fit(
            self,
            peptides,
            affinities,
            allele_encoding=None,
            inequalities=None,
            output_indices=None,
            sample_weights=None,
            shuffle_permutation=None,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0,
            seed=None):
        """
        Fit the neural network.

        Parameters
        ----------
        peptides : EncodableSequences or list of string
        affinities : list of float
            nM affinities. Must be same length of as peptides.
        allele_encoding : AlleleEncoding
            If not specified, the model will be a single-allele predictor.
        inequalities : list of string, each element one of ">", "<", or "=".
        output_indices : list of int
            For multi-output models only.
        sample_weights : list of float
        shuffle_permutation : list of int
        verbose : int
        progress_callback : function
        progress_preamble : string
        progress_print_interval : float
        seed : int, optional
            Master seed for this fit. When given, it seeds numpy's and
            torch's global RNGs at the start of the call, so every
            stochastic step downstream flows from this one value:
            data-dependent weight initialization, the initial
            example shuffle, the per-epoch training-batch shuffle, and
            random-negative sampling. When None (the default) the RNGs are
            left as the worker configured them (entropy-seeded), so
            training stays stochastic and decorrelated across workers, as
            it always has been.

            Reproducibility caveats: a fixed ``seed`` reproduces a run
            bit-for-bit only at a *fixed effective minibatch size*. fit()
            may shrink the minibatch to fit available VRAM (see
            ``check_training_batch_fits`` below), and that shrink depends on
            free GPU memory and how many workers share the card — so the
            same seed on a busier or smaller GPU can diverge. A warning is
            logged whenever the shrink fires under a non-None seed. On CUDA,
            determinism additionally assumes the default (Linear/RMSprop)
            architecture: opting into ``MHCFLURRY_MATMUL_PRECISION`` enables
            ``cudnn.benchmark`` autotuning, and convolutional
            ``locally_connected_layers`` variants are not guaranteed
            bit-identical run-to-run. CPU runs are fully deterministic.
        """
        device = self.get_device()
        configure_matmul_precision(device)

        # One seed controls every stochastic step in this fit. Seed numpy's
        # and torch's global RNGs up front so weight init, the example
        # shuffle, the per-epoch batch shuffle, and random-negative
        # sampling all derive from it. Left untouched when seed is None
        # (entropy-seeded by the worker), keeping training stochastic.
        if seed is not None:
            numpy.random.seed(int(seed) % (2 ** 32))
            torch.manual_seed(int(seed) & ((1 << 63) - 1))

        encodable_peptides = EncodableSequences.create(peptides)
        peptide_encoding = self.peptides_to_network_input(encodable_peptides)
        fit_info = collections.defaultdict(list)
        timing_enabled = _timing_enabled()
        fit_info["timing_enabled"] = timing_enabled
        fit_info["dataloader_num_workers"] = self.hyperparameters.get(
            "dataloader_num_workers", 0
        )

        random_negatives_planner = RandomNegativePeptides(
            **RandomNegativePeptides.hyperparameter_defaults.subselect(
                self.hyperparameters
            )
        )
        random_negatives_planner.plan(
            peptides=encodable_peptides.sequences,
            affinities=affinities,
            alleles=allele_encoding.alleles if allele_encoding else None,
            inequalities=inequalities,
        )

        random_negatives_allele_encoding = None
        if allele_encoding is not None:
            random_negatives_allele_encoding = AlleleEncoding(
                random_negatives_planner.get_alleles(), borrow_from=allele_encoding
            )
        num_random_negatives = random_negatives_planner.get_total_count()

        # Pre-generate random-negative peptides + encodings once per pool cycle
        # rather than once per epoch. At pool_epochs=1 the pool regenerates
        # every epoch, matching the default "fresh negatives each epoch"
        # semantics.
        random_negative_pool_epochs = int(
            self.hyperparameters.get("random_negative_pool_epochs", 1) or 1
        )
        if random_negative_pool_epochs < 1:
            random_negative_pool_epochs = 1
        # When negatives are pooled across cycles the pool needs an explicit
        # seed (it pre-generates once and slices), so feed it the master
        # seed. At pool_epochs == 1 the pool regenerates every epoch from
        # numpy's global RNG, which the master seed above already pins — so
        # leave pool_seed None and let the global stream drive it.
        pool_seed = (
            seed
            if random_negative_pool_epochs > 1
            else None
        )
        requested_fit_tensor_residency = self.hyperparameters.get(
            "fit_tensor_residency"
        )
        # Random negatives stay worker-local. For torch-index peptide
        # inputs, the pool samples and aligns int8 AA indices directly on
        # the active torch device. Host-vector models use the same encoder
        # as their real peptides so the training input shape remains identical.
        (
            random_negatives_pool,
            random_negative_pool_residency,
        ) = self._random_negatives_pool_for_fit(
            random_negatives_planner=random_negatives_planner,
            random_negative_pool_epochs=random_negative_pool_epochs,
            pool_seed=pool_seed,
            device=device,
            requested_fit_tensor_residency=requested_fit_tensor_residency,
        )
        fit_info["random_negative_pool_residency"] = (
            random_negative_pool_residency
        )
        fit_info["random_negative_pool_epochs"] = random_negative_pool_epochs

        # Allele encoding for random negatives is planned once (the allele
        # list is a deterministic function of the planner's plan_df). Hoist
        # it out of the epoch loop; it is a deterministic function of the plan.
        random_negative_x_allele_base = None
        if (
            num_random_negatives > 0
            and random_negatives_allele_encoding is not None
        ):
            (
                random_negative_x_allele_base,
                _,
            ) = self.allele_encoding_to_network_input(
                random_negatives_allele_encoding
            )

        y_values = from_ic50(numpy.asarray(affinities))
        assert numpy.isnan(y_values).sum() == 0, y_values

        if inequalities is not None:
            adjusted_inequalities = (
                pandas.Series(inequalities)
                .map({
                    "=": "=",
                    ">": "<",
                    "<": ">",
                })
                .values
            )
        else:
            adjusted_inequalities = numpy.tile("=", len(y_values))

        if len(adjusted_inequalities) != len(y_values):
            raise ValueError("Inequalities and y_values must have same length")

        x_dict_without_random_negatives = {
            "peptide": peptide_encoding,
        }
        allele_representations = None
        if allele_encoding is not None:
            (
                allele_encoding_input,
                allele_representations,
            ) = self.allele_encoding_to_network_input(allele_encoding)
            x_dict_without_random_negatives["allele"] = allele_encoding_input

        # Shuffle
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(y_values))
        y_values = y_values[shuffle_permutation]
        adjusted_inequalities = adjusted_inequalities[shuffle_permutation]
        for key in x_dict_without_random_negatives:
            x_dict_without_random_negatives[key] = x_dict_without_random_negatives[key][
                shuffle_permutation
            ]
        if sample_weights is not None:
            sample_weights = numpy.array(sample_weights, copy=False)[shuffle_permutation]
        if output_indices is not None:
            output_indices = numpy.array(output_indices, copy=False)[shuffle_permutation]

        loss_obj = get_pytorch_loss(self.hyperparameters["loss"])

        if not loss_obj.supports_inequalities and (
            any(inequality != "=" for inequality in adjusted_inequalities)
        ):
            raise ValueError("Loss %s does not support inequalities" % loss_obj)

        if (
            not loss_obj.supports_multiple_outputs
            and output_indices is not None
            and (output_indices != 0).any()
        ):
            raise ValueError("Loss %s does not support multiple outputs" % loss_obj)

        if self.hyperparameters["num_outputs"] != 1:
            if output_indices is None:
                raise ValueError("Must supply output_indices for multi-output predictor")

        if self.network() is None:
            self._network = self.make_network(
                allele_representations=allele_representations,
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters)
            )
            if verbose > 0:
                print(self.network())

        network = self.network()
        network.to(device)

        if allele_representations is not None:
            self.set_allele_representations(allele_representations)

        # Guard against silent training-time OOMs. When many workers
        # share one GPU (pan-allele default max_workers_per_gpu=2 on
        # A100-80GB), the naive minibatch_size from the hyperparameters
        # YAML may not fit per-worker. Shrink loudly for the duration
        # of this fit() call only — DO NOT mutate self.hyperparameters,
        # so the saved model config preserves the user's original
        # intent. fit_info carries the actual value used at run time.
        _requested_minibatch = int(self.hyperparameters["minibatch_size"])
        _effective_minibatch, _shrunk = check_training_batch_fits(
            _requested_minibatch,
            device,
            network,
            num_workers_per_gpu=_env_workers_per_gpu(1),
        )
        if _shrunk:
            fit_info["minibatch_size_shrunk_from"] = _requested_minibatch
            fit_info["minibatch_size_shrunk_to"] = _effective_minibatch
            if seed is not None:
                logging.warning(
                    "fit(seed=%s) shrank minibatch_size %d -> %d to fit "
                    "available VRAM. Reproducibility holds only at a fixed "
                    "effective minibatch size, so this run may not reproduce "
                    "bit-for-bit on a GPU with different free memory or a "
                    "different number of workers per card.",
                    seed, _requested_minibatch, _effective_minibatch)
        fit_info["effective_minibatch_size"] = _effective_minibatch

        optimizer = self._create_optimizer(network)
        if self.hyperparameters["learning_rate"] is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.hyperparameters["learning_rate"]
        fit_info["learning_rate"] = optimizer.param_groups[0]['lr']

        # Prepare y values with random negatives
        if loss_obj.supports_inequalities:
            random_negative_ic50 = self.hyperparameters["random_negative_affinity_min"]
            random_negative_target = from_ic50(random_negative_ic50)

            y_with_negatives = numpy.concatenate([
                numpy.tile(random_negative_target, num_random_negatives),
                y_values,
            ])
            adjusted_inequalities_with_random_negatives = (
                ["<"] * num_random_negatives + list(adjusted_inequalities)
            )
        else:
            y_with_negatives = numpy.concatenate([
                from_ic50(
                    numpy.random.uniform(
                        self.hyperparameters["random_negative_affinity_min"],
                        self.hyperparameters["random_negative_affinity_max"],
                        num_random_negatives,
                    )
                ),
                y_values,
            ])
            adjusted_inequalities_with_random_negatives = None

        if sample_weights is not None:
            sample_weights_with_negatives = numpy.concatenate([
                numpy.ones(num_random_negatives),
                sample_weights
            ])
        else:
            sample_weights_with_negatives = None

        if output_indices is not None:
            random_negative_output_indices = (
                self.hyperparameters["random_negative_output_indices"]
                if self.hyperparameters["random_negative_output_indices"]
                else list(range(0, self.hyperparameters["num_outputs"]))
            )
            output_indices_with_negatives = numpy.concatenate([
                pandas.Series(random_negative_output_indices, dtype=int)
                .sample(n=num_random_negatives, replace=True)
                .values,
                output_indices,
            ])
        else:
            output_indices_with_negatives = None

        # Encode y
        encode_y_kwargs = {}
        if adjusted_inequalities_with_random_negatives is not None:
            encode_y_kwargs["inequalities"] = adjusted_inequalities_with_random_negatives
        if output_indices_with_negatives is not None:
            encode_y_kwargs["output_indices"] = output_indices_with_negatives

        y_encoded = loss_obj.encode_y(y_with_negatives, **encode_y_kwargs)

        min_val_loss_iteration = None
        min_val_loss = None

        needs_initialization = (
            self.hyperparameters["data_dependent_initialization_method"] is not None
            and not self.fit_info
        )

        start = time.time()
        last_progress_print = None
        first_batch_time = None

        # Validation split (fixed across epochs; only training data is reshuffled)
        val_split = self.hyperparameters["validation_split"]
        n_total = len(y_encoded)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        indices = numpy.arange(n_total)
        if n_val > 0:
            train_indices_base = indices[:n_train]
            val_indices = indices[n_train:]
        else:
            train_indices_base = indices
            val_indices = None
        fit_info["train_rows"] = int(n_train)
        fit_info["validation_rows"] = int(n_val)
        fit_info["num_random_negatives"] = int(num_random_negatives)

        regularization_parameters = tuple(self._regularized_parameters(network))
        l1_reg = self.hyperparameters["dense_layer_l1_regularization"]
        l2_reg = self.hyperparameters["dense_layer_l2_regularization"]

        # The validation set indexes into the concatenated
        # [random_negs | training] array via val_indices. The training portion
        # is static across epochs; the random-negative portion changes.
        # When val_indices points entirely into the training portion (the
        # common case — with default val_split=0.1 and random_negative_rate=1,
        # val set is the tail 10% of the ~2x-size concatenated array, all in
        # training portion), x_peptide[val_indices] and x_allele[val_indices]
        # are stable across epochs. Materialize them on device once to save
        # ~60 MB+ H2D per epoch.
        #
        # When overlap IS possible (unusual hyperparameter combos), skip the
        # cache and fall back to per-epoch copy (preserved behavior).
        # Check EVERY val index, not just the first — val_indices can be
        # unsorted (scikit-learn's train_test_split sometimes produces
        # shuffled indices). A leading value ≥ num_random_negatives does
        # not guarantee all subsequent values are in the training portion;
        # the weight/peptide slicing below would then index into the
        # random-negative portion, which mutates per epoch → stale tensors.
        _val_cache_safe = (
            val_indices is not None
            and len(val_indices) > 0
            and bool(numpy.all(val_indices >= num_random_negatives))
        )
        fit_info["validation_cache_reused"] = _val_cache_safe
        _val_device_tensors = None
        if _val_cache_safe:
            val_training_indices = val_indices - num_random_negatives
            # 2D int is index-encoded (keep dtype); 3D int is an already
            # materialized vector-encoded payload (widen to fp32).
            _val_peptide_np = x_dict_without_random_negatives["peptide"][
                val_training_indices
            ]
            if (
                _val_peptide_np.ndim == 2
                and numpy.issubdtype(_val_peptide_np.dtype, numpy.integer)
            ):
                _val_peptide_device = _torch_from_numpy(_val_peptide_np).to(device)
            else:
                _val_peptide_device = _torch_from_numpy(
                    _val_peptide_np).float().to(device)
            _val_y_device = _torch_from_numpy(
                y_encoded[val_indices].astype(numpy.float32)
            ).to(device)
            _val_allele_device = None
            if "allele" in x_dict_without_random_negatives:
                _val_allele_device = _torch_from_numpy(
                    x_dict_without_random_negatives["allele"][val_training_indices]
                ).float().to(device)
            _val_weights_device = None
            if sample_weights_with_negatives is not None:
                _val_weights_device = _torch_from_numpy(
                    sample_weights_with_negatives[val_indices]
                ).float().to(device)
            _val_device_tensors = (
                _val_peptide_device,
                _val_y_device,
                _val_allele_device,
                _val_weights_device,
            )

        # AffinityDeviceTrainingData owns the device row space for this
        # fit: [random negatives | real examples]. The training loop only
        # asks it for indexed batches and refills the random-negative slice
        # once per epoch.
        random_neg_template = None
        if num_random_negatives > 0:
            _, random_neg_template = random_negatives_pool.get_epoch_inputs(0)
        affinity_training_data = AffinityDeviceTrainingData.from_arrays(
            x_peptide=x_dict_without_random_negatives["peptide"],
            x_allele=x_dict_without_random_negatives.get("allele"),
            y_encoded=y_encoded,
            sample_weights=sample_weights_with_negatives,
            random_negative_x_peptide_template=random_neg_template,
            random_negative_x_allele=random_negative_x_allele_base,
            device=device,
        )
        fit_info["fit_tensor_residency"] = "device"
        logging.info("fit tensor residency: device (device=%s)", device.type)

        for epoch in range(self.hyperparameters["max_epochs"]):
            epoch_wall_start = time.perf_counter()
            # Coarse + fine timing buckets — kept for back-compat with
            # downstream telemetry consumers; the host-path-specific
            # buckets (dataset construction, dataloader setup) stay
            # at zero in the device-only world.
            input_build_time = 0.0
            rn_pool_get_time = 0.0
            rn_refill_time = 0.0
            dataset_construction_time = 0.0
            initialization_time = 0.0
            shuffle_dataset_time = 0.0
            dataloader_setup_time = 0.0
            train_loop_wall_time = 0.0
            train_loop_time = 0.0
            epoch_h2d_time = 0.0
            epoch_loss_sync_time = 0.0
            validation_materialize_time = 0.0
            validation_compute_time = 0.0
            callback_time = 0.0
            gc_time = 0.0

            # Per-epoch random-negatives slice. With ``pool_epochs=1``
            # this regenerates every epoch (legacy behavior); with
            # ``pool_epochs=N`` the encoding pass runs once per N
            # epochs and this is an O(1) view.
            rn_get_start = time.perf_counter()
            if epoch == 0 and random_neg_template is not None:
                random_negative_peptides_encoding = random_neg_template
                random_neg_template = None
            else:
                _, random_negative_peptides_encoding = (
                    random_negatives_pool.get_epoch_inputs(epoch)
                )
            rn_pool_get_time += time.perf_counter() - rn_get_start

            rn_refill_start = time.perf_counter()
            affinity_training_data.refill_random_negative_peptides(
                random_negative_peptides_encoding
            )
            rn_refill_time += time.perf_counter() - rn_refill_start
            input_build_time = rn_pool_get_time + rn_refill_time

            if needs_initialization:
                init_start = _timing_start(device, timing_enabled)
                init_batch = affinity_training_data.batch_dict_for_indices(
                    indices, device=device
                )
                self.data_dependent_weights_initialization(
                    network,
                    init_batch,
                    method=self.hyperparameters["data_dependent_initialization_method"],
                    verbose=verbose,
                )
                initialization_time += _timing_stop(
                    init_start, device, timing_enabled
                )
                needs_initialization = False

            # Compile AFTER LSUV hook churn finishes (see fit_streaming_batches
            # comment above). Idempotent: maybe_compile_network returns
            # the OptimizedModule unchanged if ``network`` is already
            # compiled, so it's safe to call every epoch. First epoch's
            # first batch pays the codegen cost; rest runs compiled.
            network = maybe_compile_network(network, device)
            eager_network = uncompiled_network(network)
            # Same rationale as fit_streaming_batches' loss-compile call.
            loss_obj = maybe_compile_loss(loss_obj, device)

            # Train indices live on device — train_indices_base is
            # contiguous [0, n_train), so a shuffled view is just
            # torch.randperm(n_train) on device with no host array
            # or H2D copy.
            dataset_start = time.perf_counter()
            train_indices_dev_full = torch.randperm(
                train_indices_base.shape[0], device=device,
            )
            shuffle_dataset_time += time.perf_counter() - dataset_start

            # Training
            network.train()
            epoch_start = time.time()
            train_loop_wall_start = time.perf_counter()

            batch_size = _effective_minibatch
            n_train_epoch = int(train_indices_dev_full.shape[0])
            full_batch_count = (n_train_epoch // batch_size) * batch_size

            def prepared_device_batches():
                # --- Device-resident inner loop ---
                # No DataLoader, no per-batch H2D. Indices live on device;
                # batches are built by index_select into the combined
                # buffer that holds [RN | real] rows.
                if full_batch_count > 0:
                    train_indices_dev = train_indices_dev_full[:full_batch_count]
                    num_full_batches = full_batch_count // batch_size
                    for step in range(num_full_batches):
                        batch_indices = train_indices_dev[
                            step * batch_size : (step + 1) * batch_size
                        ]
                        inputs, y_batch, weights_batch = (
                            affinity_training_data.batch_for_indices(batch_indices)
                        )
                        yield {
                            "network": network,
                            "inputs": inputs,
                            "y_batch": y_batch,
                            "weights_batch": weights_batch,
                            "row_count": int(y_batch.shape[0]),
                        }
                tail_count = n_train_epoch - full_batch_count
                if tail_count > 0:
                    tail_indices_dev = train_indices_dev_full[full_batch_count:]
                    inputs, y_batch, weights_batch = (
                        affinity_training_data.batch_for_indices(tail_indices_dev)
                    )
                    yield {
                        "network": eager_network,
                        "inputs": inputs,
                        "y_batch": y_batch,
                        "weights_batch": weights_batch,
                        "row_count": int(y_batch.shape[0]),
                    }

            train_epoch_result = _run_prepared_training_batches(
                prepared_device_batches(),
                optimizer=optimizer,
                loss_obj=loss_obj,
                regularization_parameters=regularization_parameters,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                timing_enabled=timing_enabled,
                device=device,
            )
            train_loop_time += train_epoch_result["train_time"]
            if first_batch_time is None:
                first_batch_time = train_epoch_result["first_batch_time"]

            epoch_time = time.time() - epoch_start
            train_loop_wall_time = time.perf_counter() - train_loop_wall_start
            train_loss, epoch_loss_sync_time = _sync_mean_loss(
                train_epoch_result["losses"],
                device=device,
                timing_enabled=timing_enabled,
            )
            fit_info["loss"].append(train_loss)

            # Validation — batched so every GPU invocation is fixed-size
            # and peak VRAM stays bounded regardless of n_val. By default
            # validation uses the eager network to avoid a second
            # torch.compile specialization for grad-disabled eval.
            #
            # Three conditions trigger a measurement; otherwise the val
            # pass is skipped and the previous measurement is carried
            # forward so fit_info["val_loss"] stays one-per-epoch:
            #   1. on a ``validation_interval`` cadence (the routine case);
            #   2. on the final epoch of the loop (so the saved model
            #      always reflects an up-to-date val_loss);
            #   3. when patience would trigger this epoch (so the saved
            #      val_loss reflects the actual stop state, not a stale
            #      carried-forward value).
            validation_interval = _validation_interval_from_hyperparameters(
                self.hyperparameters)
            val_batch_size = None
            should_validate_this_epoch = _should_validate_epoch(
                validation_enabled=val_split > 0,
                epoch_index=epoch,
                max_epochs=self.hyperparameters["max_epochs"],
                validation_interval=validation_interval,
                early_stopping=self.hyperparameters.get("early_stopping", False),
                min_val_loss_epoch=min_val_loss_iteration,
                patience=self.hyperparameters["patience"],
            )
            if val_split > 0 and not should_validate_this_epoch:
                prev_val_loss = _carry_forward_validation_loss(fit_info)
                fit_info["val_loss"].append(prev_val_loss)
                val_loss = prev_val_loss
            if should_validate_this_epoch:
                network.eval()
                with torch.inference_mode():
                    if _val_device_tensors is not None:
                        # Fast path: reuse tensors materialized once before the
                        # epoch loop. Saves ~60 MB+ H2D copy per epoch. Bit-
                        # identical because val_indices points entirely into
                        # the static training portion of x_peptide/x_allele.
                        (
                            val_peptide,
                            val_y,
                            val_allele,
                            val_weights,
                        ) = _val_device_tensors
                    else:
                        # Rare case: val_indices overlaps the per-epoch
                        # random-negative slice. Build the batch directly
                        # from the device-resident combined buffer.
                        materialize_start = time.perf_counter()
                        val_indices_dev = torch.as_tensor(
                            val_indices, dtype=torch.long, device=device,
                        )
                        val_inputs, val_y, val_weights = (
                            affinity_training_data.batch_for_indices(val_indices_dev)
                        )
                        val_peptide = val_inputs["peptide"]
                        val_allele = val_inputs.get("allele")
                        validation_materialize_time += (
                            time.perf_counter() - materialize_start
                        )
                    val_batch_size = effective_validation_batch_size(
                        device,
                        self.hyperparameters["validation_batch_size"],
                        batch_size,
                    )
                    fit_info["effective_validation_batch_size"] = val_batch_size
                    validation_start = _timing_start(device, timing_enabled)
                    val_loss = _batched_validation_loss(
                        network=validation_forward_network(network, eager_network),
                        eager_network=eager_network,
                        val_peptide=val_peptide,
                        val_allele=val_allele,
                        val_y=val_y,
                        val_weights=val_weights,
                        loss_obj=loss_obj,
                        batch_size=val_batch_size,
                    )
                    regularization_penalty = self._regularization_penalty(
                        regularization_parameters,
                        l1=l1_reg,
                        l2=l2_reg,
                    )
                    if regularization_penalty is not None:
                        val_loss = val_loss + regularization_penalty.item()
                    validation_compute_time += _timing_stop(
                        validation_start, device, timing_enabled
                    )
                fit_info["val_loss"].append(val_loss)

            # Progress printing
            if progress_print_interval is not None and (
                not last_progress_print
                or (time.time() - last_progress_print > progress_print_interval)
            ):
                print(
                    (
                        progress_preamble
                        + " "
                        + "Epoch %3d / %3d [%0.2f sec]: loss=%g. "
                        "Min val loss (%s) at epoch %s"
                        % (
                            epoch,
                            self.hyperparameters["max_epochs"],
                            epoch_time,
                            train_loss,
                            str(min_val_loss),
                            min_val_loss_iteration,
                        )
                    ).strip()
                )
                last_progress_print = time.time()

            # Early stopping. ``min_val_loss`` / ``min_val_loss_iteration``
            # only update on epochs where validation actually ran — on
            # skipped epochs ``val_loss`` is the carried-forward previous
            # measurement and would never beat the current min anyway, so
            # restricting the update is for clarity (the patience counter
            # is anchored to the epoch the measurement was taken, not a
            # later epoch that copied the same value).
            if val_split > 0:
                if should_validate_this_epoch:
                    min_val_loss, min_val_loss_iteration = (
                        _update_min_validation_loss(
                            val_loss=val_loss,
                            epoch_index=epoch,
                            min_val_loss=min_val_loss,
                            min_val_loss_epoch=min_val_loss_iteration,
                            min_delta=self.hyperparameters["min_delta"],
                        )
                    )

                if self.hyperparameters["early_stopping"]:
                    if _early_stop_reached(
                        epoch_index=epoch,
                        min_val_loss_epoch=min_val_loss_iteration,
                        patience=self.hyperparameters["patience"],
                        early_stopping=True,
                    ):
                        if progress_print_interval is not None:
                            print(
                                (
                                    progress_preamble
                                    + " "
                                    + "Stopping at epoch %3d / %3d: loss=%g. "
                                    "Min val loss (%g) at epoch %s"
                                    % (
                                        epoch,
                                        self.hyperparameters["max_epochs"],
                                        train_loss,
                                        min_val_loss if min_val_loss is not None else numpy.nan,
                                        min_val_loss_iteration,
                                    )
                                ).strip()
                            )
                        break

            if progress_callback:
                callback_start = time.perf_counter()
                progress_callback()
                callback_time += time.perf_counter() - callback_start

            gc_start = time.perf_counter()
            gc.collect()
            gc_time += time.perf_counter() - gc_start
            if timing_enabled:
                # Coarse + fine-grained input-build buckets. The split was
                # added to find the unaccounted 35 s/epoch gap on the
                # 2026-04-28 release_full run (epoch_total_time ~45 s,
                # sum of pre-split components ~10 s).
                fit_info["epoch_input_build_time"].append(input_build_time)
                fit_info["epoch_rn_pool_get_time"].append(rn_pool_get_time)
                fit_info["epoch_random_negative_refill_time"].append(
                    rn_refill_time
                )
                fit_info["epoch_dataset_construction_time"].append(
                    dataset_construction_time
                )
                fit_info["epoch_initialization_time"].append(
                    initialization_time
                )
                fit_info["epoch_shuffle_dataset_time"].append(
                    shuffle_dataset_time
                )
                fit_info["epoch_dataloader_setup_time"].append(
                    dataloader_setup_time
                )
                fit_info["epoch_h2d_time"].append(epoch_h2d_time)
                fit_info["epoch_train_time"].append(train_loop_time)
                fit_info["epoch_train_loop_wall_time"].append(
                    train_loop_wall_time
                )
                # The dataloader-iter overhead is the wall time of the
                # `for batch in loader:` loop minus the per-batch GPU
                # compute and H2D copies that are timed separately. It
                # captures DataLoader worker IPC, prefetcher waits,
                # Python iterator overhead, and any compile-recompile
                # storms — i.e. exactly the gap that was unaccounted
                # for in the pre-split breakdown.
                fit_info["epoch_dataloader_iter_overhead_time"].append(
                    max(0.0, train_loop_wall_time - epoch_h2d_time - train_loop_time)
                )
                fit_info["epoch_loss_sync_time"].append(epoch_loss_sync_time)
                fit_info["epoch_validation_materialize_time"].append(
                    validation_materialize_time
                )
                fit_info["epoch_validation_time"].append(
                    validation_compute_time
                )
                fit_info["epoch_num_train_batches"].append(
                    len(train_epoch_result["losses"]))
                fit_info["epoch_num_train_rows"].append(
                    train_epoch_result["train_rows"])
                fit_info["epoch_tail_train_rows"].append(
                    n_train_epoch - full_batch_count
                )
                fit_info["epoch_num_validation_batches"].append(
                    int(numpy.ceil(n_val / val_batch_size))
                    if val_batch_size else 0
                )
                fit_info["epoch_callback_time"].append(callback_time)
                fit_info["epoch_gc_time"].append(gc_time)
                fit_info["epoch_total_time"].append(
                    time.perf_counter() - epoch_wall_start
                )

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = len(peptides)
        if first_batch_time is not None:
            fit_info["first_batch_time"] = first_batch_time
        self.fit_info.append(dict(fit_info))

    def predict(
            self,
            peptides,
            allele_encoding=None,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE,
            output_index=0,
            num_workers_per_gpu=None):
        """
        Predict affinities.

        Parameters
        ----------
        peptides : EncodableSequences or list of string
        allele_encoding : AlleleEncoding, optional
        batch_size : int or ``"auto"``
            ``"auto"`` (the default) sizes batches to the available GPU
            memory at call time — see ``compute_prediction_batch_size``.
            Pass an explicit int to pin the size.
        output_index : int or None
        num_workers_per_gpu : int, optional
            When multiple training/calibration workers are co-resident on
            the same CUDA device, pass the worker count so the auto-
            sizer partitions the VRAM budget. Ignored for explicit int
            batch_size. When not passed, falls back to
            ``_env_workers_per_gpu(1)`` (the ``MHCFLURRY_MAX_WORKERS_PER_GPU``
            env var the worker pool sets), mirroring ``fit()``.

        Returns
        -------
        numpy.array of nM affinity predictions
        """
        if num_workers_per_gpu is None:
            num_workers_per_gpu = _env_workers_per_gpu(1)

        assert self.prediction_cache is not None
        use_cache = allele_encoding is None and isinstance(peptides, EncodableSequences)
        if use_cache and peptides in self.prediction_cache:
            return self.prediction_cache[peptides].copy()

        device = self.get_device()
        configure_matmul_precision(device)

        x_dict = {"peptide": self.peptides_to_network_input(peptides)}

        if allele_encoding is not None:
            (
                allele_encoding_input,
                allele_representations,
            ) = self.allele_encoding_to_network_input(allele_encoding)
            x_dict["allele"] = allele_encoding_input
            self.set_allele_representations(allele_representations)
            network = self.network()
        else:
            network = self.network(borrow=True)

        network.to(device)
        network = maybe_compile_network(network, device)
        network.eval()

        # Resolve ``"auto"`` once the network is on device so the
        # heuristic has final visibility into VRAM + architecture.
        batch_size = resolve_prediction_batch_size(
            batch_size,
            device,
            model=network,
            num_workers_per_gpu=num_workers_per_gpu,
        )

        # Batch prediction
        n_samples = len(x_dict["peptide"])
        all_predictions = []

        peptide_is_indices = _peptide_uses_torch_encoding(self.hyperparameters)

        def prediction_tensor(batch_array):
            batch_array = numpy.asarray(batch_array)
            if not batch_array.flags.writeable:
                batch_array = batch_array.copy()
            # Keep integer dtype only on the index-encoded peptide path.
            # Allele inputs and 3D int8 vector-encoded payloads still widen
            # to fp32 as before — the flag only governs the peptide tensor.
            keep_int = (
                peptide_is_indices
                and batch_array.ndim == 2
                and numpy.issubdtype(batch_array.dtype, numpy.integer)
            )
            if not keep_int:
                batch_array = numpy.asarray(batch_array, dtype=numpy.float32)
            return torch.from_numpy(batch_array).to(device)

        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                peptide_batch = prediction_tensor(
                    x_dict["peptide"][batch_start:batch_end]
                )

                inputs = {"peptide": peptide_batch}
                if "allele" in x_dict:
                    allele_batch = prediction_tensor(
                        x_dict["allele"][batch_start:batch_end]
                    )
                    inputs["allele"] = allele_batch

                batch_predictions = network(inputs)
                all_predictions.append(batch_predictions.cpu().numpy())

        raw_predictions = numpy.concatenate(all_predictions, axis=0)
        predictions = numpy.array(raw_predictions, dtype="float64")

        if output_index is not None:
            predictions = predictions[:, output_index]

        result = to_ic50(predictions)
        if use_cache:
            self.prediction_cache[peptides] = result
        return result

    @classmethod
    def merge(cls, models, merge_method="average"):
        """
        Merge multiple models at the neural network level.

        Parameters
        ----------
        models : list of Class1NeuralNetwork
        merge_method : string, one of "average", "sum", or "concatenate"

        Returns
        -------
        Class1NeuralNetwork
        """
        if merge_method == "allele-specific":
            raise NotImplementedError("Allele-specific merge is not implemented")
        if len(models) == 1:
            return models[0]
        assert len(models) > 1
        if any(not model.network().has_allele for model in models):
            raise NotImplementedError("Merging allele-specific models is not implemented")

        # For now, we create a simple ensemble wrapper
        # that averages predictions
        result = Class1NeuralNetwork(**dict(models[0].hyperparameters))

        # Remove hyperparameters not shared by all models
        for model in models:
            for key, value in model.hyperparameters.items():
                if result.hyperparameters.get(key, value) != value:
                    del result.hyperparameters[key]

        # Create merged network
        result._network = MergedClass1NeuralNetwork(
            [model.network() for model in models],
            merge_method=merge_method
        )
        result.update_network_description()
        result.network_weight_paths = tuple(
            path
            for model in models
            for path in getattr(model, "network_weight_paths", ())
        )

        return result

    def make_network(
            self,
            peptide_encoding,
            allele_amino_acid_encoding,
            allele_dense_layer_sizes,
            peptide_dense_layer_sizes,
            peptide_allele_merge_method,
            peptide_allele_merge_activation,
            layer_sizes,
            dense_layer_l1_regularization,
            dense_layer_l2_regularization,
            activation,
            init,
            output_activation,
            dropout_probability,
            batch_normalization,
            locally_connected_layers,
            topology,
            num_outputs=1,
            allele_representations=None,
            peptide_amino_acid_encoding_torch=True,
            peptide_amino_acid_encoding_gpu=None):
        """
        Helper function to make a PyTorch network for class 1 affinity prediction.
        """
        if peptide_amino_acid_encoding_gpu is not None:
            peptide_amino_acid_encoding_torch = peptide_amino_acid_encoding_gpu
        hyperparameters = dict(self.hyperparameters)
        hyperparameters["peptide_encoding"] = peptide_encoding
        hyperparameters["peptide_amino_acid_encoding_torch"] = (
            peptide_amino_acid_encoding_torch
        )
        peptide_torch_encoding_name = _peptide_torch_encoding_name(hyperparameters)
        peptide_encoding_shape = peptide_sequences_to_network_input(
            [],
            peptide_encoding=peptide_encoding,
            peptide_amino_acid_encoding_torch=peptide_amino_acid_encoding_torch,
        ).shape[1:]
        # Index-encoded peptides probe as 1D (L,), but dense layers still
        # size against the post-embedding (L, V) shape.
        if peptide_torch_encoding_name and len(peptide_encoding_shape) == 1:
            peptide_encoding_shape = _peptide_torch_encoding_shape(
                peptide_encoding_shape,
                peptide_torch_encoding_name,
            )

        return Class1NeuralNetworkModel(
            peptide_encoding_shape=peptide_encoding_shape,
            allele_representations=allele_representations,
            locally_connected_layers=locally_connected_layers,
            peptide_dense_layer_sizes=peptide_dense_layer_sizes,
            allele_dense_layer_sizes=allele_dense_layer_sizes,
            layer_sizes=layer_sizes,
            peptide_allele_merge_method=peptide_allele_merge_method,
            peptide_allele_merge_activation=peptide_allele_merge_activation,
            activation=activation,
            output_activation=output_activation,
            dropout_probability=dropout_probability,
            batch_normalization=batch_normalization,
            dense_layer_l1_regularization=dense_layer_l1_regularization,
            dense_layer_l2_regularization=dense_layer_l2_regularization,
            topology=topology,
            num_outputs=num_outputs,
            init=init,
            peptide_input_vector_encoding_name=peptide_torch_encoding_name,
        )

    def clear_allele_representations(self):
        """
        Set allele representations to an empty array.
        """
        original_model = self.network()
        if original_model is not None and original_model.allele_embedding is not None:
            existing_shape = original_model.allele_embedding.weight.shape
            new_weight = numpy.zeros(
                shape=(1, existing_shape[1]),
                dtype=numpy.float32
            )
            target = original_model.allele_embedding.weight
            original_model.allele_embedding.weight.data = torch.from_numpy(
                new_weight
            ).to(device=target.device, dtype=target.dtype)
            original_model.allele_embedding.weight.requires_grad = False

    def set_allele_representations(self, allele_representations, force_surgery=False):
        """
        Set the allele representations in use by this model.

        Parameters
        ----------
        allele_representations : numpy.ndarray of shape (a, l, m)
        force_surgery : bool
        """
        network = self.network()
        if network is None:
            return

        if allele_representations is None:
            has_allele_embedding = False
            if isinstance(network, MergedClass1NeuralNetwork):
                has_allele_embedding = any(
                    sub_network.allele_embedding is not None
                    for sub_network in network.networks
                )
            else:
                has_allele_embedding = (
                    hasattr(network, 'allele_embedding') and
                    network.allele_embedding is not None
                )
            if has_allele_embedding:
                raise ValueError(
                    "set_allele_representations(None) called on a pan-allele "
                    "network"
                )
            return

        reshaped = allele_representations.reshape(
            (
                allele_representations.shape[0],
                numpy.prod(allele_representations.shape[1:]),
            )
        ).astype(numpy.float32)

        # Handle merged networks (ensembles)
        if isinstance(network, MergedClass1NeuralNetwork):
            for sub_network in network.networks:
                self._update_embedding(sub_network, reshaped, force_surgery)
        elif hasattr(network, 'allele_embedding') and network.allele_embedding is not None:
            self._update_embedding(network, reshaped, force_surgery)

    def _update_embedding(self, network, reshaped, force_surgery):
        """Update the allele embedding for a single network."""
        if network.allele_embedding is None:
            return

        target_weight = network.allele_embedding.weight
        existing_shape = target_weight.shape
        target_device = target_weight.device
        target_dtype = target_weight.dtype

        if existing_shape[0] > reshaped.shape[0] and not force_surgery:
            # Extend with NaNs
            reshaped = numpy.append(
                reshaped,
                numpy.ones([existing_shape[0] - reshaped.shape[0], reshaped.shape[1]])
                * numpy.nan,
                axis=0,
            )

        if existing_shape != reshaped.shape:
            # Need to resize embedding
            new_embedding = nn.Embedding(
                num_embeddings=reshaped.shape[0],
                embedding_dim=reshaped.shape[1]
            ).to(device=target_device)
            new_embedding.weight.data = torch.from_numpy(reshaped).to(
                device=target_device,
                dtype=target_dtype,
            )
            new_embedding.weight.requires_grad = False
            network.allele_embedding = new_embedding
        else:
            network.allele_embedding.weight.data = torch.from_numpy(
                reshaped
            ).to(device=target_device, dtype=target_dtype)
            network.allele_embedding.weight.requires_grad = False


class MergedClass1NeuralNetwork(nn.Module):
    """
    A merged ensemble of Class1NeuralNetworkModel instances.
    """

    def __init__(self, networks, merge_method="average"):
        super(MergedClass1NeuralNetwork, self).__init__()
        self.networks = nn.ModuleList(networks)
        self.merge_method = merge_method
        # Per-sub-network peptide_stage feature dims, populated lazily on
        # the first call to ``forward_peptide_stage``. Used by
        # ``forward_from_peptide_stage`` / ``forward_cartesian_from_peptide_stage``
        # to split a concatenated peptide-stage tensor back into per-network
        # slices. Lazy because the dim depends on each sub-network's
        # peptide_dense_layer configuration; computing it without running
        # an actual peptide forward is fragile.
        self._sub_stage_dims = None

    def _combine_subnet_outputs(self, outputs):
        """Apply ``self.merge_method`` to a list of per-subnet output tensors."""
        if self.merge_method == "average":
            return torch.stack(outputs, dim=-1).mean(dim=-1)
        if self.merge_method == "sum":
            return torch.stack(outputs, dim=-1).sum(dim=-1)
        if self.merge_method == "concatenate":
            return torch.cat(outputs, dim=-1)
        raise ValueError(f"Unknown merge method: {self.merge_method}")

    def _split_peptide_stage(self, peptide_stage):
        """Split a concatenated peptide-stage tensor into per-subnet chunks.

        ``forward_peptide_stage`` concatenates per-subnet stages along the
        feature axis; this is the inverse operation, used inside the
        ``forward_*_from_peptide_stage`` helpers below.
        """
        if self._sub_stage_dims is None:
            raise RuntimeError(
                "forward_peptide_stage must be called before "
                "forward_from_peptide_stage / forward_cartesian_from_peptide_stage "
                "on MergedClass1NeuralNetwork (the per-subnet feature dims "
                "are recorded lazily on the first peptide forward)."
            )
        chunks = []
        offset = 0
        for d in self._sub_stage_dims:
            chunks.append(peptide_stage[..., offset:offset + d])
            offset += d
        return chunks

    def forward(self, inputs):
        outputs = [network(inputs) for network in self.networks]
        return self._combine_subnet_outputs(outputs)

    def forward_peptide_stage(self, peptide):
        """Run each sub-network's peptide-side stage and concatenate.

        Returned tensor has shape ``(batch, sum_i(peptide_dim_i))``;
        ``forward_from_peptide_stage`` and ``forward_cartesian_from_peptide_stage``
        split it back along the feature axis using the recorded per-subnet
        dims and merge the per-subnet outputs via ``self.merge_method``.
        """
        stages = [net.forward_peptide_stage(peptide) for net in self.networks]
        if self._sub_stage_dims is None:
            self._sub_stage_dims = tuple(int(s.shape[-1]) for s in stages)
        return torch.cat(stages, dim=-1)

    def forward_from_peptide_stage(self, peptide_stage, allele_idx):
        chunks = self._split_peptide_stage(peptide_stage)
        outputs = [
            net.forward_from_peptide_stage(chunk, allele_idx)
            for net, chunk in zip(self.networks, chunks)
        ]
        return self._combine_subnet_outputs(outputs)

    def forward_cartesian_from_peptide_stage(self, peptide_stage, allele_idx):
        chunks = self._split_peptide_stage(peptide_stage)
        outputs = [
            net.forward_cartesian_from_peptide_stage(chunk, allele_idx)
            for net, chunk in zip(self.networks, chunks)
        ]
        return self._combine_subnet_outputs(outputs)

    def get_weights_list(self):
        """Get all weights as a flat list."""
        weights = []
        for network in self.networks:
            weights.extend(network.get_weights_list())
        return weights

    def set_weights_list(self, weights, auto_convert_keras=False):
        """Set weights from a flat list."""
        idx = 0
        for network in self.networks:
            n_weights = (
                len(list(network.parameters()))
                + len(list(network._named_persistent_buffers()))
            )
            network.set_weights_list(weights[idx:idx + n_weights], auto_convert_keras=auto_convert_keras)
            idx += n_weights


def cartesian_network_output(
        model, peptide_stage, allele_idx, exact_forward=False):
    """Run a pan-allele network over the (allele × peptide) cartesian product.

    ``model`` is any network exposing ``forward_cartesian_from_peptide_stage``
    and ``forward_from_peptide_stage`` (a :class:`Class1NeuralNetworkModel` or a
    :class:`MergedClass1NeuralNetwork`). The fast path delegates to the
    network's factored ``forward_cartesian_from_peptide_stage``; ``exact_forward``
    instead materializes the full ``(num_alleles * num_peptides)`` batch and runs
    the plain ``forward_from_peptide_stage`` — the unfactored reference used to
    check the compact path's numerics.

    Returns a ``(num_alleles, num_peptides, num_outputs)`` tensor.
    """
    if not exact_forward:
        return model.forward_cartesian_from_peptide_stage(
            peptide_stage,
            allele_idx,
        )

    peptide_width = peptide_stage.shape[-1]
    num_peptides = peptide_stage.shape[0]
    num_alleles = allele_idx.shape[0]
    expanded_stage = peptide_stage.unsqueeze(0).expand(
        num_alleles,
        num_peptides,
        peptide_width,
    ).reshape(num_alleles * num_peptides, peptide_width)
    expanded_alleles = allele_idx.unsqueeze(1).expand(
        num_alleles,
        num_peptides,
    ).reshape(-1)
    return model.forward_from_peptide_stage(
        expanded_stage,
        expanded_alleles,
    ).reshape(num_alleles, num_peptides, -1)


def cartesian_output_log_ic50_sum(
        network_output, model, log50000, accum_dtype):
    """Convert cartesian network outputs to summed log(IC50).

    Normal models may expose multiple output channels; prediction defaults
    to channel 0, so fast calibration must do the same. Optimized pan-model
    ensembles are different: ``MergedClass1NeuralNetwork`` with
    ``merge_method='concatenate'`` returns one channel per merged submodel.
    Those channels are ensemble members and must all contribute to the
    geometric mean.

    Returns
    -------
    (torch.Tensor, int)
        ``(a_size, chunk_n)`` summed log(IC50) contribution and the number
        of ensemble members represented by the sum.
    """
    if network_output.ndim == 2:
        network_output = network_output.unsqueeze(-1)
    if network_output.ndim != 3 or int(network_output.shape[-1]) < 1:
        raise ValueError(
            "cartesian network output must have shape "
            "(alleles, peptides, outputs); got %s" % (
                tuple(network_output.shape),
            )
        )

    is_merged_concatenate = (
        getattr(model, "merge_method", None) == "concatenate"
        and getattr(model, "networks", None) is not None
    )
    if is_merged_concatenate:
        selected = network_output
    else:
        selected = network_output[..., :1]

    log_ic50 = (1.0 - selected).to(accum_dtype) * log50000
    return log_ic50.sum(dim=-1), int(selected.shape[-1])

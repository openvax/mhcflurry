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

import time
import collections
import json
import weakref
import itertools
import os
import logging

import numpy
import torch
import torch.nn as nn

from . import keras_model_config
from . import pytorch_sizing
from .hyperparameters import HyperparameterDefaults
from .encodable_sequences import EncodableSequences, EncodingError
from .regression_target import to_ic50
from .common import get_pytorch_device
from .pytorch_layers import KerasBatchNorm1d, LocallyConnected1D, get_activation
from .pytorch_losses import get_pytorch_loss
from .keras_optimizers import create_optimizer
from .data_dependent_weights_initialization import lsuv_init
from .random_negative_peptides import (
    RandomNegativePeptides,
    RandomNegativesPool,
    supports_device_random_negative_encoding,
)
from .class1_encoding import (
    _peptide_torch_encoding_name,
    _peptide_torch_encoding_shape,
    _peptide_torch_encoding_table,
    _peptide_uses_torch_encoding,
    peptide_sequences_to_network_input,
)
from .class1_training import (
    EarlyStoppingState,
    StreamingIteratorState,
    _StreamingBatchIterableDataset,
    _make_streaming_batch_dataloader,
    _materialize_repeated_peptide_batch,
    _timing_enabled,
    _validation_interval_from_hyperparameters,
    build_affinity_device_data,
    compute_regularization_penalty,
    prepare_affinity_inputs,
    prepare_streaming_validation_tensors,
    record_fit_summary,
    run_affinity_training_epoch,
    run_streaming_training_epoch,
    seed_training_rngs,
)
from .pytorch_training import (
    configure_matmul_precision,
    maybe_compile_loss,
    maybe_compile_network,
    uncompiled_network,
)


DEFAULT_PREDICT_BATCH_SIZE = pytorch_sizing.DEFAULT_PREDICT_BATCH_SIZE
CHECKPOINT_POLICIES = ("terminal", "best")
check_training_batch_fits = pytorch_sizing.check_training_batch_fits
compute_prediction_batch_size = pytorch_sizing.compute_prediction_batch_size
env_workers_per_gpu = pytorch_sizing.env_workers_per_gpu
resolve_prediction_batch_size = pytorch_sizing.resolve_prediction_batch_size

KERAS_BATCH_NORM_EPSILON = 1e-3
# Keras uses moving = moving * 0.99 + batch * 0.01. PyTorch's momentum is the
# new-batch coefficient, so the equivalent value is 0.01.
KERAS_BATCH_NORM_MOMENTUM = 0.01


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
            peptide_input_vector_encoding_name="BLOSUM62",
            normalization="none"):
        super(Class1NeuralNetworkModel, self).__init__()

        self.peptide_encoding_shape = peptide_encoding_shape
        if peptide_input_vector_encoding_name is None:
            peptide_input_vector_encoding_name = "BLOSUM62"
        self.peptide_input_vector_encoding_name = peptide_input_vector_encoding_name
        # Device-side fixed amino-acid encoding: peptide input is (N, L) int
        # indices and ``forward`` widens to (N, L, V) fp32 via a frozen
        # embedding table. The table is a non-persistent buffer: it moves with
        # ``.to(device)`` but is fully determined by hyperparameters and is not
        # serialized in custom NPZ weight lists.
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
        normalization = str(normalization or "none").lower()
        if normalization not in {"none", "batch", "layer"}:
            raise ValueError(
                "normalization must be one of none, batch, or layer; got %r" %
                normalization
            )
        if batch_normalization:
            if normalization not in {"none", "batch"}:
                raise ValueError(
                    "batch_normalization=True conflicts with normalization=%r" %
                    normalization
                )
            normalization = "batch"
        self.normalization = normalization
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
        self.layer_norm_early = None
        if normalization == "batch":
            self.batch_norm_early = KerasBatchNorm1d(
                peptide_layer_input,
                eps=KERAS_BATCH_NORM_EPSILON,
                momentum=KERAS_BATCH_NORM_MOMENTUM,
            )
        elif normalization == "layer":
            self.layer_norm_early = nn.LayerNorm(peptide_layer_input)

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
        self.layer_norms = nn.ModuleList()
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

            if normalization == "batch":
                self.batch_norms.append(KerasBatchNorm1d(
                    size,
                    eps=KERAS_BATCH_NORM_EPSILON,
                    momentum=KERAS_BATCH_NORM_MOMENTUM,
                ))
            else:
                self.batch_norms.append(None)
            if normalization == "layer":
                self.layer_norms.append(nn.LayerNorm(size))
            else:
                self.layer_norms.append(None)

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

    def activation_output_for_layer(self, layer, output):
        """Apply the activation belonging to a trainable affine layer.

        This exposes the same layer-output boundary that Keras reports for a
        ``Dense`` layer. Data-dependent initialization uses it to measure
        post-activation variance even though PyTorch represents the linear
        transform and activation as separate operations.
        """
        hidden_layers = (
            list(self.peptide_dense_layers)
            + list(self.allele_dense_layers)
            + list(self.dense_layers)
        )
        if any(layer is hidden_layer for hidden_layer in hidden_layers):
            return self.activation(output) if self.activation is not None else output
        if layer is self.output_layer:
            return (
                self.output_activation(output)
                if self.output_activation is not None
                else output
            )
        return output

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
                elif init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _embed_peptide_indices(self, peptide):
        if peptide.dim() != 2:
            raise ValueError(
                "Expected 2D index-encoded peptide tensor (N, L); got shape %s"
                % (tuple(peptide.shape),)
            )
        return torch.nn.functional.embedding(
            peptide.long(), self.peptide_embedding_table
        )

    def _forward_peptide_stage_before_early_batch_norm(self, peptide):
        x = self._embed_peptide_indices(peptide)
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
            ``(N, L)`` int amino-acid indices.

        Returns
        -------
        torch.Tensor of shape (N, peptide_representation_dim) — the
        input the ``forward_from_peptide_stage`` fast path expects.
        """
        x = self._forward_peptide_stage_before_early_batch_norm(peptide)
        if self.batch_norm_early is not None:
            x = self.batch_norm_early(x)
        if self.layer_norm_early is not None:
            x = self.layer_norm_early(x)
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
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)
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
        if self.layer_norm_early is not None:
            peptide_stage = self.layer_norm_early(peptide_stage)

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
            if self.layer_norms[0] is not None:
                x = self.layer_norms[0](x)
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
                if self.layer_norms[i] is not None:
                    x = self.layer_norms[i](x)
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
            if self.layer_norms[0] is not None:
                x = self.layer_norms[0](x)
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
                    if self.layer_norms[i] is not None:
                        new_x = self.layer_norms[i](new_x)
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
                    if self.layer_norms[i] is not None:
                        x = self.layer_norms[i](x)
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

        # Device-side fixed amino-acid encoding: (N, L) int indices ->
        # (N, L, V) fp32 via frozen embedding table.
        x = self._embed_peptide_indices(peptide)

        # Locally connected layers
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
        if self.layer_norm_early is not None:
            x = self.layer_norm_early(x)

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
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)
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
        if auto_convert_keras:
            keras_model_config.load_keras_weights_into_model(
                self,
                weights,
                keras_config=getattr(self, "_keras_config", None),
                keras_metadata=getattr(self, "_keras_metadata", None),
            )
            return

        idx = 0
        keras_metadata = getattr(self, "_keras_metadata", None)
        skip_keras_embedding = bool(
            keras_metadata and keras_metadata.get("skip_embedding_weights", False)
        )
        for name, param in self.named_parameters():
            if skip_keras_embedding and "allele_embedding" in name:
                while (
                    idx < len(weights)
                    and len(weights[idx].shape) == 2
                    and weights[idx].shape[0] == 0
                ):
                    idx += 1
                continue
            w = weights[idx].astype(numpy.float32)
            if "allele_embedding" in name and w.shape != param.shape:
                idx += 1
                continue
            if w.shape != param.shape:
                raise ValueError(
                    f"Weight shape mismatch for {name}: "
                    f"got {weights[idx].shape}, expected {param.shape}"
                )
            param.data = torch.from_numpy(w).to(
                device=param.device,
                dtype=param.dtype,
            )
            idx += 1

        named_modules_dict = dict(self.named_modules())
        for name, buffer in self._named_persistent_buffers():
            tensor = torch.from_numpy(weights[idx]).to(
                device=buffer.device,
                dtype=buffer.dtype,
            )
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
            'normalization': self.normalization,
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
        normalization="none",
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
        optimizer_implementation="keras",
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
        data_dependent_initialization_target="post_activation",
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
        # False preserves historical Keras EarlyStopping behavior. Set True
        # to retain the state from the minimum measured validation loss.
        restore_best_weights=False,
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
        self.checkpoint_weights = {}
        self.checkpoint_weights_loaders = {}
        self.checkpoint_weight_paths = {}

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
        is_keras_format = keras_model_config.is_keras_model_config(config)
        key = (
            klass.model_cache_key(network_json),
            keras_model_config.infer_allele_representation_dim_from_weights(
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
        """Parse a legacy Keras model JSON config (see ``keras_model_config``)."""
        return keras_model_config.parse_keras_json_config(config)

    @classmethod
    def _infer_allele_representation_dim_from_weights(
            cls, network_weights, hyperparameters, network_config=None):
        """Infer flattened allele representation width (see ``keras_model_config``)."""
        return keras_model_config.infer_allele_representation_dim_from_weights(
            network_weights, hyperparameters, network_config=network_config
        )

    @classmethod
    def _placeholder_allele_representations(
            cls, hyperparameters, network_weights=None, network_config=None,
            default_sequence_length=39):
        """Create a shape-compatible placeholder allele array (see ``keras_model_config``)."""
        return keras_model_config.placeholder_allele_representations(
            hyperparameters,
            network_weights=network_weights,
            network_config=network_config,
            default_sequence_length=default_sequence_length,
        )

    @classmethod
    def _create_model_from_config(
            cls, config, instance_hyperparameters=None, network_weights=None):
        """Create a model from a config dict (see ``keras_model_config``)."""
        return keras_model_config.create_model_from_config(
            config,
            model_cls=Class1NeuralNetworkModel,
            merged_model_cls=MergedClass1NeuralNetwork,
            network_cls=cls,
            instance_hyperparameters=instance_hyperparameters,
            network_weights=network_weights,
        )

    @classmethod
    def _create_merged_model_from_config(
            cls, config, instance_hyperparameters=None, network_weights=None):
        """Create a merged model from a config dict (see ``keras_model_config``)."""
        return keras_model_config.create_merged_model_from_config(
            config,
            model_cls=Class1NeuralNetworkModel,
            merged_model_cls=MergedClass1NeuralNetwork,
            network_cls=cls,
            instance_hyperparameters=instance_hyperparameters,
            network_weights=network_weights,
        )

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
                is_keras_format = keras_model_config.is_keras_model_config(config)
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
                        {
                            'filters': layer.out_channels,
                            'kernel_size': layer.kernel_size,
                            'activation': layer.activation_name,
                        }
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
                        getattr(subnet, 'batch_norm_early', None) is not None or (
                            hasattr(subnet, 'batch_norms')
                            and bool(subnet.batch_norms)
                            and any(bn is not None for bn in subnet.batch_norms)
                        )
                    )
                    sub_config['normalization'] = getattr(
                        subnet, 'normalization', 'none')
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
        result["checkpoint_weights"] = {}
        result["checkpoint_weights_loaders"] = {}
        result.pop("checkpoint_weight_paths", None)
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
        if keras_model_config.is_keras_model_config(config):
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
        # Checkpoint sidecars are attached by the predictor persistence layer.
        # They deliberately stay out of config_json because weight arrays do
        # not belong in JSON and older predictor directories have none.
        instance.checkpoint_weights = {}
        instance.checkpoint_weights_loaders = {}
        instance.checkpoint_weight_paths = {}
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
        if self.prediction_cache is not None:
            self.prediction_cache.clear()

    @staticmethod
    def _copy_weights_list(weights):
        """Return owned numpy arrays, avoiding tensor/numpy view aliasing."""
        return [numpy.array(value, copy=True) for value in weights]

    def available_checkpoint_policies(self):
        """Return retained checkpoint policies in stable order."""
        available = set(getattr(self, "checkpoint_weights", {}))
        available.update(getattr(self, "checkpoint_weights_loaders", {}))
        return [policy for policy in CHECKPOINT_POLICIES if policy in available]

    def get_checkpoint_weights(self, policy):
        """Return retained weights for ``policy``, loading lazily if needed."""
        if policy not in CHECKPOINT_POLICIES:
            raise ValueError("Unknown checkpoint policy: %s" % policy)
        checkpoint_weights = getattr(self, "checkpoint_weights", {})
        if policy not in checkpoint_weights:
            loader = getattr(
                self, "checkpoint_weights_loaders", {}).get(policy)
            if loader is None:
                raise KeyError("Checkpoint policy not retained: %s" % policy)
            checkpoint_weights[policy] = loader()
            self.checkpoint_weights = checkpoint_weights
        return checkpoint_weights[policy]

    def set_checkpoint_weights_loader(self, policy, loader, weight_path=None):
        """Attach a lazy loader for a retained training checkpoint."""
        if policy not in CHECKPOINT_POLICIES:
            raise ValueError("Unknown checkpoint policy: %s" % policy)
        if not hasattr(self, "checkpoint_weights"):
            self.checkpoint_weights = {}
        if not hasattr(self, "checkpoint_weights_loaders"):
            self.checkpoint_weights_loaders = {}
        if not hasattr(self, "checkpoint_weight_paths"):
            self.checkpoint_weight_paths = {}
        self.checkpoint_weights.pop(policy, None)
        self.checkpoint_weights_loaders[policy] = loader
        if weight_path is not None:
            self.checkpoint_weight_paths[policy] = os.fspath(weight_path)

    def _clear_training_checkpoints(self):
        """Discard checkpoint sidecars made stale by a new fit."""
        self.checkpoint_weights = {}
        self.checkpoint_weights_loaders = {}
        self.checkpoint_weight_paths = {}

    def _finalize_training_checkpoints(
            self, eager_network, best_state_dict, save_all_checkpoints):
        """Select the primary state and optionally retain both fit states."""
        restore_best = bool(
            self.hyperparameters["restore_best_weights"]
            and best_state_dict is not None
        )
        terminal_weights = None
        if save_all_checkpoints:
            terminal_weights = self._copy_weights_list(
                eager_network.get_weights_list())

        if best_state_dict is not None and (restore_best or save_all_checkpoints):
            eager_network.load_state_dict(best_state_dict)

        if save_all_checkpoints:
            retained = {"terminal": terminal_weights}
            if best_state_dict is not None:
                retained["best"] = self._copy_weights_list(
                    eager_network.get_weights_list())
            self.checkpoint_weights = retained
            self.checkpoint_weights_loaders = {}
            self.checkpoint_weight_paths = {}

            # Loading the best state above was only for capture when terminal
            # remains primary. Restore from an owned copy, never a numpy view
            # of the live module parameters.
            if not restore_best:
                eager_network.set_weights_list(
                    terminal_weights, auto_convert_keras=False)

        return restore_best

    def __getstate__(self):
        """
        serialize to a dict. Model weights are included. For pickle support.

        Returns
        -------
        dict

        """
        self.update_network_description()
        self.load_weights()
        for policy in self.available_checkpoint_policies():
            self.get_checkpoint_weights(policy)
        result = dict(self.__dict__)
        result["_network"] = None
        result["checkpoint_weights_loaders"] = {}
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

    def peptides_to_network_input(self, peptides, device=None):
        """
        Encode peptides to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Return (N, L) int8 amino-acid indices; the network looks up fixed
        vectors on device. NumericSequences inputs return a Torch tensor
        directly, while string inputs retain the existing NumPy path.

        Parameters
        ----------
        peptides : EncodableSequences, NumericSequences or list of string
        device : torch.device or str, optional
            Device for NumericSequences alignment; string encoding stays on CPU.

        Returns
        -------
        numpy.ndarray or torch.Tensor
        """
        encoded = peptide_sequences_to_network_input(
            peptides,
            peptide_encoding=self.hyperparameters["peptide_encoding"],
            peptide_amino_acid_encoding_torch=(
                self.hyperparameters.get("peptide_amino_acid_encoding_torch")
            ),
            device=device,
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
    def data_dependent_weights_initialization(
        network,
        x_dict=None,
        method="lsuv",
        target="post_activation",
        verbose=1,
    ):
        """
        Data dependent weights initialization.

        Parameters
        ----------
        network : Class1NeuralNetworkModel
        x_dict : dict of string -> numpy.ndarray
            Training data
        method : string
            Initialization method. Currently only "lsuv" is supported.
        target : {"post_activation", "pre_activation"}
            Affine-layer output whose variance LSUV normalizes.
        verbose : int
            Status updates printed to stdout if verbose > 0
        """
        if verbose:
            print("Performing data-dependent init: ", method)
        if method == "lsuv":
            assert x_dict is not None, "Data required for LSUV init"
            lsuv_init(network, x_dict, verbose=verbose > 0, target=target)
        else:
            raise RuntimeError("Unsupported init method: ", method)

    @staticmethod
    def _regularized_parameters(network):
        """
        Parameters subject to dense kernel regularization.
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
        """Match Keras kernel_regularizer semantics used on dense kernels."""
        return compute_regularization_penalty(parameters, l1=l1, l2=l2)

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
            seed=None,
            save_all_checkpoints=False):
        """Pretrain from a stream of already batched examples.

        This is the streaming pretraining path used by pan-allele training. It
        intentionally has a narrower contract than :meth:`fit`:

        * training examples arrive as complete batches from ``generator``;
        * validation data is supplied explicitly rather than split from the
          training rows;
        * random negatives, sample weights, and the affinity ``fit()`` row
          layout are not part of this API.

        The method shares the same low-level optimizer step, loss object,
        regularization, validation, and early-stopping phase helpers in
        :mod:`mhcflurry.class1_training` as :meth:`fit`.
        """
        self._clear_training_checkpoints()
        if self.prediction_cache is not None:
            self.prediction_cache.clear()
        device = self.get_device()
        configure_matmul_precision(device)
        seed_training_rngs(seed)

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

        optimizer = self._create_optimizer(network)
        if self.hyperparameters["learning_rate"] is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.hyperparameters["learning_rate"]
        fit_info["learning_rate"] = optimizer.param_groups[0]["lr"]
        regularization_parameters = tuple(self._regularized_parameters(network))
        l1_reg = self.hyperparameters["dense_layer_l1_regularization"]
        l2_reg = self.hyperparameters["dense_layer_l2_regularization"]

        (
            val_peptide_device,
            val_y_device,
            val_allele_device,
        ) = prepare_streaming_validation_tensors(
            validation_peptide_encoding=validation_peptide_encoding,
            validation_affinities=validation_affinities,
            validation_allele_input=validation_allele_input,
            validation_inequalities=validation_inequalities,
            validation_output_indices=validation_output_indices,
            loss_obj=loss_obj,
            device=device,
            peptides_to_network_input=self.peptides_to_network_input,
        )

        dataloader_num_workers = self.hyperparameters.get(
            "dataloader_num_workers", 0
        )
        non_blocking_h2d = False
        fit_info["dataloader_num_workers"] = dataloader_num_workers
        fit_info["validation_rows"] = len(validation_affinities)
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

        stream_state = StreamingIteratorState(
            iterator=make_iterator(),
            make_iterator=make_iterator,
            generator_factory=generator_factory,
        )

        data_dependent_init = self.hyperparameters[
            "data_dependent_initialization_method"
        ]
        if data_dependent_init and not self.fit_info:
            first_chunk = next(stream_state.iterator)
            init_chunk = _materialize_repeated_peptide_batch(first_chunk)
            first_inputs = {"peptide": init_chunk["peptide"]}
            if "allele" in init_chunk:
                first_inputs["allele"] = init_chunk["allele"]
            self.data_dependent_weights_initialization(
                network,
                first_inputs,
                method=data_dependent_init,
                target=self.hyperparameters[
                    "data_dependent_initialization_target"
                ],
                verbose=verbose,
            )
            stream_state.iterator = itertools.chain(
                [first_chunk], stream_state.iterator
            )

        # Compile AFTER LSUV hook churn finishes.
        network = maybe_compile_network(network, device)
        eager_network = uncompiled_network(network)
        loss_obj = maybe_compile_loss(loss_obj, device)

        early_stopping_state = EarlyStoppingState()
        last_progress_print = 0
        first_batch_time = None
        validation_interval = _validation_interval_from_hyperparameters(
            self.hyperparameters
        )

        for epoch_index in range(int(epochs)):
            epoch_outcome = run_streaming_training_epoch(
                epoch_index=epoch_index,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                min_epochs=min_epochs,
                patience=patience,
                min_delta=min_delta,
                network=network,
                eager_network=eager_network,
                optimizer=optimizer,
                loss_obj=loss_obj,
                regularization_parameters=regularization_parameters,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                stream_state=stream_state,
                val_peptide_device=val_peptide_device,
                val_y_device=val_y_device,
                val_allele_device=val_allele_device,
                validation_count=len(validation_affinities),
                early_stopping_state=early_stopping_state,
                hyperparameters=self.hyperparameters,
                fit_info=fit_info,
                device=device,
                non_blocking_h2d=non_blocking_h2d,
                num_workers_per_gpu=env_workers_per_gpu(),
                timing_enabled=timing_enabled,
                validation_interval=validation_interval,
                save_all_checkpoints=save_all_checkpoints,
                progress_callback=progress_callback,
                progress_preamble=progress_preamble,
                progress_print_interval=progress_print_interval,
                last_progress_print=last_progress_print,
            )
            last_progress_print = epoch_outcome["last_progress_print"]
            if first_batch_time is None:
                first_batch_time = epoch_outcome["first_batch_time"]
            if epoch_outcome["should_stop"]:
                break

        restored_best_weights = self._finalize_training_checkpoints(
            eager_network,
            early_stopping_state.best_state_dict,
            save_all_checkpoints,
        )
        self.fit_info.append(
            record_fit_summary(
                fit_info=fit_info,
                early_stopping_state=early_stopping_state,
                restored_best_weights=restored_best_weights,
                saved_checkpoint_policies=self.available_checkpoint_policies(),
                elapsed_seconds=time.time() - start,
                num_points=stream_state.yielded_values,
                first_batch_time=first_batch_time,
            )
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
        """Create the configured optimizer using Keras or PyTorch equations."""
        return create_optimizer(
            network.parameters(),
            optimizer_name=self.hyperparameters["optimizer"],
            implementation=self.hyperparameters["optimizer_implementation"],
            learning_rate=self.hyperparameters["learning_rate"],
        )

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
            seed=None,
            save_all_checkpoints=False):
        """
        Fit the neural network.

        Parameters
        ----------
        peptides : EncodableSequences or list of string
        affinities : list of float
            nM affinities. Must have the same length as peptides.
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
            left unchanged. The caller or worker initializer is responsible
            for setting independent RNG states.

            A seed does not guarantee identical weights across hardware,
            dependency versions or numerical settings. Automatic minibatch
            reduction can also change a trajectory; release workflows reject
            that reduction. Preserve the effective batch, source, dependencies,
            device and precision settings alongside the seed. See PyTorch's
            reproducibility notes for backend-specific determinism controls.
        save_all_checkpoints : bool
            Retain both terminal and minimum-validation weights from this fit.
            This does not change the primary state selected by the
            ``restore_best_weights`` hyperparameter.
        """
        self._clear_training_checkpoints()
        if self.prediction_cache is not None:
            self.prediction_cache.clear()
        device = self.get_device()
        configure_matmul_precision(device)
        seed_training_rngs(seed)

        fit_info = collections.defaultdict(list)
        timing_enabled = _timing_enabled()
        fit_info["timing_enabled"] = timing_enabled
        fit_info["dataloader_num_workers"] = self.hyperparameters.get(
            "dataloader_num_workers", 0
        )

        loss_obj = get_pytorch_loss(self.hyperparameters["loss"])
        prepared = prepare_affinity_inputs(
            peptides=peptides,
            affinities=affinities,
            allele_encoding=allele_encoding,
            inequalities=inequalities,
            output_indices=output_indices,
            sample_weights=sample_weights,
            shuffle_permutation=shuffle_permutation,
            hyperparameters=self.hyperparameters,
            loss_obj=loss_obj,
            device=device,
            seed=seed,
            peptides_to_network_input=self.peptides_to_network_input,
            allele_encoding_to_network_input=self.allele_encoding_to_network_input,
            random_negatives_pool_factory=self._random_negatives_pool_for_fit,
        )
        fit_info["random_negative_pool_residency"] = (
            prepared.random_negative_pool_residency
        )
        fit_info["random_negative_pool_epochs"] = (
            prepared.random_negative_pool_epochs
        )

        if self.network() is None:
            self._network = self.make_network(
                allele_representations=prepared.allele_representations,
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters)
            )
            if verbose > 0:
                print(self.network())

        network = self.network()
        network.to(device)
        if prepared.allele_representations is not None:
            self.set_allele_representations(prepared.allele_representations)
        eager_network = uncompiled_network(network)

        requested_minibatch = int(self.hyperparameters["minibatch_size"])
        effective_minibatch, shrunk = check_training_batch_fits(
            requested_minibatch,
            device,
            network,
            num_workers_per_gpu=env_workers_per_gpu(1),
        )
        if shrunk:
            fit_info["minibatch_size_shrunk_from"] = requested_minibatch
            fit_info["minibatch_size_shrunk_to"] = effective_minibatch
            if seed is not None:
                logging.warning(
                    "fit(seed=%s) shrank minibatch_size %d -> %d to fit "
                    "available VRAM. Reproducibility holds only at a fixed "
                    "effective minibatch size, so this run may not reproduce "
                    "bit-for-bit on a GPU with different free memory or a "
                    "different number of workers per card.",
                    seed,
                    requested_minibatch,
                    effective_minibatch,
                )
        fit_info["effective_minibatch_size"] = effective_minibatch

        optimizer = self._create_optimizer(network)
        if self.hyperparameters["learning_rate"] is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.hyperparameters["learning_rate"]
        fit_info["learning_rate"] = optimizer.param_groups[0]["lr"]

        device_data = build_affinity_device_data(
            prepared=prepared,
            hyperparameters=self.hyperparameters,
            loss_obj=loss_obj,
            device=device,
        )
        fit_info["train_rows"] = device_data.n_train
        fit_info["validation_rows"] = device_data.n_val
        fit_info["num_random_negatives"] = int(prepared.num_random_negatives)
        fit_info["validation_cache_reused"] = device_data.val_cache_reused
        fit_info["fit_tensor_residency"] = "device"
        logging.info("fit tensor residency: device (device=%s)", device.type)

        regularization_parameters = tuple(self._regularized_parameters(network))
        l1_reg = self.hyperparameters["dense_layer_l1_regularization"]
        l2_reg = self.hyperparameters["dense_layer_l2_regularization"]

        early_stopping_state = EarlyStoppingState()
        needs_initialization = (
            self.hyperparameters["data_dependent_initialization_method"] is not None
            and not self.fit_info
        )
        start = time.time()
        last_progress_print = None
        first_batch_time = None

        for epoch in range(self.hyperparameters["max_epochs"]):
            epoch_outcome = run_affinity_training_epoch(
                epoch=epoch,
                network=network,
                eager_network=eager_network,
                optimizer=optimizer,
                loss_obj=loss_obj,
                regularization_parameters=regularization_parameters,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                prepared=prepared,
                device_data=device_data,
                early_stopping_state=early_stopping_state,
                hyperparameters=self.hyperparameters,
                fit_info=fit_info,
                device=device,
                effective_minibatch=effective_minibatch,
                num_workers_per_gpu=env_workers_per_gpu(),
                timing_enabled=timing_enabled,
                needs_initialization=needs_initialization,
                data_dependent_init_fn=self.data_dependent_weights_initialization,
                verbose=verbose,
                save_all_checkpoints=save_all_checkpoints,
                progress_callback=progress_callback,
                progress_preamble=progress_preamble,
                progress_print_interval=progress_print_interval,
                last_progress_print=last_progress_print,
            )
            network = epoch_outcome["network"]
            eager_network = epoch_outcome["eager_network"]
            loss_obj = epoch_outcome["loss_obj"]
            needs_initialization = epoch_outcome["needs_initialization"]
            last_progress_print = epoch_outcome["last_progress_print"]
            if first_batch_time is None:
                first_batch_time = epoch_outcome["first_batch_time"]
            if epoch_outcome["should_stop"]:
                break

        restored_best_weights = self._finalize_training_checkpoints(
            eager_network,
            early_stopping_state.best_state_dict,
            save_all_checkpoints,
        )
        self.fit_info.append(
            record_fit_summary(
                fit_info=fit_info,
                early_stopping_state=early_stopping_state,
                restored_best_weights=restored_best_weights,
                saved_checkpoint_policies=self.available_checkpoint_policies(),
                elapsed_seconds=time.time() - start,
                num_points=len(peptides),
                first_batch_time=first_batch_time,
            )
        )

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
            ``env_workers_per_gpu(1)`` (the ``MHCFLURRY_MAX_WORKERS_PER_GPU``
            env var the worker pool sets), mirroring ``fit()``.

        Returns
        -------
        numpy.array of nM affinity predictions
        """
        auto_batch_size = batch_size in (None, "auto")
        if num_workers_per_gpu is None:
            num_workers_per_gpu = env_workers_per_gpu(1)

        assert self.prediction_cache is not None
        use_cache = allele_encoding is None and isinstance(peptides, EncodableSequences)
        if use_cache:
            cached_by_output = self.prediction_cache.get(peptides)
            if cached_by_output is not None and output_index in cached_by_output:
                return cached_by_output[output_index].copy()

        device = self.get_device()
        configure_matmul_precision(device)

        x_dict = {"peptide": self.peptides_to_network_input(peptides, device=device)}

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
            total_rows=len(x_dict["peptide"]),
        )

        # Batch prediction
        n_samples = len(x_dict["peptide"])
        peptide_is_indices = _peptide_uses_torch_encoding(self.hyperparameters)

        def prediction_tensor(batch_array):
            if isinstance(batch_array, torch.Tensor):
                return batch_array.to(device)
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

        if n_samples == 0:
            if output_index is not None:
                return numpy.empty(0, dtype="float64")
            if hasattr(network, "output_layer"):
                out_dim = int(network.output_layer.out_features)
            elif hasattr(network, "networks"):
                out_dim = (
                    sum(int(n.output_layer.out_features) for n in network.networks)
                    if getattr(network, "merge_method", None) == "concatenate"
                    else int(network.networks[0].output_layer.out_features)
                )
            else:
                out_dim = int(self.hyperparameters.get("num_outputs", 1))
            return numpy.empty((0, out_dim), dtype="float64")

        while True:
            all_predictions = []
            try:
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
                break
            except RuntimeError as exc:
                if (
                        not auto_batch_size
                        or batch_size <= 1
                        or not pytorch_sizing.is_device_out_of_memory_error(exc)):
                    raise
                previous_batch_size = batch_size
                batch_size = max(1, batch_size // 2)
                all_predictions = []
                peptide_batch = None
                allele_batch = None
                inputs = None
                batch_predictions = None
                pytorch_sizing.release_device_memory_after_oom(device)
                logging.warning(
                    "Auto prediction batch OOM at %d rows; retrying at %d.",
                    previous_batch_size,
                    batch_size,
                )

        raw_predictions = numpy.concatenate(all_predictions, axis=0)
        predictions = numpy.array(raw_predictions, dtype="float64")

        if output_index is not None:
            predictions = predictions[:, output_index]

        result = to_ic50(predictions)
        if use_cache:
            cached_by_output = self.prediction_cache.get(peptides)
            if cached_by_output is None:
                cached_by_output = {}
                self.prediction_cache[peptides] = cached_by_output
            cached_by_output[output_index] = result
            return result.copy()
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

        shared_encoding_keys = (
            "allele_amino_acid_encoding",
            "peptide_encoding",
            "peptide_amino_acid_encoding",
            "peptide_amino_acid_encoding_torch",
        )
        first_hp = models[0].hyperparameters
        for key in shared_encoding_keys:
            first_val = first_hp.get(key)
            if any(model.hyperparameters.get(key) != first_val for model in models[1:]):
                raise NotImplementedError(
                    "Cannot merge models with mismatched %s" % key
                )

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
            peptide_amino_acid_encoding_gpu=None,
            normalization="none"):
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
            normalization=normalization,
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

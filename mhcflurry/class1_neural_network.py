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

from .hyperparameters import HyperparameterDefaults
from .encodable_sequences import EncodableSequences, EncodingError
from .allele_encoding import AlleleEncoding
from .regression_target import to_ic50, from_ic50
from .common import configure_pytorch, get_pytorch_device
from .pytorch_layers import LocallyConnected1D, get_activation
from .pytorch_losses import get_pytorch_loss
from .data_dependent_weights_initialization import lsuv_init
from .random_negative_peptides import RandomNegativePeptides


DEFAULT_PREDICT_BATCH_SIZE = 4096
if os.environ.get("MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"):
    DEFAULT_PREDICT_BATCH_SIZE = int(os.environ["MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"])
    logging.info(
        "Configured default predict batch size: %d" % DEFAULT_PREDICT_BATCH_SIZE
    )


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
            init="glorot_uniform"):
        super(Class1NeuralNetworkModel, self).__init__()

        self.peptide_encoding_shape = peptide_encoding_shape
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
        peptide = inputs['peptide']

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
        if self.has_allele and 'allele' in inputs:
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
        # Also include buffers (running mean/var for batch norm)
        for name, buffer in self.named_buffers():
            weights.append(buffer.detach().cpu().numpy())
        return weights

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
            for name, buffer in self.named_buffers():
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
        minibatch_size=128,
        data_dependent_initialization_method=None,
        random_negative_affinity_min=20000.0,
        random_negative_affinity_max=50000.0,
        random_negative_output_indices=None,
    ).extend(RandomNegativePeptides.hyperparameter_defaults)
    """
    Hyperparameters for neural network training.
    """

    early_stopping_hyperparameter_defaults = HyperparameterDefaults(
        patience=20,
        min_delta=0.0,
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

    # Hyperparameter renames.
    hyperparameter_renames = {
        "use_embedding": None,
        "pseudosequence_use_embedding": None,
        "monitor": None,
        "min_delta": None,
        "verbose": None,
        "mode": None,
        "take_best_epoch": None,
        "kmer_size": None,
        "peptide_amino_acid_encoding": None,
        "embedding_input_dim": None,
        "embedding_output_dim": None,
        "embedding_init_method": None,
        "left_edge": None,
        "right_edge": None,
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
        key = klass.model_cache_key(network_json)
        config = json.loads(network_json)
        # Detect if weights are from Keras or PyTorch format
        # Keras JSON has 'class_name': 'Model' or 'Functional'; PyTorch has 'hyperparameters'
        is_keras_format = config.get('class_name') in ('Model', 'Functional')

        if key not in klass.MODELS_CACHE:
            # Cache miss - create new model
            network = klass._create_model_from_config(config)
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
    def _create_model_from_config(cls, config, instance_hyperparameters=None):
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
            return cls._create_merged_model_from_config(config, instance_hyperparameters)

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

        # Create a temporary instance to get encoding shape
        temp = cls(**hyperparameters)
        peptide_encoding_shape = temp.peptides_to_network_input([]).shape[1:]

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
            # Compute embedding dimension from encoding
            from .amino_acid import ENCODING_DATA_FRAMES
            encoding_name = hyperparameters['allele_amino_acid_encoding']
            encoding_df = ENCODING_DATA_FRAMES.get(encoding_name)
            if encoding_df is not None:
                # Standard allele pseudosequence length is 37 amino acids
                allele_seq_length = 37
                embedding_dim = allele_seq_length * len(encoding_df.columns)
                allele_representations = numpy.zeros((1, embedding_dim), dtype=numpy.float32)

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
        )

        # Store keras metadata and config for weight loading
        if keras_metadata is not None:
            model._keras_metadata = keras_metadata
            model._keras_config = config

        return model

    @classmethod
    def _create_merged_model_from_config(cls, config, instance_hyperparameters=None):
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
        temp = cls(**base_hyperparameters)
        peptide_encoding_shape = temp.peptides_to_network_input([]).shape[1:]

        # Create placeholder allele representations for pan-allele models
        allele_representations = None
        if base_hyperparameters.get('allele_amino_acid_encoding'):
            from .amino_acid import ENCODING_DATA_FRAMES
            encoding_name = base_hyperparameters['allele_amino_acid_encoding']
            encoding_df = ENCODING_DATA_FRAMES.get(encoding_name)
            if encoding_df is not None:
                allele_seq_length = 37
                embedding_dim = allele_seq_length * len(encoding_df.columns)
                allele_representations = numpy.zeros((1, embedding_dim), dtype=numpy.float32)

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
                    config, instance_hyperparameters=self.hyperparameters)
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
        result["prediction_cache"] = None
        result["_device"] = None
        return result

    @classmethod
    def from_config(cls, config, weights=None, weights_loader=None):
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
        instance.prediction_cache = weakref.WeakKeyDictionary()
        return instance

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
        result["_device"] = None
        return result

    def __setstate__(self, state):
        """
        Deserialize. For pickle support.
        """
        self.__dict__.update(state)
        self.prediction_cache = weakref.WeakKeyDictionary()

    def peptides_to_network_input(self, peptides):
        """
        Encode peptides to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Parameters
        ----------
        peptides : EncodableSequences or list of string

        Returns
        -------
        numpy.array
        """
        encoder = EncodableSequences.create(peptides)
        encoded = encoder.variable_length_to_fixed_length_vector_encoding(
            **self.hyperparameters["peptide_encoding"]
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
            progress_print_interval=5.0):
        """
        Fit using a generator. Does not support many of the features of fit(),
        such as random negative peptides.
        """
        configure_pytorch()
        device = self.get_device()

        fit_info = collections.defaultdict(list)

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
            "allele": validation_allele_input,
        }
        encode_y_kwargs = {}
        if validation_inequalities is not None:
            encode_y_kwargs["inequalities"] = validation_inequalities
        if validation_output_indices is not None:
            encode_y_kwargs["output_indices"] = validation_output_indices

        output = loss_obj.encode_y(from_ic50(validation_affinities), **encode_y_kwargs)

        mutable_generator_state = {
            "yielded_values": 0
        }

        def wrapped_generator():
            for alleles, peptides, affinities in generator:
                (allele_encoding_input, _) = self.allele_encoding_to_network_input(
                    alleles
                )
                x_dict = {
                    "peptide": self.peptides_to_network_input(peptides),
                    "allele": allele_encoding_input,
                }
                y = from_ic50(affinities)
                yield (x_dict, y)
                mutable_generator_state["yielded_values"] += len(affinities)

        start = time.time()
        iterator = wrapped_generator()

        # Data dependent init
        data_dependent_init = self.hyperparameters[
            "data_dependent_initialization_method"
        ]
        if data_dependent_init and not self.fit_info:
            first_chunk = next(iterator)
            self.data_dependent_weights_initialization(
                network,
                first_chunk[0],
                method=data_dependent_init,
                verbose=verbose,
            )
            iterator = itertools.chain([first_chunk], iterator)

        min_val_loss_iteration = None
        min_val_loss = None
        last_progress_print = 0
        epoch = 1

        while True:
            epoch_start_time = time.time()
            network.train()

            epoch_losses = []
            for step in range(steps_per_epoch):
                try:
                    x_dict, y = next(iterator)
                except StopIteration:
                    break

                # Convert to tensors
                peptide_tensor = torch.from_numpy(x_dict["peptide"]).float().to(device)
                allele_tensor = torch.from_numpy(x_dict["allele"]).float().to(device)
                y_tensor = torch.from_numpy(y.astype(numpy.float32)).to(device)

                optimizer.zero_grad()
                inputs = {"peptide": peptide_tensor, "allele": allele_tensor}
                predictions = network(inputs)
                loss = loss_obj(predictions, y_tensor)
                regularization_penalty = self._regularization_penalty(
                    regularization_parameters,
                    l1=l1_reg,
                    l2=l2_reg,
                )
                if regularization_penalty is not None:
                    loss = loss + regularization_penalty
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            # Compute validation loss
            network.eval()
            with torch.no_grad():
                val_peptide = torch.from_numpy(validation_x_dict["peptide"]).float().to(device)
                val_allele = torch.from_numpy(validation_x_dict["allele"]).float().to(device)
                val_y = torch.from_numpy(output.astype(numpy.float32)).to(device)

                val_inputs = {"peptide": val_peptide, "allele": val_allele}
                val_predictions = network(val_inputs)
                val_loss = loss_obj(val_predictions, val_y)
                regularization_penalty = self._regularization_penalty(
                    regularization_parameters,
                    l1=l1_reg,
                    l2=l2_reg,
                )
                if regularization_penalty is not None:
                    val_loss = val_loss + regularization_penalty
                val_loss = val_loss.item()

            epoch_time = time.time() - epoch_start_time
            train_loss = numpy.mean(epoch_losses) if epoch_losses else float('nan')
            fit_info["loss"].append(train_loss)
            fit_info["val_loss"].append(val_loss)

            if min_val_loss is None or val_loss < min_val_loss - min_delta:
                min_val_loss = val_loss
                min_val_loss_iteration = epoch

            patience_epoch_threshold = min(
                epochs, max(min_val_loss_iteration + patience, min_epochs)
            )

            progress_message = (
                "epoch %3d/%3d [%0.2f sec.]: loss=%g val_loss=%g. Min val "
                "loss %g at epoch %s. Cum. points: %d. Stop at epoch %d."
                % (
                    epoch,
                    epochs,
                    epoch_time,
                    train_loss,
                    val_loss,
                    min_val_loss,
                    min_val_loss_iteration,
                    mutable_generator_state["yielded_values"],
                    patience_epoch_threshold,
                )
            ).strip()

            if progress_print_interval is not None and (
                time.time() - last_progress_print > progress_print_interval
            ):
                print(progress_preamble, progress_message)
                last_progress_print = time.time()

            if progress_callback:
                progress_callback()

            if epoch >= patience_epoch_threshold:
                if progress_print_interval is not None:
                    print(progress_preamble, "STOPPING", progress_message)
                break
            epoch += 1

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = mutable_generator_state["yielded_values"]
        self.fit_info.append(dict(fit_info))

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
            progress_print_interval=5.0):
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
        """
        configure_pytorch()
        device = self.get_device()

        encodable_peptides = EncodableSequences.create(peptides)
        peptide_encoding = self.peptides_to_network_input(encodable_peptides)
        fit_info = collections.defaultdict(list)

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
        peptide_encoding = peptide_encoding[shuffle_permutation]
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

        regularization_parameters = tuple(self._regularized_parameters(network))
        l1_reg = self.hyperparameters["dense_layer_l1_regularization"]
        l2_reg = self.hyperparameters["dense_layer_l2_regularization"]

        for epoch in range(self.hyperparameters["max_epochs"]):
            random_negative_peptides = EncodableSequences.create(
                random_negatives_planner.get_peptides()
            )
            random_negative_peptides_encoding = self.peptides_to_network_input(
                random_negative_peptides
            )

            # Build x_dict with random negatives
            if len(random_negative_peptides) > 0:
                x_peptide = numpy.concatenate([
                    random_negative_peptides_encoding,
                    x_dict_without_random_negatives["peptide"],
                ])
                if "allele" in x_dict_without_random_negatives:
                    x_allele = numpy.concatenate([
                        self.allele_encoding_to_network_input(
                            random_negatives_allele_encoding
                        )[0],
                        x_dict_without_random_negatives["allele"],
                    ])
                else:
                    x_allele = None
            else:
                x_peptide = x_dict_without_random_negatives["peptide"]
                x_allele = x_dict_without_random_negatives.get("allele")

            if needs_initialization:
                x_init = {"peptide": x_peptide}
                if x_allele is not None:
                    x_init["allele"] = x_allele
                self.data_dependent_weights_initialization(
                    network,
                    x_init,
                    method=self.hyperparameters["data_dependent_initialization_method"],
                    verbose=verbose,
                )
                needs_initialization = False

            # Train/val split (keep validation fixed)
            train_indices = train_indices_base.copy()
            numpy.random.shuffle(train_indices)

            # Training
            network.train()
            epoch_start = time.time()

            # Create batches
            batch_size = self.hyperparameters["minibatch_size"]
            train_losses = []

            for batch_start in range(0, n_train, batch_size):
                batch_idx = train_indices[batch_start:batch_start + batch_size]

                peptide_batch = torch.from_numpy(x_peptide[batch_idx]).float().to(device)
                y_batch = torch.from_numpy(y_encoded[batch_idx].astype(numpy.float32)).to(device)

                inputs = {"peptide": peptide_batch}
                if x_allele is not None:
                    allele_batch = torch.from_numpy(x_allele[batch_idx]).float().to(device)
                    inputs["allele"] = allele_batch

                optimizer.zero_grad()
                predictions = network(inputs)
                weights_batch = None
                if sample_weights_with_negatives is not None:
                    weights_batch = torch.from_numpy(
                        sample_weights_with_negatives[batch_idx]
                    ).float().to(device)
                loss = loss_obj(predictions, y_batch, sample_weights=weights_batch)
                regularization_penalty = self._regularization_penalty(
                    regularization_parameters,
                    l1=l1_reg,
                    l2=l2_reg,
                )
                if regularization_penalty is not None:
                    loss = loss + regularization_penalty
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            epoch_time = time.time() - epoch_start
            train_loss = numpy.mean(train_losses)
            fit_info["loss"].append(train_loss)

            # Validation
            if val_split > 0:
                network.eval()
                with torch.no_grad():
                    val_peptide = torch.from_numpy(x_peptide[val_indices]).float().to(device)
                    val_y = torch.from_numpy(y_encoded[val_indices].astype(numpy.float32)).to(device)
                    val_inputs = {"peptide": val_peptide}
                    if x_allele is not None:
                        val_allele = torch.from_numpy(x_allele[val_indices]).float().to(device)
                        val_inputs["allele"] = val_allele
                    val_predictions = network(val_inputs)
                    val_weights = None
                    if sample_weights_with_negatives is not None:
                        val_weights = torch.from_numpy(
                            sample_weights_with_negatives[val_indices]
                        ).float().to(device)
                    val_loss = loss_obj(
                        val_predictions,
                        val_y,
                        sample_weights=val_weights,
                    )
                    regularization_penalty = self._regularization_penalty(
                        regularization_parameters,
                        l1=l1_reg,
                        l2=l2_reg,
                    )
                    if regularization_penalty is not None:
                        val_loss = val_loss + regularization_penalty
                    val_loss = val_loss.item()
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

            # Early stopping
            if val_split > 0:
                if min_val_loss is None or (
                    val_loss < min_val_loss - self.hyperparameters["min_delta"]
                ):
                    min_val_loss = val_loss
                    min_val_loss_iteration = epoch

                if self.hyperparameters["early_stopping"]:
                    threshold = min_val_loss_iteration + self.hyperparameters["patience"]
                    if epoch > threshold:
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
                progress_callback()

            gc.collect()

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = len(peptides)
        self.fit_info.append(dict(fit_info))

    def predict(
            self,
            peptides,
            allele_encoding=None,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE,
            output_index=0):
        """
        Predict affinities.

        Parameters
        ----------
        peptides : EncodableSequences or list of string
        allele_encoding : AlleleEncoding, optional
        batch_size : int
        output_index : int or None

        Returns
        -------
        numpy.array of nM affinity predictions
        """
        assert self.prediction_cache is not None
        use_cache = allele_encoding is None and isinstance(peptides, EncodableSequences)
        if use_cache and peptides in self.prediction_cache:
            return self.prediction_cache[peptides].copy()

        configure_pytorch()
        device = self.get_device()

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
        network.eval()

        # Batch prediction
        n_samples = len(x_dict["peptide"])
        all_predictions = []

        def prediction_tensor(batch_array):
            batch_array = numpy.asarray(batch_array, dtype=numpy.float32)
            if not batch_array.flags.writeable:
                batch_array = batch_array.copy()
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
            allele_representations=None):
        """
        Helper function to make a PyTorch network for class 1 affinity prediction.
        """
        configure_pytorch()

        peptide_encoding_shape = self.peptides_to_network_input([]).shape[1:]

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

    def forward(self, inputs):
        outputs = [network(inputs) for network in self.networks]
        stacked = torch.stack(outputs, dim=-1)

        if self.merge_method == "average":
            return stacked.mean(dim=-1)
        elif self.merge_method == "sum":
            return stacked.sum(dim=-1)
        elif self.merge_method == "concatenate":
            return torch.cat(outputs, dim=-1)
        else:
            raise ValueError(f"Unknown merge method: {self.merge_method}")

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
            n_weights = len(list(network.parameters())) + len(list(network.buffers()))
            network.set_weights_list(weights[idx:idx + n_weights], auto_convert_keras=auto_convert_keras)
            idx += n_weights

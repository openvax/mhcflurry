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

"""Legacy Keras JSON config parsing, weight conversion, and model reconstruction.

Historical MHCflurry releases serialized ``Class1NeuralNetwork`` architectures
as Keras Functional/Model JSON configs paired with Keras-layout NPZ weight
arrays. Newer PyTorch releases serialize native hyperparameter/merged configs
with PyTorch-layout weight lists. This module isolates the config inspection,
allele-representation width inference, legacy Keras weight conversion, and
architecture reconstruction helpers used by :class:`Class1NeuralNetwork`.
"""

import logging

import numpy
import torch

from .amino_acid import get_vector_encoding_df
from .class1_encoding import (
    _peptide_torch_encoding_name,
    _peptide_torch_encoding_shape,
)


def is_keras_model_config(config):
    """Return True when ``config`` is a legacy Keras Model/Functional JSON dict."""
    return config.get("class_name") in ("Model", "Functional")


def parse_keras_json_config(config):
    """Parse a legacy Keras model JSON config to extract hyperparameters.

    Parameters
    ----------
    config : dict
        Keras model JSON config with ``'class_name'``, ``'config'``, etc.

    Returns
    -------
    tuple of (dict, dict)
        First dict: hyperparameters compatible with ``Class1NeuralNetwork``.
        Second dict: metadata about Keras model structure (e.g., embedding info).
    """
    layers = config.get("config", {}).get("layers", [])

    hyperparameters = {
        "locally_connected_layers": [],
        "layer_sizes": [],
        "activation": "tanh",
        "output_activation": "sigmoid",
        "dropout_probability": 0.0,
        "batch_normalization": False,
        "dense_layer_l1_regularization": 0.001,
        "dense_layer_l2_regularization": 0.0,
        "peptide_allele_merge_method": "multiply",  # Default
    }

    # Metadata about Keras structure
    keras_metadata = {
        "has_embedding": False,
        "embedding_input_dim": 0,
        "embedding_output_dim": 0,
        "skip_embedding_weights": False,
    }

    dense_layers = []
    peptide_dense_sizes = []
    allele_dense_sizes = []
    concatenate_count = 0
    for layer in layers:
        layer_class = layer.get("class_name", "")
        layer_config = layer.get("config", {})

        if layer_class == "LocallyConnected1D":
            kernel_size = layer_config.get("kernel_size", [3])
            lc_config = {
                "filters": layer_config.get("filters", 8),
                "kernel_size": (
                    kernel_size[0] if isinstance(kernel_size, list) else kernel_size
                ),
                "activation": layer_config.get("activation", "tanh"),
            }
            hyperparameters["locally_connected_layers"].append(lc_config)
            hyperparameters["activation"] = lc_config["activation"]

        elif layer_class == "Dense":
            units = layer_config.get("units", 32)
            activation = layer_config.get("activation", "tanh")
            layer_name = layer_config.get("name", "")
            if layer_name.startswith("peptide_dense_"):
                peptide_dense_sizes.append(units)
            elif layer_name.startswith("allele_dense_"):
                allele_dense_sizes.append(units)
            else:
                dense_layers.append({"units": units, "activation": activation})

            # Extract regularization from first dense layer
            kernel_reg = layer_config.get("kernel_regularizer")
            if kernel_reg and isinstance(kernel_reg, dict):
                reg_config = kernel_reg.get("config", {})
                if "l1" in reg_config:
                    hyperparameters["dense_layer_l1_regularization"] = reg_config["l1"]
                if "l2" in reg_config:
                    hyperparameters["dense_layer_l2_regularization"] = reg_config["l2"]

        elif layer_class == "Dropout":
            rate = layer_config.get("rate", 0.0)
            hyperparameters["dropout_probability"] = 1.0 - rate

        elif layer_class == "BatchNormalization":
            hyperparameters["batch_normalization"] = True

        elif layer_class == "LayerNormalization":
            hyperparameters["normalization"] = "layer"

        elif layer_class == "Embedding":
            keras_metadata["has_embedding"] = True
            keras_metadata["embedding_input_dim"] = layer_config.get("input_dim", 0)
            keras_metadata["embedding_output_dim"] = layer_config.get("output_dim", 0)
            # If input_dim is 0, it's a placeholder and weights should be skipped
            if layer_config.get("input_dim", 0) == 0:
                keras_metadata["skip_embedding_weights"] = True

        elif layer_class == "Concatenate":
            concatenate_count += 1
            # Only set merge_method to concatenate if there's just one Concatenate
            # and it's likely for peptide-allele merging (not DenseNet skip connections)
            if concatenate_count == 1:
                hyperparameters["peptide_allele_merge_method"] = "concatenate"

        elif layer_class == "Multiply":
            hyperparameters["peptide_allele_merge_method"] = "multiply"

    # Multiple Concatenate layers indicate DenseNet topology with skip connections.
    # The first Concatenate is typically for peptide-allele merging;
    # subsequent ones are for DenseNet skip connections.
    if concatenate_count > 1:
        hyperparameters["topology"] = "with-skip-connections"

    # The last Dense layer is the output layer
    if dense_layers:
        hyperparameters["output_activation"] = dense_layers[-1]["activation"]
        hyperparameters["num_outputs"] = dense_layers[-1]["units"]
        hyperparameters["layer_sizes"] = [d["units"] for d in dense_layers[:-1]]
        if dense_layers[:-1]:
            hyperparameters["activation"] = dense_layers[0]["activation"]

    if peptide_dense_sizes:
        hyperparameters["peptide_dense_layer_sizes"] = peptide_dense_sizes
    if allele_dense_sizes:
        hyperparameters["allele_dense_layer_sizes"] = allele_dense_sizes

    return hyperparameters, keras_metadata


def infer_allele_representation_dim_from_weights(
    network_weights, hyperparameters, network_config=None
):
    """Infer flattened allele representation width from serialized weights.

    Strategy:
      1. Compute the position of the first allele-stage weight from
         the architecture config. In ``named_parameters()`` order the
         layout is::

            locally_connected -> peptide_dense -> batch_norm_early?
              -> allele_dense -> allele_embedding -> dense_layers
              -> output

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
    if not network_weights or not hyperparameters.get("allele_amino_acid_encoding"):
        return None
    try:
        encoding_dim = len(
            get_vector_encoding_df(
                hyperparameters["allele_amino_acid_encoding"]
            ).columns
        )
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
        "allele_dense_layer_sizes",
        hyperparameters.get("allele_dense_layer_sizes", []),
    )

    peptide_offset = 0
    peptide_offset += 2 * len(
        architecture.get(
            "locally_connected_layers",
            hyperparameters.get("locally_connected_layers", []),
        )
    )
    peptide_offset += 2 * len(
        architecture.get(
            "peptide_dense_layer_sizes",
            hyperparameters.get("peptide_dense_layer_sizes", []),
        )
    )
    if architecture.get(
        "batch_normalization",
        hyperparameters.get("batch_normalization", False),
    ) or (
        architecture.get(
            "normalization",
            hyperparameters.get("normalization", "none"),
        )
        in ("batch", "layer")
    ):
        peptide_offset += 2

    n = len(network_weights)

    # Position-based primary lookup.
    primary_dim = (
        candidate_dim(network_weights[peptide_offset])
        if peptide_offset < n
        else None
    )

    # Cross-check: when allele_dense layers exist, the same shape[1]
    # must appear at the embedding position (peptide_offset +
    # 2*n_allele_dense_layers). If they disagree, the offset is
    # wrong and we fall back to a duplicate-shape scan.
    if primary_dim is not None and allele_dense_sizes:
        embedding_idx = peptide_offset + 2 * len(allele_dense_sizes)
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


def placeholder_allele_representations(
    hyperparameters,
    network_weights=None,
    network_config=None,
    default_sequence_length=39,
):
    """Create a shape-compatible placeholder for pan-allele models.

    Public artifacts have used several pseudosequence widths over time
    (including 2.1.x-era weights). Prefer the width encoded in the saved
    weights; fall back to the current 39-residue sequence only when no
    weights are available to guide reconstruction.
    """
    if not hyperparameters.get("allele_amino_acid_encoding"):
        return None
    encoding_name = hyperparameters["allele_amino_acid_encoding"]
    try:
        encoding_df = get_vector_encoding_df(encoding_name)
    except KeyError:
        return None
    embedding_dim = infer_allele_representation_dim_from_weights(
        network_weights,
        hyperparameters,
        network_config=network_config,
    )
    if embedding_dim is None:
        embedding_dim = default_sequence_length * len(encoding_df.columns)
    return numpy.zeros((1, embedding_dim), dtype=numpy.float32)


def _assign_keras_dense(layer, w, b):
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


def _assign_keras_locally_connected(layer, w, b):
    w = w.astype(numpy.float32)
    b = b.astype(numpy.float32)
    if len(w.shape) == 5 and w.shape[1] == 1:
        out_len, _, k, in_ch, out_ch = w.shape
        w = w.squeeze(1)
        w = w.reshape(out_len, k * in_ch, out_ch)
        w = w.transpose(0, 2, 1)
    elif len(w.shape) == 3 and w.shape[0] == layer.output_length:
        # Keras (out_len, k*in_ch, out_ch) -> PyTorch (out_len, out_ch, in_ch*k)
        if (
            w.shape[1] == layer.weight.shape[2]
            and w.shape[2] == layer.weight.shape[1]
        ):
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


def _assign_keras_batch_norm(layer, gamma, beta, mean, var):
    layer.weight.data = torch.from_numpy(gamma.astype(numpy.float32)).to(
        device=layer.weight.device, dtype=layer.weight.dtype
    )
    layer.bias.data = torch.from_numpy(beta.astype(numpy.float32)).to(
        device=layer.bias.device, dtype=layer.bias.dtype
    )
    layer.running_mean.data = torch.from_numpy(mean.astype(numpy.float32)).to(
        device=layer.running_mean.device,
        dtype=layer.running_mean.dtype,
    )
    layer.running_var.data = torch.from_numpy(var.astype(numpy.float32)).to(
        device=layer.running_var.device,
        dtype=layer.running_var.dtype,
    )


def load_keras_weights_into_model(
    model, weights, *, keras_config=None, keras_metadata=None
):
    """Convert and assign Keras-layout weight arrays into ``model``.

    When ``keras_config`` is provided, weights are assigned in the layer order
    recorded in the Keras JSON graph. Otherwise, weights are matched against
    ``model.named_parameters()`` with Keras-to-PyTorch shape transposition and
    BatchNormalization moving-statistic unpacking.
    """
    skip_keras_embedding = bool(
        keras_metadata and keras_metadata.get("skip_embedding_weights", False)
    )

    if keras_config:
        keras_layers = keras_config.get("config", {}).get("layers", [])
        idx = 0
        for layer in keras_layers:
            layer_class = layer.get("class_name", "")
            layer_name = layer.get("config", {}).get("name", "")

            if layer_class == "Dense":
                w = weights[idx]
                b = weights[idx + 1]
                idx += 2
                if layer_name == "output":
                    _assign_keras_dense(model.output_layer, w, b)
                elif layer_name.startswith("dense_"):
                    dense_idx = int(layer_name.split("_")[1])
                    _assign_keras_dense(model.dense_layers[dense_idx], w, b)
                elif layer_name.startswith("peptide_dense_"):
                    dense_idx = int(layer_name.split("_")[2])
                    _assign_keras_dense(
                        model.peptide_dense_layers[dense_idx], w, b
                    )
                elif layer_name.startswith("allele_dense_"):
                    dense_idx = int(layer_name.split("_")[2])
                    _assign_keras_dense(
                        model.allele_dense_layers[dense_idx], w, b
                    )
            elif layer_class == "LocallyConnected1D":
                w = weights[idx]
                b = weights[idx + 1]
                idx += 2
                lc_idx = int(layer_name.split("_")[1])
                _assign_keras_locally_connected(model.lc_layers[lc_idx], w, b)
            elif layer_class == "Embedding":
                w = weights[idx]
                idx += 1
                if skip_keras_embedding or model.allele_embedding is None:
                    continue
                if w.shape == model.allele_embedding.weight.shape:
                    target = model.allele_embedding.weight
                    model.allele_embedding.weight.data = torch.from_numpy(
                        w.astype(numpy.float32)
                    ).to(device=target.device, dtype=target.dtype)
            elif layer_class == "BatchNormalization":
                gamma = weights[idx]
                beta = weights[idx + 1]
                mean = weights[idx + 2]
                var = weights[idx + 3]
                idx += 4
                if layer_name == "batch_norm_early":
                    if model.batch_norm_early is not None:
                        _assign_keras_batch_norm(
                            model.batch_norm_early, gamma, beta, mean, var
                        )
                elif layer_name.startswith("batch_norm_"):
                    bn_idx = int(layer_name.split("_")[2])
                    if model.batch_norms[bn_idx] is not None:
                        _assign_keras_batch_norm(
                            model.batch_norms[bn_idx], gamma, beta, mean, var
                        )
        return

    idx = 0
    named_modules = dict(model.named_modules())

    for name, param in model.named_parameters():
        # Skip allele_embedding when loading Keras weights with placeholder
        if skip_keras_embedding and "allele_embedding" in name:
            # Placeholder embeddings have shape (0, embed_dim)
            while (
                idx < len(weights)
                and len(weights[idx].shape) == 2
                and weights[idx].shape[0] == 0
            ):
                idx += 1
            continue
        w = weights[idx].astype(numpy.float32)
        extra_keras_skip = 0
        module = None
        if "." in name:
            module_name = name.rsplit(".", 1)[0]
            module = named_modules.get(module_name)

        # Skip allele_embedding if shapes don't match (pan-allele models);
        # the embedding will be set by set_allele_representations later.
        if "allele_embedding" in name and w.shape != param.shape:
            idx += 1
            continue

        # Dense/Linear layer: Keras (in, out) -> PyTorch (out, in).
        # Must transpose even when shapes match (square matrices).
        is_linear_weight = (
            "weight" in name and "embedding" not in name and len(w.shape) == 2
        )
        if is_linear_weight and (
            w.shape == param.shape[::-1]
            or (w.shape == param.shape and w.shape[0] == w.shape[1])
        ):
            w = w.T
        # LocallyConnected1D weight: Keras (out_len, 1, k, in_ch, out_ch)
        # -> PyTorch (out_len, out_ch, in_ch * k)
        elif len(w.shape) == 5 and w.shape[1] == 1:
            out_len, _, k, in_ch, out_ch = w.shape
            w = w.squeeze(1)
            w = w.reshape(out_len, k * in_ch, out_ch)
            w = w.transpose(0, 2, 1)
        # LocallyConnected1D weight (3D): Keras (out_len, k*in_ch, out_ch)
        # -> PyTorch (out_len, out_ch, k*in_ch)
        elif (
            len(w.shape) == 3
            and w.shape[0] == param.shape[0]
            and w.shape[1] == param.shape[2]
            and w.shape[2] == param.shape[1]
        ):
            w = w.transpose(0, 2, 1)
        # LocallyConnected1D bias: Keras (out_len * out_ch,) -> PyTorch (out_len, out_ch)
        elif (
            len(w.shape) == 1
            and len(param.shape) == 2
            and w.shape[0] == param.shape[0] * param.shape[1]
        ):
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
                    module.running_var.data = torch.from_numpy(running_var).to(
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


def create_model_from_config(
    config,
    *,
    model_cls,
    merged_model_cls,
    network_cls,
    instance_hyperparameters=None,
    network_weights=None,
):
    """Create a ``Class1NeuralNetworkModel`` (or merged model) from config.

    Parameters
    ----------
    config : dict
        Configuration dictionary (either legacy Keras JSON, native
        hyperparameters dict, or merged-network dict).
    model_cls : type
        ``Class1NeuralNetworkModel`` class.
    merged_model_cls : type
        ``MergedClass1NeuralNetwork`` class.
    network_cls : type
        ``Class1NeuralNetwork`` class.
    instance_hyperparameters : dict, optional
        Hyperparameters from the ``Class1NeuralNetwork`` instance.
        These take precedence for things like ``peptide_encoding``.
    network_weights : list of numpy.ndarray, optional
        Saved weights used to infer the placeholder allele representation.
    """
    keras_metadata = None

    # Check if this is a merged network config
    if config.get("merged_networks"):
        return create_merged_model_from_config(
            config,
            model_cls=model_cls,
            merged_model_cls=merged_model_cls,
            network_cls=network_cls,
            instance_hyperparameters=instance_hyperparameters,
            network_weights=network_weights,
        )

    # Check if this is a legacy Keras JSON config
    if is_keras_model_config(config):
        hyperparameters, keras_metadata = parse_keras_json_config(config)
    else:
        hyperparameters = config.get("hyperparameters", config)

    # Merge with instance hyperparameters if provided; instance
    # hyperparameters take precedence for things like peptide_encoding.
    if instance_hyperparameters:
        merged = dict(instance_hyperparameters)
        for key in (
            "layer_sizes",
            "locally_connected_layers",
            "dropout_probability",
            "batch_normalization",
            "normalization",
            "activation",
            "output_activation",
            "peptide_allele_merge_method",
        ):
            if key in hyperparameters:
                merged[key] = hyperparameters[key]
        hyperparameters = merged

    hyperparameters = network_cls.hyperparameter_defaults.with_defaults(
        network_cls.apply_hyperparameter_renames(dict(hyperparameters))
    )

    temp = network_cls(**hyperparameters)
    peptide_encoding_shape = temp.peptides_to_network_input([]).shape[1:]
    peptide_torch_encoding_name = _peptide_torch_encoding_name(hyperparameters)
    if peptide_torch_encoding_name and len(peptide_encoding_shape) == 1:
        peptide_encoding_shape = _peptide_torch_encoding_shape(
            peptide_encoding_shape,
            peptide_torch_encoding_name,
        )

    allele_representations = config.get("allele_representations")
    if allele_representations is not None:
        allele_representations = numpy.array(allele_representations)

    # For pan-allele Keras models with placeholder embedding (input_dim=0),
    # create a placeholder allele representation to ensure correct architecture.
    if (
        allele_representations is None
        and keras_metadata is not None
        and keras_metadata.get("has_embedding", False)
        and keras_metadata.get("embedding_output_dim", 0) > 0
    ):
        embedding_dim = keras_metadata["embedding_output_dim"]
        allele_representations = numpy.zeros(
            (1, embedding_dim), dtype=numpy.float32
        )

    # For PyTorch-format configs without allele_representations but with
    # allele_amino_acid_encoding (pan-allele models), create placeholder.
    has_allele = config.get("has_allele", True)
    if (
        allele_representations is None
        and keras_metadata is None
        and has_allele
        and hyperparameters.get("allele_amino_acid_encoding")
    ):
        allele_representations = placeholder_allele_representations(
            hyperparameters,
            network_weights=network_weights,
            network_config=config,
        )

    model = model_cls(
        peptide_encoding_shape=peptide_encoding_shape,
        allele_representations=allele_representations,
        locally_connected_layers=hyperparameters.get(
            "locally_connected_layers", []
        ),
        peptide_dense_layer_sizes=hyperparameters.get(
            "peptide_dense_layer_sizes", []
        ),
        allele_dense_layer_sizes=hyperparameters.get(
            "allele_dense_layer_sizes", []
        ),
        layer_sizes=hyperparameters.get("layer_sizes", [32]),
        peptide_allele_merge_method=hyperparameters.get(
            "peptide_allele_merge_method", "multiply"
        ),
        peptide_allele_merge_activation=hyperparameters.get(
            "peptide_allele_merge_activation", ""
        ),
        activation=hyperparameters.get("activation", "tanh"),
        output_activation=hyperparameters.get("output_activation", "sigmoid"),
        dropout_probability=hyperparameters.get("dropout_probability", 0.0),
        batch_normalization=hyperparameters.get("batch_normalization", False),
        normalization=hyperparameters.get("normalization", "none"),
        dense_layer_l1_regularization=hyperparameters.get(
            "dense_layer_l1_regularization", 0.001
        ),
        dense_layer_l2_regularization=hyperparameters.get(
            "dense_layer_l2_regularization", 0.0
        ),
        topology=hyperparameters.get("topology", "feedforward"),
        num_outputs=hyperparameters.get("num_outputs", 1),
        init=hyperparameters.get("init", "glorot_uniform"),
        peptide_input_vector_encoding_name=peptide_torch_encoding_name,
    )

    if keras_metadata is not None:
        model._keras_metadata = keras_metadata
        model._keras_config = config

    return model


def create_merged_model_from_config(
    config,
    *,
    model_cls,
    merged_model_cls,
    network_cls,
    instance_hyperparameters=None,
    network_weights=None,
):
    """Create a ``MergedClass1NeuralNetwork`` from a merged configuration dict.

    Parameters
    ----------
    config : dict
        Configuration dictionary with ``'merged_networks'`` key.
    model_cls : type
        ``Class1NeuralNetworkModel`` class.
    merged_model_cls : type
        ``MergedClass1NeuralNetwork`` class.
    network_cls : type
        ``Class1NeuralNetwork`` class.
    instance_hyperparameters : dict, optional
        Hyperparameters from the ``Class1NeuralNetwork`` instance.
    network_weights : list of numpy.ndarray, optional
        Saved weights used to infer the placeholder allele representation.
    """
    merged_configs = config["merged_networks"]
    merge_method = config.get("merge_method", "average")

    base_hyperparameters = config.get("hyperparameters", {})
    if instance_hyperparameters:
        base_hyperparameters = dict(instance_hyperparameters)
        base_hyperparameters.update(config.get("hyperparameters", {}))
    base_hyperparameters = network_cls.hyperparameter_defaults.with_defaults(
        network_cls.apply_hyperparameter_renames(dict(base_hyperparameters))
    )
    temp = network_cls(**base_hyperparameters)
    peptide_encoding_shape = temp.peptides_to_network_input([]).shape[1:]
    peptide_torch_encoding_name = _peptide_torch_encoding_name(
        base_hyperparameters
    )
    if peptide_torch_encoding_name and len(peptide_encoding_shape) == 1:
        peptide_encoding_shape = _peptide_torch_encoding_shape(
            peptide_encoding_shape,
            peptide_torch_encoding_name,
        )

    allele_representations = placeholder_allele_representations(
        base_hyperparameters,
        network_weights=network_weights,
        network_config=config,
    )

    sub_networks = []
    for sub_config in merged_configs:
        model = model_cls(
            peptide_encoding_shape=peptide_encoding_shape,
            allele_representations=allele_representations,
            locally_connected_layers=sub_config.get(
                "locally_connected_layers", []
            ),
            peptide_dense_layer_sizes=sub_config.get(
                "peptide_dense_layer_sizes", []
            ),
            allele_dense_layer_sizes=sub_config.get(
                "allele_dense_layer_sizes", []
            ),
            layer_sizes=sub_config.get("layer_sizes", [32]),
            peptide_allele_merge_method=sub_config.get(
                "peptide_allele_merge_method", "multiply"
            ),
            peptide_allele_merge_activation=sub_config.get(
                "peptide_allele_merge_activation", ""
            ),
            activation=sub_config.get("activation", "tanh"),
            output_activation=sub_config.get("output_activation", "sigmoid"),
            dropout_probability=sub_config.get("dropout_probability", 0.0),
            batch_normalization=sub_config.get("batch_normalization", False),
            normalization=sub_config.get("normalization", "none"),
            dense_layer_l1_regularization=base_hyperparameters.get(
                "dense_layer_l1_regularization", 0.001
            ),
            dense_layer_l2_regularization=base_hyperparameters.get(
                "dense_layer_l2_regularization", 0.0
            ),
            topology=sub_config.get("topology", "feedforward"),
            num_outputs=sub_config.get("num_outputs", 1),
            init=base_hyperparameters.get("init", "glorot_uniform"),
            peptide_input_vector_encoding_name=peptide_torch_encoding_name,
        )
        sub_networks.append(model)

    return merged_model_cls(sub_networks, merge_method=merge_method)

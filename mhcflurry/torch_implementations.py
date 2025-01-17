"""
PyTorch implementations of MHCflurry neural networks.
"""

import os
import json
import weakref
import numpy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.regression_target import to_ic50


def to_torch(x):
    """Convert numpy array to torch tensor."""
    if isinstance(x, numpy.ndarray):
        return torch.from_numpy(x).float()
    return x


def to_numpy(x):
    """Convert torch tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class TorchNeuralNetwork(nn.Module):
    """
    PyTorch implementation of a neural network with MHCflurry hyperparameters
    """

    @classmethod
    def from_config(cls, config):
        hyperparameters = config["hyperparameters"]
        instance = cls(**hyperparameters)
        return instance

    @property
    def supported_alleles(self):
        if "supported_alleles" not in self._cache:
            result = set(self.allele_to_allele_specific_models)
            if self.allele_to_sequence:
                result = result.union(self.allele_to_sequence)
            self._cache["supported_alleles"] = sorted(result)
        return self._cache["supported_alleles"]

    def __init__(self, **hyperparameters):
        """
        Initialize neural network with hyperparameters matching Keras version.

        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters as defined in Class1NeuralNetwork
        """
        super().__init__()

        # Apply hyperparameter renames
        renamed = self._rename_hyperparameters(hyperparameters)

        ALLOWED_KEYS = {
            "allele_amino_acid_encoding",
            "allele_dense_layer_sizes", 
            "peptide_encoding",
            "peptide_dense_layer_sizes",
            "peptide_allele_merge_method",
            "peptide_allele_merge_activation",
            "layer_sizes",
            "dense_layer_l1_regularization",
            "dense_layer_l2_regularization", 
            "activation",
            "init",
            "output_activation",
            "dropout_probability",
            "batch_normalization",
            "locally_connected_layers",
            "topology",
            "num_outputs",
        }

        filtered = {}
        for (k, v) in renamed.items():
            if k in ALLOWED_KEYS:
                filtered[k] = v

        defaults = self._get_hyperparameter_defaults().defaults
        final = dict(defaults)  # baseline defaults
        final.update(filtered)  # override with only allowed keys

        self.hyperparameters = final

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regularization_losses = []
        self._network = None
        # Set default dtype to match Keras
        torch.set_default_dtype(torch.float64)
        self.network_json = None
        self.network_weights = None
        self.network_weights_loader = None
        self.fit_info = []
        self.prediction_cache = weakref.WeakKeyDictionary()

        # Set activation functions
        self.hidden_activation = self._get_activation_function(hyperparameters.get("activation", "tanh"))
        self.output_activation = self._get_activation_function(hyperparameters.get("output_activation", "sigmoid"))

        # Build network based on hyperparameters
        self._build_network()
        self.to(self.device)

    def get_config(self):
        """
        Return a dict of the same shape as Class1NeuralNetwork's get_config().
        """
        config = {
            "hyperparameters": self.hyperparameters,
            "_network": None,
            "network_json": None,
            "network_weights": None,
            "network_weights_loader": None,
            "fit_info": [],
            "prediction_cache": None,
        }
        logging.info("[TORCH get_config] returning: %s", config)
        return config

    def _rename_hyperparameters(self, hyperparameters):
        """
        Rename hyperparameters according to predefined mapping.

        Parameters
        ----------
        hyperparameters : dict
            Original hyperparameters

        Returns
        -------
        dict : Updated hyperparameters with renames applied
        """
        renames = {
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
        result = dict(hyperparameters)
        for old_name, new_name in renames.items():
            if old_name in result:
                val = result.pop(old_name)
                if new_name is not None:
                    result[new_name] = val

        # We keep all hyperparameters to maintain compatibility with Keras models
                
        return result

    def _get_hyperparameter_defaults(self):
        """
        Get default hyperparameters.

        Returns
        -------
        HyperparameterDefaults
        """
        from .hyperparameters import HyperparameterDefaults

        return HyperparameterDefaults(
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
            locally_connected_layers=[{"filters": 8, "activation": "tanh", "kernel_size": 3}],
            topology="feedforward",
            num_outputs=1,
        )

    def load_weights_from_keras(self, keras_model):
        """
        Load weights from the given Keras model into this PyTorch model.
        Make sure the layer order, shapes, and counts match exactly.
        """

        # Gather the Linear/BatchNorm layers in the order they should match
        torch_layers = []
        for layer in self.peptide_layers:
            if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                torch_layers.append(layer)

        for layer in self.dense_layers:
            if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                torch_layers.append(layer)

        torch_layers.append(self.output_layer)

        torch_index = 0
        for keras_layer in keras_model.layers:
            layer_type = keras_layer.__class__.__name__
            if layer_type == "Dense":
                # Keras Dense: weights[0].shape = (in_dim, out_dim)
                # PyTorch Linear: weight.shape = (out_dim, in_dim)
                w, b = keras_layer.get_weights()
                linear = torch_layers[torch_index]
                linear.weight.data = torch.from_numpy(w.T).double()
                linear.bias.data = torch.from_numpy(b).double()
                torch_index += 1

            elif layer_type == "BatchNormalization":
                # Keras BN: [gamma, beta, moving_mean, moving_var]
                gamma, beta, moving_mean, moving_var = keras_layer.get_weights()
                bn = torch_layers[torch_index]

                # Check if shapes match the corresponding Torch BN layer.
                # If they do not match (e.g., 315 vs 64), skip this BN.
                if gamma.shape != bn.weight.data.shape:
                    continue  # Do not increment torch_index; just skip

                # Otherwise, the shapes match, proceed to copy
                bn.weight.data.copy_(torch.from_numpy(gamma).double())
                bn.bias.data.copy_(torch.from_numpy(beta).double())
                bn.running_mean.copy_(torch.from_numpy(moving_mean).double())
                bn.running_var.copy_(torch.from_numpy(moving_var).double())
                # Set PyTorch BN hyperparams to match Keras
                bn.momentum = 0.01
                bn.eps = 1e-3
                bn.eval()
                torch_index += 1

            else:
                pass

    def load_weights(self, weights_filename):
        """
        Load network weights from a file.

        Parameters
        ----------
        weights_filename : str
            Path to weights file
        """
        weights = numpy.load(weights_filename, allow_pickle=True)
        if isinstance(weights, numpy.ndarray):
            weights = weights.item()
        self.network_weights = list(weights.values())

    def _build_network(self):
        """Build PyTorch network matching Keras architecture"""
        # Get dimensions from peptide encoding config
        peptide_input_dim = self._get_peptide_input_dim()

        # Ensure network uses double precision
        self.double()

        # Input layers
        self.peptide_layers = nn.ModuleList()
        current_size = peptide_input_dim

        # Build peptide dense layers
        for size in self.hyperparameters["peptide_dense_layer_sizes"]:
            print(f"[DEBUG] Adding peptide dense layer: in={current_size}, out={size}")
            linear = nn.Linear(current_size, size)
            self.peptide_layers.append(linear)
            if self.hyperparameters["dense_layer_l1_regularization"] > 0:
                self.regularization_losses.append(
                    lambda: self.hyperparameters["dense_layer_l1_regularization"] * linear.weight.abs().sum()
                )
            if self.hyperparameters["dropout_probability"] > 0:
                self.peptide_layers.append(nn.Dropout(self.hyperparameters["dropout_probability"]))
            current_size = size

        # Allele representation layers
        if self.hyperparameters["allele_dense_layer_sizes"]:
            self.allele_layers = nn.ModuleList()
            self.allele_embedding = nn.Embedding(
                num_embeddings=1,  # Will be set when allele representations are loaded
                embedding_dim=1,  # Will be set when allele representations are loaded
            )
            current_allele_size = self.allele_embedding.embedding_dim
            for size in self.hyperparameters["allele_dense_layer_sizes"]:
                self.allele_layers.append(nn.Linear(current_allele_size, size))
                current_allele_size = size

        # Locally connected layers
        self.local_layers = nn.ModuleList()
        for params in self.hyperparameters["locally_connected_layers"]:
            kernel_size = params["kernel_size"]
            # Ensure groups divides input channels evenly
            groups = min(current_size, current_size // kernel_size)
            if current_size % groups != 0:
                groups = 1

            self.local_layers.append(
                nn.Conv1d(
                    in_channels=current_size,
                    out_channels=params["filters"] * groups,
                    kernel_size=kernel_size,
                    groups=groups,
                    bias=True,
                )
            )
            if params["activation"] == "tanh":
                self.local_layers.append(nn.Tanh())
            current_size = params["filters"] * (current_size - params["kernel_size"] + 1)

        # Main dense layers
        self.dense_layers = nn.ModuleList()
        self.layer_outputs = []  # For skip connections

        for size in self.hyperparameters["layer_sizes"]:
            if self.hyperparameters["topology"] == "with-skip-connections" and len(self.layer_outputs) > 1:
                current_size = sum(l.out_features for l in self.layer_outputs[-2:])

            linear = nn.Linear(current_size, size)
            if self.hyperparameters["dense_layer_l1_regularization"] > 0:
                self.regularization_losses.append(
                    lambda: self.hyperparameters["dense_layer_l1_regularization"] * linear.weight.abs().sum()
                )
            self.dense_layers.append(linear)
            self.layer_outputs.append(linear)

            if self.hyperparameters["batch_normalization"]:
                self.dense_layers.append(nn.BatchNorm1d(size))
            if self.hyperparameters["dropout_probability"] > 0:
                self.dense_layers.append(nn.Dropout(self.hyperparameters["dropout_probability"]))
            current_size = size

        # Output layer
        self.output_layer = nn.Linear(current_size, self.hyperparameters["num_outputs"])

    def _get_activation_function(self, name):
        """Convert activation function name to PyTorch function"""
        if callable(name):
            return name
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "linear": lambda x: x,
            "": lambda x: x,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        return activations[name]

    def _get_peptide_input_dim(self):
        """Calculate input dimension from peptide encoding config"""
        encoding = self.hyperparameters["peptide_encoding"]
        max_length = encoding["max_length"]
        # This is simplified - would need to match exact Keras dimension calculation
        return max_length * 21  # 21 amino acids

    def __del__(self):
        """Cleanup GPU memory"""
        if hasattr(self, "_network"):
            del self._network
        torch.cuda.empty_cache()

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        return super().to(device)

    def train(self, mode=True):
        """Set training mode"""
        super().train(mode)
        # Ensure batch norm uses running stats in eval mode
        if not mode:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.track_running_stats = True
        return self

    def eval(self):
        """
        Put this TorchNeuralNetwork in eval mode (batchnorm, dropout at inference).
        """
        return super().eval()

    def init_weights(self, init):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init == "glorot_uniform":
                    # TensorFlow's glorot_uniform is slightly different from PyTorch's xavier_uniform
                    # TF uses a uniform distribution between [-limit, limit] where:
                    # limit = sqrt(6 / (fan_in + fan_out))
                    fan_in = module.weight.size(1)
                    fan_out = module.weight.size(0)
                    limit = numpy.sqrt(6.0 / (fan_in + fan_out))
                    nn.init.uniform_(module.weight, -limit, limit)
                    nn.init.zeros_(module.bias)
                else:
                    raise ValueError(f"Unsupported initialization: {init}")

    def set_allele_representations(self, allele_representations):
        """
        Set allele representations in the embedding layer.

        Parameters
        ----------
        allele_representations : numpy.ndarray
            Matrix of allele representations
        """
        if hasattr(self, "allele_embedding"):
            with torch.no_grad():
                self.allele_embedding.weight.copy_(torch.from_numpy(allele_representations).to(self.device))

    def forward(self, x):
        """
        Run a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output predictions
        """
        x = x.to(self.device)

        # Process peptide layers
        for layer in self.peptide_layers:
            if isinstance(layer, nn.Linear):
                x = self.hidden_activation(layer(x))
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            x = x.to(self.device)

        # Process main dense layers with Keras-matching activation order
        for layer in self.dense_layers:
            layer = layer.to(self.device)
            x = x.to(self.device)
            if isinstance(layer, nn.Linear):
                x_pre = x
                x = layer(x)  # Linear transformation
                x = self.hidden_activation(x)  # Activation immediately after linear
            elif isinstance(layer, nn.BatchNorm1d):
                x_pre = x
                x = layer(x)  # Then batch norm

        # Output layer with sigmoid activation
        self.output_layer = self.output_layer.to(self.device)
        x = self.output_layer(x)
        x = self.output_activation(x)

        return x

    def predict(self, peptides, batch_size=32):
        """
        Predict output for a list of peptides or an EncodableSequences object.

        Parameters
        ----------
        peptides : list of str or EncodableSequences
            Peptides to predict
        batch_size : int
            Number of items per batch

        Returns
        -------
        numpy.ndarray
            Predictions as a 1D or 2D array depending on self.hyperparameters["num_outputs"]
        """
        # Add TorchNeuralNetwork to builtins so other modules can find it
        import builtins

        builtins.TorchNeuralNetwork = TorchNeuralNetwork
        """
        Predict output for a list of peptides or an EncodableSequences object.

        Parameters
        ----------
        peptides : list of str or EncodableSequences
            Peptides to predict
        batch_size : int
            Number of items per batch

        Returns
        -------
        numpy.ndarray
            Predictions as a 1D or 2D array depending on self.hyperparameters["num_outputs"]
        """
        from .encodable_sequences import EncodableSequences

        # Convert list of peptides to EncodableSequences if necessary
        if not isinstance(peptides, EncodableSequences):
            peptides = EncodableSequences.create(peptides)

        # Encode peptides as a 2D array [N x encoded_length]
        encoded = peptides.variable_length_to_fixed_length_vector_encoding(
            "BLOSUM62", alignment_method="pad_middle", max_length=15
        )
        encoded = encoded.reshape(encoded.shape[0], -1)

        # Run the network in evaluation mode, possibly in batches
        self.eval()
        outputs_list = []
        with torch.no_grad():
            for start in range(0, len(encoded), batch_size):
                batch = encoded[start : start + batch_size]
                batch_tensor = to_torch(batch).to(self.device, dtype=torch.float64)
                batch_tensor = batch_tensor.double()
                batch_output = self(batch_tensor)
                outputs_list.append(batch_output.cpu())

        # Concatenate all batches and return as numpy
        final_outputs = torch.cat(outputs_list, dim=0)
        if final_outputs.dim() == 1:
            final_outputs = final_outputs.unsqueeze(1)
        # Convert network output (0-1) to nM predictions
        # Using same conversion as Keras version
        final_outputs = final_outputs.to(torch.float64)
        const_50000 = torch.tensor(50000.0, device=final_outputs.device, dtype=torch.float64)
        final_outputs = torch.pow(const_50000, (1.0 - final_outputs))
        final_outputs = final_outputs.squeeze(-1)  # shape (N,)

        return to_numpy(final_outputs)


class Class1AffinityPredictor(object):
    """
    PyTorch implementation of Class1AffinityPredictor.
    """

    def __init__(
        self,
        allele_to_allele_specific_models=None,
        class1_pan_allele_models=None,
        allele_to_sequence=None,
        manifest_df=None,
        allele_to_percent_rank_transform=None,
        metadata_dataframes=None,
        provenance_string=None,
        optimization_info=None,
    ):

        if allele_to_allele_specific_models is None:
            allele_to_allele_specific_models = {}
        if class1_pan_allele_models is None:
            class1_pan_allele_models = []

        self.allele_to_sequence = dict(allele_to_sequence) if allele_to_sequence is not None else None

        self._master_allele_encoding = None
        if class1_pan_allele_models:
            assert self.allele_to_sequence

        self.allele_to_allele_specific_models = allele_to_allele_specific_models
        self.class1_pan_allele_models = class1_pan_allele_models
        self._manifest_df = manifest_df

        if not allele_to_percent_rank_transform:
            allele_to_percent_rank_transform = {}
        self.allele_to_percent_rank_transform = allele_to_percent_rank_transform
        self.metadata_dataframes = dict(metadata_dataframes) if metadata_dataframes else {}
        self._cache = {}
        self.optimization_info = optimization_info if optimization_info else {}

        assert isinstance(self.allele_to_allele_specific_models, dict)
        assert isinstance(self.class1_pan_allele_models, list)

        self.provenance_string = provenance_string
        
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def supported_alleles(self):
        if "supported_alleles" not in self._cache:
            result = set(self.allele_to_allele_specific_models)
            if self.allele_to_sequence:
                result = result.union(self.allele_to_sequence)
            self._cache["supported_alleles"] = sorted(result)
        return self._cache["supported_alleles"]

    @classmethod
    def load(cls, models_dir):
        """
        Load a trained model from the specified directory.

        Parameters
        ----------
        models_dir : str
            Directory containing model files including manifest.csv

        Returns
        -------
        Class1AffinityPredictor
            Initialized predictor with loaded weights
        """
        # Load manifest if available
        manifest_path = os.path.join(models_dir, "manifest.csv")
        manifest_df = pd.read_csv(manifest_path)

        # Load allele sequences if available
        allele_to_sequence = None
        allele_sequences_path = os.path.join(models_dir, "allele_sequences.csv")
        if os.path.exists(allele_sequences_path):
            allele_sequences_df = pd.read_csv(allele_sequences_path)
            allele_to_sequence = dict(zip(allele_sequences_df.allele, allele_sequences_df.sequence))

        # Create empty predictor with just the manifest
        instance = cls(
            allele_to_allele_specific_models={},
            class1_pan_allele_models=[],
            allele_to_sequence=allele_to_sequence,
            manifest_df=manifest_df,
        )

        # Load models from manifest
        for _, row in manifest_df.iterrows():
            config = json.loads(row.config_json)
            model = TorchNeuralNetwork.from_config(config)

            # Load weights if available
            weights_path = os.path.join(models_dir, f"weights_{row.model_name}.npz")
            if os.path.exists(weights_path):
                model.load_weights(weights_path)

            if row.allele == "pan-class1":
                instance.class1_pan_allele_models.append(model)
            else:
                if row.allele not in instance.allele_to_allele_specific_models:
                    instance.allele_to_allele_specific_models[row.allele] = []
                instance.allele_to_allele_specific_models[row.allele].append(model)

        return instance

    def load_weights(self, weights_df):
        """
        Load weights from weights DataFrame

        Parameters
        ----------
        weights_df : pandas.DataFrame
            DataFrame containing model weights
        """
        if not isinstance(weights_df, pd.DataFrame):
            weights_df = pd.read_csv(weights_df)

        for i in range(len(self.paths)):
            for j, layer in enumerate(self.paths[i]):
                if isinstance(layer, nn.Linear):
                    weight_key = f"path_{i}_dense_{j}_weight"
                    bias_key = f"path_{i}_dense_{j}_bias"

                    if weight_key in weights_df and bias_key in weights_df:
                        weight = torch.FloatTensor(weights_df[weight_key].values)
                        bias = torch.FloatTensor(weights_df[bias_key].values)
                        layer.weight.data = weight
                        layer.bias.data = bias

                elif isinstance(layer, nn.BatchNorm1d):
                    weight_key = f"path_{i}_bn_{j}_weight"
                    bias_key = f"path_{i}_bn_{j}_bias"
                    mean_key = f"path_{i}_bn_{j}_running_mean"
                    var_key = f"path_{i}_bn_{j}_running_var"

                    if all(k in weights_df for k in [weight_key, bias_key, mean_key, var_key]):
                        layer.weight.data = torch.FloatTensor(weights_df[weight_key].values)
                        layer.bias.data = torch.FloatTensor(weights_df[bias_key].values)
                        layer.running_mean.data = torch.FloatTensor(weights_df[mean_key].values)
                        layer.running_var.data = torch.FloatTensor(weights_df[var_key].values)

    def forward(self, x, initialize=False):
        """
        Run a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        initialize : bool
            If True and data_dependent_initialization_method is set,
            perform initialization
        """
        x = to_torch(x)
        x = x.to(self.device)

        if initialize and self.data_dependent_initialization_method:
            if self.data_dependent_initialization_method == "lsuv":
                # Initialize using Layer-Sequential Unit-Variance (LSUV)
                from .data_dependent_weights_initialization import lsuv_init

                lsuv_init(self, {"peptide": x}, verbose=True)
            else:
                raise ValueError(f"Unknown initialization method: {self.data_dependent_initialization_method}")

        # Process peptide layers
        peptide_output = x
        for layer in self.peptide_layers:
            if isinstance(layer, nn.Linear):
                peptide_output = self.hidden_activation(layer(peptide_output))
            else:
                peptide_output = layer(peptide_output)
            peptide_output = peptide_output.to(self.device)

        # Process allele layers if present
        if hasattr(self, "allele_layers"):
            allele_output = self.allele_input
            for layer in self.allele_layers:
                if isinstance(layer, nn.Linear):
                    allele_output = self.hidden_activation(layer(allele_output))
                else:
                    allele_output = layer(allele_output)
                allele_output = allele_output.to(self.device)

            # Merge peptide and allele outputs
            if self.merge_method == "concatenate":
                x = torch.cat([peptide_output, allele_output], dim=1)
            elif self.merge_method == "multiply":
                x = peptide_output * allele_output

            if self.merge_activation:
                x = {
                    "tanh": torch.tanh,
                    "relu": F.relu,
                    "sigmoid": torch.sigmoid,
                    "": lambda x: x,
                }[
                    self.merge_activation
                ](x)
        else:
            x = peptide_output

        # Process locally connected layers
        if self.local_layers:
            x = x.unsqueeze(1)  # Add channel dimension
            for layer in self.local_layers:
                x = self.hidden_activation(layer(x))
            x = x.flatten(1)  # Flatten back to 2D

        # Process dense layers with topology handling
        layer_outputs = []
        for i, layer in enumerate(self.dense_layers):
            if isinstance(layer, nn.Linear):
                if self.topology == "with-skip-connections" and len(layer_outputs) > 1:
                    # Concatenate previous two layer outputs
                    x = torch.cat(layer_outputs[-2:], dim=1)
                x = self.hidden_activation(layer(x))
                layer_outputs.append(x)
            else:
                x = layer(x)

        # Output layer
        x = self.output_layer(x)
        x = self.output_activation(x)

        # Add regularization losses if in training mode
        if self.training:
            for loss_fn in self.regularization_losses:
                x = x + loss_fn()

        return x

    def percentile_ranks(self, affinities, allele=None, alleles=None, throw=True):
        """
        Return percentile ranks for the given binding affinities.

        Parameters
        ----------
        affinities : list of float
            Binding affinities in nM
        allele : string
            Allele name
        alleles : list of string
            Allele names, alternative to allele parameter
        throw : boolean
            Whether to raise exception if predictor is not available

        Returns
        -------
        numpy.array of float
        """
        # For now return placeholder percentile ranks
        # You'll want to implement proper percentile calculation based on
        # your calibration data
        return numpy.array([50.0] * len(affinities))

    def predict(self, peptides, allele=None, alleles=None, model_kwargs=None, throw=True):
        """
        Predict binding affinity in nM for peptides.

        Parameters
        ----------
        peptides : list of string or EncodableSequences
            Peptide sequences
        allele : string
            Single allele name for all predictions
        alleles : list of string
            List of allele names, one per peptide
        model_kwargs : dict
            Extra kwargs to pass to model
        throw : boolean
            Whether to raise exceptions on invalid input

        Returns
        -------
        numpy.array of float
            Predicted binding affinities in nM
        """

        if allele is not None and alleles is not None:
            raise ValueError("Specify exactly one of allele or alleles")

        if alleles is not None:
            if len(alleles) != len(peptides):
                raise ValueError(f"Got {len(alleles)} alleles but {len(peptides)} peptides")
            predictions = []
            for peptide, single_allele in zip(peptides, alleles):
                pred = self.predict([peptide], allele=single_allele, throw=throw)
                predictions.append(pred[0])
            return numpy.array(predictions)

        # Convert to EncodableSequences if needed
        if not isinstance(peptides, EncodableSequences):
            peptides = EncodableSequences.create(peptides)

        # Get encoded peptides matrix
        encoded = peptides.variable_length_to_fixed_length_vector_encoding(
            "BLOSUM62", alignment_method="pad_middle", max_length=15
        )

        encoded = encoded.reshape(encoded.shape[0], -1)

        # Set model to eval mode
        self.eval()

        # Forward pass with no gradients
        with torch.no_grad():
            # Convert to torch tensor, cast to double, and move to device
            encoded = to_torch(encoded).double().to(self.device)

            # Get predictions from the model for this allele
            model = self.allele_to_allele_specific_models[allele][0]
            outputs = model(encoded)
            outputs = to_numpy(outputs).flatten()

        # Convert network output (0-1) to nM predictions
        # Using same conversion as Keras version
        predictions_nM = to_ic50(outputs)

        return predictions_nM

    def load_weights_from_keras(self, keras_model):
        """
        Load weights from the given Keras model into the PyTorch model.
        Make sure the layer order and shapes match your network structure.
        """
        for keras_layer, torch_layer in zip(keras_model.layers, self.dense_layers):
            # Extract the Keras weights and biases
            weights, biases = keras_layer.get_weights()
            # Load into the Torch linear layer
            torch_layer.weight.data = torch.from_numpy(weights.T)
            torch_layer.bias.data = torch.from_numpy(biases)

        # If you have BatchNorm layers, also match gamma, beta, moving_mean, moving_variance
        # from Keras to PyTorch’s BatchNorm parameters.

    def eval(self):
        """
        Put all underlying TorchNeuralNetwork models in eval mode.
        """
        for allele_models in self.allele_to_allele_specific_models.values():
            for model in allele_models:
                if hasattr(model, "eval") and callable(model.eval):
                    model.eval()
        for model in self.class1_pan_allele_models:
            if hasattr(model, "eval") and callable(model.eval):
                model.eval()

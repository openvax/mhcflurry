"""
PyTorch implementations of MHCflurry neural networks.
"""
import collections
import logging
import weakref
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas

from .class1_neural_network import Class1NeuralNetwork
from .encodable_sequences import EncodableSequences
from .allele_encoding import AlleleEncoding
from .common import normalize_allele_name
from .percent_rank_transform import PercentRankTransform

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
    PyTorch implementation of Class1NeuralNetwork that exactly matches the Keras architecture
    """
    def __init__(self, **hyperparameters):
        """
        Initialize neural network with hyperparameters matching Keras version.
        
        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters as defined in Class1NeuralNetwork
        """
        super().__init__()
        
        # Use same hyperparameter defaults as Keras version
        self.hyperparameters = Class1NeuralNetwork.hyperparameter_defaults.with_defaults(
            Class1NeuralNetwork.apply_hyperparameter_renames(hyperparameters)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regularization_losses = []
        self._network = None
        self.network_json = None
        self.network_weights = None
        self.network_weights_loader = None
        self.fit_info = []
        self.prediction_cache = weakref.WeakKeyDictionary()

        # Set activation functions
        self.hidden_activation = self._get_activation_function(
            hyperparameters.get('activation', 'tanh'))
        self.output_activation = self._get_activation_function(
            hyperparameters.get('output_activation', 'sigmoid'))

        # Build network based on hyperparameters
        self._build_network()
        self.to(self.device)
        
    def _build_network(self):
        """Build PyTorch network matching Keras architecture"""
        # Get dimensions from peptide encoding config
        peptide_input_dim = self._get_peptide_input_dim()
        
        # Input layers
        self.peptide_layers = nn.ModuleList()
        current_size = peptide_input_dim
        
        # Peptide dense layers
        for size in self.hyperparameters["peptide_dense_layer_sizes"]:
            linear = nn.Linear(current_size, size)
            if self.hyperparameters["dense_layer_l1_regularization"] > 0:
                self.regularization_losses.append(
                    lambda: self.hyperparameters["dense_layer_l1_regularization"] * 
                           linear.weight.abs().sum()
                )
            self.peptide_layers.append(linear)
            if self.hyperparameters["batch_normalization"]:
                self.peptide_layers.append(nn.BatchNorm1d(size))
            if self.hyperparameters["dropout_probability"] > 0:
                self.peptide_layers.append(
                    nn.Dropout(self.hyperparameters["dropout_probability"]))
            current_size = size

        # Allele representation layers
        if self.hyperparameters["allele_dense_layer_sizes"]:
            self.allele_layers = nn.ModuleList()
            self.allele_embedding = nn.Embedding(
                num_embeddings=1,  # Will be set when allele representations are loaded
                embedding_dim=1    # Will be set when allele representations are loaded
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
                    bias=True
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
                    lambda: self.hyperparameters["dense_layer_l1_regularization"] * 
                           linear.weight.abs().sum()
                )
            self.dense_layers.append(linear)
            self.layer_outputs.append(linear)
            
            if self.hyperparameters["batch_normalization"]:
                self.dense_layers.append(nn.BatchNorm1d(size))
            if self.hyperparameters["dropout_probability"] > 0:
                self.dense_layers.append(
                    nn.Dropout(self.hyperparameters["dropout_probability"]))
            current_size = size

        # Output layer
        self.output_layer = nn.Linear(
            current_size, 
            self.hyperparameters["num_outputs"]
        )

    def _get_activation_function(self, name):
        """Convert activation function name to PyTorch function"""
        if callable(name):
            return name
        activations = {
            'tanh': torch.tanh,
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'linear': lambda x: x,
            '': lambda x: x,
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
        if hasattr(self, '_network'):
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
                    limit = numpy.sqrt(6. / (fan_in + fan_out))
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
        if hasattr(self, 'allele_embedding'):
            with torch.no_grad():
                self.allele_embedding.weight.copy_(
                    torch.from_numpy(allele_representations).to(self.device))

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
        
        # Process dense layers with Keras-matching activation order
        for i, layer in enumerate(self.dense_layers):
            layer = layer.to(self.device)
            x = x.to(self.device)
            if isinstance(layer, nn.Linear):
                x = x.to(self.device, dtype=torch.float32)  # Ensure float32
                x_pre = x
                x = layer(x)  # Linear transformation
                x = self.hidden_activation(x)  # Activation immediately after linear
            elif isinstance(layer, nn.BatchNorm1d):
                x = x.to(self.device, dtype=torch.float32)  # Ensure float32
                x_pre = x
                x = layer(x)  # Then batch norm
        
        # Output layer with sigmoid activation
        self.output_layer = self.output_layer.to(self.device)
        x = self.output_layer(x)
        x = self.output_activation(x)

        return x

    def load_weights_from_keras(self, keras_model):
        """
        Load weights from a Keras model into this PyTorch model.
        
        Parameters
        ----------
        keras_model : keras.Model
            Keras model with matching architecture
        """
        from tf_keras.layers import Dense, BatchNormalization
        keras_layers = [l for l in keras_model.layers 
                       if isinstance(l, (Dense, BatchNormalization))]
        torch_layers = [l for l in self.modules()
                       if isinstance(l, (nn.Linear, nn.BatchNorm1d))]
        
        assert len(keras_layers) == len(torch_layers), "Model architectures do not match"
        
        for k_layer, t_layer in zip(keras_layers, torch_layers):
            weights = k_layer.get_weights()
            
            if isinstance(t_layer, nn.Linear):
                # Keras stores weights as (input_dim, output_dim)
                # PyTorch stores as (output_dim, input_dim)
                t_layer.weight.data = torch.from_numpy(weights[0].T).float()
                t_layer.bias.data = torch.from_numpy(weights[1]).float()

                print(f"[DEBUG] LINEAR layer => weight min/max/mean: "
                      f"{t_layer.weight.data.min().item()}/{t_layer.weight.data.max().item()}/{t_layer.weight.data.mean().item()}, "
                      f"bias: {t_layer.bias.data.min().item()}/{t_layer.bias.data.max().item()}/{t_layer.bias.data.mean().item()}")
                
            elif isinstance(t_layer, nn.BatchNorm1d):
                if len(weights) == 4:  # Has learned parameters
                    # In Keras: [gamma, beta, moving_mean, moving_variance]
                    # In PyTorch: weight=gamma, bias=beta
                    with torch.no_grad():
                        t_layer.weight.data.copy_(torch.from_numpy(weights[0]).float())
                        t_layer.bias.data.copy_(torch.from_numpy(weights[1]).float())
                        t_layer.running_mean.data.copy_(torch.from_numpy(weights[2]).float())
                        t_layer.running_var.data.copy_(torch.from_numpy(weights[3]).float())
                    
                    # Configure batch norm settings to exactly match Keras
                    t_layer.momentum = 0.01  # PyTorch momentum = 1 - Keras momentum (0.99)
                    t_layer.eps = 1e-3  # Keras default
                    t_layer.track_running_stats = True
                    t_layer.training = False
                    t_layer.eval()  # Double ensure eval mode

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
            optimization_info=None):

        if allele_to_allele_specific_models is None:
            allele_to_allele_specific_models = {}
        if class1_pan_allele_models is None:
            class1_pan_allele_models = []

        self.allele_to_sequence = (
            dict(allele_to_sequence)
            if allele_to_sequence is not None else None)

        self._master_allele_encoding = None
        if class1_pan_allele_models:
            assert self.allele_to_sequence

        self.allele_to_allele_specific_models = allele_to_allele_specific_models
        self.class1_pan_allele_models = class1_pan_allele_models
        self._manifest_df = manifest_df

        if not allele_to_percent_rank_transform:
            allele_to_percent_rank_transform = {}
        self.allele_to_percent_rank_transform = allele_to_percent_rank_transform
        self.metadata_dataframes = (
            dict(metadata_dataframes) if metadata_dataframes else {})
        self._cache = {}
        self.optimization_info = optimization_info if optimization_info else {}

        assert isinstance(self.allele_to_allele_specific_models, dict)
        assert isinstance(self.class1_pan_allele_models, list)

        self.provenance_string = provenance_string

    @classmethod
    def load(cls, models_dir):
        """
        Load a trained model from the specified directory.
        
        Parameters
        ----------
        models_dir : str
            Directory containing model files including weights.csv
            
        Returns
        -------
        Class1AffinityPredictor
            Initialized predictor with loaded weights
        """
        import os
        import pandas as pd
        import json
        
        # Load manifest if available
        manifest_path = os.path.join(models_dir, "manifest.csv")
        if os.path.exists(manifest_path):
            manifest_df = pd.read_csv(manifest_path)
            # Get network config from first model
            config = json.loads(manifest_df.iloc[0].config_json)
            
            instance = cls(
                input_size=config.get('input_size', 315),
                peptide_dense_layer_sizes=config.get('peptide_dense_layer_sizes', []),
                layer_sizes=config.get('layer_sizes', []),
                dropout_probability=config.get('dropout_probability', 0.0),
                batch_normalization=config.get('batch_normalization', False),
                activation=config.get('activation', 'tanh'),
                init=config.get('init', 'glorot_uniform'),
                output_activation=config.get('output_activation', 'sigmoid'),
                num_outputs=config.get('num_outputs', 1)
            )
        else:
            # Fallback to default architecture
            instance = cls(
                input_size=315,
                peptide_dense_layer_sizes=[],
                layer_sizes=[],
                dropout_probability=0.0,
                batch_normalization=False,
                activation='tanh',
                init='glorot_uniform',
                output_activation='sigmoid',
                num_outputs=1
            )
        
        # Load weights if available
        weights_path = os.path.join(models_dir, "weights.csv")
        if os.path.exists(weights_path):
            weights_df = pd.read_csv(weights_path)
            instance.load_weights(weights_df)
        
        instance = instance.to(instance.device)
        return instance
    def load_weights(self, weights_df):
        """
        Load weights from weights DataFrame
        
        Parameters
        ----------
        weights_df : pandas.DataFrame
            DataFrame containing model weights
        """
        if not isinstance(weights_df, pandas.DataFrame):
            weights_df = pandas.read_csv(weights_df)
            
        weights = []
        for i in range(len(self.paths)):
            path_weights = []
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
                raise ValueError(
                    f"Unknown initialization method: {self.data_dependent_initialization_method}")
        
        # Process peptide layers
        peptide_output = x
        for layer in self.peptide_layers:
            if isinstance(layer, nn.Linear):
                peptide_output = self.hidden_activation(layer(peptide_output))
            else:
                peptide_output = layer(peptide_output)
            peptide_output = peptide_output.to(self.device)
                
        # Process allele layers if present
        if hasattr(self, 'allele_layers'):
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
                }[self.merge_activation](x)
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
        from mhcflurry.encodable_sequences import EncodableSequences
        
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
            "BLOSUM62",
            alignment_method="pad_middle",
            max_length=15)

        encoded = encoded.reshape(encoded.shape[0], -1)

        # Set model to eval mode
        self.eval()

        # Forward pass with no gradients
        with torch.no_grad():
            # Convert to torch tensor and move to device
            encoded = to_torch(encoded).to(self.device)
            
            # Get predictions
            outputs = self(encoded)
            outputs = to_numpy(outputs).flatten()

        # Convert network output (0-1) to nM predictions
        # Using same conversion as Keras version
        max_ic50 = 50000.0
        predictions_nM = max_ic50 ** (1.0 - outputs)

        return predictions_nM

    def load_weights_from_keras(self, keras_model):
        """
        Load weights from a Keras model into this PyTorch model.
        
        Parameters
        ----------
        keras_model : keras.Model
            Keras model with matching architecture
        """
        from tf_keras.layers import Dense, BatchNormalization
        
        # Get all dense and batch norm layers from both models
        keras_layers = [l for l in keras_model.layers 
                       if isinstance(l, (Dense, BatchNormalization))]
        
        # Get corresponding PyTorch layers
        torch_layers = []
        torch_layers.extend([l for l in self.dense_layers if isinstance(l, (nn.Linear, nn.BatchNorm1d))])
        if hasattr(self, 'output_layer'):
            torch_layers.append(self.output_layer)
        
        assert len(keras_layers) == len(torch_layers), (
            f"Model architectures do not match: Keras has {len(keras_layers)} layers, "
            f"PyTorch has {len(torch_layers)} layers")
        
        for k_layer, t_layer in zip(keras_layers, torch_layers):
            weights = k_layer.get_weights()
            
            if isinstance(t_layer, nn.Linear):
                # Keras stores weights as (input_dim, output_dim)
                # PyTorch stores as (output_dim, input_dim)
                t_layer.weight.data = torch.from_numpy(weights[0].T).float()
                t_layer.bias.data = torch.from_numpy(weights[1]).float()

                print(f"[DEBUG] LINEAR layer => weight min/max/mean: "
                      f"{t_layer.weight.data.min().item()}/{t_layer.weight.data.max().item()}/{t_layer.weight.data.mean().item()}, "
                      f"bias: {t_layer.bias.data.min().item()}/{t_layer.bias.data.max().item()}/{t_layer.bias.data.mean().item()}")
                    
            elif isinstance(t_layer, nn.BatchNorm1d):
                if len(weights) == 4:  # Has learned parameters
                    # In Keras: [gamma, beta, moving_mean, moving_variance]
                    # In PyTorch: weight=gamma, bias=beta
                    with torch.no_grad():
                        t_layer.weight.data.copy_(torch.from_numpy(weights[0]).float())
                        t_layer.bias.data.copy_(torch.from_numpy(weights[1]).float())
                        t_layer.running_mean.data.copy_(torch.from_numpy(weights[2]).float())
                        t_layer.running_var.data.copy_(torch.from_numpy(weights[3]).float())
                    
                    # Configure batch norm settings to match Keras
                    t_layer.momentum = 0.01  # PyTorch momentum = 1 - Keras momentum (0.99)
                    t_layer.eps = 1e-3  # Keras default
                    t_layer.track_running_stats = True
                    t_layer.training = False
                    t_layer.eval()  # Double ensure eval mode

    def export_weights_to_keras(self, keras_model):
        """
        Export weights from this PyTorch model to a Keras model.
        
        Parameters
        ----------
        keras_model : keras.Model
            Keras model with matching architecture
        """
        from tf_keras.layers import Dense, BatchNormalization
        keras_layers = [l for l in keras_model.layers 
                       if isinstance(l, (Dense, BatchNormalization))]
        torch_layers = [l for l in self.layers 
                       if isinstance(l, (nn.Linear, nn.BatchNorm1d))]
        
        assert len(keras_layers) == len(torch_layers), "Model architectures do not match"
        
        for k_layer, t_layer in zip(keras_layers, torch_layers):
            if isinstance(t_layer, nn.Linear):
                # Convert PyTorch weights to Keras format
                weights = [
                    to_numpy(t_layer.weight.data.T),  # Transpose back to Keras format
                    to_numpy(t_layer.bias.data)
                ]
                k_layer.set_weights(weights)
                
            elif isinstance(t_layer, nn.BatchNorm1d):
                # Convert PyTorch batch norm parameters to Keras format
                weights = [
                    to_numpy(t_layer.weight.data),  # gamma
                    to_numpy(t_layer.bias.data),    # beta
                    to_numpy(t_layer.running_mean.data),
                    to_numpy(t_layer.running_var.data)
                ]
                k_layer.set_weights(weights)

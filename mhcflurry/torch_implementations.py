"""
PyTorch implementations of MHCflurry neural networks.
"""
import collections
import logging
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
    def __init__(
            self,
            input_size=315,
            peptide_dense_layer_sizes=[],
            allele_dense_layer_sizes=[],
            layer_sizes=[32],
            dropout_probability=0.0,
            batch_normalization=False,
            activation="tanh",
            init="glorot_uniform",
            output_activation="sigmoid",
            num_outputs=1,
            locally_connected_layers=[],
            topology="feedforward",
            peptide_allele_merge_method="multiply",
            peptide_allele_merge_activation="",
            allele_amino_acid_encoding="BLOSUM62",
            dense_layer_l1_regularization=0.001,
            dense_layer_l2_regularization=0.0):
        super().__init__()

        # Store all parameters as instance attributes
        self.input_size = input_size
        self.peptide_dense_layer_sizes = peptide_dense_layer_sizes
        self.allele_dense_layer_sizes = allele_dense_layer_sizes
        self.layer_sizes = layer_sizes
        self.dropout_prob = dropout_probability
        self.use_batch_norm = batch_normalization
        self.activation = activation
        self.init = init
        self.output_activation = output_activation
        self.num_outputs = num_outputs
        self.locally_connected_layers = locally_connected_layers
        self.topology = topology
        self.merge_method = peptide_allele_merge_method
        self.merge_activation = peptide_allele_merge_activation
        self.allele_amino_acid_encoding = allele_amino_acid_encoding
        self.dense_layer_l1_regularization = dense_layer_l1_regularization
        self.dense_layer_l2_regularization = dense_layer_l2_regularization

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Activation functions
        self.hidden_activation = {
            "tanh": torch.tanh,
            "relu": F.relu,
        }[activation]
        
        self.output_activation = {
            "sigmoid": torch.sigmoid,
            "linear": lambda x: x,
        }[output_activation]

        # Build network layers
        self.layers = nn.ModuleList()
        current_size = input_size

        # Peptide dense layers
        self.peptide_layers = nn.ModuleList()
        for size in peptide_dense_layer_sizes:
            self.peptide_layers.extend([
                nn.Linear(current_size, size),
                nn.Dropout(dropout_probability) if dropout_probability > 0 else nn.Identity()
            ])
            if batch_normalization:
                self.peptide_layers.append(nn.BatchNorm1d(size))
            current_size = size

        # Locally connected layers
        self.local_layers = nn.ModuleList()
        for params in locally_connected_layers:
            self.local_layers.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=params['filters'],
                    kernel_size=params['kernel_size'],
                    groups=input_size // params['kernel_size'],
                    bias=True
                )
            )
            current_size = params['filters'] * (input_size - params['kernel_size'] + 1)

        # Main dense layers
        self.dense_layers = nn.ModuleList()
        prev_layers = []
        for size in layer_sizes:
            if topology == "with-skip-connections" and prev_layers:
                # Concatenate previous two layers
                if len(prev_layers) > 1:
                    current_size = prev_layers[-1].out_features + prev_layers[-2].out_features
            
            layer = nn.Linear(current_size, size)
            self.dense_layers.append(layer)
            prev_layers.append(layer)
            
            if batch_normalization:
                self.dense_layers.append(nn.BatchNorm1d(size))
            if dropout_probability > 0:
                self.dense_layers.append(nn.Dropout(dropout_probability))
            current_size = size

        # Output layer
        self.output_layer = nn.Linear(current_size, num_outputs)

        # Build network layers
        self.build_layers()
        
        # Initialize weights
        self.init_weights(init)


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
    def init_weights(self, init):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if init == "glorot_uniform":
                    # TensorFlow's glorot_uniform is slightly different from PyTorch's xavier_uniform
                    # TF uses a uniform distribution between [-limit, limit] where:
                    # limit = sqrt(6 / (fan_in + fan_out))
                    fan_in = layer.weight.size(1)
                    fan_out = layer.weight.size(0)
                    limit = numpy.sqrt(6. / (fan_in + fan_out))
                    nn.init.uniform_(layer.weight, -limit, limit)
                    nn.init.zeros_(layer.bias)
                else:
                    raise ValueError(f"Unsupported initialization: {init}")
                    
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

    def forward(self, x):
        """
        Run a forward pass through parallel paths and combine results.
        """
        # Convert input to torch tensor if needed
        x = to_torch(x)
        
        # Process each path
        path_outputs = []
        for path in self.paths:
            path_x = x
            
            # Pass through layers in order
            for i, layer in enumerate(path):
                if isinstance(layer, nn.Linear):
                    path_x = layer(path_x)
                    # Apply activation unless this is the final layer
                    if i < len(path) - 1:
                        path_x = self.hidden_activation(path_x)
                else:
                    path_x = layer(path_x)
            
            path_outputs.append(path_x)
            
        # Concatenate outputs from all paths
        x = torch.cat(path_outputs, dim=1)
        
        # Apply final output activation to mean of path outputs
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.output_activation(x)
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
        keras_layers = [l for l in keras_model.layers 
                       if isinstance(l, (Dense, BatchNormalization))]
        torch_layers = [l for l in self.layers 
                       if isinstance(l, (nn.Linear, nn.BatchNorm1d))]
        
        assert len(keras_layers) == len(torch_layers), "Model architectures do not match"
        
        for k_layer, t_layer in zip(keras_layers, torch_layers):
            weights = k_layer.get_weights()
            
            if isinstance(t_layer, nn.Linear):
                # Keras stores weights as (input_dim, output_dim)
                # PyTorch stores as (output_dim, input_dim)
                t_layer.weight.data = torch.from_numpy(weights[0].T).float()
                t_layer.bias.data = torch.from_numpy(weights[1]).float()
                
            elif isinstance(t_layer, nn.BatchNorm1d):
                if len(weights) == 4:  # Has learned parameters
                    # In Keras: [gamma, beta, moving_mean, moving_variance]
                    # In PyTorch: weight=gamma, bias=beta
                    t_layer.weight.data = torch.from_numpy(weights[0]).float()
                    t_layer.bias.data = torch.from_numpy(weights[1]).float()
                        
                    # Set running statistics
                    t_layer.running_mean.data = torch.from_numpy(weights[2]).float()
                    t_layer.running_var.data = torch.from_numpy(weights[3]).float()
                        
                    # Configure batch norm settings to match Keras
                    t_layer.momentum = 0.01  # PyTorch momentum = 1 - Keras momentum (0.99)
                    t_layer.eps = 0.001  # Match Keras epsilon
                    t_layer.track_running_stats = True
                    t_layer.training = False  # Set to eval mode

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

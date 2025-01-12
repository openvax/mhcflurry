"""
PyTorch implementations of MHCflurry neural networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import numpy as np

def to_torch(x):
    """Convert numpy array to torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x

def to_numpy(x):
    """Convert torch tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class Class1AffinityPredictor(nn.Module):
    """
    PyTorch implementation of the Class1NeuralNetwork affinity predictor.
    """
    def __init__(
            self,
            peptide_dense_layer_sizes,
            layer_sizes,
            input_size=315,
            dropout_probability=0.0,
            batch_normalization=False,
            activation="tanh",
            init="glorot_uniform", 
            output_activation="sigmoid",
            num_outputs=1):
        super().__init__()
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_size = input_size
        self.dropout_prob = dropout_probability
        self.use_batch_norm = batch_normalization
        
        # Activation functions
        self.hidden_activation = {
            "tanh": torch.tanh,
            "relu": F.relu,
        }[activation]
        
        self.output_activation = {
            "sigmoid": torch.sigmoid,
            "linear": lambda x: x,
        }[output_activation]

        # Build parallel paths
        self.paths = nn.ModuleList()
        for i in range(10):  # Match Keras model's 10 parallel paths
            path_layers = []
            current_size = input_size
            
            # First dense layer varies by path
            first_layer_size = 256 if i in [0, 2, 4, 5, 9] else 1024
            path_layers.append(nn.Linear(current_size, first_layer_size))
            if batch_normalization:
                path_layers.append(nn.BatchNorm1d(first_layer_size))
            path_layers.append(nn.Dropout(dropout_probability))
            current_size = first_layer_size
            
            # Second dense layer
            second_layer_size = 512
            path_layers.append(nn.Linear(current_size, second_layer_size))
            if batch_normalization:
                path_layers.append(nn.BatchNorm1d(second_layer_size))
            path_layers.append(nn.Dropout(dropout_probability))
            
            # Output layer for this path
            path_layers.append(nn.Linear(second_layer_size, 1))
            
            self.paths.append(nn.ModuleList(path_layers))

        # Initialize weights
        self.init_weights(init)

    @classmethod
    def load(cls, models_dir):
        """
        Minimal placeholder load method. Adjust as needed to read any model
        files from models_dir. Must return an initialized Class1AffinityPredictor.
        """
        # For example, return a default instance:
        instance = cls(
            input_size=315,
            peptide_dense_layer_sizes=[],
            layer_sizes=[],
        )
        instance = instance.to(instance.device)  # Move model to appropriate device
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
                    limit = np.sqrt(6. / (fan_in + fan_out))
                    nn.init.uniform_(layer.weight, -limit, limit)
                    nn.init.zeros_(layer.bias)
                else:
                    raise ValueError(f"Unsupported initialization: {init}")

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
        return np.array([50.0] * len(affinities))

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

"""
PyTorch implementations of MHCflurry neural networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            input_size,
            peptide_dense_layer_sizes,
            layer_sizes,
            dropout_probability=0.0,
            batch_normalization=False,
            activation="tanh",
            init="glorot_uniform",
            output_activation="sigmoid",
            num_outputs=1):
        super().__init__()
        
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

        # Build network layers
        layers = []
        current_size = input_size

        # Peptide dense layers
        for size in peptide_dense_layer_sizes:
            layers.append(nn.Linear(current_size, size))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_probability))
            current_size = size

        # Main dense layers  
        for size in layer_sizes:
            layers.append(nn.Linear(current_size, size))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_probability))
            current_size = size

        # Output layer
        layers.append(nn.Linear(current_size, num_outputs))
        
        self.layers = nn.ModuleList(layers)

        # Initialize weights
        self.init_weights(init)

    def init_weights(self, init):
        """Initialize network weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if init == "glorot_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                else:
                    raise ValueError(f"Unsupported initialization: {init}")

    def forward(self, x):
        """Forward pass."""
        x = to_torch(x)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if i < len(self.layers) - 1:  # Not output layer
                    x = self.hidden_activation(x)
            elif isinstance(layer, (nn.BatchNorm1d, nn.Dropout)):
                x = layer(x)
                
        return self.output_activation(x)

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
                    t_layer.running_mean.data = torch.from_numpy(weights[2]).float()
                    # Keras uses moving_variance, PyTorch expects running_var
                    t_layer.running_var.data = torch.from_numpy(weights[3]).float()
                    # Set momentum to match Keras default of 0.99
                    # PyTorch momentum = 1 - Keras momentum
                    # Keras default momentum is 0.99, so PyTorch needs 0.01
                    t_layer.momentum = 0.01
                    t_layer.eps = 0.001  # Match Keras default epsilon
                    # PyTorch uses running_var while Keras uses moving_variance
                    # Need to convert between the two
                    t_layer.running_var.data = torch.from_numpy(weights[3]).float() 
                    t_layer.track_running_stats = True

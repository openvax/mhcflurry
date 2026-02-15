"""
PyTorch custom layers for mhcflurry.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    """
    Map activation name string to a PyTorch activation function.

    Parameters
    ----------
    name : str
        Activation name: "tanh", "sigmoid", "relu", "linear", or ""

    Returns
    -------
    callable or None
        Activation function, or None for no activation
    """
    if not name or name == "linear":
        return None
    name = name.lower()
    if name == "tanh":
        return torch.tanh
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "relu":
        return F.relu
    else:
        raise ValueError(f"Unknown activation: {name}")


class LocallyConnected1D(nn.Module):
    """
    A locally connected 1D layer (unshared convolution).

    Unlike Conv1D, this layer uses different filter weights at each position
    in the input sequence. This is equivalent to Keras' LocallyConnected1D.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (filters)
    input_length : int
        Length of the input sequence
    kernel_size : int
        Size of the convolution kernel
    activation : str
        Activation function name
    """

    def __init__(self, in_channels, out_channels, input_length, kernel_size,
                 activation="tanh"):
        super(LocallyConnected1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.activation_name = activation
        self.output_length = input_length - kernel_size + 1

        # Weight shape: (output_length, out_channels, in_channels * kernel_size)
        self.weight = nn.Parameter(
            torch.randn(self.output_length, out_channels, in_channels * kernel_size)
        )
        # Bias shape: (output_length, out_channels)
        self.bias = nn.Parameter(
            torch.zeros(self.output_length, out_channels)
        )

        self._activation = get_activation(activation)

        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, sequence_length, in_channels)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, output_length, out_channels)
        """
        batch_size = x.size(0)

        # Use unfold to extract patches and match Keras flatten order.
        # x_unfolded shape: (batch, output_length, in_channels, kernel_size)
        x_unfolded = x.unfold(1, self.kernel_size, 1)
        # Keras flattens patches with kernel positions first, then channels.
        x_unfolded = x_unfolded.permute(0, 1, 3, 2)
        # Reshape to (batch, output_length, kernel_size * in_channels)
        x_unfolded = x_unfolded.reshape(
            batch_size, self.output_length, self.kernel_size * self.in_channels
        )

        # Apply locally connected weights via einsum
        # x_unfolded: (batch, output_length, in_channels * kernel_size)
        # weight: (output_length, out_channels, in_channels * kernel_size)
        # result: (batch, output_length, out_channels)
        output = torch.einsum('boi,ofi->bof', x_unfolded, self.weight) + self.bias

        if self._activation is not None:
            output = self._activation(output)

        return output

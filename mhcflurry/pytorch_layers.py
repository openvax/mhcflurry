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
PyTorch custom layers for mhcflurry.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    """
    Map activation name string to a PyTorch activation function.

    Parameters
    ----------
    name : str
        Activation name: "tanh", "sigmoid", "relu", "silu"/"swish",
        "gelu", "linear", or ""

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
    elif name in {"silu", "swish"}:
        return F.silu
    elif name == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Unknown activation: {name}")


class KerasBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d with Keras-compatible running-statistics updates.

    PyTorch normalizes with the biased batch variance but updates
    ``running_var`` with an unbiased estimate. Keras uses the biased population
    variance for both. ``momentum`` retains PyTorch's new-batch-coefficient
    convention so ``0.01`` corresponds to Keras ``momentum=0.99``.

    Supports both ``(batch, features)`` dense outputs and
    ``(batch, channels, positions)`` sequence outputs. In the sequence case,
    statistics are accumulated over batch and position, matching Keras
    ``BatchNormalization(axis=-1)`` on the equivalent channels-last tensor.
    """

    def forward(self, inputs):
        self._check_input_dim(inputs)
        if inputs.dim() == 2:
            reduction_dims = (0,)
            broadcast_shape = (1, -1)
        elif inputs.dim() == 3:
            reduction_dims = (0, 2)
            broadcast_shape = (1, -1, 1)
        else:
            raise ValueError(
                "KerasBatchNorm1d supports 2D dense or 3D sequence outputs; "
                "got shape %s" % (tuple(inputs.shape),)
            )

        if self.training or not self.track_running_stats:
            mean = inputs.mean(dim=reduction_dims)
            variance = inputs.var(dim=reduction_dims, unbiased=False)
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    self.running_mean.lerp_(mean.detach(), self.momentum)
                    self.running_var.lerp_(variance.detach(), self.momentum)
        else:
            mean = self.running_mean
            variance = self.running_var

        mean = mean.view(broadcast_shape)
        variance = variance.view(broadcast_shape)
        result = (inputs - mean) * torch.rsqrt(variance + self.eps)
        if self.affine:
            result = (
                result * self.weight.view(broadcast_shape) +
                self.bias.view(broadcast_shape)
            )
        return result


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

        # Match Keras LocallyConnected1D GlorotUniform. Keras computes fan-in
        # and fan-out on its (output_length, flattened_input, filters) kernel;
        # applying torch's generic 3D fan calculation to our transposed storage
        # would use different fan values.
        flattened_input = in_channels * kernel_size
        fan_in = self.output_length * flattened_input
        fan_out = self.output_length * out_channels
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.weight, -bound, bound)

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

"""
Layer-sequential unit-variance initialization for neural networks.

See:
    Mishkin and Matas, "All you need is a good init". 2016.
    https://arxiv.org/abs/1511.06422
"""
#
# LSUV initialization code in this file is adapted from:
#   https://github.com/ducha-aiki/LSUV-keras/blob/master/lsuv_init.py
# by Dmytro Mishkin
#
# Here is the license for the original code:
#
#
# Copyright (C) 2017, Dmytro Mishkin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
import numpy
import torch
import torch.nn as nn


def svd_orthonormal(shape):
    """
    Generate an orthonormal matrix using SVD.

    Parameters
    ----------
    shape : tuple
        Shape of the weight matrix (must have at least 2 dimensions)

    Returns
    -------
    numpy.ndarray
        Orthonormal matrix of the given shape
    """
    # Orthonormal init code is from Lasagne
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], numpy.prod(shape[1:]))
    a = numpy.random.standard_normal(flat_shape).astype("float32")
    u, _, v = numpy.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


def get_activations_pytorch(model, layer_name, x_dict, device=None):
    """
    Get activations from a specific layer in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    layer_name : str
        Name of the layer to get activations from
    x_dict : dict
        Input dictionary with tensors
    device : torch.device, optional
        Device to run on

    Returns
    -------
    numpy.ndarray
        Activations from the specified layer
    """
    if device is None:
        device = next(model.parameters()).device

    activations = {}

    def hook_fn(module, input, output):
        activations['output'] = output.detach().cpu().numpy()

    # Find the layer by name
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    # Register hook
    handle = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        # Convert inputs to tensors
        inputs = {}
        for key, value in x_dict.items():
            if isinstance(value, numpy.ndarray):
                inputs[key] = torch.from_numpy(value).to(device)
            else:
                inputs[key] = value.to(device)

        # Run forward pass
        _ = model(inputs)

    # Remove hook
    handle.remove()

    return activations['output']


def lsuv_init(model, batch, verbose=True, margin=0.1, max_iter=100):
    """
    Initialize neural network weights using layer-sequential unit-variance
    initialization.

    See:
        Mishkin and Matas, "All you need is a good init". 2016.
        https://arxiv.org/abs/1511.06422

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    batch : dict
        Training data batch (dict of numpy arrays or tensors)
    verbose : boolean
        Whether to print progress to stdout
    margin : float
        Acceptable variance margin
    max_iter : int
        Maximum iterations per layer

    Returns
    -------
    nn.Module
        Same model, modified in-place
    """
    needed_variance = 1.0
    layers_initialized = 0

    device = next(model.parameters()).device

    # Get list of layers to initialize (Dense/Linear and Conv layers)
    layers_to_init = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            layers_to_init.append((name, module))

    for layer_name, layer in layers_to_init:
        # Get output shape
        try:
            activations = get_activations_pytorch(model, layer_name, batch, device)
            output_size = numpy.prod(activations.shape[1:])
        except Exception as e:
            if verbose:
                print(f'LSUV initialization skipping {layer_name}: {e}')
            continue

        # Skip small layers
        if output_size < 32:
            if verbose:
                print(f'LSUV initialization skipping {layer_name} (output size {output_size} < 32)')
            continue

        layers_initialized += 1

        # Apply orthonormal initialization to weights
        with torch.no_grad():
            weight = layer.weight.data.cpu().numpy()
            ortho_weight = svd_orthonormal(weight.shape)
            layer.weight.data = torch.from_numpy(ortho_weight).to(device)

        # Get activations and compute variance
        activations = get_activations_pytorch(model, layer_name, batch, device)
        variance = numpy.var(activations)

        iteration = 0
        if verbose:
            print(layer_name, variance)

        while abs(needed_variance - variance) > margin:
            if verbose:
                print(
                    'LSUV initialization',
                    layer_name,
                    iteration,
                    needed_variance,
                    margin,
                    variance)

            if numpy.abs(numpy.sqrt(variance)) < 1e-7:
                break  # avoid zero division

            # Scale weights to achieve unit variance
            with torch.no_grad():
                scale_factor = numpy.sqrt(needed_variance) / numpy.sqrt(variance)
                layer.weight.data *= scale_factor

            # Recompute activations and variance
            activations = get_activations_pytorch(model, layer_name, batch, device)
            variance = numpy.var(activations)

            iteration += 1
            if iteration >= max_iter:
                break

    if verbose:
        print('Done with LSUV: total layers initialized', layers_initialized)
    return model

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

from .common import configure_tensorflow


def svd_orthonormal(shape):
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


def get_activations(model, layer, X_batch):
    configure_tensorflow()
    from tensorflow.keras.models import Model
    intermediate_layer_model = Model(
        inputs=model.get_input_at(0),
        outputs=layer.get_output_at(0)
    )
    activations = intermediate_layer_model.predict(X_batch)
    return activations


def lsuv_init(model, batch, verbose=True, margin=0.1, max_iter=100):
    """
    Initialize neural network weights using layer-sequential unit-variance
    initialization.

    See:
        Mishkin and Matas, "All you need is a good init". 2016.
        https://arxiv.org/abs/1511.06422

    Parameters
    ----------
    model : keras.Model
    batch : dict
        Training data, as would be passed keras.Model.fit()
    verbose : boolean
        Whether to print progress to stdout
    margin : float
    max_iter : int

    Returns
    -------
    keras.Model
        Same as what was passed in.
    """
    configure_tensorflow()
    from tensorflow.keras.layers import Dense, Convolution2D
    needed_variance = 1.0
    layers_inintialized = 0
    for layer in model.layers:
        if not isinstance(layer, (Dense, Convolution2D)):
            continue
        # avoid small layers where activation variance close to zero, esp.
        # for small batches_generator
        if numpy.prod(layer.get_output_shape_at(0)[1:]) < 32:
            if verbose:
                print('LSUV initialization skipping', layer.name)
            continue
        layers_inintialized += 1
        weights_and_biases = layer.get_weights()
        weights_and_biases[0] = svd_orthonormal(weights_and_biases[0].shape)
        layer.set_weights(weights_and_biases)
        activations = get_activations(model, layer, batch)
        variance = numpy.var(activations)
        iteration = 0
        if verbose:
            print(layer.name, variance)
        while abs(needed_variance - variance) > margin:
            if verbose:
                print(
                    'LSUV initialization',
                    layer.name,
                    iteration,
                    needed_variance,
                    margin,
                    variance)

            if numpy.abs(numpy.sqrt(variance)) < 1e-7:
                break  # avoid zero division

            weights_and_biases = layer.get_weights()
            weights_and_biases[0] /= numpy.sqrt(variance) / numpy.sqrt(
                needed_variance)
            layer.set_weights(weights_and_biases)
            activations = get_activations(model, layer, batch)
            variance = numpy.var(activations)

            iteration += 1
            if iteration >= max_iter:
                break
    if verbose:
        print('Done with LSUV: total layers initialized', layers_inintialized)
    return model
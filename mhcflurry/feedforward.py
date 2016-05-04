# Copyright (c) 2016. Mount Sinai School of Medicine
#
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

from __future__ import (
    print_function,
    division,
    absolute_import,
)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
import theano

from .feedforward_hyperparameters import OPTIMIZER, LOSS, ACTIVATION

theano.config.exception_verbosity = 'high'


def make_network(
        input_size,
        embedding_input_dim=None,
        embedding_output_dim=None,
        layer_sizes=[100],
        activation=ACTIVATION,
        init="lecun_uniform",
        output_activation="sigmoid",
        dropout_probability=0.0,
        model=None,
        optimizer=OPTIMIZER,
        loss=LOSS):

    if model is None:
        model = Sequential()

    if embedding_input_dim:
        if not embedding_output_dim:
            raise ValueError(
                "Both embedding_input_dim and embedding_output_dim must be set")

        model.add(Embedding(
            input_dim=embedding_input_dim,
            output_dim=embedding_output_dim,
            input_length=input_size,
            init=init))
        model.add(Flatten())

        if dropout_probability > 0:
            model.add(Dropout(dropout_probability))

        input_size = input_size * embedding_output_dim

    layer_sizes = (input_size,) + tuple(layer_sizes)

    for i, dim in enumerate(layer_sizes):
        if i == 0:
            # input is only conceptually a layer of the network,
            # don't need to actually do anything
            continue

        previous_dim = layer_sizes[i - 1]

        # hidden layer fully connected layer
        model.add(
            Dense(
                input_dim=previous_dim,
                output_dim=dim,
                init=init))
        model.add(Activation(activation))
        if dropout_probability > 0:
            model.add(Dropout(dropout_probability))

    # output
    model.add(Dense(
        input_dim=layer_sizes[-1],
        output_dim=1,
        init=init))
    model.add(Activation(output_activation))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def make_hotshot_network(
        peptide_length=9,
        n_amino_acids=20,
        **kwargs):
    """
    Construct a feed-forward neural network whose inputs are binary vectors
    representing a "one-hot" or "hot-shot" encoding of a fixed length amino
    acid sequence.
    """
    return make_network(
        input_size=peptide_length * n_amino_acids,
        **kwargs)

def make_embedding_network(
        peptide_length=9,
        n_amino_acids=20,
        embedding_output_dim=20,
        **kwargs):
    """
    Construct a feed-forward neural network whose inputs are vectors of integer
    indices.
    """
    return make_network(
        input_size=peptide_length,
        embedding_input_dim=n_amino_acids,
        embedding_output_dim=embedding_output_dim,
        **kwargs)

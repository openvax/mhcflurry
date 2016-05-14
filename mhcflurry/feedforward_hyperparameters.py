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

from __future__ import print_function, division, absolute_import
from collections import namedtuple

from .common import all_combinations

Params = namedtuple("Params", [
    "activation",
    "initialization_method",
    "embedding_dim",
    "dropout_probability",
    "learning_rate",
    "hidden_layer_size",
    "loss",
    "optimizer",
    "n_training_epochs",
])


# keeping these for compatibility with old code
N_EPOCHS = 250
ACTIVATION = "tanh"
INITIALIZATION_METHOD = "lecun_uniform"
EMBEDDING_DIM = 32
HIDDEN_LAYER_SIZE = 100
DROPOUT_PROBABILITY = 0.1
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
LOSS = "mse"

default_hyperparameters = Params(
    activation=ACTIVATION,
    initialization_method=INITIALIZATION_METHOD,
    embedding_dim=EMBEDDING_DIM,
    dropout_probability=DROPOUT_PROBABILITY,
    learning_rate=LEARNING_RATE,
    hidden_layer_size=HIDDEN_LAYER_SIZE,
    loss=LOSS,
    optimizer=OPTIMIZER,
    n_training_epochs=N_EPOCHS)


def all_combinations_of_hyperparameters(**kwargs):
    # enusre that all parameters are members of the Params object
    for arg_name in set(kwargs.keys()):
        if arg_name not in Params._fields:
            raise ValueError("Invalid parameter '%s'" % arg_name)
    # if any values aren't specified then just fill them with a single
    # element list containing the default value
    for arg_name in Params._fields:
        if arg_name not in kwargs:
            default_value = default_hyperparameters.__dict__[arg_name]
            kwargs[arg_name] = [default_value]
    for d in all_combinations(**kwargs):
        yield Params(**d)

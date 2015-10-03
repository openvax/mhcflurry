# Copyright (c) 2015. Mount Sinai School of Medicine
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

from collections import namedtuple

ModelConfig = namedtuple(
    "ModelConfig",
    [
        "embedding_size",
        "hidden_layer_size",
        "activation",
        "loss",
        "init",
        "n_pretrain_epochs",
        "n_epochs",
        "dropout_probability",
        "max_ic50",
    ])

HIDDEN1_LAYER_SIZES = [
    50,
    400,
]

INITILIZATION_METHODS = [
    "uniform",
    "glorot_uniform",
]

ACTIVATIONS = [
    "relu",
    "tanh",
]

MAX_IC50_VALUES = [
    5000,
    20000,
]


def generate_all_model_configs(
        embedding_sizes=[0, 64],
        n_training_epochs=100,
        max_dropout=0.25):
    configurations = []
    for activation in ACTIVATIONS:
        for loss in ["mse"]:
            for init in INITILIZATION_METHODS:
                for n_pretrain_epochs in [0, 10]:
                    for hidden_layer_size in HIDDEN1_LAYER_SIZES:
                        for embedding_size in embedding_sizes:
                            for dropout in [0, max_dropout]:
                                for max_ic50 in MAX_IC50_VALUES:
                                    config = ModelConfig(
                                        embedding_size=embedding_size,
                                        hidden_layer_size=hidden_layer_size,
                                        activation=activation,
                                        init=init,
                                        loss=loss,
                                        dropout_probability=dropout,
                                        n_pretrain_epochs=n_pretrain_epochs,
                                        n_epochs=n_training_epochs,
                                        max_ic50=max_ic50)
                                    print(config)
                                    configurations.append(config)
    return configurations

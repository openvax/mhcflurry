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
        "minibatch_size",
    ])

HIDDEN_LAYER_SIZES = [
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
    "prelu",
]

MAX_IC50_VALUES = [
    5000,
    20000,
]

EMBEDDING_SIZES = [
    0,
    64,
]


def generate_all_model_configs(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        embedding_sizes=EMBEDDING_SIZES,
        init_methods=INITILIZATION_METHODS,
        activations=ACTIVATIONS,
        n_training_epochs_values=[100],
        n_pretrain_epochs_values=[0, 10],
        minibatch_sizes=[128],
        dropout_values=[0.0, 0.25],
        max_ic50_values=[5000, 20000],
        losses=["mse"]):
    configurations = []
    for activation in activations:
        for loss in losses:
            for init in init_methods:
                for n_pretrain in n_pretrain_epochs_values:
                    for n_training_epochs in n_training_epochs_values:
                        for hidden in hidden_layer_sizes:
                            for embedding_size in embedding_sizes:
                                for dropout in dropout_values:
                                    for max_ic50 in max_ic50_values:
                                        for minibatch_size in minibatch_sizes:
                                            config = ModelConfig(
                                                embedding_size=embedding_size,
                                                hidden_layer_size=hidden,
                                                activation=activation,
                                                init=init,
                                                loss=loss,
                                                dropout_probability=dropout,
                                                n_pretrain_epochs=n_pretrain,
                                                n_epochs=n_training_epochs,
                                                max_ic50=max_ic50,
                                                minibatch_size=minibatch_size)
                                            print(config)
                                            configurations.append(config)
    return configurations

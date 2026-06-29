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
Generate grid of hyperparameters
"""
from __future__ import print_function
from sys import stdout, stderr
from copy import deepcopy
from yaml import dump

base_hyperparameters = dict(
    convolutional_filters=64,
    convolutional_kernel_size=8,
    convolutional_kernel_l1_l2=(0.00, 0.0),
    flanking_averages=True,
    n_flank_length=15,
    c_flank_length=15,
    post_convolutional_dense_layer_sizes=[],
    minibatch_size=512,
    dropout_rate=0.5,
    convolutional_activation="relu",
    patience=20,
    learning_rate=0.001)

grid = []


def hyperparrameters_grid():
    for learning_rate in [0.001]:
        for convolutional_activation in ["tanh", "relu"]:
            for convolutional_filters in [256, 512]:
                for flanking_averages in [True]:
                    for convolutional_kernel_size in [11, 13, 15, 17]:
                        for l1 in [0.0, 1e-6]:
                            for s in [[8], [16]]:
                                for d in [0.3, 0.5]:
                                    new = deepcopy(base_hyperparameters)
                                    new["learning_rate"] = learning_rate
                                    new["convolutional_activation"] = convolutional_activation
                                    new["convolutional_filters"] = convolutional_filters
                                    new["flanking_averages"] = flanking_averages
                                    new["convolutional_kernel_size"] = convolutional_kernel_size
                                    new["convolutional_kernel_l1_l2"] = (l1, 0.0)
                                    new["post_convolutional_dense_layer_sizes"] = s
                                    new["dropout_rate"] = d
                                    yield new


for new in hyperparrameters_grid():
    if new not in grid:
        grid.append(new)

print("Hyperparameters grid size: %d" % len(grid), file=stderr)
dump(grid, stdout)

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
    learning_rate=0.001)

grid = []


def hyperparrameters_grid():
    for learning_rate in [0.001, 0.0001]:
        for convolutional_activation in ["relu", "tanh"]:
            for convolutional_filters in [64, 128]:
                for flanking_averages in [True]:
                    for convolutional_kernel_size in [5, 6, 7, 8, 9]:
                        for l1 in [0.0, 0.0001, 0.000001]:
                            for s in [[], [8]]:
                                for d in [0.5]:
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

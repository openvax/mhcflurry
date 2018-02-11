"""
Generate grid of hyperparameters
"""

from sys import stdout
from copy import deepcopy
from yaml import dump

base_hyperparameters = {
    ##########################################
    # ENSEMBLE SIZE
    ##########################################
    "n_models": 4,

    ##########################################
    # OPTIMIZATION
    ##########################################
    "max_epochs": 500,
    "patience": 20,
    "early_stopping": True,
    "validation_split": 0.1,
    "minibatch_size": None,
    "loss": "custom:mse_with_inequalities",

    ##########################################
    # RANDOM NEGATIVE PEPTIDES
    ##########################################
    "random_negative_rate": 0.2,
    "random_negative_constant": 25,
    "random_negative_affinity_min": 20000.0,
    "random_negative_affinity_max": 50000.0,

    ##########################################
    # PEPTIDE REPRESENTATION
    ##########################################
    # One of "one-hot", "embedding", or "BLOSUM62".
    "peptide_amino_acid_encoding": "BLOSUM62",
    "use_embedding": False,  # maintained for backward compatability
    "embedding_output_dim": 8,  # only used if using embedding
    "kmer_size": 15,

    ##########################################
    # NEURAL NETWORK ARCHITECTURE
    ##########################################
    "locally_connected_layers": [
        {
            "filters": 8,
            "activation": "tanh",
            "kernel_size": 3
        }
    ],
    "activation": "tanh",
    "output_activation": "sigmoid",
    "layer_sizes": [16],
    "dense_layer_l1_regularization": None,
    "batch_normalization": False,
    "dropout_probability": 0.0,

    ##########################################
    # TRAINING Data
    ##########################################
    "train_data": {"subset": "all"},
}

grid = []
for train_subset in ["all", "quantitative"]:
    for minibatch_size in [128, 512]:
        for dense_layer_size in [8, 16, 32, 64, 128]:
            for l1 in [0.0, 0.001]:
                for num_lc in [0, 1, 2]:
                    for lc_kernel_size in [3, 5]:
                        new = deepcopy(base_hyperparameters)
                        new["train_data"]["subset"] = train_subset
                        new["layer_sizes"] = [dense_layer_size]
                        new["dense_layer_l1_regularization"] = l1
                        (lc_layer,) = new["locally_connected_layers"]
                        lc_layer['kernel_size'] = lc_kernel_size
                        if num_lc == 0:
                            new["locally_connected_layers"] = []
                        elif num_lc == 1:
                            new["locally_connected_layers"] = [lc_layer]
                        elif num_lc == 2:
                            new["locally_connected_layers"] = [lc_layer, deepcopy(lc_layer)]
                        if not grid or new not in grid:
                            grid.append(new)

dump(grid, stdout)

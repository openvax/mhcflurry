"""
Generate grid of hyperparameters
"""

from sys import stdout
from copy import deepcopy
from yaml import dump

base_hyperparameters = {
    'activation': 'tanh',
    'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 0.0,
    'dense_layer_l2_regularization': 0.0,
    'dropout_probability': 0.5,
    'early_stopping': True,
    'init': 'glorot_uniform',
    'layer_sizes': [1024, 512],
    'learning_rate': None,
    'locally_connected_layers': [],
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': 5000,
    'minibatch_size': 128,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    "patience": 20,
    'peptide_encoding': {
        'vector_encoding_name': 'BLOSUM62',
        'alignment_method': 'left_pad_centered_right_pad',
        'max_length': 15,
    },
    'peptide_allele_merge_activation': '',
    'peptide_allele_merge_method': 'concatenate',
    'peptide_amino_acid_encoding': 'BLOSUM62',
    'peptide_dense_layer_sizes': [],
    'random_negative_affinity_max': 50000.0,
    'random_negative_affinity_min': 20000.0,
    'random_negative_constant': 25,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 0.2,
    'train_data': {
        'pretrain': True,
        'pretrain_peptides_per_epoch': 1024,
        'pretrain_patience': 10,
    },
    'validation_split': 0.1,
}

grid = []
for layer_sizes in [[512, 256], [512, 512], [1024, 512]]:
    for pretrain in [True]:
        l1_base = 0.000001
        for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
            new = deepcopy(base_hyperparameters)
            new["layer_sizes"] = layer_sizes
            new["dense_layer_l1_regularization"] = l1
            new["train_data"]["pretrain"] = pretrain
            if not grid or new not in grid:
                grid.append(new)

dump(grid, stdout)

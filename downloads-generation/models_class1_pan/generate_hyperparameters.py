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
    'learning_rate': 0.001,
    'locally_connected_layers': [],
    'topology': 'feedfoward',
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': 5000,
    # See scripts/training/release_exact/generate_hyperparameters.py for
    # the rationale on this bump.
    'minibatch_size': 4096,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    "patience": 20,
    "min_delta": 0.0,
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
    'random_negative_affinity_min': 30000.0,
    'random_negative_constant': 1,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 1.0,
    'random_negative_method': 'by_allele_equalize_nonbinders',
    'random_negative_binder_threshold': 500.0,
    # Phase 1 of issue openvax/mhcflurry#268: amortize the random-negative
    # generation + BLOSUM62 encoding cost across 100 epochs. The per-epoch
    # encode was ~17 s on the release_exact 8xA100 run (~44% of epoch
    # wall-clock); with the pool, each worker pays it once per 100 epochs
    # and does an O(1) array slice otherwise. Within a cycle the same
    # peptides recycle in a fixed order — the user has signed off on that
    # semantic change. Cross-worker diversity is preserved because the
    # training driver seeds each worker's pool with a SHA1 mix of
    # (arch, fold, replicate, work_item_name).
    'random_negative_pool_epochs': 100,
    'train_data': {
        'pretrain': True,
        'pretrain_peptides_per_epoch': 64,
        'pretrain_steps_per_epoch': 256,
        'pretrain_patience': 2,
        'pretrain_min_delta': 0.0001,
        'pretrain_max_val_loss': 0.10,
        'pretrain_max_epochs': 50,
        'pretrain_min_epochs': 5,
    },
    'validation_split': 0.1,
    'data_dependent_initialization_method': "lsuv",
}

grid = []
for layer_sizes in [[512, 256], [512, 512], [1024, 512], [1024, 1024]]:
    l1_base = 0.0000001
    for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
        new = deepcopy(base_hyperparameters)
        new["topology"] = 'feedforward'
        new["layer_sizes"] = layer_sizes
        new["dense_layer_l1_regularization"] = l1
        if not grid or new not in grid:
            grid.append(new)

for layer_sizes in [[256, 512], [256, 256, 512], [256, 512, 512]]:
    l1_base = 0.0000001
    for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
        new = deepcopy(base_hyperparameters)
        new["topology"] = 'with-skip-connections'
        new["layer_sizes"] = layer_sizes
        new["dense_layer_l1_regularization"] = l1
        if not grid or new not in grid:
            grid.append(new)

dump(grid, stdout)

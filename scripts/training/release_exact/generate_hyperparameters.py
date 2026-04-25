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
    # Bumped 512 → 4096 to close the A100 utilization gap. At 512 rows per
    # step the 2-layer MLP was <1 ms of actual compute sandwiched inside
    # ~33 ms of Python/IPC glue (measured: 33.7 ms/step across 37
    # completed models on the 2026-04-24 run). A 4096-row batch shifts
    # the compute:overhead ratio ~8× in compute's favor. RMSprop absorbs
    # the dynamics change; this breaks bit-exactness with the 2.2.0
    # release weights but not the training recipe's semantics.
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
    # Phase 1 of issue openvax/mhcflurry#268: random-negative pool
    # framework. ``pool_epochs=1`` preserves the pre-Phase-1 memory
    # profile (one epoch of encoded negatives in heap at a time) and
    # is the production-safe default. Setting >1 amortizes the
    # generation+encoding cost across that many epochs but materializes
    # ``pool_epochs × per_epoch_count`` peptides simultaneously per
    # worker — at ``100`` on the 8xA100 release_exact run that was
    # ~7.5 GB int8 per worker theoretically, but in practice ballooned
    # to ~199 GB/worker (tooling overhead + intermediate Series) and
    # OOM'd the 944 GB box. Hold at 1 until a streaming-rebuild fix
    # lands that doesn't materialize the full N-epoch buffer at once.
    'random_negative_pool_epochs': 1,
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

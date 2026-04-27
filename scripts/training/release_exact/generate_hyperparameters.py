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
    # Hard absolute cap (down from 5000). Median per-task epoch count
    # observed on the 2026-04-25 verda_a100x8 run was 67 with max=174;
    # 500 leaves comfortable headroom while preventing the runaway
    # "patience-reset" tail where tiny noise improvements could keep a
    # task alive for thousands of epochs.
    'max_epochs': 500,
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
    # ``min_delta=0.0`` lets a 1e-9 RMSprop noise improvement reset the
    # patience counter, which on the 2026-04-25 run caused tasks to
    # stretch to 174 epochs against a median of 67. 1e-7 sits two
    # orders above the observed noise-floor improvement rate
    # (~4e-9 per epoch) so it cuts that pattern cleanly, while still
    # preserving genuine late-escape trajectories — when an escape is
    # real, the per-epoch val_loss drop is ≥1e-3, four orders above
    # this threshold. 1e-6 (the prior draft) was more aggressive and
    # would have killed the late-escape tasks visible on the live
    # cohort (~3 of 16 workers at any given moment).
    "min_delta": 1e-7,
    # Run the validation pass every N epochs instead of every epoch.
    # Validation is ~150 ms on a 244K-row val set with bs=16384 and
    # represents a per-epoch GPU-sync barrier that prevents pipelining
    # the next epoch's CPU prep with the current epoch's training tail.
    # Early-stop check still fires reliably because patience=20 is far
    # larger than ``validation_interval=5``. A final validation pass is
    # forced before any patience-triggered break (see fit() loop).
    "validation_interval": 5,
    'peptide_encoding': {
        'vector_encoding_name': 'BLOSUM62',
        'alignment_method': 'left_pad_centered_right_pad',
        'max_length': 15,
    },
    # Phase 2 of issue openvax/mhcflurry#268: fixed peptide vector
    # expansion as a frozen torch embedding table in the network's
    # forward pass instead of a numpy lookup at peptide-encoding time.
    # ``peptides_to_network_input`` returns int8 indices (cheap dict
    # lookup) and torch widens to the configured fp32 vectors on CUDA,
    # MPS, or CPU via the embedding lookup. Works for BLOSUM62, one-hot,
    # PMBEC, contact, physchem, atchley, and +joined composites.
    # Eliminates the ~17 sec/epoch CPU
    # bottleneck in random-negative regeneration (with
    # random_negative_pool_epochs=1 the CPU encoding fires every epoch).
    # Forward parity vs numpy path verified by
    # ``test_peptide_amino_acid_encoding_gpu_forward_parity``.
    'peptide_amino_acid_encoding_gpu': True,
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

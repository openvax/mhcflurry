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
Generate grid of hyperparameters.
"""

import argparse
from copy import deepcopy
from sys import stdout

from yaml import safe_dump


DEFAULT_MINIBATCH_SIZE = 1024


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=DEFAULT_MINIBATCH_SIZE,
        help=(
            "Training minibatch size to write into every architecture. "
            "Default: %(default)s"
        ),
    )
    return parser

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
    # Release training uses a shared default across model families. Keep this
    # script-level CLI knob so sweeps and remote workflows can override it
    # without patching the recipe.
    'minibatch_size': DEFAULT_MINIBATCH_SIZE,
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
    # Validation represents a per-epoch GPU-sync barrier that prevents
    # pipelining the next epoch's CPU prep with the current epoch's
    # training tail. The effective validation batch size scales with the
    # configured training minibatch, so callers that change --minibatch-size
    # also change validation-pass chunking. Early-stop check still fires
    # reliably because patience=20 is far larger than
    # ``validation_interval=5``. A final validation pass is forced
    # before any patience-triggered break (see fit() loop).
    "validation_interval": 5,
    'peptide_encoding': {
        'vector_encoding_name': 'BLOSUM62',
        'alignment_method': 'left_pad_centered_right_pad',
        'max_length': 15,
    },
    # Fixed peptide vector expansion as a frozen torch embedding table
    # in the network's forward pass instead of a numpy lookup at
    # peptide-encoding time. ``peptides_to_network_input`` returns int8
    # indices (cheap dict lookup) and torch widens to the configured
    # fp32 vectors on CUDA, MPS, or CPU via the embedding lookup. Works
    # for BLOSUM62, one-hot, PMBEC, contact, physchem, atchley, and
    # +joined composites. Forward parity vs numpy path verified by
    # ``test_peptide_amino_acid_encoding_torch_forward_parity``.
    'peptide_amino_acid_encoding_torch': True,
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
    # Random-negative pool framework. ``pool_epochs=1`` keeps a single
    # epoch of encoded negatives in heap at a time (production-safe
    # default). Setting >1 amortizes the generation+encoding cost across
    # that many epochs but materializes ``pool_epochs × per_epoch_count``
    # peptides simultaneously per worker — on the 8xA100 release_exact
    # run, ``100`` was ~7.5 GB int8 per worker in theory but ballooned to
    # ~199 GB/worker in practice (tooling overhead + intermediate Series)
    # and OOM'd the 944 GB box. Hold at 1 until a streaming-rebuild fix
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

def build_grid(minibatch_size=DEFAULT_MINIBATCH_SIZE):
    grid = []
    base = deepcopy(base_hyperparameters)
    base["minibatch_size"] = minibatch_size
    for layer_sizes in [[512, 256], [512, 512], [1024, 512], [1024, 1024]]:
        l1_base = 0.0000001
        for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
            new = deepcopy(base)
            new["topology"] = 'feedforward'
            new["layer_sizes"] = layer_sizes
            new["dense_layer_l1_regularization"] = l1
            if not grid or new not in grid:
                grid.append(new)

    for layer_sizes in [[256, 512], [256, 256, 512], [256, 512, 512]]:
        l1_base = 0.0000001
        for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
            new = deepcopy(base)
            new["topology"] = 'with-skip-connections'
            new["layer_sizes"] = layer_sizes
            new["dense_layer_l1_regularization"] = l1
            if not grid or new not in grid:
                grid.append(new)
    return grid


def main(argv=None):
    args = make_parser().parse_args(argv)
    safe_dump(build_grid(minibatch_size=args.minibatch_size), stdout)


if __name__ == "__main__":
    main()

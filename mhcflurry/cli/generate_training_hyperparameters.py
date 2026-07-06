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

"""Generate release-training hyperparameter grids."""
import argparse
import sys
from copy import deepcopy

import yaml


DEFAULT_MINIBATCH_SIZE = 1024
PROCESSING_VARIANT_CHOICES = (
    "with_flanks",
    "no_n_flank",
    "no_c_flank",
    "no_flank",
    "short_flanks",
)


AFFINITY_BASE_HYPERPARAMETERS = {
    "activation": "tanh",
    "allele_dense_layer_sizes": [],
    "batch_normalization": False,
    "dense_layer_l1_regularization": 0.0,
    "dense_layer_l2_regularization": 0.0,
    "dropout_probability": 0.5,
    "early_stopping": True,
    "init": "glorot_uniform",
    "layer_sizes": [1024, 512],
    "learning_rate": 0.001,
    "locally_connected_layers": [],
    "topology": "feedfoward",
    "loss": "custom:mse_with_inequalities",
    # Hard cap for runaway patience-reset tails. Recent full release
    # cohorts had median task lengths under 100 epochs; 500 leaves broad
    # headroom while preventing accidental multi-thousand-epoch tasks.
    "max_epochs": 500,
    "minibatch_size": DEFAULT_MINIBATCH_SIZE,
    "optimizer": "rmsprop",
    "output_activation": "sigmoid",
    "patience": 20,
    # Keep min_delta above the observed RMSprop noise floor so tiny
    # numerical improvements do not reset patience indefinitely, while
    # still preserving genuine late-escape trajectories.
    "min_delta": 1e-7,
    # Validation is a GPU-sync barrier. Checking every 5 epochs keeps
    # early stopping responsive relative to patience=20 while cutting
    # repeated validation overhead.
    "validation_interval": 5,
    "peptide_encoding": {
        "vector_encoding_name": "BLOSUM62",
        "alignment_method": "left_pad_centered_right_pad",
        "max_length": 15,
    },
    "peptide_amino_acid_encoding_torch": True,
    "peptide_allele_merge_activation": "",
    "peptide_allele_merge_method": "concatenate",
    "peptide_amino_acid_encoding": "BLOSUM62",
    "peptide_dense_layer_sizes": [],
    "random_negative_affinity_max": 50000.0,
    "random_negative_affinity_min": 30000.0,
    "random_negative_constant": 1,
    "random_negative_distribution_smoothing": 0.0,
    "random_negative_match_distribution": True,
    "random_negative_rate": 1.0,
    "random_negative_method": "by_allele_equalize_nonbinders",
    "random_negative_binder_threshold": 500.0,
    # Keep random-negative pooling memory bounded. Larger values can
    # amortize generation cost, but prior full runs ballooned memory
    # far beyond the nominal encoded peptide tensor size.
    "random_negative_pool_epochs": 1,
    "train_data": {
        "pretrain": True,
        "pretrain_peptides_per_epoch": 64,
        "pretrain_steps_per_epoch": 256,
        "pretrain_patience": 2,
        "pretrain_min_delta": 0.0001,
        "pretrain_max_val_loss": 0.10,
        "pretrain_max_epochs": 50,
        "pretrain_min_epochs": 5,
    },
    "validation_split": 0.1,
    "data_dependent_initialization_method": "lsuv",
}


PROCESSING_BASE_HYPERPARAMETERS = {
    "convolutional_filters": 64,
    "convolutional_kernel_size": 8,
    "convolutional_kernel_l1_l2": [0.00, 0.0],
    "flanking_averages": True,
    "n_flank_length": 15,
    "c_flank_length": 15,
    "post_convolutional_dense_layer_sizes": [],
    "minibatch_size": DEFAULT_MINIBATCH_SIZE,
    "dropout_rate": 0.5,
    "convolutional_activation": "relu",
    "patience": 20,
    "learning_rate": 0.001,
}


def unique_hyperparameters(items):
    """Return ``items`` with duplicate dicts removed, preserving order."""
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def build_affinity_grid(minibatch_size=DEFAULT_MINIBATCH_SIZE):
    """Return the 35-architecture Class I pan-allele affinity grid."""
    grid = []
    base = deepcopy(AFFINITY_BASE_HYPERPARAMETERS)
    base["minibatch_size"] = minibatch_size
    for layer_sizes in [[512, 256], [512, 512], [1024, 512], [1024, 1024]]:
        l1_base = 0.0000001
        for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
            new = deepcopy(base)
            new["topology"] = "feedforward"
            new["layer_sizes"] = layer_sizes
            new["dense_layer_l1_regularization"] = l1
            if not grid or new not in grid:
                grid.append(new)

    for layer_sizes in [[256, 512], [256, 256, 512], [256, 512, 512]]:
        l1_base = 0.0000001
        for l1 in [l1_base, l1_base / 10, l1_base / 100, l1_base / 1000, 0.0]:
            new = deepcopy(base)
            new["topology"] = "with-skip-connections"
            new["layer_sizes"] = layer_sizes
            new["dense_layer_l1_regularization"] = l1
            if not grid or new not in grid:
                grid.append(new)
    return grid


def processing_base_grid_iter(minibatch_size=DEFAULT_MINIBATCH_SIZE):
    """Yield the base processing architecture grid before flank variants."""
    base = deepcopy(PROCESSING_BASE_HYPERPARAMETERS)
    base["minibatch_size"] = minibatch_size
    for learning_rate in [0.001]:
        for convolutional_activation in ["tanh", "relu"]:
            for convolutional_filters in [256, 512]:
                for flanking_averages in [True]:
                    for convolutional_kernel_size in [11, 13, 15, 17]:
                        for l1 in [0.0, 1e-6]:
                            for dense_sizes in [[8], [16]]:
                                for dropout_rate in [0.3, 0.5]:
                                    new = deepcopy(base)
                                    new["learning_rate"] = learning_rate
                                    new["convolutional_activation"] = (
                                        convolutional_activation)
                                    new["convolutional_filters"] = (
                                        convolutional_filters)
                                    new["flanking_averages"] = flanking_averages
                                    new["convolutional_kernel_size"] = (
                                        convolutional_kernel_size)
                                    new["convolutional_kernel_l1_l2"] = [
                                        l1, 0.0]
                                    new[
                                        "post_convolutional_dense_layer_sizes"
                                    ] = dense_sizes
                                    new["dropout_rate"] = dropout_rate
                                    yield new


def build_processing_base_grid(minibatch_size=DEFAULT_MINIBATCH_SIZE):
    """Return the processing architecture grid before flank variants."""
    return unique_hyperparameters(
        processing_base_grid_iter(minibatch_size=minibatch_size))


def transform_processing_hyperparameters(kind, hyperparameters):
    """Return one processing flank variant for a hyperparameter dict."""
    new_hyperparameters = deepcopy(hyperparameters)
    if kind in ("no_n_flank", "no_flank"):
        new_hyperparameters["n_flank_length"] = 0
    if kind in ("no_c_flank", "no_flank"):
        new_hyperparameters["c_flank_length"] = 0
    if kind == "short_flanks":
        new_hyperparameters["c_flank_length"] = 5
        new_hyperparameters["n_flank_length"] = 5
    return new_hyperparameters


def build_processing_variant_grid(production_hyperparameters, kind):
    """Return a flank-mode variant of a processing hyperparameter grid."""
    return unique_hyperparameters(
        transform_processing_hyperparameters(kind, item)
        for item in production_hyperparameters
    )


def read_hyperparameters_yaml(path):
    """Read a YAML hyperparameter list from ``path``."""
    with open(path) as fd:
        hyperparameters = yaml.safe_load(fd)
    if hyperparameters is None:
        return []
    if not isinstance(hyperparameters, list):
        raise ValueError(
            "Expected YAML list in %s; got %s" % (
                path, type(hyperparameters).__name__))
    return hyperparameters


def dump_hyperparameters(hyperparameters, stream=None):
    """Write hyperparameter dictionaries as safe YAML."""
    yaml.safe_dump(hyperparameters, stream or sys.stdout)


def add_minibatch_argument(parser):
    """Add the common training-minibatch-size argument to ``parser``."""
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=DEFAULT_MINIBATCH_SIZE,
        help=(
            "Training minibatch size to write into every architecture. "
            "Default: %(default)s"
        ),
    )


def make_parser(prog=None):
    """Build the argparse parser for the unified generator command."""
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    subparsers = parser.add_subparsers(dest="recipe", required=True)

    affinity = subparsers.add_parser(
        "affinity",
        help="Generate the Class I pan-allele affinity grid.")
    add_minibatch_argument(affinity)

    processing_base = subparsers.add_parser(
        "processing-base",
        help="Generate the base Class I processing grid.")
    add_minibatch_argument(processing_base)

    processing_variant = subparsers.add_parser(
        "processing-variant",
        help="Generate a flank-mode processing grid from a base grid.")
    processing_variant.add_argument(
        "production_hyperparameters",
        metavar="YAML",
        help="Base processing hyperparameter grid YAML.")
    processing_variant.add_argument(
        "kind",
        choices=PROCESSING_VARIANT_CHOICES,
        help="Processing flank variant to output.")

    return parser


def run_argv(argv=None, prog=None):
    """Run the unified generator command."""
    args = make_parser(prog=prog).parse_args(argv)
    if args.recipe == "affinity":
        grid = build_affinity_grid(minibatch_size=args.minibatch_size)
    elif args.recipe == "processing-base":
        grid = build_processing_base_grid(minibatch_size=args.minibatch_size)
    elif args.recipe == "processing-variant":
        grid = build_processing_variant_grid(
            read_hyperparameters_yaml(args.production_hyperparameters),
            args.kind)
    else:
        raise ValueError("Unsupported recipe: %s" % args.recipe)
    print("Hyperparameters grid size: %d" % len(grid), file=sys.stderr)
    dump_hyperparameters(grid)


def run_affinity_argv(argv=None, prog=None):
    """Compatibility entry point for the old affinity generator script."""
    if argv is None:
        argv = sys.argv[1:]
    return run_argv(["affinity"] + list(argv), prog=prog)


def run_processing_base_argv(argv=None, prog=None):
    """Compatibility entry point for the old processing-base script."""
    if argv is None:
        argv = sys.argv[1:]
    return run_argv(["processing-base"] + list(argv), prog=prog)


def run_processing_variant_argv(argv=None, prog=None):
    """Compatibility entry point for the old processing-variant script."""
    if argv is None:
        argv = sys.argv[1:]
    return run_argv(["processing-variant"] + list(argv), prog=prog)


if __name__ == "__main__":
    run_argv()

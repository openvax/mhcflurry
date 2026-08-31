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

from ..common import positive_int_arg


AFFINITY_DEFAULT_MINIBATCH_SIZE = 128
PROCESSING_DEFAULT_MINIBATCH_SIZE = 512
DEFAULT_OPTIMIZER_IMPLEMENTATION = "keras"
OPTIMIZER_IMPLEMENTATION_CHOICES = ("keras", "pytorch")
LSUV_TARGET_CHOICES = ("post_activation", "pre_activation")
AFFINITY_INIT_CHOICES = (
    "glorot_uniform",
    "glorot_normal",
    "he_uniform",
    "he_normal",
    "orthogonal",
)
PROCESSING_INIT_CHOICES = ("glorot_uniform", "kaiming_uniform_fan_in")
# Compatibility name used by the historical affinity generator wrapper.
DEFAULT_MINIBATCH_SIZE = AFFINITY_DEFAULT_MINIBATCH_SIZE
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
    # Preserve the published 2.1.x/2.2.x scientific recipe. Early stopping
    # normally terminates well before this safety ceiling.
    "max_epochs": 5000,
    "minibatch_size": AFFINITY_DEFAULT_MINIBATCH_SIZE,
    "optimizer": "rmsprop",
    "optimizer_implementation": DEFAULT_OPTIMIZER_IMPLEMENTATION,
    "output_activation": "sigmoid",
    "patience": 20,
    "min_delta": 0.0,
    "validation_interval": 1,
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
    "data_dependent_initialization_target": "post_activation",
}


PROCESSING_BASE_HYPERPARAMETERS = {
    "convolutional_filters": 64,
    "convolutional_kernel_size": 8,
    "convolutional_kernel_l1_l2": [0.00, 0.0],
    "flanking_averages": True,
    "n_flank_length": 15,
    "c_flank_length": 15,
    "post_convolutional_dense_layer_sizes": [],
    "minibatch_size": PROCESSING_DEFAULT_MINIBATCH_SIZE,
    "dropout_rate": 0.5,
    "convolutional_activation": "relu",
    "patience": 20,
    "learning_rate": 0.001,
    "optimizer_implementation": DEFAULT_OPTIMIZER_IMPLEMENTATION,
    "init": "glorot_uniform",
}


def unique_hyperparameters(items):
    """Return ``items`` with duplicate dicts removed, preserving order."""
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def build_affinity_grid(
    minibatch_size=AFFINITY_DEFAULT_MINIBATCH_SIZE,
    optimizer_implementation=DEFAULT_OPTIMIZER_IMPLEMENTATION,
    data_dependent_initialization_target="post_activation",
    init="glorot_uniform",
):
    """Return the 35-architecture Class I pan-allele affinity grid."""
    if init not in AFFINITY_INIT_CHOICES:
        raise ValueError(
            "Unknown affinity initializer %r; expected one of: %s" % (
                init,
                ", ".join(AFFINITY_INIT_CHOICES),
            )
        )
    grid = []
    base = deepcopy(AFFINITY_BASE_HYPERPARAMETERS)
    base["minibatch_size"] = minibatch_size
    base["optimizer_implementation"] = optimizer_implementation
    base["init"] = init
    base["data_dependent_initialization_target"] = (
        data_dependent_initialization_target
    )
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


def processing_base_grid_iter(
    minibatch_size=PROCESSING_DEFAULT_MINIBATCH_SIZE,
    optimizer_implementation=DEFAULT_OPTIMIZER_IMPLEMENTATION,
    init="glorot_uniform",
):
    """Yield the base processing architecture grid before flank variants."""
    base = deepcopy(PROCESSING_BASE_HYPERPARAMETERS)
    base["minibatch_size"] = minibatch_size
    base["optimizer_implementation"] = optimizer_implementation
    base["init"] = init
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


def build_processing_base_grid(
    minibatch_size=PROCESSING_DEFAULT_MINIBATCH_SIZE,
    optimizer_implementation=DEFAULT_OPTIMIZER_IMPLEMENTATION,
    init="glorot_uniform",
):
    """Return the processing architecture grid before flank variants."""
    return unique_hyperparameters(
        processing_base_grid_iter(
            minibatch_size=minibatch_size,
            optimizer_implementation=optimizer_implementation,
            init=init,
        ))


def transform_processing_hyperparameters(kind, hyperparameters):
    """Return one processing flank variant for a hyperparameter dict."""
    if kind not in PROCESSING_VARIANT_CHOICES:
        raise ValueError(
            "Unknown processing variant %r; expected one of: %s" % (
                kind,
                ", ".join(PROCESSING_VARIANT_CHOICES),
            )
        )
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


def build_affinity_ablation_panels():
    """Return the paired, representative affinity audit panels."""
    architecture_keys = (
        ("feedforward", (512, 512), 1e-8),
        ("with-skip-connections", (256, 512, 512), 1e-8),
    )
    conditions = {
        "published_parity": dict(
            minibatch_size=128,
            optimizer_implementation="keras",
            data_dependent_initialization_target="post_activation",
        ),
        "proposed_release": dict(
            minibatch_size=1024,
            optimizer_implementation="keras",
            data_dependent_initialization_target="post_activation",
        ),
        "pre_activation_lsuv": dict(
            minibatch_size=1024,
            optimizer_implementation="keras",
            data_dependent_initialization_target="pre_activation",
        ),
        "no_lsuv": dict(
            minibatch_size=1024,
            optimizer_implementation="keras",
            data_dependent_initialization_target="post_activation",
        ),
        "pytorch_rmsprop": dict(
            minibatch_size=1024,
            optimizer_implementation="pytorch",
            data_dependent_initialization_target="post_activation",
        ),
        "pytorch_rmsprop_batch128": dict(
            minibatch_size=128,
            optimizer_implementation="pytorch",
            data_dependent_initialization_target="post_activation",
        ),
    }
    result = {}
    for condition, options in conditions.items():
        grid = build_affinity_grid(**options)
        selected = [
            item for item in grid
            if (
                item["topology"],
                tuple(item["layer_sizes"]),
                item["dense_layer_l1_regularization"],
            ) in architecture_keys
        ]
        if len(selected) != len(architecture_keys):
            raise RuntimeError(
                "Affinity ablation architecture selection returned %d rows"
                % len(selected)
            )
        if condition == "no_lsuv":
            for item in selected:
                item["data_dependent_initialization_method"] = None
        result[condition] = selected
    return result


def build_processing_ablation_panels():
    """Return the paired, representative processing audit panels."""
    architecture_keys = (
        ("tanh", 256, 11, (0.0, 0.0), (8,), 0.3),
        ("relu", 512, 17, (1e-6, 0.0), (16,), 0.5),
    )
    conditions = {
        "glorot_keras_adam": ("glorot_uniform", "keras"),
        "kaiming_keras_adam": ("kaiming_uniform_fan_in", "keras"),
        "glorot_pytorch_adam": ("glorot_uniform", "pytorch"),
        "kaiming_pytorch_adam": ("kaiming_uniform_fan_in", "pytorch"),
    }
    result = {}
    for condition, (init, optimizer_implementation) in conditions.items():
        grid = build_processing_base_grid(
            init=init,
            optimizer_implementation=optimizer_implementation,
        )
        selected = [
            item for item in grid
            if (
                item["convolutional_activation"],
                item["convolutional_filters"],
                item["convolutional_kernel_size"],
                tuple(item["convolutional_kernel_l1_l2"]),
                tuple(item["post_convolutional_dense_layer_sizes"]),
                item["dropout_rate"],
            ) in architecture_keys
        ]
        if len(selected) != len(architecture_keys):
            raise RuntimeError(
                "Processing ablation architecture selection returned %d rows"
                % len(selected)
            )
        result[condition] = selected
    return result


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


def add_minibatch_argument(parser, default=DEFAULT_MINIBATCH_SIZE):
    """Add the common training-minibatch-size argument to ``parser``."""
    parser.add_argument(
        "--minibatch-size",
        type=positive_int_arg,
        default=default,
        help=(
            "Training minibatch size to write into every architecture. "
            "Default: %(default)s"
        ),
    )


def add_optimizer_implementation_argument(parser):
    """Add the optimizer-equation implementation argument."""
    parser.add_argument(
        "--optimizer-implementation",
        choices=OPTIMIZER_IMPLEMENTATION_CHOICES,
        default=DEFAULT_OPTIMIZER_IMPLEMENTATION,
        help=(
            "Optimizer update equations to write into every architecture. "
            "Default: %(default)s (published 2.1.x parity)"
        ),
    )


def make_parser(prog=None):
    """Build the argparse parser for the unified generator command."""
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    subparsers = parser.add_subparsers(dest="recipe", required=True)

    affinity = subparsers.add_parser(
        "affinity",
        help="Generate the Class I pan-allele affinity grid.")
    add_minibatch_argument(affinity, AFFINITY_DEFAULT_MINIBATCH_SIZE)
    add_optimizer_implementation_argument(affinity)
    affinity.add_argument(
        "--data-dependent-initialization-target",
        choices=LSUV_TARGET_CHOICES,
        default="post_activation",
        help="LSUV activation boundary. Default: %(default)s (2.1.x parity)",
    )
    affinity.add_argument(
        "--init",
        choices=AFFINITY_INIT_CHOICES,
        default="glorot_uniform",
        help="Affinity-layer initializer. Default: %(default)s (2.1.x parity)",
    )

    processing_base = subparsers.add_parser(
        "processing-base",
        help="Generate the base Class I processing grid.")
    add_minibatch_argument(processing_base, PROCESSING_DEFAULT_MINIBATCH_SIZE)
    add_optimizer_implementation_argument(processing_base)
    processing_base.add_argument(
        "--init",
        choices=PROCESSING_INIT_CHOICES,
        default="glorot_uniform",
        help="Processing-layer initializer. Default: %(default)s (2.1.x parity)",
    )

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
        grid = build_affinity_grid(
            minibatch_size=args.minibatch_size,
            optimizer_implementation=args.optimizer_implementation,
            data_dependent_initialization_target=(
                args.data_dependent_initialization_target
            ),
            init=args.init,
        )
    elif args.recipe == "processing-base":
        grid = build_processing_base_grid(
            minibatch_size=args.minibatch_size,
            optimizer_implementation=args.optimizer_implementation,
            init=args.init,
        )
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

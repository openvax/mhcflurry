#!/usr/bin/env python
"""Generate a controlled affinity optimizer/LSUV/batch/initializer sweep."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import yaml

from mhcflurry.cli.generate_training_hyperparameters import (
    AFFINITY_INIT_CHOICES,
    LSUV_TARGET_CHOICES,
    OPTIMIZER_IMPLEMENTATION_CHOICES,
    build_affinity_grid,
)


DEFAULT_MINIBATCH_SIZES = (128, 256, 512, 1024)
DEFAULT_INITIALIZERS = ("glorot_uniform", "he_uniform", "orthogonal")
DESIGN_CHOICES = ("optimizer-lsuv-batch-init", "regularization-activation")
REGULARIZATION_BASE_RECIPES = {
    "native-pre-1024": {
        "minibatch_size": 1024,
        "optimizer_implementation": "pytorch",
        "data_dependent_initialization_method": "lsuv",
        "lsuv_target": "pre_activation",
    },
    "native-post-1024": {
        "minibatch_size": 1024,
        "optimizer_implementation": "pytorch",
        "data_dependent_initialization_method": "lsuv",
        "lsuv_target": "post_activation",
    },
    "keras-no-lsuv-1024": {
        "minibatch_size": 1024,
        "optimizer_implementation": "keras",
        "data_dependent_initialization_method": None,
        "lsuv_target": "not_applicable",
    },
}
REGULARIZATION_ACTIVATION_VARIANTS = (
    ("control", "tanh", "none", 0.50, False, 20),
    ("activation-relu", "relu", "none", 0.50, False, 20),
    ("activation-silu", "silu", "none", 0.50, False, 20),
    ("activation-gelu", "gelu", "none", 0.50, False, 20),
    ("normalization-batch", "tanh", "batch", 0.50, False, 20),
    ("normalization-layer", "tanh", "layer", 0.50, False, 20),
    ("dropout-keep-075", "tanh", "none", 0.75, False, 20),
    ("dropout-keep-100", "tanh", "none", 1.00, False, 20),
    ("restore-best", "tanh", "none", 0.50, True, 20),
    ("restore-best-patience-40", "tanh", "none", 0.50, True, 40),
)
REPRESENTATIVE_ARCHITECTURES = (
    ("feedforward", (512, 512), 1e-8),
    ("with-skip-connections", (256, 512, 512), 1e-8),
)
MANIFEST_FIELDS = (
    "condition",
    "minibatch_size",
    "optimizer_implementation",
    "data_dependent_initialization_method",
    "lsuv_target",
    "init",
    "effective_hidden_initializer",
    "architecture_count",
    "fold_count",
    "network_count",
    "hyperparameters_path",
    "hyperparameters_sha256",
    "design",
    "base_recipe",
    "activation",
    "normalization",
    "dropout_keep_probability",
    "restore_best_weights",
    "patience",
)


def condition_name(minibatch_size, optimizer_implementation, lsuv_target, init):
    """Return the stable filesystem label for one factorial condition."""
    return "__".join((
        "mb_%d" % minibatch_size,
        "rmsprop_%s" % optimizer_implementation,
        "lsuv_%s" % lsuv_target,
        "init_%s" % init,
    ))


def select_representative_architectures(grid):
    """Select the paired architectures used in the release parity audit."""
    selected = [
        item
        for item in grid
        if (
            item["topology"],
            tuple(item["layer_sizes"]),
            item["dense_layer_l1_regularization"],
        ) in REPRESENTATIVE_ARCHITECTURES
    ]
    if len(selected) != len(REPRESENTATIVE_ARCHITECTURES):
        raise RuntimeError(
            "Expected %d representative architectures; found %d" % (
                len(REPRESENTATIVE_ARCHITECTURES),
                len(selected),
            )
        )
    return selected


def build_conditions(
    mode="representative",
    minibatch_sizes=DEFAULT_MINIBATCH_SIZES,
    optimizer_implementations=OPTIMIZER_IMPLEMENTATION_CHOICES,
    lsuv_targets=LSUV_TARGET_CHOICES,
    initializers=DEFAULT_INITIALIZERS,
):
    """Return ``(condition, hyperparameters, axes)`` sweep records.

    LSUV replaces eligible hidden-layer weights with orthonormal matrices
    before variance scaling. Crossing a nominal Glorot/He initializer with
    LSUV would therefore be a misleading mostly-inactive axis. Instead, this
    design crosses the two LSUV activation targets using their canonical
    orthogonal replacement, then compares Glorot, He, and orthogonal weights
    in separate no-LSUV arms.
    """
    records = []
    shared_axes = itertools.product(
        minibatch_sizes,
        optimizer_implementations,
    )
    initialization_recipes = [
        {
            "data_dependent_initialization_method": "lsuv",
            "lsuv_target": target,
            "init": "glorot_uniform",
            "effective_hidden_initializer": "orthogonal_then_lsuv",
        }
        for target in lsuv_targets
    ] + [
        {
            "data_dependent_initialization_method": None,
            "lsuv_target": "not_applicable",
            "init": init,
            "effective_hidden_initializer": init,
        }
        for init in initializers
    ]
    for (minibatch_size, optimizer), recipe in itertools.product(
        shared_axes,
        initialization_recipes,
    ):
        grid = build_affinity_grid(
            minibatch_size=minibatch_size,
            optimizer_implementation=optimizer,
            data_dependent_initialization_target=(
                recipe["lsuv_target"]
                if recipe["lsuv_target"] != "not_applicable"
                else "post_activation"
            ),
            init=recipe["init"],
        )
        for item in grid:
            item["data_dependent_initialization_method"] = recipe[
                "data_dependent_initialization_method"
            ]
        if mode == "representative":
            grid = select_representative_architectures(grid)
        elif mode != "full":
            raise ValueError("Unknown factorial mode: %s" % mode)
        axes = {
            "minibatch_size": minibatch_size,
            "optimizer_implementation": optimizer,
            **recipe,
        }
        records.append((condition_name(
            minibatch_size,
            optimizer,
            recipe["lsuv_target"],
            recipe["init"],
        ), grid, axes))
    return records


def build_regularization_activation_conditions(base_recipe="native-pre-1024"):
    """Return one-factor-at-a-time variants of a fixed affinity recipe."""
    if base_recipe not in REGULARIZATION_BASE_RECIPES:
        raise ValueError("Unknown regularization base recipe: %s" % base_recipe)
    recipe = REGULARIZATION_BASE_RECIPES[base_recipe]
    grid = build_affinity_grid(
        minibatch_size=recipe["minibatch_size"],
        optimizer_implementation=recipe["optimizer_implementation"],
        data_dependent_initialization_target=(
            recipe["lsuv_target"]
            if recipe["lsuv_target"] != "not_applicable"
            else "post_activation"
        ),
        init="glorot_uniform",
    )
    grid = select_representative_architectures(grid)
    records = []
    for (
        variant,
        activation,
        normalization,
        dropout_keep,
        restore_best_weights,
        patience,
    ) in (
            REGULARIZATION_ACTIVATION_VARIANTS):
        condition = "%s__%s" % (base_recipe, variant)
        condition_grid = []
        for source in grid:
            item = dict(source)
            item.update({
                "activation": activation,
                "batch_normalization": False,
                "normalization": normalization,
                "dropout_probability": dropout_keep,
                "restore_best_weights": restore_best_weights,
                "patience": patience,
                "data_dependent_initialization_method": recipe[
                    "data_dependent_initialization_method"],
            })
            condition_grid.append(item)
        records.append((condition, condition_grid, {
            "design": "regularization-activation",
            "base_recipe": base_recipe,
            "minibatch_size": recipe["minibatch_size"],
            "optimizer_implementation": recipe[
                "optimizer_implementation"],
            "data_dependent_initialization_method": recipe[
                "data_dependent_initialization_method"],
            "lsuv_target": recipe["lsuv_target"],
            "init": "glorot_uniform",
            "effective_hidden_initializer": (
                "orthogonal_then_lsuv"
                if recipe["data_dependent_initialization_method"] == "lsuv"
                else "glorot_uniform"
            ),
            "activation": activation,
            "normalization": normalization,
            "dropout_keep_probability": dropout_keep,
            "restore_best_weights": restore_best_weights,
            "patience": patience,
        }))
    return records


def write_conditions(
        out_dir, mode="representative",
        design="optimizer-lsuv-batch-init",
        regularization_base_recipe="native-pre-1024",
        **factorial_axes):
    """Write condition YAML files and checksummed JSON/CSV manifests."""
    out_dir = Path(out_dir)
    conditions_dir = out_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if design == "optimizer-lsuv-batch-init":
        conditions = build_conditions(mode=mode, **factorial_axes)
        baseline_condition = condition_name(
            128, "keras", "post_activation", "glorot_uniform")
    elif design == "regularization-activation":
        if mode != "representative":
            raise ValueError(
                "regularization-activation supports representative mode only")
        conditions = build_regularization_activation_conditions(
            regularization_base_recipe)
        baseline_condition = "%s__control" % regularization_base_recipe
    else:
        raise ValueError("Unknown affinity factorial design: %s" % design)
    for condition, grid, axes in conditions:
        relative_path = Path("conditions") / (condition + ".yaml")
        payload = yaml.safe_dump(grid, sort_keys=True)
        (out_dir / relative_path).write_text(payload)
        records.append({
            "condition": condition,
            **axes,
            "architecture_count": len(grid),
            "fold_count": 4,
            "network_count": 4 * len(grid),
            "hyperparameters_path": str(relative_path),
            "hyperparameters_sha256": hashlib.sha256(
                payload.encode()
            ).hexdigest(),
        })

    fixed_controls = {
        "fold_count": 4,
        "learning_rate": 0.001,
        "optimizer": "rmsprop",
        "random_seed": 42,
        "lsuv_replaces_eligible_weights_with_orthogonal": True,
    }
    if design == "regularization-activation":
        fixed_controls.update({
            "activation_screen": ["tanh", "relu", "silu", "gelu"],
            "dropout_probability_semantics": "keep_probability",
        })
    manifest = {
        "schema_version": 1,
        "design": design,
        "mode": mode,
        "baseline_condition": baseline_condition,
        "fixed_controls": fixed_controls,
        "records": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (out_dir / "manifest.csv").open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return manifest


def make_parser():
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument(
        "--mode", choices=("representative", "full"), default="representative"
    )
    parser.add_argument(
        "--design", choices=DESIGN_CHOICES,
        default="optimizer-lsuv-batch-init",
    )
    parser.add_argument(
        "--regularization-base-recipe",
        choices=tuple(REGULARIZATION_BASE_RECIPES),
        default="native-pre-1024",
    )
    parser.add_argument(
        "--minibatch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_MINIBATCH_SIZES,
    )
    parser.add_argument(
        "--optimizer-implementations",
        nargs="+",
        choices=OPTIMIZER_IMPLEMENTATION_CHOICES,
        default=OPTIMIZER_IMPLEMENTATION_CHOICES,
    )
    parser.add_argument(
        "--lsuv-targets",
        nargs="+",
        choices=LSUV_TARGET_CHOICES,
        default=LSUV_TARGET_CHOICES,
    )
    parser.add_argument(
        "--initializers",
        nargs="+",
        choices=AFFINITY_INIT_CHOICES,
        default=DEFAULT_INITIALIZERS,
    )
    return parser


def main(argv=None):
    """Write the requested factorial grid."""
    args = make_parser().parse_args(argv)
    if any(value <= 0 for value in args.minibatch_sizes):
        raise ValueError("Minibatch sizes must be positive")
    manifest = write_conditions(
        args.out_dir,
        mode=args.mode,
        design=args.design,
        regularization_base_recipe=args.regularization_base_recipe,
        minibatch_sizes=args.minibatch_sizes,
        optimizer_implementations=args.optimizer_implementations,
        lsuv_targets=args.lsuv_targets,
        initializers=args.initializers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

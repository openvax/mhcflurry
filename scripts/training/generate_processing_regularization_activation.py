#!/usr/bin/env python
"""Generate matched 5-aa processing regularization/activation experiments."""

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from mhcflurry.cli.generate_training_hyperparameters import (
    build_processing_ablation_panels,
    build_processing_variant_grid,
)


ACTIVATIONS = ("tanh", "relu", "silu", "gelu")
NORMALIZATIONS = ("none", "batch", "layer")
ARCHITECTURE_NAMES = ("small", "large")
MANIFEST_FIELDS = (
    "condition",
    "architecture",
    "baseline_condition",
    "variant",
    "activation",
    "normalization",
    "dropout_rate",
    "restore_best_weights",
    "patience",
    "minibatch_size",
    "optimizer_implementation",
    "init",
    "fold_count",
    "network_count",
    "hyperparameters_path",
    "hyperparameters_sha256",
)


def _architecture_controls():
    grid = build_processing_variant_grid(
        build_processing_ablation_panels()["glorot_keras_adam"],
        "short_flanks",
    )
    keys = {
        ("tanh", 256, 11): "small",
        ("relu", 512, 17): "large",
    }
    result = {
        keys.get((
            item["convolutional_activation"],
            item["convolutional_filters"],
            item["convolutional_kernel_size"],
        )): item
        for item in grid
    }
    if None in result or set(result) != set(ARCHITECTURE_NAMES):
        raise RuntimeError("Could not identify representative architectures")
    return result


def build_conditions():
    """Return independent, one-factor-at-a-time processing conditions."""
    records = []
    for architecture, control in _architecture_controls().items():
        control_activation = control["convolutional_activation"]
        control_dropout = control["dropout_rate"]
        control_patience = control["patience"]
        variants = [(
            "control", control_activation, "none", control_dropout,
            False, control_patience)]
        variants.extend(
            (
                "activation-%s" % activation,
                activation,
                "none",
                control_dropout,
                False,
                control_patience,
            )
            for activation in ACTIVATIONS
            if activation != control_activation
        )
        variants.extend((
            (
                "normalization-batch", control_activation, "batch",
                control_dropout, False, control_patience),
            (
                "normalization-layer", control_activation, "layer",
                control_dropout, False, control_patience),
            (
                "dropout-half", control_activation, "none",
                control_dropout / 2.0, False, control_patience),
            (
                "dropout-none", control_activation, "none", 0.0,
                False, control_patience),
            (
                "restore-best", control_activation, "none", control_dropout,
                True, control_patience),
            (
                "restore-best-patience-40", control_activation, "none",
                control_dropout, True, 40),
        ))
        baseline = "%s__control" % architecture
        for (
            variant,
            activation,
            normalization,
            dropout_rate,
            restore_best_weights,
            patience,
        ) in variants:
            item = deepcopy(control)
            item.update({
                "convolutional_activation": activation,
                "normalization": normalization,
                "dropout_rate": dropout_rate,
                "restore_best_weights": restore_best_weights,
                "patience": patience,
            })
            records.append(("%s__%s" % (architecture, variant), [item], {
                "architecture": architecture,
                "baseline_condition": baseline,
                "variant": variant,
                "activation": activation,
                "normalization": normalization,
                "dropout_rate": dropout_rate,
                "restore_best_weights": restore_best_weights,
                "patience": patience,
                "minibatch_size": item["minibatch_size"],
                "optimizer_implementation": item["optimizer_implementation"],
                "init": item["init"],
            }))
    return records


def write_conditions(out_dir):
    """Write condition YAMLs and a checksummed manifest."""
    out_dir = Path(out_dir)
    conditions_dir = out_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for condition, grid, axes in build_conditions():
        relative_path = Path("conditions") / (condition + ".yaml")
        payload = yaml.safe_dump(grid, sort_keys=True)
        (out_dir / relative_path).write_text(payload)
        records.append({
            "condition": condition,
            **axes,
            "fold_count": 4,
            "network_count": 4,
            "hyperparameters_path": str(relative_path),
            "hyperparameters_sha256": hashlib.sha256(
                payload.encode()).hexdigest(),
        })
    manifest = {
        "schema_version": 1,
        "design": "processing-5aa-regularization-activation",
        "fixed_controls": {
            "flank_length_each_side": 5,
            "fold_count": 4,
            "held_out_samples_per_fold": 10,
            "minibatch_size": 512,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "optimizer_implementation": "keras",
            "init": "glorot_uniform",
            "random_seed": 42,
        },
        "architectures": list(ARCHITECTURE_NAMES),
        "activation_screen": list(ACTIVATIONS),
        "normalization_screen": list(NORMALIZATIONS),
        "records": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (out_dir / "manifest.csv").open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return manifest


def main(argv=None):
    """Write the experiment design requested on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    args = parser.parse_args(argv)
    print(json.dumps(write_conditions(args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

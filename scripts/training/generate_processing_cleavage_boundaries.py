#!/usr/bin/env python
"""Write paired compact/extended cleavage-boundary processing panels."""

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


ARCHITECTURES = {
    ("tanh", 256, 11): "small_tanh",
    ("relu", 512, 17): "large_relu",
}
WINDOWS = {
    "compact_5x2": 2,
    "extended_5x5": 5,
}
CONTEXT_DROPOUT = 0.25
MANIFEST_FIELDS = (
    "condition", "architecture", "model_kind", "window",
    "baseline_5aa_condition", "baseline_no_flank_condition",
    "external_context_length", "peptide_context_length", "context_dropout",
    "fold_count", "network_count", "hyperparameters_path",
    "hyperparameters_sha256",
)


def _controls():
    controls = build_processing_variant_grid(
        build_processing_ablation_panels()["glorot_keras_adam"],
        "short_flanks",
    )
    result = {}
    for item in controls:
        key = (
            item["convolutional_activation"],
            item["convolutional_filters"],
            item["convolutional_kernel_size"],
        )
        architecture = ARCHITECTURES.get(key)
        if architecture is None or architecture in result:
            raise RuntimeError("Unexpected representative architecture: %r" % (key,))
        result[architecture] = item
    if set(result) != set(ARCHITECTURES.values()):
        raise RuntimeError("Missing representative processing architecture")
    return result


def build_conditions():
    """Return boundary conditions and their exactly matched legacy controls."""
    records = []
    for architecture, control in _controls().items():
        baseline_5aa = "%s__legacy_5aa" % architecture
        baseline_no_flank = "%s__legacy_no_flank" % architecture
        common_axes = {
            "architecture": architecture,
            "baseline_5aa_condition": baseline_5aa,
            "baseline_no_flank_condition": baseline_no_flank,
        }

        records.append((baseline_5aa, [deepcopy(control)], {
            **common_axes,
            "model_kind": "legacy_5aa",
            "window": "legacy_5aa",
            "external_context_length": 5,
            "peptide_context_length": 0,
            "context_dropout": 0.0,
        }))

        no_flank = deepcopy(control)
        no_flank.update({"n_flank_length": 0, "c_flank_length": 0})
        records.append((baseline_no_flank, [no_flank], {
            **common_axes,
            "model_kind": "legacy_no_flank",
            "window": "legacy_no_flank",
            "external_context_length": 0,
            "peptide_context_length": 0,
            "context_dropout": 0.0,
        }))

        for window, peptide_context_length in WINDOWS.items():
            item = deepcopy(control)
            item.update({
                "flanking_averages": False,
                "cleavage_boundary_flank_length": 5,
                "cleavage_boundary_peptide_length": peptide_context_length,
                "cleavage_boundary_hidden_size": 32,
                "cleavage_boundary_context_dropout": CONTEXT_DROPOUT,
            })
            condition = "%s__%s" % (architecture, window)
            records.append((condition, [item], {
                **common_axes,
                "model_kind": "cleavage_boundary",
                "window": window,
                "external_context_length": 5,
                "peptide_context_length": peptide_context_length,
                "context_dropout": CONTEXT_DROPOUT,
            }))
    return records


def write_conditions(out_dir):
    """Write condition YAMLs and a checksummed experiment manifest."""
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
        "design": "processing-cleavage-boundaries",
        "fixed_controls": {
            "flank_length_each_side": 5,
            "boundary_hidden_size": 32,
            "fold_count": 4,
            "held_out_samples_per_fold": 10,
            "minibatch_size": 512,
            "optimizer_implementation": "keras",
            "init": "glorot_uniform",
            "random_seed": 42,
        },
        "network_budget": {
            "boundary_networks": 16,
            "legacy_control_networks": 16,
            "total_networks": 32,
        },
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
    """Write the experiment design."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    args = parser.parse_args(argv)
    print(json.dumps(write_conditions(args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

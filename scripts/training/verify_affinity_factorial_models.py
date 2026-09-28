#!/usr/bin/env python
"""Verify a completed affinity-factorial model collection."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml


CONTROL_KEYS = (
    "data_dependent_initialization_method",
    "data_dependent_initialization_target",
    "init",
    "learning_rate",
    "minibatch_size",
    "optimizer",
    "optimizer_implementation",
)
ARCHITECTURE_KEYS = (
    "topology",
    "layer_sizes",
    "dense_layer_l1_regularization",
)


def architecture_signature(hyperparameters):
    """Return the stable architecture identity used for fold accounting."""
    return tuple(
        tuple(hyperparameters[key])
        if isinstance(hyperparameters[key], list)
        else hyperparameters[key]
        for key in ARCHITECTURE_KEYS
    )


def verify(models_dir, hyperparameters_path, num_folds=4):
    """Raise on missing, shrunken, duplicated, or misconfigured models."""
    models_dir = Path(models_dir)
    expected = yaml.safe_load(Path(hyperparameters_path).read_text())
    expected_by_arch = {
        architecture_signature(hyperparameters): hyperparameters
        for hyperparameters in expected
    }
    if len(expected_by_arch) != len(expected):
        raise ValueError("Expected hyperparameter YAML has duplicate architectures")

    with (models_dir / "manifest.csv").open(newline="") as fd:
        rows = list(csv.DictReader(fd))
    expected_count = len(expected) * num_folds
    if len(rows) != expected_count:
        raise ValueError(
            "Expected %d models; found %d" % (expected_count, len(rows))
        )

    observed = Counter()
    fold_hashes = {}
    for row in rows:
        weights_path = models_dir / ("weights_%s.npz" % row["model_name"])
        if not weights_path.is_file():
            raise ValueError("Missing weights: %s" % weights_path)
        config = json.loads(row["config_json"])
        actual = config["hyperparameters"]
        signature = architecture_signature(actual)
        if signature not in expected_by_arch:
            raise ValueError("Unexpected architecture: %r" % (signature,))
        expected_hyperparameters = expected_by_arch[signature]
        for key in CONTROL_KEYS:
            expected_value = expected_hyperparameters.get(key)
            # The published pan-allele recipe intentionally fine-tunes at one
            # tenth of the pretraining learning rate. Training serializes that
            # effective value in the fitted model while the input YAML retains
            # the pretraining value. This behavior is present in v2.1.x and is
            # not a recipe mismatch.
            if (
                    key == "learning_rate"
                    and expected_hyperparameters.get(
                        "train_data", {}).get("pretrain", False)
                    and expected_value is not None):
                expected_value /= 10
            if actual.get(key) != expected_value:
                raise ValueError(
                    "%s mismatch for %r: expected %r; found %r" % (
                        key,
                        signature,
                        expected_value,
                        actual.get(key),
                    )
                )
        fit_info = config["fit_info"][-1]
        configured_batch = int(actual["minibatch_size"])
        if int(fit_info["effective_minibatch_size"]) != configured_batch:
            raise ValueError(
                "Training minibatch shrank for %s: configured %d; effective %s"
                % (
                    row["model_name"],
                    configured_batch,
                    fit_info["effective_minibatch_size"],
                )
            )
        training_info = fit_info["training_info"]
        fold = int(training_info["fold_num"])
        if int(training_info["num_folds"]) != num_folds:
            raise ValueError("Wrong fold count for %s" % row["model_name"])
        observed[(signature, fold)] += 1
        fold_hash = training_info["train_peptide_hash"]
        previous_hash = fold_hashes.setdefault(fold, fold_hash)
        if fold_hash != previous_hash:
            raise ValueError("Training-row hash differs within fold %d" % fold)

    expected_pairs = {
        (signature, fold)
        for signature in expected_by_arch
        for fold in range(num_folds)
    }
    if set(observed) != expected_pairs or set(observed.values()) != {1}:
        raise ValueError("Architecture/fold coverage is not exactly one each")
    return {
        "architecture_count": len(expected),
        "fold_count": num_folds,
        "model_count": len(rows),
        "fold_training_peptide_hashes": fold_hashes,
    }


def main(argv=None):
    """Run model verification and print its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models_dir")
    parser.add_argument("hyperparameters")
    parser.add_argument("--num-folds", type=int, default=4)
    args = parser.parse_args(argv)
    print(json.dumps(
        verify(args.models_dir, args.hyperparameters, args.num_folds),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()

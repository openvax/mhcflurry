#!/usr/bin/env python
"""Split a trained pan-allele ensemble into fold-complete architectures."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas


ARCHITECTURE_FIELDS = (
    "topology",
    "layer_sizes",
    "dense_layer_l1_regularization",
)
SHARED_METADATA_NAMES = (
    "allele_sequences.csv",
    "info.txt",
    "percent_ranks.csv",
    "train_data.csv.bz2",
)


def sha256_file(path):
    """Return the SHA256 digest of ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as fd:
        for chunk in iter(lambda: fd.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_record(row):
    """Return architecture/fold metadata from one manifest row."""
    config = json.loads(row.config_json)
    hyperparameters = config["hyperparameters"]
    training_info = None
    for fit in reversed(config.get("fit_info", [])):
        candidate = fit.get("training_info", {})
        if "architecture_num" in candidate and "fold_num" in candidate:
            training_info = candidate
            break
    if training_info is None:
        raise ValueError(
            "Model %s has no architecture_num/fold_num training metadata"
            % row.model_name
        )
    return {
        "model_name": row.model_name,
        "architecture_num": int(training_info["architecture_num"]),
        "fold_num": int(training_info["fold_num"]),
        "architecture": {
            field: hyperparameters[field] for field in ARCHITECTURE_FIELDS
        },
    }


def link_or_copy(source, target):
    """Hard-link a file when possible, otherwise copy it."""
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def shared_metadata_paths(models_dir):
    """Yield predictor metadata required for loading and overlap auditing."""
    models_dir = Path(models_dir)
    seen = set()
    for name in SHARED_METADATA_NAMES:
        path = models_dir / name
        if path.is_file():
            seen.add(path.name)
            yield path
    for path in sorted(models_dir.glob("pseudosequences.*.csv")):
        if path.name not in seen:
            yield path


def split_models(models_dir, out_dir, expected_folds=4):
    """Write one loadable model directory per architecture."""
    models_dir = Path(models_dir)
    out_dir = Path(out_dir)
    manifest_path = models_dir / "manifest.csv"
    manifest = pandas.read_csv(manifest_path)
    missing = {"model_name", "allele", "config_json"} - set(manifest.columns)
    if missing:
        raise ValueError(
            "Manifest missing required column(s): %s" % ", ".join(sorted(missing))
        )
    if set(manifest.allele) != {"pan-class1"}:
        raise ValueError("Architecture split requires pan-class1 models only")

    records = [model_record(row) for row in manifest.itertuples(index=False)]
    by_architecture = {}
    for record in records:
        by_architecture.setdefault(record["architecture_num"], []).append(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    manifest_sha256 = sha256_file(manifest_path)
    for architecture_num, architecture_records in sorted(by_architecture.items()):
        descriptors = {
            json.dumps(record["architecture"], sort_keys=True)
            for record in architecture_records
        }
        if len(descriptors) != 1:
            raise ValueError(
                "Architecture %d has inconsistent hyperparameters"
                % architecture_num
            )
        folds = sorted(record["fold_num"] for record in architecture_records)
        if folds != list(range(expected_folds)):
            raise ValueError(
                "Architecture %d has folds %s; expected %s"
                % (architecture_num, folds, list(range(expected_folds)))
            )
        names = [record["model_name"] for record in architecture_records]
        target = out_dir / ("architecture_%d" % architecture_num)
        provenance = {
            "schema_version": 1,
            "source_models_dir": str(models_dir.resolve()),
            "source_manifest_sha256": manifest_sha256,
            "architecture_num": architecture_num,
            "architecture": architecture_records[0]["architecture"],
            "folds": folds,
            "model_names": names,
        }
        if target.exists():
            existing_path = target / "subset_provenance.json"
            if not existing_path.is_file() or json.loads(
                    existing_path.read_text()) != provenance:
                raise ValueError(
                    "Existing architecture subset does not match: %s" % target
                )
            results.append({"models_dir": str(target), **provenance})
            continue

        temp = Path(tempfile.mkdtemp(
            dir=out_dir, prefix=".%s-" % target.name
        ))
        try:
            subset = manifest[manifest.model_name.isin(names)].copy()
            if len(subset) != expected_folds:
                raise ValueError(
                    "Architecture %d selected %d manifest rows; expected %d"
                    % (architecture_num, len(subset), expected_folds)
                )
            subset.to_csv(temp / "manifest.csv", index=False)
            link_modes = set()
            for name in names:
                source = models_dir / ("weights_%s.npz" % name)
                if not source.is_file():
                    raise ValueError("Missing model weights: %s" % source)
                link_modes.add(link_or_copy(source, temp / source.name))
            for source in shared_metadata_paths(models_dir):
                link_modes.add(link_or_copy(source, temp / source.name))
            (temp / "subset_provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n"
            )
            os.replace(temp, target)
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        results.append({
            "models_dir": str(target),
            "link_modes": sorted(link_modes),
            **provenance,
        })
    return results


def main(argv=None):
    """Run the architecture splitter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--expected-folds", type=int, default=4)
    args = parser.parse_args(argv)
    result = split_models(
        args.models_dir, args.out_dir, expected_folds=args.expected_folds
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

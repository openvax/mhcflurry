#!/usr/bin/env python3
"""Copy a selected predictor for distribution without training checkpoints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

csv.field_size_limit(sys.maxsize)


def export_models(source, destination):
    """Preserve prediction inputs and audit tables; omit unused network states.

    The source is never modified. Each exported manifest records the digest of
    its original and the checkpoint-only columns removed for inference loading.
    Active NPZ files are copied byte-for-byte and must all exist.
    """
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists():
        raise ValueError("Export destination already exists: %s" % destination)
    if source == destination.resolve() or source in destination.resolve().parents:
        raise ValueError("Export destination must be outside the source")
    records = []

    def copy_directory(src, dst):
        dst.mkdir(parents=True)
        manifest = src / "manifest.csv"
        selected = None
        if manifest.exists():
            with manifest.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            if not fields or "model_name" not in fields or not rows:
                raise ValueError("Empty or invalid model manifest: %s" % manifest)
            names = [row["model_name"] for row in rows]
            if len(set(names)) != len(names) or any(
                    not name or Path(name).name != name or "\\" in name
                    for name in names):
                raise ValueError("Invalid or duplicate model name: %s" % manifest)
            selected = {"weights_%s.npz" % name for name in names}
            missing = sorted(name for name in selected if not (src / name).is_file())
            if missing:
                raise ValueError("Missing selected weights in %s: %s" % (src, missing))
            removed = [f for f in fields if f.startswith("checkpoint_") and f.endswith("_weights")]
            with (dst / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[f for f in fields if f not in removed],
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            records.append({
                "manifest": str(manifest.relative_to(source)),
                "source_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "exported_sha256": hashlib.sha256((dst / "manifest.csv").read_bytes()).hexdigest(),
                "removed_checkpoint_columns": removed,
                "selected_weights": sorted(selected),
            })
        for item in sorted(src.iterdir()):
            if item.is_symlink():
                raise ValueError("Symlinks are not supported in model exports: %s" % item)
            if item.name == "manifest.csv" and selected is not None:
                continue
            if item.name == "checkpoints":
                continue
            if selected is not None and item.name.startswith("weights_") and item.suffix == ".npz" and item.name not in selected:
                continue
            if item.is_dir():
                copy_directory(item, dst / item.name)
            else:
                shutil.copy2(item, dst / item.name)

    try:
        copy_directory(source, destination)
        (destination / "inference_export.json").write_text(json.dumps({
            "format": 1,
            "description": "Selected inference weights; original training metadata retained.",
            "manifests": records,
        }, indent=2) + "\n")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    export_models(args.source, args.destination)


if __name__ == "__main__":
    main()

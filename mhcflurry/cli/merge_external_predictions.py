# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Merge precomputed external-predictor benchmark groups into one table."""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path

import pandas

from mhcflurry.experiment_archive import sha256_file


def make_parser(prog="mhcflurry eval merge-external-predictions"):
    """Build the external-prediction merge parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="PREDICTOR=MANIFEST.csv",
        help=(
            "Predictor name and data_evaluation group manifest. Repeat once "
            "per predictor. The manifests must describe the same base rows."
        ),
    )
    parser.add_argument("--out", required=True, metavar="CSV[.bz2]")
    return parser


def _parse_group(spec):
    if "=" not in spec:
        raise ValueError(
            "--group must have the form PREDICTOR=MANIFEST.csv: %s" % spec)
    predictor, manifest = spec.split("=", 1)
    predictor = predictor.strip()
    manifest = Path(manifest).expanduser().resolve()
    if not predictor or not manifest.is_file():
        raise ValueError("Invalid --group: %s" % spec)
    return predictor, manifest


def _member_key(filename, predictor):
    marker = ".%s." % predictor
    if marker not in filename:
        raise ValueError(
            "Group member does not contain predictor marker %s: %s" % (
                marker, filename))
    return filename.replace(marker, ".<predictor>.", 1)


def _load_group(predictor, manifest):
    frame = pandas.read_csv(manifest)
    if list(frame.columns) != ["filename"]:
        raise ValueError(
            "Group manifest must contain only a filename column: %s" %
            manifest)
    records = []
    seen = set()
    for filename in frame.filename:
        path = (manifest.parent / str(filename)).resolve()
        if not path.is_file():
            raise ValueError("Missing group member: %s" % path)
        key = _member_key(path.name, predictor)
        if key in seen:
            raise ValueError("Duplicate group member identity: %s" % key)
        seen.add(key)
        records.append((key, path))
    return records


def merge_external_prediction_groups(group_specs, out_path):
    """Merge aligned per-predictor group members and return provenance."""
    groups = [_parse_group(spec) for spec in group_specs]
    predictor_names = [predictor for predictor, _manifest in groups]
    if len(set(predictor_names)) != len(predictor_names):
        raise ValueError("Predictor names in --group must be unique")

    loaded = {
        predictor: (manifest, _load_group(predictor, manifest))
        for predictor, manifest in groups
    }
    first_predictor = predictor_names[0]
    first_records = loaded[first_predictor][1]
    expected_keys = [key for key, _path in first_records]
    expected_key_set = set(expected_keys)
    for predictor in predictor_names[1:]:
        actual_keys = {key for key, _path in loaded[predictor][1]}
        if actual_keys != expected_key_set:
            raise ValueError(
                "Group %s describes different members: missing=%s extra=%s" % (
                    predictor,
                    sorted(expected_key_set - actual_keys)[:5],
                    sorted(actual_keys - expected_key_set)[:5],
                ))

    path_by_predictor = {
        predictor: dict(records)
        for predictor, (_manifest, records) in loaded.items()
    }
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = bz2.open if out_path.suffix == ".bz2" else open
    input_records = []
    total_rows = 0
    metadata_columns = None
    with opener(out_path, "wt", newline="") as out_fd:
        for member_index, key in enumerate(expected_keys):
            merged = None
            for predictor in predictor_names:
                path = path_by_predictor[predictor][key]
                frame = pandas.read_csv(path)
                required = [predictor]
                missing = [column for column in required if column not in frame]
                if missing:
                    raise ValueError(
                        "%s lacks predictor column %s" % (path, predictor))
                current_metadata = [
                    column for column in frame.columns
                    if column not in (
                        predictor, "%s_best_allele" % predictor)
                ]
                if merged is None:
                    metadata_columns = metadata_columns or current_metadata
                    if current_metadata != metadata_columns:
                        raise ValueError(
                            "Benchmark metadata columns changed in %s" % path)
                    merged = frame[current_metadata].copy()
                else:
                    if current_metadata != metadata_columns:
                        raise ValueError(
                            "Benchmark metadata columns differ in %s" % path)
                    if not frame[current_metadata].equals(
                            merged[metadata_columns]):
                        raise ValueError(
                            "Benchmark rows are not aligned for %s in %s" % (
                                predictor, path))
                merged[predictor] = frame[predictor].to_numpy()
                input_records.append({
                    "predictor": predictor,
                    "member_identity": key,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rows": len(frame),
                })
            merged.to_csv(
                out_fd, index=False, header=member_index == 0)
            total_rows += len(merged)
            completed = member_index + 1
            if completed == len(expected_keys) or completed % 10 == 0:
                print(
                    "Merged %d/%d benchmark members (%d rows)" % (
                        completed, len(expected_keys), total_rows),
                    flush=True,
                )

    manifest_records = [
        {
            "predictor": predictor,
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        }
        for predictor, manifest in groups
    ]
    provenance = {
        "schema_version": 1,
        "predictors": predictor_names,
        "metadata_columns": metadata_columns,
        "member_count": len(expected_keys),
        "row_count": total_rows,
        "group_manifests": manifest_records,
        "input_members": input_records,
        "output": {
            "path": str(out_path),
            "sha256": sha256_file(out_path),
        },
    }
    provenance_path = Path("%s.provenance.json" % out_path)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    provenance["provenance_path"] = str(provenance_path)
    return provenance


def run(args):
    """Merge external predictions specified on the command line."""
    provenance = merge_external_prediction_groups(args.group, args.out)
    print(provenance["output"]["path"])
    return 0


def run_argv(argv=None, prog="mhcflurry eval merge-external-predictions"):
    """Parse arguments and merge external prediction groups."""
    return run(make_parser(prog).parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(run_argv())

#!/usr/bin/env python3
"""Score a fixed weighted ensemble from preserved processing predictions."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy
import pandas

from mhcflurry.cli.processing_affinity_control import (
    score_risk_sets,
    summarize_metrics,
)


def sha256_file(path):
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_member(value):
    """Parse a COLUMN=WEIGHT member specification."""
    try:
        column, weight = value.rsplit("=", 1)
        weight = float(weight)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "members must have the form COLUMN=WEIGHT"
        ) from error
    if not column or not numpy.isfinite(weight) or weight < 0:
        raise argparse.ArgumentTypeError(
            "member columns must be nonempty and weights finite and nonnegative"
        )
    return column, weight


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog=os.environ.get("MHCFLURRY_CLI_PROG"), description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--member", action="append", type=parse_member, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.predictions).resolve()
    frame = pandas.read_csv(source, dtype={"sample_id": str}, low_memory=False)

    columns = [column for column, _ in args.member]
    missing = sorted(set(columns + [args.baseline]) - set(frame.columns))
    if missing:
        raise ValueError("Missing score columns: %s" % ", ".join(missing))
    if args.name in frame.columns:
        raise ValueError("Output score column already exists: %s" % args.name)
    weights = numpy.asarray([weight for _, weight in args.member], dtype="float64")
    if weights.sum() <= 0:
        raise ValueError("At least one member weight must be positive")
    weights /= weights.sum()
    frame[args.name] = numpy.average(
        frame[columns].to_numpy(dtype="float64"), axis=1, weights=weights
    )

    score_columns = list(dict.fromkeys([args.baseline] + columns + [args.name]))
    metrics = score_risk_sets(frame, score_columns)
    summary, comparisons = summarize_metrics(metrics, args.baseline)

    predictions_path = out / "matched_predictions.csv.bz2"
    frame.to_csv(predictions_path, index=False, compression="bz2")
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    comparisons.to_csv(out / "comparisons.csv", index=False)
    provenance = {
        "format": 1,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "rows": int(len(frame)),
        },
        "name": args.name,
        "baseline": args.baseline,
        "members": [
            {"column": column, "normalized_weight": float(weight)}
            for column, weight in zip(columns, weights)
        ],
        "outputs": {
            "matched_predictions_sha256": sha256_file(predictions_path),
        },
    }
    with open(out / "provenance.json", "w") as fd:
        json.dump(provenance, fd, indent=2, sort_keys=True)
        fd.write("\n")
    print(summary.loc[summary.score.isin(score_columns)].to_string(index=False))


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

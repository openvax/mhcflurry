#!/usr/bin/env python3
"""Compose a processing ensemble from complete predictor directories."""

import argparse
import json
import os
from pathlib import Path

from mhcflurry import Class1ProcessingPredictor
from mhcflurry.experiment_archive import fingerprint_directory


def parse_predictor(value):
    """Parse a LABEL=PATH predictor specification."""
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "predictors must have the form LABEL=PATH"
        ) from error
    if not label or not path:
        raise argparse.ArgumentTypeError("predictor label and path must be nonempty")
    return label, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog=os.environ.get("MHCFLURRY_CLI_PROG"), description=__doc__)
    parser.add_argument(
        "--predictor", action="append", type=parse_predictor, required=True
    )
    parser.add_argument("--require-equal-counts", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def run(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError("Output already exists: %s" % out)

    loaded = []
    sources = []
    for label, raw_path in args.predictor:
        path = Path(raw_path).resolve()
        predictor = Class1ProcessingPredictor.load(str(path))
        loaded.append(predictor)
        sources.append(
            {
                "label": label,
                "path": str(path),
                "models": int(len(predictor.models)),
                "fingerprint": fingerprint_directory(path),
            }
        )
    counts = [len(predictor.models) for predictor in loaded]
    if args.require_equal_counts and len(set(counts)) != 1:
        raise ValueError("Predictor model counts differ: %s" % counts)

    result = Class1ProcessingPredictor(
        models=[model for predictor in loaded for model in predictor.models],
        provenance_string="fixed ensemble: %s" % ", ".join(
            source["label"] for source in sources
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out))
    reloaded = Class1ProcessingPredictor.load(str(out))
    if len(reloaded.models) != sum(counts):
        raise RuntimeError("Reloaded ensemble has the wrong model count")

    provenance = {
        "format": 1,
        "averaging": "equal weight per network",
        "sources": sources,
        "assembled_predictor": {
            "path": str(out),
            "models": int(len(reloaded.models)),
            "source_model_counts": counts,
            "effective_source_weights": [
                float(count) / sum(counts) for count in counts
            ],
            "fingerprint": fingerprint_directory(out),
        },
    }
    with open(out / "ensemble_provenance.json", "w") as fd:
        json.dump(provenance, fd, indent=2, sort_keys=True)
        fd.write("\n")
    print(json.dumps(provenance["assembled_predictor"], sort_keys=True))


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

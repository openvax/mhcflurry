#!/usr/bin/env python3
"""Shrink release-pipeline inputs for MHCFLURRY_RELEASE_SMOKE runs.

A smoke run exercises every stage and code path of the release pipeline at
tiny scale, to catch integration failures before an expensive run. It is never
a release artifact. The release scripts call this helper only when
MHCFLURRY_RELEASE_SMOKE=1.
"""

import argparse
from pathlib import Path

import numpy
import pandas
import yaml


SMOKE_PRETRAIN_STEPS_PER_EPOCH = 16


def cap_hyperparameters(items, max_architectures, max_epochs):
    """Keep the first architectures and cap every training and pretraining epoch count."""
    if not isinstance(items, list) or not items:
        raise ValueError("Expected a non-empty list of hyperparameter dicts")
    if max_architectures < 1 or max_epochs < 1:
        raise ValueError("max_architectures and max_epochs must be positive")
    capped = []
    for item in items[:max_architectures]:
        item = dict(item)
        item["max_epochs"] = min(int(item.get("max_epochs", max_epochs)), max_epochs)
        train_data = item.get("train_data")
        if isinstance(train_data, dict):
            train_data = dict(train_data)
            if "pretrain_max_epochs" in train_data:
                train_data["pretrain_max_epochs"] = min(int(train_data["pretrain_max_epochs"]), max_epochs)
            if "pretrain_min_epochs" in train_data:
                train_data["pretrain_min_epochs"] = min(
                    int(train_data["pretrain_min_epochs"]),
                    int(train_data.get("pretrain_max_epochs", max_epochs)))
            if "pretrain_steps_per_epoch" in train_data:
                train_data["pretrain_steps_per_epoch"] = min(
                    int(train_data["pretrain_steps_per_epoch"]), SMOKE_PRETRAIN_STEPS_PER_EPOCH)
            item["train_data"] = train_data
        capped.append(item)
    return capped


def smoke_sample_exclusions(hit_samples, holdout_samples, keep, seed, min_hits=200):
    """Exclude the holdout plus every training sample except a seeded subset.

    Candidates need at least min_hits hits so matched decoy generation and the
    processing fold assignment have enough rows. Returns (excluded, kept).
    """
    counts = pandas.Series(hit_samples).astype(str).value_counts()
    holdout = set(pandas.Series(holdout_samples).astype(str))
    candidates = sorted(set(counts.index[counts >= min_hits]) - holdout)
    if len(candidates) < keep:
        raise ValueError("Only %d training samples have at least %d hits; need %d"
                         % (len(candidates), min_hits, keep))
    kept = sorted(numpy.random.default_rng(seed).choice(candidates, size=keep, replace=False))
    excluded = sorted(holdout | (set(counts.index) - set(kept)))
    return excluded, kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("cap-hyperparameters", help="Truncate and cap a hyperparameter YAML in place.")
    cap.add_argument("path")
    cap.add_argument("--max-architectures", type=int, required=True)
    cap.add_argument("--max-epochs", type=int, required=True)
    exclude = sub.add_parser("sample-exclusions", help="Write a smoke processing sample exclusion CSV.")
    exclude.add_argument("--hits", required=True)
    exclude.add_argument("--holdout", required=True)
    exclude.add_argument("--keep", type=int, required=True)
    exclude.add_argument("--seed", type=int, required=True)
    exclude.add_argument("--min-hits", type=int, default=200)
    exclude.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "cap-hyperparameters":
        path = Path(args.path)
        items = cap_hyperparameters(yaml.safe_load(path.read_text()), args.max_architectures, args.max_epochs)
        path.write_text(yaml.safe_dump(items, sort_keys=True))
        print("smoke: %s capped to %d architecture(s), max_epochs %d" % (path, len(items), args.max_epochs))
    else:
        hits = pandas.read_csv(args.hits, usecols=["sample_id"])
        holdout = pandas.read_csv(args.holdout, usecols=["sample_id"])
        excluded, kept = smoke_sample_exclusions(
            hits.sample_id, holdout.sample_id, args.keep, args.seed, args.min_hits)
        pandas.DataFrame({"sample_id": excluded}).to_csv(args.out, index=False)
        print("smoke: keeping %d processing samples: %s" % (len(kept), " ".join(kept)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

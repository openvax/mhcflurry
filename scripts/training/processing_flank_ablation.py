#!/usr/bin/env python3
"""Diagnose external-flank reliance of fixed processing ensembles on risk sets."""

import argparse
import json
from pathlib import Path

import numpy
import pandas

from mhcflurry.cli.processing_affinity_control import (
    IDENTITY_COLUMNS, score_risk_sets, summarize_metrics)
from mhcflurry.common import configure_pytorch
from mhcflurry.experiment_archive import sha256_file
from mhcflurry.processing_matching import MATCHING_POLICY, validate_matching_assignments


def perturbation_table(frame, seed):
    """Keep one deterministic input per source row, including reused decoys."""
    required = ["source_row", *IDENTITY_COLUMNS]
    if frame.empty or frame[required].drop(columns=["n_flank", "c_flank"]).isna().any().any():
        raise ValueError("Require nonempty, identified source rows")
    normalized = frame[required].copy()
    for column in ("n_flank", "c_flank"):
        normalized[column] = normalized[column].fillna("")
    unique = normalized.drop_duplicates()
    if unique.source_row.duplicated().any():
        raise ValueError("A source row maps to inconsistent peptide/label/flanks")
    unique = unique.sort_values("source_row").reset_index(drop=True)
    unique["peptide_len"] = unique.peptide.str.len()
    rng = numpy.random.default_rng(seed)
    donor = numpy.arange(len(unique))
    for indices in unique.groupby(["sample_id", "peptide_len"], sort=True).indices.values():
        donor[indices] = rng.permutation(indices)
    unique["donor_source_row"] = unique.source_row.to_numpy()[donor]
    for column in ("n_flank", "c_flank"):
        unique["shuffled_" + column] = unique[column].to_numpy()[donor]
    return unique


def flank_audit(frame, radius=5):
    """Count usable local context; stored X padding is not observed sequence."""
    records = []
    for (sample, hit), group in frame.groupby(["sample_id", "hit"], sort=True):
        record = {"sample_id": sample, "hit": hit, "n": len(group)}
        for side in ("n", "c"):
            sequence = group[side + "_flank"].fillna("")
            context = sequence.str[-radius:] if side == "n" else sequence.str[:radius]
            complete = context.str.len().eq(radius) & ~context.str.contains("[^ACDEFGHIKLMNPQRSTVWY]")
            record[side + "_complete_fraction"] = complete.mean()
            record[side + "_empty_fraction"] = sequence.eq("").mean()
        records.append(record)
    return pandas.DataFrame(records)


def predictor_files(directory):
    """Fingerprint only the manifest and the primary weights actually loaded."""
    directory = Path(directory)
    manifest = directory / "manifest.csv"
    names = pandas.read_csv(manifest, usecols=["model_name"]).model_name
    if names.empty or names.duplicated().any():
        raise ValueError("Require nonempty unique model names")
    paths = [manifest]
    for name in names:
        if Path(name).name != name:
            raise ValueError("Invalid model name")
        paths.append(directory / ("weights_" + name + ".npz"))
    return {p.name: sha256_file(p) for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Saved processing-affinity-control matched_predictions table.")
    parser.add_argument("--predictor", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("cpu", "mps", "gpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("Threads and batch size must be positive")
    if args.out.exists():
        raise ValueError("Use a fresh output directory to preserve prior experiments")
    models = {}
    for spec in args.predictor:
        name, directory = spec.split("=", 1)
        if not name or Path(name).name != name or name in models:
            raise ValueError("Require distinct nonempty predictor names")
        models[name] = Path(directory).resolve()
    provenance = {"design": "fixed-weights-flank-perturbation-v1", "seed": args.seed,
                  "input": str(args.input.resolve()), "input_sha256": sha256_file(args.input),
                  "backend": args.backend, "threads": args.threads, "batch_size": args.batch_size,
                  "predictors": {name: {"path": str(path), "files": predictor_files(path)}
                                 for name, path in models.items()},
                  "training": False, "release_accepted": False}
    frame = pandas.read_csv(args.input)
    counts = frame.groupby(["sample_id", "risk_set_id"]).size() - 1
    validate_matching_assignments(frame, "mhcflurry_production_affinity", counts.iloc[0])
    provenance["processing_matching_policy"] = MATCHING_POLICY
    unique = perturbation_table(frame, args.seed)
    args.out.mkdir(parents=True)
    (args.out / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    unique.to_csv(args.out / "perturbed_inputs.csv.bz2", index=False)
    flank_audit(unique).to_csv(args.out / "flank_audit.csv", index=False)
    configure_pytorch(backend=args.backend, num_threads=args.threads)
    from mhcflurry import Class1ProcessingPredictor
    score_columns = []
    for name, path in models.items():
        predictor = Class1ProcessingPredictor.load(str(path))
        for mode in ("real", "masked", "shuffled"):
            print("Scoring", name, mode, len(unique), "unique rows", flush=True)
            if mode == "masked":
                n_flanks = c_flanks = [""] * len(unique)
            else:
                prefix = "shuffled_" if mode == "shuffled" else ""
                n_flanks = unique[prefix + "n_flank"].tolist()
                c_flanks = unique[prefix + "c_flank"].tolist()
            values = predictor.predict(unique.peptide.tolist(), n_flanks, c_flanks,
                                       batch_size=args.batch_size)
            if not numpy.isfinite(values).all():
                raise ValueError("Nonfinite predictions")
            score = name + "__" + mode
            mapped = pandas.Series(values, index=unique.source_row)
            frame[score] = frame.source_row.map(mapped)
            score_columns.append(score)
            # Persist each inference result before starting the next condition.
            pandas.DataFrame({"source_row": unique.source_row, score: values}).to_csv(
                args.out / (score + ".csv.bz2"), index=False)
        del predictor
    metrics = score_risk_sets(frame, score_columns)
    summary, comparisons = summarize_metrics(metrics, score_columns[0])
    frame.to_csv(args.out / "matched_predictions.csv.bz2", index=False)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    comparisons.to_csv(args.out / "comparisons.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    (args.out / "completed.json").write_text(json.dumps({"complete": True}) + "\n")


if __name__ == "__main__":
    main()

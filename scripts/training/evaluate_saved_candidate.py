#!/usr/bin/env python3
"""Finish held-out evaluation of saved ensembles, without any network training."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

from compose_processing_ensemble import fingerprint_directory, sha256_file
from run_exact_public_processing import (
    CONDITIONS, Driver, fit_combiners, utc_now, verify_public_inputs, write_json)


def model_inventory(directory):
    """Require primary weights for every manifest model, not merely a directory."""
    directory = Path(directory)
    with (directory / "manifest.csv").open() as fd:
        rows = list(csv.DictReader(fd))
    if not rows:
        raise ValueError("Empty predictor: %s" % directory)
    for row in rows:
        name = row["model_name"]
        if Path(name).name != name or not (directory / ("weights_" + name + ".npz")).is_file():
            raise ValueError("Missing or invalid model weights: %s / %s" % (directory, name))
    return {"path": str(directory), "models": len(rows),
            "fingerprint": fingerprint_directory(directory)}


def copy_predictor(source, destination):
    """Make a portable evaluation copy; resume only byte-identical directories."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if fingerprint_directory(source) != fingerprint_directory(destination):
            raise ValueError("Changed or incomplete predictor copy: %s" % destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".copying")
    # A failed copy may be resumed, but never expose it as a complete predictor.
    shutil.copytree(source, staging, dirs_exist_ok=True)
    if fingerprint_directory(source) != fingerprint_directory(staging):
        raise ValueError("Predictor copy verification failed: %s" % staging)
    staging.rename(destination)


def comparison_command(args, candidate, out, components, extra=()):
    """Use the frozen release policy and always save prediction-level outputs."""
    return ["mhcflurry", "eval", "compare-models", "--a", str(candidate),
            "--a-label", candidate.name, "--b", "public:2.2.0",
            "--b-label", "selected-public-2.1.x", "--include", components,
            "--data-dir", str(args.public_root / "data_evaluation"),
            "--release-holdout-dir", str(args.release_holdout_dir),
            "--affinity-training-overlap-policy", "audit",
            "--out", str(out), "--backend", args.backend,
            "--gpus", str(args.gpus), "--num-jobs", "1",
            "--max-workers-per-gpu", "1", "--torch-compile", "0",
            "--matmul-precision", "highest", *map(str, extra)]


def evaluate(driver, args, name, candidate, components, extra=()):
    comparison = args.out / "comparisons" / name
    driver.run(name + "-evaluate", comparison_command(
        args, candidate, comparison, components, extra))
    driver.run(name + "-plots", ["mhcflurry", "eval", "plot-comparison",
        "--input", comparison, "--components", components,
        "--summary-pdf", comparison / "plots/model_comparison_figures.pdf"])


def finish_exact_replay(args, driver):
    """Assemble saved four-per-family fits in a new evaluation-only directory."""
    source = args.exact_processing_run
    processing_root = args.public_root / "models_class1_processing"
    processing_data = processing_root / "models.selected.short_flanks/train_data.csv.bz2"
    presentation_data = args.public_root / "models_class1_presentation/train_data.csv.bz2"
    verify_public_inputs(processing_data, presentation_data, args.release_holdout_dir)
    out = args.out / "exact-public-replay"
    out.mkdir(exist_ok=True)
    processors = {"public_refit": processing_root / "models.selected.short_flanks"}
    for name in CONDITIONS:
        for stage in ("train", "select"):
            marker = source / "stages" / (name + "-" + stage + ".json")
            if json.loads(marker.read_text()).get("exit_code") != 0:
                raise ValueError("Incomplete saved stage: %s" % marker)
        identity = json.loads((source / name / "training_identity.json").read_text())
        if not identity["same_ordered_rows"]:
            raise ValueError("Exact-public training identity failed: " + name)
        original = source / name / "processing/models.selected.short_flanks"
        if model_inventory(original)["models"] != 4:
            raise ValueError("Require four saved networks per exact-data condition")
        selected = out / name / "processing/models.selected.short_flanks"
        copy_predictor(original, selected)
        processors[name] = selected
    hybrid = out / "hybrid/processing/models.selected.short_flanks"
    driver.run("exact-compose-hybrid", ["mhcflurry", "train", "compose-processing-ensemble",
        "--predictor", "legacy=" + str(processors[CONDITIONS[0]]),
        "--predictor", "boundary=" + str(processors[CONDITIONS[1]]),
        "--require-equal-counts", "--out", hybrid])
    processors["hybrid"] = hybrid
    fit_combiners(out, presentation_data,
                  args.public_root / "models_class1_pan/models.combined",
                  processing_root / "models.selected.no_flank", processors)
    for name in processors:
        evaluate(driver, args, "exact-" + name, out / name, "processing,presentation", [
            "--a-processing-dir", processing_root if name == "public_refit" else out / name / "processing",
            "--processing-modes", "short_flanks", "--presentation-score-kinds", "presentation_score"])
    write_json(out / "completed.json", {"at": utc_now(), "networks_trained_here": 0,
        "scope": "eight-network exact-data replay, not full architecture selection"})


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "out", "public-root", "release-holdout-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exact-processing-run", type=Path)
    parser.add_argument("--source-commit", required=True, help="Evaluator commit, not the training commit.")
    parser.add_argument("--backend", choices=("gpu", "cpu", "mps", "auto"), default="gpu")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--phase", choices=("all", "full", "exact"), default="all")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.phase == "exact" and args.exact_processing_run is None:
        raise ValueError("--phase exact requires --exact-processing-run")
    for name in ("candidate", "out", "public_root", "release_holdout_dir", "exact_processing_run"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    for source in (args.candidate, args.public_root, args.release_holdout_dir, args.exact_processing_run):
        if source is not None and (args.out == source or source in args.out.parents or args.out in source.parents):
            raise ValueError("Evaluation output must be separate from all source directories")
    bundle = args.candidate / "presentation/models"
    sources = {
        "affinity": bundle / "affinity_predictor",
        "hybrid_processing": bundle / "processing_predictor_with_flanks",
        "no_flank_processing": bundle / "processing_predictor_without_flanks",
        "legacy_5aa": args.candidate / "processing/models.selected.short_flanks",
        "legacy_15aa": args.candidate / "processing/models.selected.with_flanks",
    }
    inventory = {key: model_inventory(path) for key, path in sources.items()}
    public_models = {name: fingerprint_directory(args.public_root / relative) for name, relative in (
        ("affinity", "models_class1_pan/models.combined"),
        ("affinity_train_excluded", "models_class1_pan_variants/models.no_additional_ms"),
        ("processing", "models_class1_processing"),
        ("presentation", "models_class1_presentation/models"))}
    if any(value["files"] == 0 for value in public_models.values()):
        raise ValueError("Missing required public model bundle")
    if args.exact_processing_run:
        for name in CONDITIONS:
            inventory["exact-" + name] = model_inventory(
                args.exact_processing_run / name / "processing/models.selected.short_flanks")
    provenance = {"evaluator_commit": args.source_commit, "candidate_run": str(args.candidate),
        "training_provenance": json.loads((args.candidate / "provenance.json").read_text()),
        "models": inventory, "presentation_bundle": fingerprint_directory(bundle),
        "public_root": str(args.public_root), "public_models": public_models,
        "affinity_training_sha256": sha256_file(args.candidate / "affinity/models.combined/train_data.csv.bz2"),
        "holdout": fingerprint_directory(args.release_holdout_dir),
        "networks_trained_here": 0, "release_accepted": False, "phase": args.phase,
        "exact_processing_run": str(args.exact_processing_run) if args.exact_processing_run else None}
    args.out.mkdir(parents=True, exist_ok=True)
    document = args.out / "evaluation.json"
    if document.exists() and json.loads(document.read_text()) != provenance:
        raise ValueError("Refusing changed evaluation inputs or evaluator source")
    write_json(document, provenance)
    shutil.copytree(args.release_holdout_dir, args.out / "release_holdout", dirs_exist_ok=True)
    if args.prepare_only:
        print(json.dumps(provenance, indent=2))
        return 0
    os.environ.update({"MHCFLURRY_TORCH_COMPILE": "0", "MHCFLURRY_TORCH_COMPILE_LOSS": "0",
                       "MHCFLURRY_MATMUL_PRECISION": "highest"})
    driver = Driver(args.out)
    if args.phase == "exact":
        finish_exact_replay(args, driver)
        write_json(args.out / "completed.json", {"at": utc_now(), "release_accepted": False})
        return 0
    candidate = args.out / "candidate"
    copy_predictor(bundle, candidate / "presentation/models")
    # Use the bundle's actual components for both standalone and joint evaluation.
    for source, destination in (
            (sources["affinity"], candidate / "affinity/models.combined"),
            (sources["hybrid_processing"], candidate / "processing/models.selected.short_flanks"),
            (sources["no_flank_processing"], candidate / "processing/models.selected.no_flank"),
            (sources["legacy_15aa"], candidate / "processing/models.selected.with_flanks")):
        copy_predictor(source, destination)
    affinity_data = args.candidate / "affinity/models.combined/train_data.csv.bz2"
    target_data = candidate / "affinity/models.combined/train_data.csv.bz2"
    if not target_data.exists():
        shutil.copy2(affinity_data, target_data)
    if sha256_file(affinity_data) != sha256_file(target_data):
        raise ValueError("Candidate affinity training audit table differs")
    evaluate(driver, args, "full-presentation", candidate, "presentation")
    evaluate(driver, args, "full-processing", candidate, "processing")
    evaluate(driver, args, "full-legacy-processing", candidate, "processing", [
        "--a-processing-dir", args.candidate / "processing", "--processing-modes", "short_flanks"])
    baseline = args.public_root / "models_class1_pan_variants/models.no_additional_ms"
    fair = comparison_command(args, candidate, args.out / "comparisons/full-affinity-train-excluded", "affinity")
    for flag, value in (("--b", str(baseline)), ("--b-label", "public-no-additional-MS"),
                        ("--affinity-training-overlap-policy", "exclude")):
        fair[fair.index(flag) + 1] = value
    fair += ["--b-affinity-dir", str(baseline), "--affinity-source", "no_additional_ms"]
    driver.run("full-affinity-train-excluded-evaluate", fair)
    driver.run("full-affinity-train-excluded-plots", ["mhcflurry", "eval", "plot-comparison",
        "--input", args.out / "comparisons/full-affinity-train-excluded", "--components", "affinity",
        "--summary-pdf", args.out / "comparisons/full-affinity-train-excluded/plots/model_comparison_figures.pdf"])
    evaluate(driver, args, "full-affinity-descriptive", candidate, "affinity")
    write_json(args.out / "full_candidate_completed.json", {"at": utc_now(), "release_accepted": False})
    if args.exact_processing_run and args.phase == "all":
        finish_exact_replay(args, driver)
    write_json(args.out / "completed.json", {"at": utc_now(), "release_accepted": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

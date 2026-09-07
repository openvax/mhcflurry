#!/usr/bin/env python3
"""Replay the frozen processing/presentation candidate on exact public data."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy
import pandas
import yaml

from audit_training_data_identity import compare_rows
from compose_processing_ensemble import fingerprint_directory, sha256_file
from generate_processing_cleavage_boundaries import build_conditions


PUBLIC_PROCESSING_SHA256 = "f875773ec6ca75824e831a677301545f70519bf0f7f8f4a058f64eee90cdc3af"
PUBLIC_PRESENTATION_SHA256 = "5b1f8fec4d8a1756e3fa49dd048f27dec20ed9fffa94089778adb4d6c1e5abb1"
CONDITIONS = ("large_relu__legacy_5aa", "large_relu__extended_5x5")
IDENTITY_COLUMNS = ["peptide", "n_flank", "c_flank", "sample_id", "hit", "hla"]


def write_json(path, value):
    """Atomically publish a metadata document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_public_inputs(processing, presentation, holdout):
    """Reject changed public tables or training/evaluation sample overlap."""
    from mhcflurry.release_holdout import load_excluded_samples
    from mhcflurry.training_folds import extract_training_folds

    files = {}
    for name, path, expected in (
            ("processing", processing, PUBLIC_PROCESSING_SHA256),
            ("presentation", presentation, PUBLIC_PRESENTATION_SHA256)):
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError("Not the archived public %s training file: %s" % (name, path))
        data = pandas.read_csv(path, dtype={"sample_id": str})
        excluded = set(load_excluded_samples(str(holdout / (name + "_samples.csv"))))
        overlap = sorted(set(data.sample_id) & excluded)
        if overlap:
            raise ValueError("%s training overlaps release holdout: %s" % (name, overlap))
        if name == "processing":
            extract_training_folds(data, 4, reuse=True)
        files[name] = {"path": str(path), "sha256": digest, "rows": len(data)}
    return files


class Driver:
    """Log resumable command stages without hiding a failing subprocess."""

    def __init__(self, out):
        self.out = out

    def run(self, stage, command):
        command = list(map(str, command))
        marker = self.out / "stages" / (stage + ".json")
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["command"] != command:
                raise ValueError("Refusing changed command for completed stage: " + stage)
            return
        event = {"stage": stage, "command": command, "started_at": utc_now()}
        write_json(self.out / "commands" / (stage + ".json"), event)
        log_path = self.out / "logs" / (stage + ".log")
        log_path.parent.mkdir(exist_ok=True)
        print("Starting", stage, command, flush=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            status = process.wait()
        event.update(exit_code=status, finished_at=utc_now())
        write_json(self.out / "commands" / (stage + ".json"), event)
        if status:
            raise subprocess.CalledProcessError(status, command)
        write_json(marker, event)


def cached_scores(out, name, data, input_hash, model_dir, calculate):
    """Bind every cached vector to both its row ordering and model files."""
    cache = out / "component_scores"
    cache.mkdir(exist_ok=True)
    path = cache / (name + ".npy")
    metadata_path = cache / (name + ".json")
    identity = {"input_sha256": input_hash, "rows": len(data),
                "ordering": "original archived presentation table order",
                "predictor": fingerprint_directory(model_dir)}
    if metadata_path.exists() and path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity") == identity and metadata.get("sha256") == sha256_file(path):
            values = numpy.load(path, allow_pickle=False)
            if values.shape == (len(data),) and numpy.isfinite(values).all():
                return values
    values = numpy.asarray(calculate(), dtype="float64")
    if values.shape != (len(data),) or not numpy.isfinite(values).all():
        raise ValueError("Invalid component scores: " + name)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as fd:
        numpy.save(fd, values, allow_pickle=False)
    temporary.replace(path)
    write_json(metadata_path, {"identity": identity, "sha256": sha256_file(path)})
    return values


def fit_combiners(out, training_data, affinity_dir, no_flank_dir, processing_dirs):
    """Refit identical combiners, reusing the shared public component scores."""
    from mhcflurry import (
        Class1AffinityPredictor, Class1PresentationPredictor, Class1ProcessingPredictor)

    data = pandas.read_csv(training_data, dtype={"sample_id": str}).fillna("")
    if not data.peptide.str.len().between(8, 15).all():
        raise ValueError("Exact-data replay cannot silently filter presentation rows")
    data.to_csv(out / "presentation_training_rows.csv.bz2", index=False)
    input_hash = sha256_file(training_data)
    affinity = Class1AffinityPredictor.load(str(affinity_dir), optimization_level=0)
    no_flank = Class1ProcessingPredictor.load(str(no_flank_dir))
    stack = Class1PresentationPredictor(
        affinity_predictor=affinity, processing_predictor_without_flanks=no_flank)
    affinities = cached_scores(out, "public_affinity", data, input_hash, affinity_dir,
        lambda: stack.predict_affinity(
            peptides=data.peptide.values,
            alleles={hla: hla.split() for hla in data.hla.unique()},
            sample_names=data.hla.values, include_affinity_percentile=False).affinity.values)
    without = cached_scores(out, "public_no_flank", data, input_hash, no_flank_dir,
        lambda: stack.predict_processing(data.peptide.values))
    for name, model_dir in processing_dirs.items():
        processor = Class1ProcessingPredictor.load(str(model_dir))
        stack.processing_predictor_with_flanks = processor
        scores = cached_scores(out, name, data, input_hash, model_dir,
            lambda: stack.predict_processing(
                data.peptide.values, data.n_flank.values, data.c_flank.values))
        stack.fit_from_scores(targets=data.hit.values, affinities=affinities,
            processing_scores_by_model={"with_flanks": scores, "without_flanks": without},
            verbose=1)
        models_dir = out / name / "presentation/models"
        models_dir.parent.mkdir(parents=True, exist_ok=True)
        stack.save(str(models_dir), write_percent_ranks=False)
        joined = data.copy()
        joined["affinity"] = affinities
        joined["processing_with_flanks"] = scores
        joined["processing_without_flanks"] = without
        joined.to_csv(out / name / "presentation/training_component_scores.csv.bz2", index=False)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True,
                        help="Public download root containing the original model bundles.")
    parser.add_argument("--release-holdout-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    args.out = args.out.resolve()
    args.public_root = args.public_root.resolve()
    args.release_holdout_dir = args.release_holdout_dir.resolve()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    processing_root = args.public_root / "models_class1_processing"
    processing_data = processing_root / "models.selected.short_flanks/train_data.csv.bz2"
    presentation_data = args.public_root / "models_class1_presentation/train_data.csv.bz2"
    files = verify_public_inputs(processing_data, presentation_data, args.release_holdout_dir)
    config = {"design": "exact-public-processing", "inputs": files,
              "source_commit": args.source_commit, "random_seed": args.random_seed,
              "conditions": CONDITIONS, "networks": 8, "reuse_original_folds": True,
              "primary_contrast": "public affinity + fixed legacy/boundary processing vs public",
              "extra_data_training_authorized_by_driver": False,
              "holdout_sha256": {p.name: sha256_file(p) for p in
                  sorted(args.release_holdout_dir.glob("*")) if p.is_file()}}
    config = json.loads(json.dumps(config))
    if (out / "experiment.json").exists():
        if json.loads((out / "experiment.json").read_text()) != config:
            raise ValueError("Refusing to reuse output directory with different experiment inputs")
    write_json(out / "experiment.json", config)
    for name, path in (("processing", processing_data), ("presentation", presentation_data)):
        destination = out / "inputs" / (name + ".csv.bz2")
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
    shutil.copytree(args.release_holdout_dir, out / "release_holdout", dirs_exist_ok=True)
    for name, grid, _axes in build_conditions(
            architectures=["large_relu"], peptide_context_lengths=[5]):
        if name in CONDITIONS:
            (out / "inputs" / (name + ".yaml")).write_text(yaml.safe_dump(grid, sort_keys=True))
    if args.prepare_only:
        print(json.dumps(config, indent=2))
        return 0
    os.environ.update({"MHCFLURRY_TORCH_COMPILE": "0", "MHCFLURRY_TORCH_COMPILE_LOSS": "0",
                       "MHCFLURRY_MATMUL_PRECISION": "highest"})
    driver = Driver(out)
    parallel = ["--gpus", args.gpus, "--num-jobs", args.num_jobs,
                "--max-workers-per-gpu", "1", "--torch-compile", "0",
                "--matmul-precision", "highest"]
    training_parallel = [*parallel, "--dataloader-num-workers", "1"]
    for name in CONDITIONS:
        processing = out / name / "processing"
        processing.mkdir(parents=True, exist_ok=True)
        unselected = processing / "models.unselected.short_flanks"
        selected = processing / "models.selected.short_flanks"
        # Always use the same resume command: initialize only before its first invocation.
        if not (unselected / "training_init_info.pkl").exists():
            driver.run(name + "-initialize", ["mhcflurry", "class1-train-processing-models",
                "--data", processing_data, "--reuse-folds", "--num-folds", "4",
                "--random-seed", args.random_seed, "--hyperparameters", out / "inputs" / (name + ".yaml"),
                "--out-models-dir", unselected, "--only-initialize", *training_parallel])
        driver.run(name + "-train", ["mhcflurry", "class1-train-processing-models",
            "--out-models-dir", unselected, "--continue-incomplete", *training_parallel])
        columns = IDENTITY_COLUMNS + ["fold_%d" % i for i in range(4)]
        reference = pandas.read_csv(processing_data, dtype=str, keep_default_na=False)
        replay = pandas.read_csv(unselected / "train_data.csv.bz2", dtype=str, keep_default_na=False)
        identity, _ = compare_rows(reference, replay, columns)
        write_json(out / name / "training_identity.json", identity)
        if not identity["same_ordered_rows"]:
            raise ValueError("Training command changed public rows/folds")
        driver.run(name + "-select", ["mhcflurry", "class1-select-processing-models",
            "--data", unselected / "train_data.csv.bz2", "--models-dir", unselected,
            "--out-models-dir", selected, "--min-models-per-fold", "1",
            "--max-models-per-fold", "1", "--save-validation-predictions", *training_parallel])
        driver.run(name + "-loss-plots", ["mhcflurry", "train", "plot-loss-curves",
            "--selected-dir", selected, "--unselected-dir", unselected,
            "--out", out / name / "loss_plots"])
    hybrid = out / "hybrid/processing/models.selected.short_flanks"
    driver.run("compose-hybrid", ["mhcflurry", "train", "compose-processing-ensemble",
        "--predictor", "legacy=" + str(out / CONDITIONS[0] / "processing/models.selected.short_flanks"),
        "--predictor", "boundary=" + str(out / CONDITIONS[1] / "processing/models.selected.short_flanks"),
        "--require-equal-counts", "--out", hybrid])
    processors = {"public_refit": processing_root / "models.selected.short_flanks",
                  **{name: out / name / "processing/models.selected.short_flanks" for name in CONDITIONS},
                  "hybrid": hybrid}
    fit_combiners(out, presentation_data,
                  args.public_root / "models_class1_pan/models.combined",
                  processing_root / "models.selected.no_flank", processors)
    for name in processors:
        comparison = out / "comparisons" / (name + "-vs-public")
        driver.run(name + "-evaluate", ["mhcflurry", "eval", "compare-models",
            "--a", out / name, "--a-label", name,
            "--a-presentation-dir", out / name / "presentation/models",
            "--a-processing-dir", processing_root if name == "public_refit" else out / name / "processing",
            "--b", "public:2.2.0", "--b-label", "selected-public-2.1.x",
            "--data-dir", args.public_root / "data_evaluation",
            "--release-holdout-dir", args.release_holdout_dir,
            "--include", "processing,presentation", "--processing-modes", "short_flanks",
            "--presentation-score-kinds", "presentation_score",
            "--presentation-modes", "with_flanks,without_flanks", "--out", comparison, *parallel])
        driver.run(name + "-plots", ["mhcflurry", "eval", "plot-comparison",
            "--input", comparison, "--components", "processing,presentation",
            "--summary-pdf", comparison / "plots/model_comparison_figures.pdf"])
    write_json(out / "completed.json", {"at": utc_now(), "extra_data_gate": "requires review of held-out results"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

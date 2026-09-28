"""Single-A100 runplz/Modal launcher for the matched processing width sweep.

Launch from a frozen source extraction using ``runplz modal --detach``.
Use ``runplz status/collect --outputs-dir ...`` with the resulting receipt.
"""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from runplz import App, Image


RUN_ID = os.environ["PROCESSING_KERNEL_RUN_ID"]
SOURCE_COMMIT = os.environ["PROCESSING_KERNEL_SOURCE_COMMIT"]
SOURCE_SHA256 = os.environ["PROCESSING_KERNEL_SOURCE_SHA256"]
RESUME_MATCHING_DIR = os.environ.get("PROCESSING_KERNEL_RESUME_MATCHING_DIR", "")
PRIOR_SWEEP = os.environ.get("PROCESSING_KERNEL_PRIOR_SWEEP", "")
TRAIN_DATA = os.environ.get("PROCESSING_KERNEL_TRAIN_DATA", "")
EVALUATION = os.environ.get("PROCESSING_KERNEL_EVALUATION", "none")
RECIPE_AFTER_WIDTHS = os.environ.get("PROCESSING_RECIPE_AFTER_WIDTHS", "0") == "1"
RANKING_CONFIRMATION = os.environ.get("PROCESSING_RANKING_CONFIRMATION", "0") == "1"
TIMEOUT = int(os.environ.get("PROCESSING_KERNEL_TIMEOUT_SECONDS", str(8 * 60 * 60)))
DEADLINE = os.environ.get("MHCFLURRY_EXPERIMENT_DEADLINE_EPOCH", "")
if not 1 <= TIMEOUT <= int(15.5 * 60 * 60):
    raise ValueError("Processing campaign timeout exceeds the approved first allocation")
if RECIPE_AFTER_WIDTHS and not DEADLINE:
    raise ValueError("A combined campaign requires an absolute budget deadline")
if RANKING_CONFIRMATION and (not DEADLINE or not TRAIN_DATA or RECIPE_AFTER_WIDTHS or TIMEOUT > 4 * 60 * 60):
    raise ValueError("Ranking confirmation requires frozen data, an absolute deadline and at most four GPU-hours")
if EVALUATION not in ("all", "none"):
    raise ValueError("PROCESSING_KERNEL_EVALUATION must be all or none")
VOLUME = "mhcflurry-230-final-weights"
PRIOR_VOLUME = "mhcflurry-processing-cleavage-f50c70f8c-20260904"
app = App("mhcflurry-" + RUN_ID)
app.repo_root = Path(__file__).resolve().parents[2]
if not (app.repo_root / "setup.py").is_file():
    raise ValueError("Launch from a complete frozen mhcflurry source extraction")
image = (Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
         .apt_install("python-is-python3", "bzip2", "build-essential", "git")
         .pip_install("runplz==4.4.2", "pyarrow", "pypdf", "reportlab")
         .pip_install_local_dir(".", editable=True))


@app.function(
    image=image, gpu="A100-40GB", num_gpus=1, min_cpu=8, min_memory=64,
    timeout=TIMEOUT, volumes={"/out": VOLUME, "/prior": PRIOR_VOLUME},
    env={"PROCESSING_KERNEL_RUN_ID": RUN_ID,
         "PROCESSING_KERNEL_SOURCE_COMMIT": SOURCE_COMMIT,
         "PROCESSING_KERNEL_SOURCE_SHA256": SOURCE_SHA256,
         "PROCESSING_KERNEL_RESUME_MATCHING_DIR": RESUME_MATCHING_DIR,
         "PROCESSING_KERNEL_PRIOR_SWEEP": PRIOR_SWEEP,
         "PROCESSING_KERNEL_TRAIN_DATA": TRAIN_DATA,
         "PROCESSING_KERNEL_EVALUATION": EVALUATION,
         "PROCESSING_KERNEL_TIMEOUT_SECONDS": str(TIMEOUT),
         "PROCESSING_RECIPE_AFTER_WIDTHS": "1" if RECIPE_AFTER_WIDTHS else "0",
         "PROCESSING_RANKING_CONFIRMATION": "1" if RANKING_CONFIRMATION else "0",
         "MHCFLURRY_EXPERIMENT_DEADLINE_EPOCH": DEADLINE,
         "MHCFLURRY_DATA_DIR": "/out/downloads",
         "MHCFLURRY_DOWNLOADS_CURRENT_RELEASE": "2.2.0",
         "PYTHONUNBUFFERED": "1", "MHCFLURRY_TORCH_COMPILE": "0",
         "MHCFLURRY_TORCH_COMPILE_LOSS": "0", "MHCFLURRY_MATMUL_PRECISION": "highest",
         "MHCFLURRY_ENABLE_TIMING": "1", "MHCFLURRY_FAIL_ON_TRAINING_BATCH_SHRINK": "1",
         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
def sweep():
    """Preserve preparation failures and never fall back to unmatched negatives."""
    out = Path(os.environ["RUNPLZ_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_exact_public_processing import Driver, write_json
    driver = Driver(out)
    archive = Path("/out/inputs") / RUN_ID / "source.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Source archive hash mismatch")
    shutil.copy2(archive, out / "source.tar.gz")
    write_json(out / "launch_provenance.json", {
        "source_commit": SOURCE_COMMIT, "source_archive_sha256": digest,
        "modal_volume": VOLUME, "run_id": RUN_ID, "output_path": str(out),
        "gpu": "A100-40GB", "num_gpus": 1, "timeout_seconds": TIMEOUT,
        "resume_matching_dir": RESUME_MATCHING_DIR or None,
        "prior_sweep": PRIOR_SWEEP or None, "frozen_train_data": TRAIN_DATA or None,
        "evaluation": EVALUATION, "recipe_after_widths": RECIPE_AFTER_WIDTHS,
        "ranking_confirmation": RANKING_CONFIRMATION,
        "absolute_budget_deadline_epoch": DEADLINE or None})
    with (out / "pip_freeze.txt").open("w") as fd:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=fd, check=True)
    public = Path("/out/downloads/2.2.0")
    holdout = Path("/out/inputs/release_holdout")
    shared = out / "processing.shared"
    shared.mkdir(exist_ok=True)
    with (out / "gpu_occupancy.csv").open("a") as telemetry_file:
        telemetry = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits", "--loop-ms=5000"], stdout=telemetry_file)
        try:
            resume_args = ["--resume-matching-dir", RESUME_MATCHING_DIR] if RESUME_MATCHING_DIR else []
            if TRAIN_DATA:
                source = Path(TRAIN_DATA)
                if not source.is_file():
                    raise ValueError("Missing frozen matched training data: " + TRAIN_DATA)
                train_data = source
            else:
                train_data = shared / "train_data.csv"
                driver.run("matched-training-data", ["mhcflurry", "train", "processing-data",
                    *resume_args,
                    "--hits", "/prior/processing.shared/hits_with_tpm.csv.bz2",
                    "--affinity-predictor", public / "models_class1_pan/models.combined",
                    "--proteome-reference-csv", public / "data_references/uniprot_proteins.csv.bz2",
                    "--ppv-multiplier", 100, "--negative-policy", "matched", "--decoys-per-hit", 1,
                    "--max-affinity-distance", 0.25, "--exclude-samples-file", holdout / "processing_samples.csv",
                    "--random-seed", 42, "--out", train_data, "--gpus", 1,
                    "--num-jobs", 1, "--max-workers-per-gpu", 1,
                    "--max-tasks-per-worker", 100, "--torch-compile", 0, "--matmul-precision", "highest"])
            recovery_args = ["--resume-from", PRIOR_SWEEP] if PRIOR_SWEEP else []
            if RANKING_CONFIRMATION:
                driver.run("ranking-confirmation", ["mhcflurry", "train", "processing-hyperparameter-sweep",
                    "--design", "ranking-confirmation", "--out", out / "ranking_confirmation",
                    "--train-data", train_data, "--folds-from", train_data,
                    "--public-root", public, "--release-holdout-dir", holdout,
                    "--source-commit", SOURCE_COMMIT, "--gpus", 1, "--num-jobs", 1,
                    "--evaluation", "none", "--save-all-checkpoints"])
                return
            driver.run("kernel-sweep", ["mhcflurry", "train", "processing-kernel-sweep",
                "--out", out / "kernel_sweep", "--train-data", train_data,
                "--public-root", public, "--release-holdout-dir", holdout,
                "--source-commit", SOURCE_COMMIT, "--gpus", 1, "--num-jobs", 1,
                "--evaluation", EVALUATION, "--save-all-checkpoints", *recovery_args])
            figures = out / "kernel_sweep"
            panels = [{"title": "Processing kernel-width sweep",
                       "text": "48 fits: six widths, two unmixed families, four paired folds. "
                               "Best validation checkpoints; matched negatives. Validation has one "
                               "negative per hit. Release evaluation mode: " + EVALUATION + ". "
                               "If included, release results are exploratory, not an untouched test."}]
            for path in sorted(figures.glob("*/loss_plots/*.png")):
                panels.append({"title": path.parent.parent.name + " / " + path.stem,
                               "text": "Per-condition epoch traces; full fit histories are in model manifests.",
                               "image": str(path)})
            write_json(out / "figure_panels.json", panels)
            evaluation_pdf = (["--pdf", figures / "evaluation/kernel_width.pdf"]
                              if EVALUATION == "all" else [])
            driver.run("all-figures", ["mhcflurry", "eval", "collate-figures",
                "--pdf", figures / "validation_kernel_width.pdf", *evaluation_pdf,
                "--addendum", out / "figure_panels.json", "--out", out / "all-figures.pdf"])
            if RECIPE_AFTER_WIDTHS:
                frozen = figures / "inputs/training_with_folds.csv.bz2"
                driver.run("training-recipe-sweep", ["mhcflurry", "train", "processing-hyperparameter-sweep",
                    "--design", "training-recipe", "--out", out / "recipe_sweep",
                    "--train-data", frozen, "--folds-from", frozen,
                    "--public-root", public, "--release-holdout-dir", holdout,
                    "--source-commit", SOURCE_COMMIT, "--gpus", 1, "--num-jobs", 1,
                    "--evaluation", "none", "--save-all-checkpoints"])
                panels.append({"title": "Processing optimizer / initialization / batch screen",
                               "text": "64 fits on two paired development folds. Learning rate 0.001; "
                                       "one matched negative per hit. Confirm finalists on four folds. "
                                       "No new release evaluation or final ensemble selection in this stage."})
                for path in sorted((out / "recipe_sweep").glob("*/loss_plots/*.png")):
                    panels.append({"title": path.parent.parent.name + " / " + path.stem,
                                   "text": "Epoch histories, best/terminal checkpoints and member predictions retained.",
                                   "image": str(path)})
                write_json(out / "campaign_figure_panels.json", panels)
                driver.run("campaign-all-figures", ["mhcflurry", "eval", "collate-figures",
                    "--pdf", figures / "validation_kernel_width.pdf",
                    "--pdf", out / "recipe_sweep/validation_training_recipe.pdf",
                    "--addendum", out / "campaign_figure_panels.json", "--out", out / "campaign-all-figures.pdf"])
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=10)


@app.local_entrypoint()
def main():
    sweep.remote()

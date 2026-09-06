"""Bounded runplz/Modal launcher for the exact-public-data processing replay.

Run from a clean ``git archive`` extraction, not the working tree containing
large experiment outputs. See docs/exact_public_data_experiment.md.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from runplz import App, Image
from runplz.backends import modal as runplz_modal


VOLUME = "mhcflurry-230-final-weights"
RUN_ID = os.environ["EXACT_PUBLIC_RUN_ID"]
SOURCE_COMMIT = os.environ["EXACT_PUBLIC_SOURCE_COMMIT"]
SOURCE_SHA256 = os.environ["EXACT_PUBLIC_SOURCE_SHA256"]

# runplz #165: its current Modal CLI has no detached-run switch. Keep this
# adapter explicit until the upstream backend exposes the durable-run option.
_subprocess_run = runplz_modal.subprocess.run


def _modal_detached(command, *args, **kwargs):
    command = list(command)
    if command[:2] == ["modal", "run"] and "--detach" not in command:
        command.insert(2, "--detach")
    return _subprocess_run(command, *args, **kwargs)


runplz_modal.subprocess.run = _modal_detached
app = App("mhcflurry-" + RUN_ID)
image = (Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
         .apt_install("python-is-python3", "bzip2", "build-essential", "git")
         .pip_install("runplz==4.4.2", "pyarrow", "pypdf", "reportlab")
         .pip_install_local_dir(".", editable=True))


@app.function(
    image=image, gpu="A100-40GB", num_gpus=1, min_cpu=8, min_memory=64,
    timeout=8 * 60 * 60, volumes={"/persist": VOLUME},
    env={"EXACT_PUBLIC_RUN_ID": RUN_ID, "EXACT_PUBLIC_SOURCE_COMMIT": SOURCE_COMMIT,
         "EXACT_PUBLIC_SOURCE_SHA256": SOURCE_SHA256,
         "MHCFLURRY_DATA_DIR": "/persist/downloads",
         "MHCFLURRY_DOWNLOADS_CURRENT_RELEASE": "2.2.0",
         "PYTHONUNBUFFERED": "1", "MHCFLURRY_TORCH_COMPILE": "0",
         "MHCFLURRY_TORCH_COMPILE_LOSS": "0", "MHCFLURRY_MATMUL_PRECISION": "highest",
         "MHCFLURRY_ENABLE_TIMING": "1", "MHCFLURRY_FAIL_ON_TRAINING_BATCH_SHRINK": "1",
         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
def replay():
    """Train, evaluate and preserve results without launching any extra-data run."""
    out = Path("/persist/runs") / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    archive = Path("/persist/inputs") / RUN_ID / "source.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Exact source archive hash mismatch")
    provenance = {"source_commit": SOURCE_COMMIT, "source_archive": str(archive),
                  "source_archive_sha256": digest, "modal_volume": VOLUME,
                  "gpu": "A100-40GB", "num_gpus": 1, "timeout_seconds": 8 * 60 * 60}
    (out / "launch_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (out / "pip_freeze.txt").open("w") as fd:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=fd, check=True)
    # Unlike the prior factorial's private staging area, this replay resolves
    # the actual public baseline from its normal, release-specific bundle.
    subprocess.run(["mhcflurry-downloads", "fetch", "--release", "2.2.0",
                    "models_class1_pan", "models_class1_processing",
                    "models_class1_presentation", "data_evaluation"], check=True)
    command = ["mhcflurry", "train", "exact-public-processing", "--out", str(out),
               "--public-root", "/persist/downloads/2.2.0",
               "--release-holdout-dir", "/persist/inputs/release_holdout",
               "--source-commit", SOURCE_COMMIT, "--gpus", "1", "--num-jobs", "1"]
    with (out / "gpu_occupancy.csv").open("a") as telemetry_file:
        telemetry = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits", "--loop-ms=5000"], stdout=telemetry_file)
        try:
            subprocess.run(command, check=True)
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=10)
    Path("/out/modal-volume-receipt.json").write_text(json.dumps({
        "status": "complete", "volume": VOLUME, "path": str(out)}, indent=2) + "\n")


@app.local_entrypoint()
def main():
    replay.remote()

"""
Run MHCflurry training jobs on Modal.

This script is intentionally generic: pass any supported training command
template and run multiple workers in parallel.

Example:
    modal run scripts/modal_train_mhcflurry.py \
      --command-template "mhcflurry-class1-train-processing-models --data /artifacts/data/train.csv --models-dir /artifacts/runs/{run_name} --verbosity 1" \
      --workers 4

The command template supports:
    {run_name}  -> unique run id per worker
    {worker}    -> worker index (0-based)
"""

from __future__ import annotations

import datetime
import uuid

import modal


REPO_URL = "https://github.com/openvax/mhcflurry.git"
REPO_REF = "master"
REPO_DIR = "/workspace/mhcflurry"
ARTIFACTS_DIR = "/artifacts"
VOLUME_NAME = "mhcflurry-training"

ALLOWED_TRAINING_COMMANDS = {
    "mhcflurry-class1-train-allele-specific-models",
    "mhcflurry-class1-train-pan-allele-models",
    "mhcflurry-class1-train-processing-models",
    "mhcflurry-class1-train-presentation-models",
}


def _install_repo():
    """Build helper used at image build time."""
    # Import is intentionally local to keep image build behavior explicit.
    import subprocess

    subprocess.run(
        [
            "bash",
            "-lc",
            (
                "set -euxo pipefail; "
                f"git clone --depth 1 --branch {REPO_REF} {REPO_URL} {REPO_DIR}; "
                f"cd {REPO_DIR}; "
                "python -m pip install --upgrade pip; "
                "python -m pip install -e ."
            ),
        ],
        check=True,
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_function(_install_repo)
    .env(
        {
            # Keep torch memory allocator behavior predictable across jobs.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Allow override of default downloads location if desired.
            "MHCFLURRY_DOWNLOADS_DIR": f"{ARTIFACTS_DIR}/downloads",
        }
    )
)

app = modal.App("mhcflurry-train", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=12 * 60 * 60,
    cpu=8,
    memory=32768,
    volumes={ARTIFACTS_DIR: volume},
)
def run_training_job(job: dict) -> dict:
    import os
    import shlex
    import subprocess
    import time

    run_name = job["run_name"]
    command = job["command"]
    run_dir = os.path.join(ARTIFACTS_DIR, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    argv = shlex.split(command)
    if not argv:
        raise ValueError("Empty command")
    if argv[0] not in ALLOWED_TRAINING_COMMANDS:
        raise ValueError(
            "Unsupported command '%s'. Allowed: %s"
            % (argv[0], sorted(ALLOWED_TRAINING_COMMANDS))
        )

    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")

    start = time.time()
    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        proc = subprocess.run(
            argv,
            cwd=REPO_DIR,
            stdout=out_f,
            stderr=err_f,
            text=True,
            check=False,
        )
    elapsed = time.time() - start

    return {
        "run_name": run_name,
        "command": command,
        "exit_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


@app.local_entrypoint()
def main(command_template: str, workers: int = 1):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    jobs = []
    for worker in range(workers):
        run_name = f"{timestamp}-w{worker:03d}-{uuid.uuid4().hex[:8]}"
        command = command_template.format(run_name=run_name, worker=worker)
        jobs.append({"run_name": run_name, "command": command})

    results = list(run_training_job.map(jobs))
    results = sorted(results, key=lambda d: d["run_name"])

    for result in results:
        print(
            "%s exit=%s elapsed=%.1fs stdout=%s stderr=%s"
            % (
                result["run_name"],
                result["exit_code"],
                result["elapsed_seconds"],
                result["stdout_path"],
                result["stderr_path"],
            )
        )


"""Run saved-candidate evaluation through runplz/Modal, without training."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from runplz import App, Image
from runplz.backends import modal as runplz_modal


VOLUME = "mhcflurry-230-final-weights"
RUN_ID = os.environ["SAVED_EVAL_RUN_ID"]
SOURCE_COMMIT = os.environ["SAVED_EVAL_SOURCE_COMMIT"]
SOURCE_SHA256 = os.environ["SAVED_EVAL_SOURCE_SHA256"]
CANDIDATE_RUN = os.environ["SAVED_EVAL_CANDIDATE_RUN"]
EXACT_RUN = os.environ.get("SAVED_EVAL_EXACT_RUN", "")
for value in (RUN_ID, CANDIDATE_RUN, EXACT_RUN):
    if value and (Path(value).name != value or value in (".", "..")):
        raise ValueError("Run IDs must be single directory names")

# Explicit adapter pending runplz #165 (durable detached execution).
_subprocess_run = runplz_modal.subprocess.run


def _modal_detached(command, *args, **kwargs):
    command = list(command)
    if command[:2] == ["modal", "run"] and "--detach" not in command:
        command.insert(2, "--detach")
    return _subprocess_run(command, *args, **kwargs)


runplz_modal.subprocess.run = _modal_detached
app = App("mhcflurry-" + RUN_ID)
app.repo_root = Path(__file__).resolve().parents[2]
if not (app.repo_root / "setup.py").is_file():
    raise ValueError("Launch from a complete clean source archive")
image = (Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
         .apt_install("python-is-python3", "bzip2", "build-essential", "git")
         .pip_install("runplz==4.4.2", "pyarrow", "pypdf", "reportlab")
         .pip_install_local_dir(".", editable=True))


@app.function(
    image=image, gpu="A100-40GB", num_gpus=1, min_cpu=8, min_memory=64,
    timeout=8 * 60 * 60, volumes={"/persist": VOLUME},
    env={"SAVED_EVAL_RUN_ID": RUN_ID, "SAVED_EVAL_SOURCE_COMMIT": SOURCE_COMMIT,
         "SAVED_EVAL_SOURCE_SHA256": SOURCE_SHA256, "SAVED_EVAL_CANDIDATE_RUN": CANDIDATE_RUN,
         "SAVED_EVAL_EXACT_RUN": EXACT_RUN,
         "MHCFLURRY_DATA_DIR": "/persist/downloads",
         "MHCFLURRY_DOWNLOADS_CURRENT_RELEASE": "2.2.0",
         "PYTHONUNBUFFERED": "1", "MHCFLURRY_TORCH_COMPILE": "0",
         "MHCFLURRY_TORCH_COMPILE_LOSS": "0", "MHCFLURRY_MATMUL_PRECISION": "highest",
         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
def evaluate():
    out = Path("/persist/runs") / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    archive = Path("/persist/inputs") / RUN_ID / "source.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Evaluator source archive hash mismatch")
    provenance = {"evaluator_commit": SOURCE_COMMIT, "source_archive_sha256": digest,
                  "candidate_run": CANDIDATE_RUN, "exact_run": EXACT_RUN,
                  "gpu": "A100-40GB", "gpus": 1, "timeout_seconds": 8 * 60 * 60,
                  "networks_trained_here": 0}
    path = out / "launch_provenance.json"
    if path.exists() and json.loads(path.read_text()) != provenance:
        raise ValueError("Refusing changed launch provenance")
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    with (out / "pip_freeze.txt").open("w") as fd:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=fd, check=True)
    subprocess.run(["mhcflurry-downloads", "fetch", "--release", "2.2.0",
                    "models_class1_pan", "models_class1_pan_variants", "models_class1_processing",
                    "models_class1_presentation", "data_evaluation"], check=True)
    command = ["mhcflurry", "eval", "saved-candidate", "--out", str(out),
               "--candidate", "/persist/runs/" + CANDIDATE_RUN,
               "--public-root", "/persist/downloads/2.2.0",
               "--release-holdout-dir", "/persist/inputs/release_holdout",
               "--source-commit", SOURCE_COMMIT]
    if EXACT_RUN:
        command += ["--exact-processing-run", "/persist/runs/" + EXACT_RUN]
    (out / "driver_command.json").write_text(json.dumps(command, indent=2) + "\n")
    monitor_stop = threading.Event()

    def monitor_memory():
        # Durable cgroup counters distinguish OOM pressure from other
        # cancellations without relying on a surviving Python traceback.
        with (out / "memory_occupancy.jsonl").open("a") as fd:
            while not monitor_stop.is_set():
                record = {"unix_time": time.time()}
                for name in ("memory.current", "memory.peak", "memory.max", "memory.events"):
                    path = Path("/sys/fs/cgroup") / name
                    if path.is_file():
                        record[name] = path.read_text().strip()
                fd.write(json.dumps(record) + "\n")
                fd.flush()
                monitor_stop.wait(5)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    with (out / "gpu_occupancy.csv").open("a") as telemetry_file, (out / "driver.log").open("a") as log:
        telemetry = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits", "--loop-ms=5000"], stdout=telemetry_file)
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except BaseException as error:
            (out / "failure.json").write_text(json.dumps({
                "unix_time": time.time(), "error": repr(error),
                "returncode": getattr(error, "returncode", None),
            }, indent=2) + "\n")
            print("Saved evaluation failed; durable driver log:", out / "driver.log", flush=True)
            raise
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=10)
            monitor_stop.set()
            monitor.join(timeout=10)
    Path("/out/modal-volume-receipt.json").write_text(json.dumps({
        "status": "complete", "volume": VOLUME, "path": str(out)}, indent=2) + "\n")


@app.local_entrypoint()
def main():
    evaluate.remote()

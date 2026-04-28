"""Exact 2.2.0 pan-allele replication on a single A100-80GB.

Same recipe as jobs/pan_allele_release_exact.py — same curated data,
same 4 folds, same 35-arch sweep, same --min-models 2 / --max-models 8
selection, same percentile-rank calibration — but pinned to the
MassedCompute 1×A100-80GB shape instead of OCI 8×A100. Chosen because
all 5 8× provisioning attempts on 2026-04-21 failed, while
MassedCompute 1×A100-80GB booted reliably in the same session.

Compute: 140 networks × ~90 min each / 2 concurrent workers ≈ 105 hr
wall time at $1.49/hr ≈ $156. Slow but cheap; if 8× shapes recover in
a different hour, switch back to pan_allele_release_exact.py for a
~28 hr / $336 run.

Launch:
    runplz brev jobs/pan_allele_release_exact_single_a100.py

Resume on ssh drop: re-launch the same command. The shell script uses
``--continue-incomplete`` so partially-finished work items resume
from ``models.unselected/``.
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-release-exact-single-a100",
    brev_config=BrevConfig(
        mode="container",
        # MassedCompute 1×A100-80GB — known-good shape on 2026-04-21.
        # $1.49/hr, 14 vCPUs, ~80 GB RAM, 625 GB disk, 80 GB VRAM,
        # 2m30s typical boot.
        instance_type="massedcompute_A100_sxm4_80G",
        auto_create_instances=True,
        ssh_ready_wait_seconds=1800,
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        "runplz>=3.9.0",
        "pandas>=2.0",
        "pyarrow",
        "appdirs",
        "scikit-learn",
        "mhcgnomes>=3.0.1",
        "numpy>=1.22.4",
        "pyyaml",
        "tqdm",
    )
    .pip_install_local_dir(".", editable=True)
)


@app.function(
    image=image,
    gpu="A100",
    # MassedCompute 1×A100-80GB spec.
    min_cpu=14,
    min_memory=80,
    min_gpu_memory=80,  # force 80GB shape — 2 workers need it
    min_disk=500,
    timeout=120 * 60 * 60,  # 120h cap; wall estimate ~105h
    env={
        # 80GB card → 2 workers (each ~16-22 GB peak during validation
        # inference). 3 workers arithmetically fits (≤66 GB) but has
        # deterministic OOM risk under validation spikes — stick with
        # the empirically-safe ceiling.
        "MAX_WORKERS_PER_GPU": "2",
        # Keep prefetch modest on 14 vCPUs; production-sized fit() datasets
        # downgrade to in-process batching to avoid spawn-pickling arrays.
        "DATALOADER_NUM_WORKERS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "auto",
        "MHCFLURRY_TORCH_COMPILE_LOSS": "1",
    },
)
def train():
    env = os.environ.copy()
    env["MHCFLURRY_OUT"] = env["RUNPLZ_OUT"]
    subprocess.run(
        ["bash", "scripts/training/pan_allele_release_affinity.sh"],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    train.remote()

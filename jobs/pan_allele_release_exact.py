"""Exact replication of the public mhcflurry pan-allele release training
recipe (models_class1_pan 2.2.0): same pretraining, same curated data,
same 4 folds, same 35-architecture sweep, same selection step
(--min-models 2 --max-models 8), same percentile-rank calibration.

Launch:
    runplz brev --instance mhcflurry-release-exact jobs/pan_allele_release_exact.py

Compute: ~140 networks × ~90 min each on A100 → ~210 GPU-hours. On the
pinned 8×A100 box that's ~28 hours wall-clock, ~$336.

If the ssh session drops mid-run, re-launch runplz with the same command;
the shell script uses `--continue-incomplete` to resume from the
unselected/ checkpoint directory.
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-release-exact",
    brev_config=BrevConfig(
        mode="container",
        # MassedCompute 8×A100-80GB DGXx8, 120 vCPUs, 10 TB disk at
        # $12.29/hr. Chose over Denvr after 3/3 Denvr failures on
        # 2026-04-21 (Brev API EOF, mid-boot SSH drop, 1200s SSH
        # timeout). 80GB cards unlock MAX_WORKERS_PER_GPU=2 for 16
        # concurrent workers (vs 8 on 40GB Denvr) — halves wall time.
        # Runplz 3.6.2 brought transient-EOF retry + SIGTERM cleanup
        # so the earlier MassedCompute shadeform failures should no
        # longer be as reliable a blocker.
        instance_type="massedcompute_A100_sxm4_80G_DGXx8",
        auto_create_instances=True,
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        "runplz>=3.6.2",
        "pandas>=2.0",
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
    min_cpu=96,          # match the 8-GPU machine's vCPU budget
    min_memory=400,      # GB RAM — full data + pretraining buffer
    min_gpu_memory=40,   # GB VRAM per GPU
    min_disk=1000,       # GB — ~140 networks × ~10MB weights + init info
    timeout=60 * 60 * 60,  # 60 hours — safety margin above 28h estimate
    env={
        # massedcompute_A100_sxm4_80G_DGXx8 is 8× A100-80GB. 80GB cards
        # fit 2 workers (each ~20 GB peak VRAM during validation
        # inference), unlocking 16 concurrent training workers across
        # 8 GPUs. With Phase 2 NonDaemonPool, each worker's DataLoader
        # prefetch path is live in production too.
        "MAX_WORKERS_PER_GPU": "2",
    },
)
def train():
    # In Brev container-mode, RUNPLZ_OUT resolves to $HOME/runplz-out,
    # which is where the backend will rsync from. The training shell
    # script reads MHCFLURRY_OUT — point it at the same dir.
    env = os.environ.copy()
    env["MHCFLURRY_OUT"] = env["RUNPLZ_OUT"]
    subprocess.run(
        ["bash", "scripts/training/pan_allele_release_exact.sh"],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    train.remote()

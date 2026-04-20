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
    brev=BrevConfig(
        auto_create=True,
        mode="container",
        # 8×A100-40GB, 120 vCPUs, 14 TB disk at $12/hr. Denvr has been
        # reliable for us (MassedCompute timed out on SSH earlier). VRAM
        # at 40GB is plenty — individual networks fit comfortably.
        instance_type="denvr_A100_sxm4x8",
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        "runplz>=1.5.0",
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
        # MHCFLURRY_OUT is wired to RUNPLZ_OUT inside train() so we
        # don't hard-code a container path that conflicts with the
        # backend's chosen output dir. Everything else is default.
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

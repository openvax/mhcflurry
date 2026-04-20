"""Full presentation-predictor subset run: affinity → processing →
presentation end-to-end, but with a small per-variant architecture
sweep so the whole pipeline fits in ~$100 on 8×A100.

Launch:
    runplz brev --instance mhcflurry-presentation-subset \
        jobs/pan_allele_presentation_subset.py

Stages (with OMP_NUM_THREADS=1 now set everywhere):
  1. Affinity — 4 folds × 8 architectures × 1 replicate (32 networks)
  2. Processing — 4 folds × 8 architectures × 2 variants
     (no_flank + short_flanks, which are the ones Class1PresentationPredictor
     actually consumes; with_flanks is skipped) = 64 networks
  3. Presentation — logistic regression over (affinity, processing);
     cheap (minutes), plus percentile-rank calibration.

Pinned to denvr_A100_sxm4x8 at $12/hr; expected wall time ~4-6 hours
including boot + data downloads.
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-presentation-subset",
    brev_config=BrevConfig(
        mode="container",
        instance_type="denvr_A100_sxm4x8",
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        "runplz>=3.0.0",
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
    min_cpu=96,
    min_memory=400,
    min_gpu_memory=40,
    min_disk=1000,
    timeout=10 * 60 * 60,  # 10h cap; expect 4-6h
)
def train():
    env = os.environ.copy()
    env["MHCFLURRY_OUT"] = env["RUNPLZ_OUT"]
    subprocess.run(
        ["bash", "scripts/training/pan_allele_presentation_subset.sh"],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    train.remote()

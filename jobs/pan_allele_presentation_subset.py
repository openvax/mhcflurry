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
        # Single-A100 fallback — org's $30/hr cap blocks GCP 8×A100, and
        # verda/massedcompute 8×A100 shadeform paths both went STATUS:
        # FAILURE. MassedCompute 1×A100-80GB is $1.49/hr, 2m30s boot,
        # reliable historically. Wall time ~8-12 hr vs 4-6 hr on 8×.
        instance_type="massedcompute_A100_sxm4_80G",
        auto_create_instances=True,
    ),
)

image = (
    # devel variant ships `nsys` (Nsight Systems CLI) at
    # /usr/local/cuda/bin/nsys — needed for the NSYS_PROFILE=1 gate in
    # scripts/training/pan_allele_presentation_subset.sh.
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        "runplz>=3.7.2",
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
    # Sized for the pinned 1×A100-80GB MassedCompute shape (14 vCPU,
    # ~80 GB RAM, 625 GB disk, 80 GB VRAM). Overstating min_cpu/min_memory
    # here has previously blocked `brev search` from matching the shape.
    min_cpu=14,
    min_memory=80,
    min_gpu_memory=40,
    min_disk=100,
    timeout=18 * 60 * 60,  # 18h cap; single-GPU wall time ~8-12h
    env={
        "DATALOADER_NUM_WORKERS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
        # Wrap stage 1 affinity training in `nsys profile` for the first
        # ~3 min to capture a representative slice (startup + pretrain +
        # first finetune epochs). Rest of the run is uninstrumented.
        "NSYS_PROFILE": "1",
    },
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

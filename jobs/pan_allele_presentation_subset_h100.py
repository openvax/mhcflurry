"""Same as pan_allele_presentation_subset.py but on a single Nebius H100.

Why single H100 vs 8×A100:
  - 5/5 8×A100 provisioning attempts failed on 2026-04-21 across
    multiple shadeform-brokered providers (Denvr, MassedCompute) and
    OCI launchpad. Root cause was Brev's control plane, not our code.
  - Nebius H100 is provided by Nebius direct (not shadeform-brokered),
    so it routes around today's broker issues.
  - H100 is ~1.5-2× A100 for our FP32 MLP workload. With
    MAX_WORKERS_PER_GPU=2 (80GB VRAM), we get 2 concurrent workers
    and ~35 min per work item. 32 items / 2 × 35 min = ~9 hr wall
    time. At $3.54/hr that's ~$32 — an order of magnitude less than
    the 8×A100 attempts would have cost.
  - The code is not yet H100-optimized (no bf16 autocast, no
    torch.compile, no CUDA graphs). Those would pull out another
    1.5-2× but are Phase 4 scope. Current code runs correctly on
    H100, just leaves some tensor-core utilization on the table.

Launch:
    runplz brev jobs/pan_allele_presentation_subset_h100.py

Stages: same as presentation_subset (affinity → processing →
presentation), same SUBSET_ARCHS knob. Differs only in hardware
target. Wall time expected ~9 hr for 32 work items.
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-presentation-subset-h100",
    brev_config=BrevConfig(
        mode="container",
        # Nebius direct, NOT shadeform-brokered. 80GB H100 SXM5,
        # 16 vCPUs, 200 GB RAM, ~50 GB disk. Boot is typically <5m.
        # $3.54/hr.
        instance_type="gpu-h100-sxm.1gpu-16vcpu-200gb",
        auto_create_instances=True,
        # H100 boots fast on Nebius; 20 min margin is plenty.
        ssh_ready_wait_seconds=1200,
    ),
)

image = (
    # :devel ships nsys at /usr/local/cuda/bin/nsys, needed for
    # NSYS_PROFILE=1 diagnostics. H100 benefits most from profiling
    # since our code isn't H100-optimized yet — worth capturing a
    # trace to see where cycles are spent.
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
    gpu="H100",
    # Nebius gpu-h100-sxm.1gpu-16vcpu-200gb: 16 vCPU, 200 GB RAM,
    # 80 GB VRAM. Overstating these blocks shape matching in brev
    # search.
    min_cpu=16,
    min_memory=200,
    min_gpu_memory=80,
    min_disk=100,
    timeout=14 * 60 * 60,  # 14h cap; H100 expected wall time ~9h
    env={
        # 2 workers × 80 GB card = fine (per-worker peak ~30 GB).
        # H100's higher compute/memory bandwidth may let us push to
        # 3 later if GPU util stays high.
        "MAX_WORKERS_PER_GPU": "2",
        # Shell default is 1. Production-sized fit() datasets downgrade to
        # in-process batching to avoid spawn-pickling arrays into children.
        "DATALOADER_NUM_WORKERS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
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

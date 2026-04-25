"""Phase-4 sweep on 1×L40S-48GB MassedCompute.

Cheaper than the A100 equivalent ($1.06/hr vs $1.49/hr) and has more
vCPUs (22 vs 14) — extra CPU headroom for bigger DataLoader prefetch
fans and CPU-overflow worker experiments.

Sweep covers:
  Phase A: max_tasks_per_worker ∈ {1, 10, 70} — compile amortization.
  Phase B: gpu_workers ∈ {1, 2, 3} — GPU-occupancy curve.
  Phase C: CPU overflow + OMP variations at the leading gpu/maxtasks.

Torch.compile + TF32 enabled via env (MHCFLURRY_TORCH_COMPILE=1).

Launch:
    runplz brev jobs/sweep_workers_l40s.py
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-sweep-l40s",
    brev_config=BrevConfig(
        mode="container",
        # MassedCompute 1×L40S-48GB — cheapest L40S on Brev ($1.06/hr),
        # 22 vCPUs, 625 GB disk, 48 GB VRAM, ~2m30s boot.
        instance_type="massedcompute_L40S",
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
    gpu="L40S",
    # MassedCompute L40S spec: 22 vCPU, ~80 GB RAM (approximate; box
    # advertises high memory), 48 GB VRAM.
    min_cpu=22,
    min_memory=60,
    min_gpu_memory=48,
    min_disk=500,
    timeout=24 * 60 * 60,  # 24h cap; sweep wall estimate ~10-14h
)
def sweep():
    env = os.environ.copy()
    env["MHCFLURRY_OUT"] = env["RUNPLZ_OUT"]
    subprocess.run(
        ["bash", "scripts/training/sweep_workers.sh"],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    sweep.remote()

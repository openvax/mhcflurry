"""Workers × OMP grid-search on 1×A100-80GB.

Runs ``scripts/training/sweep_workers.sh`` which measures wall-time /
throughput for each combination of {MAX_WORKERS_PER_GPU=1..4} ×
{OMP_NUM_THREADS=1, unset}. Output is a CSV plus the winning config.

With torch.compile enabled (``MHCFLURRY_TORCH_COMPILE=1``) the sweep
tells us the best parallelism for the compiled hot path — which may
differ from the un-compiled baseline since compile removes per-step
kernel-launch overhead and shifts the bottleneck away from GPU
occupancy toward H2D / CPU-side prep.

Launch:
    runplz brev jobs/sweep_workers.py
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-sweep-workers",
    brev_config=BrevConfig(
        mode="container",
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
    min_cpu=14,
    min_memory=80,
    min_gpu_memory=80,
    min_disk=200,
    timeout=4 * 60 * 60,  # 4h cap; expected sweep wall ~45 min
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

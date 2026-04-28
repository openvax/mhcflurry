"""Full release-exact 2.2.0 training on 1×L40S-48GB, Phase-4 config.

Same recipe as jobs/pan_allele_release_exact_single_a100.py — same
curated data, same 4 folds × 35-arch sweep = 140 networks, same
--min-models 2 / --max-models 8 selection, same percentile-rank
calibration — but pinned to MassedCompute 1×L40S-48GB + the winner
config from the v11 single-network sweep on L40S:

  * batch_size = 512 (already release default post-Phase 4)
  * dataloader_num_workers = 1
  * MAX_WORKERS_PER_GPU = 2
  * OMP/MKL/OPENBLAS auto-computed via set_cpu_threads helper
  * MHCFLURRY_TORCH_COMPILE = 1 (default on in release_exact.sh)
  * MAX_TASKS_PER_WORKER = 12 (bounded worker reuse; compile still amortized)
  * TF32 + cudnn.benchmark on by default (decoupled from compile)

Compute: 140 nets × ~120 s per net / 2 concurrent workers ≈ 2.5 hr
wall + ~1 hr for torch.compile amortization + data prep ≈ ~3-4 hr.
Cost: L40S MassedCompute is $1.06/hr → ~$4-5 for the full run.

Launch:
    runplz brev --instance mhcflurry-sweep-l40s \
        jobs/pan_allele_release_exact_l40s.py

Uses the same persistent L40S box we've been sweeping on.
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-release-exact-l40s",
    brev_config=BrevConfig(
        mode="container",
        instance_type="massedcompute_L40S",
        auto_create_instances=True,
        ssh_ready_wait_seconds=1800,
        # Brev backend kill-switch. runplz only enforces wall caps for Brev
        # through BrevConfig.max_runtime_seconds; @app.function(timeout=...)
        # is Modal-only.
        max_runtime_seconds=24 * 60 * 60,
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        "runplz>=3.9.2",
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
    min_cpu=22,
    min_memory=60,
    min_gpu_memory=48,
    min_disk=500,
    timeout=24 * 60 * 60,  # Modal-only. Brev cap set in BrevConfig above.
    env={
        "MAX_WORKERS_PER_GPU": "2",
        "MAX_TASKS_PER_WORKER": "12",
        "DATALOADER_NUM_WORKERS": "1",
        "MHCFLURRY_TORCH_COMPILE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
        # OMP/MKL/OPENBLAS auto-computed by set_cpu_threads inside
        # pan_allele_release_affinity.sh based on nproc + worker layout.
        # No manual override here → script auto-tunes.
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

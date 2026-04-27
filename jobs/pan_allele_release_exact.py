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
        # OCI 8×A100-80GB via Brev's direct launchpad integration.
        # $19.30/hr, 128 vCPUs, 27 TB disk, 15 min boot.
        #
        # Routes around the shadeform broker entirely. All 5 failed
        # 2026-04-21 provisioning attempts (3× Denvr, 2× MassedCompute
        # DGXx8) went through shadeform and failed with
        # `external nodes: skipping (list failed): not_found` — the
        # shadeform broker losing track of its own capacity pool.
        # OCI launchpad is a different provisioning path (direct from
        # NVIDIA Brev to Oracle) so it's immune to that class of
        # failure.
        #
        # 80GB cards support MAX_WORKERS_PER_GPU=2 = 16 concurrent
        # workers (vs 8 on 40GB). Cost delta over MassedCompute
        # ($19.30 vs $12.29/hr, +57%) is acceptable given 5/5
        # shadeform failures today.
        instance_type="oci.a100x8.sxm.brev-dgxc",
        auto_create_instances=True,
        # OCI boot advertised at 15 min; give 40 min margin against
        # launchpad's own provisioning queue. Closes runplz #34.
        ssh_ready_wait_seconds=2400,
        # Brev backend kill-switch. runplz only enforces wall caps for Brev
        # through BrevConfig.max_runtime_seconds; @app.function(timeout=...)
        # is Modal-only.
        max_runtime_seconds=60 * 60 * 60,
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
        # 3.11.0: detached remote bootstrap (setsid + nohup + file redirects)
        # so an SSH drop from the client can no longer SIGPIPE the remote
        # training process tree. Before this, a wifi hiccup on the laptop
        # killed the 8×A100 run at 21:09 UTC 2026-04-23 after ~1h36m and
        # 3/140 tasks. See pirl-unc/runplz#53.
        "runplz>=3.11.0",
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
    min_cpu=96,          # match the 8-GPU machine's vCPU budget
    min_memory=400,      # GB RAM — full data + pretraining buffer
    min_gpu_memory=40,   # GB VRAM per GPU
    min_disk=1000,       # GB — ~140 networks × ~10MB weights + init info
    timeout=60 * 60 * 60,  # Modal-only. Brev cap set in BrevConfig above.
    env={
        # massedcompute_A100_sxm4_80G_DGXx8 is 8× A100-80GB. 80GB cards
        # fit 2 workers (each ~20 GB peak VRAM during validation
        # inference), unlocking 16 concurrent training workers across
        # 8 GPUs. With Phase 2 NonDaemonPool, each worker's DataLoader
        # prefetch path is live in production too.
        "MAX_WORKERS_PER_GPU": "2",
        "MAX_TASKS_PER_WORKER": "12",
        # Enable torch.compile (gated in class1_neural_network.py via
        # _maybe_compile_network + _maybe_compile_loss). Pays ~60 s
        # codegen per worker process, amortized across the 12
        # tasks/worker; fuses the forward+loss into a few kernels and
        # eliminates per-step Python dispatch. Stacks with the
        # minibatch=4096 bump — bigger batches give compile more
        # arithmetic to amortize its fixed overheads over.
        "MHCFLURRY_TORCH_COMPILE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
        # One DataLoader subprocess per training worker is the measured
        # sweep winner and matches scripts/training/pan_allele_release_exact.sh.
        # It keeps the SHM/prefetch path active without multiplying process
        # count and /dev/shm pressure. Use 0 for the legacy numpy/no-SHM
        # fallback; avoid >=2 unless re-benchmarking on new hardware.
        "DATALOADER_NUM_WORKERS": "1",
        # Populate the per-epoch timing arrays in fit_info (epoch_fetch_time,
        # epoch_train_time, epoch_validation_time, etc.). Writes to the
        # persisted model's config_json so we can do post-hoc breakdown
        # of where time goes. No runtime cost beyond a few timestamp
        # records per epoch.
        "MHCFLURRY_ENABLE_TIMING": "1",
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

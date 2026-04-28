"""Exact replication of the public mhcflurry pan-allele release training
recipe (models_class1_pan 2.2.0): same pretraining, same curated data,
same 4 folds, same 35-architecture sweep, same selection step
(--min-models 2 --max-models 8), same percentile-rank calibration.

Launch:
    runplz brev --instance mhcflurry-release-exact jobs/pan_allele_release_affinity.py

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
        # 8×A100-80GB via Shadeform→Verda. 2026-04-26 confirmed working:
        # the prior `mhcflurry-release-exact-verda-a100x8` box ran a full
        # release_exact training cycle on this exact type.
        #
        # Switched off OCI (was: oci.a100x8.sxm.brev-dgxc) on 2026-04-27
        # after Brev support confirmed the OCI launchpad path is broken
        # at their backend layer (cloudCredId/workspaceGroupId rejection
        # on every create call). runplz 3.15.0 default-blocks OCI from
        # the auto-pick selector for the same reason; pinning here was
        # forcing runplz to attempt the broken path anyway.
        #
        # 80GB cards support MAX_WORKERS_PER_GPU=2 = 16 concurrent workers
        # (vs 8 on 40GB). If Shadeform goes down again (5/5 failures on
        # 2026-04-21), the next fallback worth trying by hand is
        # massedcompute_A100_sxm4_80G — same hardware class via a
        # different Shadeform sub-broker.
        instance_type="verda_A100_sxm4_80Gx8",
        auto_create_instances=True,
        # Verda boot has run ~5-10 min in practice; 40 min keeps the same
        # generous margin we used for OCI without changing other timing.
        # Closes runplz #34.
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
        # auto -> orchestrator resolves on the actual box. Lands at 4 on
        # 8x80GB Verda (per the 2026-04-28 _AUTO_MWPG_PER_WORKER_GB_DEFAULT
        # change to 4.0 GB/worker + by_jobs skip when num_jobs is also auto).
        # Was pinned to 2 historically to mirror the 2.2.0 training config,
        # but mhcflurry seeds tasks from (fold, arch, replicate) — not from
        # worker ID — so wall-time parallelism doesn't affect trained
        # weights, and torch.compile/cudnn nondeterminism already broke
        # any bit-for-bit promise the historical pin offered.
        "MAX_WORKERS_PER_GPU": "auto",
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
        # ``auto`` → orchestrator derives per-fit-worker DataLoader prefetch
        # child count from box capacity (vCPU + RAM + fit-worker plan,
        # capped at 4). On 8×A100-80GB Verda (176 vCPUs / 16 fit workers /
        # 400 GB RAM) lands at 4 — the 2026-04-26 production benchmark.
        # On L40S/T4/single-A100 boxes it steps down. See
        # ``mhcflurry.local_parallelism.auto_dataloader_num_workers`` for
        # the heuristic; the test matrix in
        # ``test/test_orchestrator_helpers.py`` covers the production
        # tiers. Pin a literal int (0/1/2/3/4) only when re-benchmarking;
        # changing the cap to >4 also requires
        # ``MHCFLURRY_AUTO_DATALOADER_HARD_CAP=N``.
        "DATALOADER_NUM_WORKERS": "auto",
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
        ["bash", "scripts/training/pan_allele_release_affinity.sh"],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    train.remote()

"""Full mhcflurry release pipeline: affinity → processing → presentation.

Replicates the public mhcflurry 2.2.0 release-equivalent pipeline at
full architecture-sweep size: trains the affinity ensemble (same recipe
as ``pan_allele_release_affinity.py``), then trains the no-flank +
short-flanks processing ensembles, then fits + calibrates the
presentation predictor on top. Final artifact is a
``Class1PresentationPredictor`` directory at
``$MHCFLURRY_OUT/presentation/models/`` along with its three input
predictors.

Launch:
    runplz brev --instance mhcflurry-release-exact jobs/pan_allele_release_full.py

Compute (rough estimate, 8x A100-80GB):
    Stage 1 — affinity:     ~28 hr  (140 networks @ ~90 min A100-hour)
    Stage 2 — processing:    ~3 hr  (no_flank + short_flanks variants)
    Stage 3 — presentation:  ~30 min (LR fit + calibrate + bundle)
    Total:                  ~32 hr  (~$380 at $12/hr).

Stage 1 supports --continue-incomplete via the affinity script's
internal logic. Stages 2-3 re-run from scratch on retry; their wall
time is small relative to stage 1 so this is acceptable.

If the ssh session drops mid-run, re-launch runplz with the same
command — stage 1 picks up where it left off; stages 2-3 re-run.
"""

import os
import subprocess

from runplz import App, BrevConfig, Image

app = App(
    "pan-allele-release-full",
    brev_config=BrevConfig(
        mode="container",
        # 8xA100-80GB via Shadeform->MassedCompute DGXx8. Verda's 8x
        # shape stopped showing up in `brev search` after 2026-04-28;
        # MassedCompute is the next-cheapest 8x80GB option ($12.29/hr,
        # ~8m boot). 80GB cards fit 2 workers/GPU = 16 concurrent
        # training workers across 8 GPUs.
        instance_type="massedcompute_A100_sxm4_80G_DGXx8",
        auto_create_instances=True,
        ssh_ready_wait_seconds=2400,
        # Headroom: stage 1 alone uses ~28h of the 60h Brev cap; the
        # full pipeline plus fault-recovery buffer fits.
        max_runtime_seconds=60 * 60 * 60,
    ),
)

image = (
    Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("bzip2", "wget", "rsync", "build-essential", "git")
    .pip_install(
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
    min_cpu=96,
    min_memory=400,
    min_gpu_memory=40,
    min_disk=1500,       # GB — affinity (~150) + processing (~50) + cache (~10) + buffer
    timeout=60 * 60 * 60,
    env={
        # Container env preset. The per-stage shell scripts also pass
        # the matching CLI flags to the orchestrator, which propagates
        # CLI-or-env -> env (resolve_local_parallelism_args). Either
        # path produces identical worker state; keeping both means
        # callers can tune via container env or via shell-script CLI
        # without divergence.
        # auto → orchestrator resolves on the actual box (8x80GB Verda
        # lands at 4 via VRAM + hard_cap, vs affinity-only's pinned 2 for
        # 2.2.0 replication). The shell pre-resolves via the same helper
        # so OMP/MKL budget calculations downstream see a numeric value.
        "MAX_WORKERS_PER_GPU": "auto",
        "MAX_TASKS_PER_WORKER": "12",
        "MHCFLURRY_TORCH_COMPILE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "auto",
        "MHCFLURRY_TORCH_COMPILE_LOSS": "1",
        "DATALOADER_NUM_WORKERS": "auto",
        "MHCFLURRY_ENABLE_TIMING": "1",
        # TF32 + cudnn.benchmark on. Free ~2x matmul speedup on Ampere+
        # with fp32 accumulation preserved (input-mantissa truncation
        # only). The affinity-only release_affinity job keeps this OFF
        # to preserve bit-for-bit replication of public 2.2.0; this is
        # a new (2.3.0) release so the small numeric drift is fine.
        "MHCFLURRY_MATMUL_PRECISION": "high",
    },
)
def train():
    # In Brev container-mode RUNPLZ_OUT resolves to $HOME/runplz-out.
    # The full-pipeline script writes its three sub-stages
    # (affinity/, processing/, presentation/) under MHCFLURRY_OUT.
    env = os.environ.copy()
    env["MHCFLURRY_OUT"] = env["RUNPLZ_OUT"]
    subprocess.run(
        ["bash", "scripts/training/pan_allele_release_full.sh"],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    train.remote()

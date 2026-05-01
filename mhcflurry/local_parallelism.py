"""
Infrastructure for "local" parallelism, i.e. multiprocess parallelism on one
compute node.
"""

import itertools
import logging
import multiprocessing
import multiprocessing.pool
import re
import traceback
import sys
import os
import time
import queue
import subprocess
from multiprocessing import Queue, cpu_count
from multiprocessing.util import Finalize
from pprint import pprint
import random

import numpy

from .common import configure_pytorch, normalize_pytorch_backend


# Per-worker VRAM upper bound (gigabytes) used by ``auto_max_workers_per_gpu``
# to budget how many workers fit on each GPU. Live diagnostics from the
# 2026-04-28 release_exact run showed actual per-worker steady-state VRAM
# is 1.85-2.4 GB on the 4096-batch torch.compile path; the 16.0 GB historical
# value was set before the fixed-encoding + indices-on-device work landed
# and was producing only 2 workers/GPU on 80 GB cards (well below the
# hard_cap of 4). Lowered to 4.0 GB which keeps a 2x safety margin over
# observed and unlocks the 4-worker tier on 80 GB cards. 40 GB cards still
# resolve to 1-2 (60% headroom × 40 / 4 = 6, hard-capped). Override with
# ``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB`` for re-benchmarking.
_AUTO_MWPG_PER_WORKER_GB_DEFAULT = 4.0

# SM-scheduler ceiling. Beyond ~4 workers/GPU the kernel queue serializes
# behind a single SM scheduler, so per-worker throughput drops faster than
# you gain from more parallelism. Tunable via env, but the empirical sweet
# spot is 2-4 for small MLP workloads.
_AUTO_MWPG_HARD_CAP_DEFAULT = 4

# Fallback free-VRAM estimate per GPU (gigabytes) when ``nvidia-smi`` is not
# available. This path deliberately does not import torch: local pools use fork
# on Linux, and touching the CUDA runtime in the parent before forking causes
# CUDA re-initialization failures in children.
_AUTO_MWPG_FREE_VRAM_FALLBACK_GB = 16.0


# ----- Auto DataLoader prefetch worker count --------------------------------
#
# Per-fit() ``dataloader_num_workers`` controls how many child processes each
# fit() worker spawns to prefetch minibatches. The trade-off:
#
# * 0 → in-process batch building. No spawn cost; no parallel prefetch.
#   GPU starves on CPU-heavy data prep (random-negative regeneration,
#   fancy-indexing). Acceptable on tiny/CPU-only configs.
# * 1 → one prefetcher per fit() worker. Hides single-step CPU prep behind
#   the GPU step, but the lone prefetcher is the bottleneck when CPU prep
#   dominates one step. This was the 2026-04-27 release recipe default and
#   is what produced the 2× slowdown vs the 4-worker run on 8×A100.
# * 2-4 → multiple prefetchers feed batches in parallel. Past 4 we hit
#   diminishing returns on most boxes (PyTorch's MultiProcessingDataLoaderIter
#   plus our shared-tensor backing share queue contention) and start losing
#   to scheduler overhead.
#
# This auto-resolver picks per-fit-worker prefetch count from:
#   * total vCPUs (default ``os.cpu_count()``)
#   * fit-worker count (NUM_JOBS or num_gpus × max_workers_per_gpu)
#   * total RAM in GB (optional, for tight-RAM boxes)
#   * SM-scheduler-style hard cap (default 4, env-overridable)
#
# Intent: the recipe pins ``DATALOADER_NUM_WORKERS=auto`` and the orchestrator
# computes the right value per box. Hardware tier mismatches like the
# L40S-tuned "1" landing on a 176-vCPU 8×A100 box can't recur.

# Per-fit-worker DataLoader child cap. Past 4, PyTorch's queue contention +
# our SHM backing's shared collator hit diminishing returns on tested boxes
# (8×A100 / L40S / single-A100). Override with
# ``MHCFLURRY_AUTO_DATALOADER_HARD_CAP``.
_AUTO_DATALOADER_HARD_CAP_DEFAULT = 4

# Approximate RSS that one DataLoader child holds: torch + mhcflurry imports
# (~400 MB) plus a small per-batch buffer. The SHM-backed datasets share
# storage handles, so we don't double-count tensor bytes.
_AUTO_DATALOADER_RAM_PER_CHILD_GB = 0.5

# Approximate RSS that the main fit() worker holds before any DataLoader
# children. Used by the RAM guard to avoid over-allocating prefetchers when
# RAM-per-fit is tight. ~2 GB covers torch, mhcflurry, the validation cache,
# and per-fold backing arrays.
_AUTO_DATALOADER_RAM_BASELINE_PER_FIT_GB = 2.0

# Effective cores per DL child for the CPU budget. Each prefetcher does
# fancy-indexing + collation in one process; with ~50% blocking on the work
# queue, two physical cores per child is a comfortable upper bound.
_AUTO_DATALOADER_CORES_PER_CHILD = 2


def _max_workers_per_gpu_arg(value):
    """argparse type for ``--max-workers-per-gpu``. Accepts ``"auto"`` or int>=1."""
    import argparse
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--max-workers-per-gpu must be 'auto' or an integer >= 1, got %r"
            % (value,)
        )
    if v < 1:
        raise argparse.ArgumentTypeError(
            "--max-workers-per-gpu must be >= 1, got %d" % (v,)
        )
    return v


def _num_jobs_arg(value):
    """argparse type for ``--num-jobs``. Accepts ``"auto"`` or int>=0."""
    import argparse
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--num-jobs must be 'auto' or an integer >= 0, got %r" % (value,)
        )
    if v < 0:
        raise argparse.ArgumentTypeError(
            "--num-jobs must be >= 0, got %d" % (v,)
        )
    return v


def _dataloader_num_workers_arg(value):
    """argparse type for ``--dataloader-num-workers``. ``"auto"`` or int>=0."""
    import argparse
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--dataloader-num-workers must be 'auto' or an integer >= 0, "
            "got %r" % (value,)
        )
    if v < 0:
        raise argparse.ArgumentTypeError(
            "--dataloader-num-workers must be >= 0, got %d" % (v,)
        )
    return v


def _random_negative_pool_epochs_arg(value):
    """argparse type for ``--random-negative-pool-epochs``. ``"auto"`` or int>=1."""
    import argparse
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "--random-negative-pool-epochs must be 'auto' or an integer >= 1, "
            "got %r" % (value,)
        )
    if v < 1:
        raise argparse.ArgumentTypeError(
            "--random-negative-pool-epochs must be >= 1, got %d" % (v,)
        )
    return v


def _free_vram_override_gb(num_gpus):
    """Return env-pinned free VRAM in GB, or ``None``.

    ``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB`` accepts either one
    value applied to every GPU or a comma/space-separated list. It exists for
    tests and unusual launchers where ``nvidia-smi`` is hidden but the caller
    knows the device budget.
    """
    value = os.environ.get("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB")
    if not value:
        return None
    parts = [p for p in value.replace(",", " ").split() if p]
    if not parts:
        return None
    vals = [float(p) for p in parts]
    return min(vals[:max(int(num_gpus), 1)])


def _detect_num_cuda_devices_no_torch():
    """Return the number of visible CUDA devices via ``nvidia-smi -L``.

    Subprocess so the orchestrator can size a fork-based worker pool
    without initializing CUDA in the parent process. Returns 0 when
    nvidia-smi is unavailable or no GPUs are visible.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-L"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return 0
    # nvidia-smi -L output is one line per device:
    #   "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-...)"
    # Match the index-prefixed form so we don't count MIG slices or stray
    # diagnostic lines that happen to start with "GPU ".
    pattern = re.compile(r"^GPU \d+:")
    n = 0
    for line in output.decode("utf-8", errors="ignore").splitlines():
        if pattern.match(line):
            n += 1
    return n


def _free_vram_from_nvidia_smi_gb(num_gpus):
    """Return minimum free VRAM across visible GPUs using ``nvidia-smi``.

    This is intentionally a subprocess call instead of ``torch.cuda`` so the
    orchestrator can size a fork-based worker pool without initializing CUDA in
    the parent process.
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return None

    vals = []
    for line in output.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            # nvidia-smi reports MiB for memory.free with nounits.
            vals.append(float(line.split()[0]) / 1024.0)
        except (ValueError, IndexError):
            continue
    if not vals:
        return None
    return min(vals[:max(int(num_gpus), 1)])


def auto_max_workers_per_gpu(
        num_jobs, num_gpus, backend="auto", per_worker_gb=None):
    """Pick ``max_workers_per_gpu`` based on detected hardware.

    Returns an int ≥ 1. Logic:

      * ``num_gpus == 0`` (CPU-only) → 1.
      * Otherwise: take the minimum of three caps:
          - ``num_jobs // num_gpus`` — don't oversubscribe a GPU beyond the
            jobs that actually exist.
          - ``floor(0.6 × free_vram_gb / per_worker_gb)`` — VRAM headroom
            with 40% slack for activation peaks and the auto-sized
            validation batch.
          - ``hard_cap`` (default 4) — SM-scheduler kernel-serialization
            wall.

    Free VRAM is read from ``nvidia-smi`` (or from
    ``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB``). It deliberately
    avoids ``torch.cuda`` so resolving local parallelism before forking does
    not initialize CUDA in the parent process. Per-worker VRAM upper bound and
    the hard cap are overridable via
    ``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB`` and
    ``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP``.

    Calibrate's per-worker footprint is dominated by the cached_stages
    tensor (~15 GB at production peptide universe / ensemble size), so
    callers in the calibrate path pass ``per_worker_gb`` explicitly to
    override the train-default of 4 GB. When given, the explicit hint
    wins over the env var. (No env var override: the workload-specific
    knowledge belongs in the workload's command, not in a global env.)

    The result is logged so the chosen value is visible in the worker
    log alongside the reasoning.
    """
    if not num_gpus or num_gpus < 1 or backend == "cpu":
        return 1

    if per_worker_gb is None:
        per_worker_gb = float(os.environ.get(
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB",
            str(_AUTO_MWPG_PER_WORKER_GB_DEFAULT),
        ))
    else:
        per_worker_gb = float(per_worker_gb)
    hard_cap = int(os.environ.get(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP",
        str(_AUTO_MWPG_HARD_CAP_DEFAULT),
    ))

    free_vram_gb = (
        _free_vram_override_gb(num_gpus)
        or _free_vram_from_nvidia_smi_gb(num_gpus)
    )

    free_vram_gb_used = (
        free_vram_gb
        if free_vram_gb is not None
        else _AUTO_MWPG_FREE_VRAM_FALLBACK_GB
    )

    # When ``num_jobs`` is user-explicit (>0) it caps how many workers/GPU
    # we can pack: e.g. --num-jobs 8 across 8 GPUs means 1/GPU regardless
    # of VRAM. When ``num_jobs <= 0`` (the default; user delegated to
    # auto_num_jobs), don't let it clamp — otherwise this is a chicken/egg
    # with auto_num_jobs (which derives num_jobs from max_workers_per_gpu)
    # and the resolver pins itself to whatever happens to be in args
    # before the resolver runs. Practical effect on the 8x80GB tier:
    # without the user-explicit guard, by_jobs would clamp MWPG to 2
    # (16//8) when production sets num_jobs=16; with the guard, MWPG can
    # rise to the hard_cap of 4 if VRAM allows.
    # ``num_jobs`` may arrive as ``"auto"`` when called from
    # ``resolve_local_parallelism_args`` (which resolves MWPG before
    # num_jobs); treat that as "user delegated" — same as ``num_jobs=0``.
    if isinstance(num_jobs, str) and num_jobs.strip().lower() == "auto":
        num_jobs_int = 0
    else:
        num_jobs_int = int(num_jobs)
    by_vram = max(1, int(free_vram_gb_used * 0.6 / per_worker_gb))
    if num_jobs_int > 0:
        by_jobs = max(1, num_jobs_int // max(int(num_gpus), 1))
        chosen = max(1, min(by_jobs, by_vram, hard_cap))
        by_jobs_log = str(by_jobs)
    else:
        chosen = max(1, min(by_vram, hard_cap))
        by_jobs_log = "skipped (num_jobs auto)"

    logging.info(
        "auto_max_workers_per_gpu: chose %d "
        "(num_jobs=%d, num_gpus=%d, by_jobs=%s, by_vram=%d at "
        "free_vram=%.1f GB%s / %.1f GB/worker, hard_cap=%d)",
        chosen,
        num_jobs_int,
        int(num_gpus),
        by_jobs_log,
        by_vram,
        free_vram_gb_used,
        "" if free_vram_gb is not None else " (fallback)",
        per_worker_gb,
        hard_cap,
    )
    return chosen


def _system_ram_gb():
    """Best-effort total system RAM in GB. ``None`` if unknown.

    Reads ``/proc/meminfo`` on Linux without importing ``psutil``. The fork-pool
    parent must avoid heavyweight imports so we use a 5-line procfs read.
    macOS does not expose /proc, so this returns ``None`` there — auto-tuning
    falls through to the CPU-only path, which is fine for dev work.
    """
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # Format: ``MemTotal:   123456 kB``
                    return float(parts[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None
    return None


def auto_dataloader_num_workers(
        num_fit_workers,
        vcpus=None,
        ram_gb=None,
        hard_cap=None):
    """Pick per-fit-worker DataLoader child count from box capacity.

    Returns an int >= 0. The result is the value that should be plugged
    into each component model's ``dataloader_num_workers`` hyperparameter.
    A return of ``0`` means in-process batching (no prefetch children),
    which is correct for serial runs and very tight CPU configs.

    Inputs
    ------
    num_fit_workers : int
        Total fit() worker processes that will share the box. Equal to
        ``num_gpus * max_workers_per_gpu`` for the canonical GPU run, or
        ``num_jobs`` for CPU-only runs.
    vcpus : int, optional
        Total vCPU count. Default ``os.cpu_count()``.
    ram_gb : float, optional
        Total system RAM in GB. ``None`` skips the RAM cap (CPU-only
        decision).
    hard_cap : int, optional
        Maximum DL children per fit-worker. Default 4 (overridable via
        ``MHCFLURRY_AUTO_DATALOADER_HARD_CAP``).

    Heuristic
    ---------
    1. **Serial / no GPU work**: if ``num_fit_workers <= 0``, return 0.
       The caller will run fit() in-process; spawning children would buy
       nothing and cost a process-fork.
    2. **CPU budget per fit-worker**: ``cpu_per_fit = vcpus // num_fit_workers``.
       Each DL child needs ~``_AUTO_DATALOADER_CORES_PER_CHILD`` (=2)
       physical cores to keep up with fancy-indexing + collate without
       starving the main fit-worker's OMP/MKL pool.
    3. **CPU cap**: ``cpu_cap = cpu_per_fit // 2``.
    4. **RAM cap** (when ``ram_gb`` is provided): each DL child holds
       ~``_AUTO_DATALOADER_RAM_PER_CHILD_GB`` (=0.5) GB of RSS for the
       torch+mhcflurry imports; the main fit-worker baseline is
       ~``_AUTO_DATALOADER_RAM_BASELINE_PER_FIT_GB`` (=2.0) GB.
       ``ram_cap = max(0, (ram_per_fit_gb - 2.0) / 0.5)``.
    5. **Hard cap**: ``min(cpu_cap, ram_cap, hard_cap)``.
    6. **Floor**: 0 only when ``cpu_cap == 0`` (i.e. fewer cores than
       fit-workers — rare, only on tight clusters). Otherwise the floor
       is 1 because in-process batching on a multi-GPU box almost always
       starves the GPU.

    Edge cases
    ----------
    * ``num_fit_workers > vcpus`` → ``cpu_per_fit = 0``, ``cpu_cap = 0``,
      result 0. The main fit-workers themselves are oversubscribed; adding
      DL children would make it worse.
    * ``ram_gb`` very small (e.g. < 2 GB / fit) → ``ram_cap = 0``, falls back
      to in-process batching to preserve correctness over throughput.
    * ``hard_cap`` env override of ``0`` → forces in-process for diagnostics.

    Cross-checks (see test_orchestrator_helpers.py)
    -----------------------------------------------
    * 8×A100-80GB Verda (176v / 16 fit / 400G) → 4
    * 8×A100-40GB (176v / 8 fit / 400G) → 4
    * 8×L40S (96v / 16 fit / 200G) → 3
    * Single A100 80G Lambda (30v / 2 fit / 200G) → 4
    * Single A100 80G tight (16v / 2 fit / 64G) → 4
    * Single T4 (8v / 1 fit / 16G) → 4
    * CPU 8-thread (8v / 0 fit) → 0
    * Tight cluster node (32v / 16 fit / 64G) → 1
    * RAM-starved (176v / 16 fit / 32G) → 0
    """
    if num_fit_workers is None or int(num_fit_workers) <= 0:
        return 0
    num_fit_workers = int(num_fit_workers)
    if vcpus is None:
        vcpus = os.cpu_count() or 1
    vcpus = int(vcpus)

    if hard_cap is None:
        hard_cap = int(os.environ.get(
            "MHCFLURRY_AUTO_DATALOADER_HARD_CAP",
            _AUTO_DATALOADER_HARD_CAP_DEFAULT,
        ))
    hard_cap = int(hard_cap)

    cpu_per_fit = max(0, vcpus // num_fit_workers)
    cpu_cap = cpu_per_fit // _AUTO_DATALOADER_CORES_PER_CHILD

    if ram_gb is not None:
        ram_per_fit = float(ram_gb) / num_fit_workers
        ram_for_dl = max(0.0, ram_per_fit - _AUTO_DATALOADER_RAM_BASELINE_PER_FIT_GB)
        ram_cap = int(ram_for_dl / _AUTO_DATALOADER_RAM_PER_CHILD_GB)
    else:
        ram_cap = None

    chosen = min(hard_cap, cpu_cap)
    if ram_cap is not None:
        chosen = min(chosen, ram_cap)

    # Floor at 1 when the box has CPU + RAM + cap headroom; floor at 0
    # when ANY of cpu_cap / ram_cap / hard_cap is itself 0 (the user or
    # the box explicitly forced in-process batching). The 1-floor exists
    # so capable boxes don't accidentally disable prefetch when integer
    # division rounds down to 0; it must not override an explicit 0.
    if (
        cpu_cap == 0
        or (ram_cap is not None and ram_cap == 0)
        or hard_cap == 0
    ):
        chosen = 0
    else:
        chosen = max(1, chosen)
    chosen = max(0, chosen)

    logging.info(
        "auto_dataloader_num_workers: chose %d (num_fit_workers=%d, vcpus=%d, "
        "ram_gb=%s, cpu_per_fit=%d, cpu_cap=%d, ram_cap=%s, hard_cap=%d)",
        chosen, num_fit_workers, vcpus,
        f"{ram_gb:.1f}" if ram_gb is not None else "?",
        cpu_per_fit, cpu_cap,
        str(ram_cap) if ram_cap is not None else "?",
        hard_cap,
    )
    return chosen


def resolve_dataloader_num_workers(
        value,
        num_fit_workers=None,
        vcpus=None,
        ram_gb=None):
    """Normalize a ``dataloader_num_workers`` value to an int.

    Accepts ``"auto"`` / unset / ``None`` (delegates to
    ``auto_dataloader_num_workers``), or any int-coercible value. Used by
    both the orchestrator-side resolver and the shell helper that injects
    ``dataloader_num_workers`` into the recipe's hyperparameters.yaml.
    """
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return auto_dataloader_num_workers(
            num_fit_workers=num_fit_workers,
            vcpus=vcpus,
            ram_gb=ram_gb,
        )
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "dataloader_num_workers must be 'auto' or a non-negative integer; "
            "got %r" % (value,)
        )
    if out < 0:
        raise ValueError(
            "dataloader_num_workers must be >= 0; got %d" % out
        )
    return out


def auto_random_negative_pool_epochs(
        num_random_negatives,
        peptide_max_length,
        num_workers,
        ram_gb=None,
        *,
        safety_fraction=None,
        per_pool_epoch_per_worker_bytes=None,
        hard_cap=None):
    """Pick ``random_negative_pool_epochs`` from box capacity.

    The pool sits in the heap of every fit-worker process. Per-pool-epoch
    memory cost is dominated by:

    * ``num_random_negatives × peptide_max_length`` int8 indices,
    * intermediate ``pandas.Series[str]`` allocations and encoder buffers
      observed in practice at ~10–40× the int8 size on 2026-04 measurements
      (the old ``pool_epochs=100`` run OOM'd a 944 GB box at ~199 GB / worker
      worth of transient pool cost).

    We budget for the pessimistic ``per_pool_epoch_per_worker_bytes`` figure
    so the auto value is safe under transient peaks; tunable via env when
    a workload is known to behave better than the empirical pessimistic
    figure.

    Returns an int >= 1. ``1`` reproduces the pre-Phase-1 every-epoch-regen
    behavior. ``> 1`` amortizes the RN gen + encode cost across N epochs.

    Inputs
    ------
    num_random_negatives : int
        Number of random-negative peptides per epoch (the planner's
        ``get_total_count()``). The size of one pool-epoch in the cycle.
    peptide_max_length : int
        Longest peptide the encoding allocates space for. With
        BLOSUM62 + ``peptide_amino_acid_encoding_torch=True`` the
        per-peptide footprint is ``peptide_max_length`` int8 bytes.
    num_workers : int
        Total fit() worker processes that will share the box. Each holds
        its own RN pool, so total RAM cost = ``num_workers × pool_epochs ×
        per_pool_epoch_per_worker_bytes``.
    ram_gb : float, optional
        Total system RAM in GB. ``None`` returns ``1`` (safe default —
        we don't know the budget so don't bump pool_epochs above legacy).
    safety_fraction : float, optional
        Fraction of total RAM available to RN pools across all workers.
        Default ``0.5`` (half of system RAM, leaving 50% for the rest).
        Override via ``MHCFLURRY_AUTO_RN_POOL_SAFETY_FRACTION``.
    per_pool_epoch_per_worker_bytes : float, optional
        Empirical per-pool-epoch per-worker RAM cost in bytes. Default
        ``1 GB`` (conservative — captures the int8 indices + transient
        pandas + encoder buffers seen on the 2026-04 run). Override via
        ``MHCFLURRY_AUTO_RN_POOL_PER_EPOCH_PER_WORKER_GB``.
    hard_cap : int, optional
        Maximum pool_epochs regardless of memory budget. Default ``10``
        (overridable via ``MHCFLURRY_AUTO_RN_POOL_HARD_CAP``). Past 10
        the diminishing-return curve flattens — RN gen overhead is
        already amortized to <1 sec/epoch by then.

    Heuristic
    ---------
    Total available bytes for RN pools across the box:
        ``ram_gb × 1e9 × safety_fraction``
    Per-worker budget:
        ``available / max(num_workers, 1)``
    Pool epochs that fit:
        ``per_worker_budget / per_pool_epoch_per_worker_bytes``
    Clamp to ``[1, hard_cap]``.

    Cross-checks
    ------------
    * 8×A100-80GB Verda (400 GB / 32 fit-workers, 1 GB/pool-epoch):
        400 × 0.5 / 32 / 1 = 6.25 → 6
    * Single A100 80G Lambda (200 GB / 2 fit-workers):
        200 × 0.5 / 2 / 1 = 50 → clamped to ``hard_cap``=10
    * Tight cluster node (64 GB / 16 fit-workers):
        64 × 0.5 / 16 / 1 = 2 → 2
    * RAM-starved (32 GB / 16 fit-workers):
        32 × 0.5 / 16 / 1 = 1 → 1 (legacy regen-every-epoch)
    """
    if num_workers is None or int(num_workers) <= 0:
        return 1
    if ram_gb is None:
        return 1
    if safety_fraction is None:
        safety_fraction = float(os.environ.get(
            "MHCFLURRY_AUTO_RN_POOL_SAFETY_FRACTION", "0.5"))
    if per_pool_epoch_per_worker_bytes is None:
        per_pool_epoch_per_worker_bytes = float(os.environ.get(
            "MHCFLURRY_AUTO_RN_POOL_PER_EPOCH_PER_WORKER_GB", "1.0"
        )) * (1024 ** 3)
    if hard_cap is None:
        hard_cap = int(os.environ.get(
            "MHCFLURRY_AUTO_RN_POOL_HARD_CAP", "10"
        ))
    available_bytes = float(ram_gb) * (1024 ** 3) * float(safety_fraction)
    per_worker_budget = available_bytes / max(int(num_workers), 1)
    by_memory = max(1, int(per_worker_budget / max(
        per_pool_epoch_per_worker_bytes, 1.0)))
    # The `num_random_negatives` and `peptide_max_length` inputs are
    # accepted for API symmetry / future extensions (e.g. variable-size
    # encodings); the empirical per-pool-epoch figure already covers them.
    del num_random_negatives, peptide_max_length
    return min(by_memory, max(1, int(hard_cap)))


def auto_num_jobs(num_gpus, max_workers_per_gpu):
    """Compute total fit-worker count from GPU plan.

    Returns ``num_gpus * max_workers_per_gpu`` for GPU runs, ``0`` when no
    GPUs are visible (caller decides serial vs explicit CPU pool). Treats
    ``"auto"`` ``max_workers_per_gpu`` as not-yet-resolved and raises;
    callers must resolve it first via ``auto_max_workers_per_gpu``.
    """
    if not num_gpus or int(num_gpus) <= 0:
        return 0
    if isinstance(max_workers_per_gpu, str):
        if max_workers_per_gpu.strip().lower() == "auto":
            raise ValueError(
                "auto_num_jobs: max_workers_per_gpu must be resolved to an int "
                "before computing num_jobs (call auto_max_workers_per_gpu first)."
            )
        max_workers_per_gpu = int(max_workers_per_gpu)
    return int(num_gpus) * int(max_workers_per_gpu)


def resolve_num_gpus_for_local_parallelism(args, backend=None):
    """Resolve ``args.gpus`` for local worker scheduling.

    ``--gpus`` historically defaulted to ``None`` and most call sites only
    knew the final device count after the worker pool was created. Resolve it
    once, before sizing max-workers-per-GPU, so every downstream resolver sees
    the same capacity.
    """
    if backend is None:
        backend = normalize_pytorch_backend(
            getattr(args, "backend", "auto") or "auto"
        )
    num_gpus_raw = getattr(args, "gpus", None)
    gpus_was_auto = num_gpus_raw is None and backend in ("auto", "gpu")
    if gpus_was_auto:
        num_gpus = _detect_num_cuda_devices_no_torch()
    else:
        num_gpus = int(num_gpus_raw or 0)
    args.gpus = int(num_gpus)
    args.gpus_was_auto = gpus_was_auto
    return int(num_gpus)


def resolve_max_workers_per_gpu(
        args, per_worker_gb=None, num_gpus=None, backend=None):
    """Resolve ``args.max_workers_per_gpu`` to an int, mutating ``args``.

    Accepts the literal string ``"auto"`` (the default) or an int. When
    ``"auto"``, calls ``auto_max_workers_per_gpu`` with the rest of the
    args' parallelism config to pick a value. Idempotent — calling
    twice on the same args is a no-op the second time.

    ``per_worker_gb`` lets workload-specific commands (e.g. calibrate,
    where cached_stages dominates per-worker VRAM at ~15 GB) override
    the train-default. Falls back to env var / the module default when
    not given.

    Returns the resolved int (also stored on ``args.max_workers_per_gpu``
    so subsequent consumers see the int).
    """
    if backend is None:
        backend = getattr(args, "backend", "auto")
    if num_gpus is None:
        num_gpus = getattr(args, "gpus", 0) or 0
    value = getattr(args, "max_workers_per_gpu", None)
    if value is None:
        value = "auto"
    if isinstance(value, str) and value.lower() == "auto":
        resolved = auto_max_workers_per_gpu(
            num_jobs=getattr(args, "num_jobs", 0),
            num_gpus=num_gpus,
            backend=backend,
            per_worker_gb=per_worker_gb,
        )
    else:
        resolved = int(value)
    args.max_workers_per_gpu = resolved
    return resolved


def resolve_local_parallelism_args(
        args, cap_auto_num_jobs=True, per_worker_gb=None):
    """Resolve and normalize local parallelism arguments in one place.

    This is the single pre-fork normalization point for local worker pools:

    * resolves ``--max-workers-per-gpu=auto`` without touching CUDA;
    * resolves ``--num-jobs=auto`` to ``gpus × max_workers_per_gpu`` (post-
      MWPG resolution), so the worker count always matches GPU capacity by
      default;
    * caps any user-explicit ``--num-jobs`` that exceeds GPU capacity. CPU
      overflow workers are almost never useful for GPU-bound mhcflurry
      workloads (calibrate / pan-allele cartesian forwards) — they take
      work items at GPU pace from the imap_unordered queue and then chew
      on them at CPU pace, blocking GPU workers from idling out properly.
      The legacy "explicit MWPG opts in to CPU overflow" exception was a
      foot-gun in practice and is gone;
    * hoists torch.compile's worker-thread cap before the Pool forks.
    """
    if getattr(args, "_local_parallelism_args_resolved", False):
        return args

    backend = normalize_pytorch_backend(getattr(args, "backend", "auto") or "auto")
    num_gpus = resolve_num_gpus_for_local_parallelism(args, backend=backend)

    original = getattr(args, "max_workers_per_gpu", None)
    was_auto = (
        original is None
        or (isinstance(original, str) and original.lower() == "auto")
    )
    resolved = resolve_max_workers_per_gpu(
        args,
        per_worker_gb=per_worker_gb,
        num_gpus=num_gpus,
        backend=backend,
    )
    args.max_workers_per_gpu_was_auto = was_auto

    num_jobs_raw = getattr(args, "num_jobs", "auto")
    num_jobs_was_auto = (
        num_jobs_raw is None
        or (isinstance(num_jobs_raw, str) and num_jobs_raw.lower() == "auto")
    )

    gpu_capacity = (
        auto_num_jobs(num_gpus, resolved)
        if backend in ("auto", "gpu") else 0
    )
    if num_jobs_was_auto:
        # CPU-only / no-GPU plan keeps the historical 0 = serial default.
        args.num_jobs = gpu_capacity
    else:
        args.num_jobs = int(num_jobs_raw)
    args.num_jobs_was_auto = num_jobs_was_auto

    num_jobs = int(args.num_jobs)
    if (
            cap_auto_num_jobs
            and num_jobs > 0
            and gpu_capacity > 0):
        if num_jobs > gpu_capacity:
            print(
                "Local parallelism: capping num_jobs from %d to %d "
                "(--gpus=%d × --max-workers-per-gpu=%d). CPU overflow "
                "workers serialize the work queue and starve the GPU "
                "workers, so excess --num-jobs is dropped rather than "
                "spilling to CPU." % (
                    num_jobs, gpu_capacity, num_gpus, resolved,
                )
            )
            args.num_jobs = gpu_capacity

    # Resolve --dataloader-num-workers=auto to an int. Done after num_jobs
    # is finalized so the auto resolver sees the post-cap fit-worker count.
    # Stored on args so each train_*_command can apply it to its work_items
    # at planning time. Non-affinity commands (processing) accept the flag
    # for argv uniformity but currently no-op when applying it; see
    # ``apply_dataloader_num_workers_to_work_items``.
    dl_value = getattr(args, "dataloader_num_workers", "auto")
    final_num_jobs = int(getattr(args, "num_jobs", 0) or 0)
    if final_num_jobs <= 0 and num_gpus > 0:
        # Serial run on a GPU box: still resolve as if 1 fit-worker exists
        # so a non-zero default lands. Tests pass num_jobs=0 frequently.
        effective_fit_workers = 1
    else:
        effective_fit_workers = max(1, final_num_jobs)
    args.dataloader_num_workers = resolve_dataloader_num_workers(
        dl_value,
        num_fit_workers=effective_fit_workers,
        ram_gb=_system_ram_gb(),
    )
    args.dataloader_num_workers_was_auto = (
        isinstance(dl_value, str) and dl_value.lower() == "auto"
    )

    # Resolve --random-negative-pool-epochs=auto to an int. Sized from
    # system RAM and the post-cap fit-worker plan. Non-resolved here when
    # we don't yet know num_random_negatives or peptide_max_length — the
    # auto resolver tolerates None for those (see
    # ``auto_random_negative_pool_epochs``).
    rn_pool_value = getattr(args, "random_negative_pool_epochs", "auto")
    if isinstance(rn_pool_value, str) and rn_pool_value.lower() == "auto":
        args.random_negative_pool_epochs = auto_random_negative_pool_epochs(
            num_random_negatives=None,
            peptide_max_length=None,
            num_workers=effective_fit_workers,
            ram_gb=_system_ram_gb(),
        )
        args.random_negative_pool_epochs_was_auto = True
    else:
        args.random_negative_pool_epochs = int(rn_pool_value)
        args.random_negative_pool_epochs_was_auto = False

    # Promote orchestrator-owned tuning knobs from CLI to env so the
    # existing call sites (torch_training_loop._maybe_compile_network,
    # _configure_matmul_precision, class1_neural_network._timing_enabled,
    # hoist_torchinductor_compile_threads) read a single source of truth.
    # Auto -> leave env untouched (preserves backward-compat for env-only
    # deploys); explicit CLI value -> overrides env. Workers inherit
    # via the multiprocessing fork done after this resolver runs.
    torch_compile_cli = getattr(args, "torch_compile", "auto")
    if torch_compile_cli in ("0", "1"):
        os.environ["MHCFLURRY_TORCH_COMPILE"] = torch_compile_cli
    torch_compile_loss_cli = getattr(args, "torch_compile_loss", "auto")
    if torch_compile_loss_cli in ("0", "1"):
        os.environ["MHCFLURRY_TORCH_COMPILE_LOSS"] = torch_compile_loss_cli
    matmul_cli = getattr(args, "matmul_precision", "none")
    if matmul_cli != "none":
        os.environ["MHCFLURRY_MATMUL_PRECISION"] = matmul_cli
    if getattr(args, "enable_timing", False):
        os.environ["MHCFLURRY_ENABLE_TIMING"] = "1"

    hoist_torchinductor_compile_threads(args)
    args._local_parallelism_args_resolved = True
    return args


def apply_random_negative_pool_epochs_to_work_items(
        work_items, pool_epochs, *, log=None):
    """Inject ``random_negative_pool_epochs`` into every work item's hyperparameters.

    Parallel to ``apply_dataloader_num_workers_to_work_items``: the orchestrator
    chooses an auto value once at startup (or honors the CLI int) and writes it
    into every per-work-item hyperparameter dict. fit() reads it from
    ``self.hyperparameters['random_negative_pool_epochs']`` when constructing
    its ``RandomNegativesPool``.

    Parameters
    ----------
    work_items : list of dict
        Each dict has a ``hyperparameters`` sub-dict.
    pool_epochs : int
        The resolved value from ``resolve_local_parallelism_args`` (>= 1).
    log : callable, optional
        Logging hook. Defaults to ``print``.
    """
    if log is None:
        log = print
    pool_epochs_int = int(pool_epochs)
    overridden = 0
    for item in work_items:
        hp = item.get("hyperparameters")
        if hp is None:
            continue
        prev = hp.get("random_negative_pool_epochs")
        hp["random_negative_pool_epochs"] = pool_epochs_int
        if prev is None or int(prev) != pool_epochs_int:
            overridden += 1
    log(
        "apply_random_negative_pool_epochs_to_work_items: set %d/%d items to "
        "random_negative_pool_epochs=%d (overridden=%d)" % (
            len(work_items), len(work_items), pool_epochs_int, overridden,
        )
    )


def apply_dataloader_num_workers_to_work_items(work_items, num_workers, *, log=None):
    """Inject ``dataloader_num_workers`` into every work item's hyperparameters.

    Generic across train_*_command modules. Affinity commands (pan-allele
    + allele-specific) build per-work-item hyperparameter dicts that are
    passed to ``Class1NeuralNetwork``; the resolver writes the integer
    chosen at orchestrator startup into each. Processing models do not yet
    consume ``dataloader_num_workers`` (their fit() loop has its own
    DataLoader plumbing without this hyperparameter); calling this on
    processing work items is a no-op write — the field is set but
    ``Class1ProcessingNeuralNetwork`` ignores it. When processing's fit()
    grows the same prefetch hyperparameter, no change is needed here.

    Parameters
    ----------
    work_items : list of dict
        Each dict has a ``hyperparameters`` sub-dict (the canonical shape
        produced by ``train_pan_allele_models_command`` /
        ``train_allele_specific_models_command``).
    num_workers : int
        The resolved value from
        ``resolve_local_parallelism_args``. Use 0 to force in-process
        batching (no prefetch children).
    log : callable, optional
        Logging hook for the human-readable summary. Defaults to ``print``.
    """
    if log is None:
        log = print
    overridden = 0
    for item in work_items:
        hp = item.get("hyperparameters")
        if hp is None:
            continue
        prev = hp.get("dataloader_num_workers")
        hp["dataloader_num_workers"] = int(num_workers)
        if prev is None or int(prev) != int(num_workers):
            overridden += 1
    log(
        "apply_dataloader_num_workers_to_work_items: set %d/%d items to "
        "dataloader_num_workers=%d (overridden=%d)" % (
            len(work_items), len(work_items), int(num_workers), overridden,
        )
    )


# Inductor's compile worker pool defaults to ``os.cpu_count()``; that's fine
# for one process but stacks badly when N fit() workers each spawn their own
# pool. Production auto-sizing budgets roughly ``cpu_count // num_jobs`` helper
# threads per worker and caps the result. The one-worker cache warmup can use a
# much larger cap because only one compiler-heavy process exists.
_INDUCTOR_THREAD_HARD_CAP = 16
_INDUCTOR_WARMUP_THREAD_HARD_CAP = 64


def _torch_compile_enabled():
    return os.environ.get("MHCFLURRY_TORCH_COMPILE", "0") == "1"


def _auto_torchinductor_compile_threads(num_jobs, phase="production"):
    """Return auto-sized Inductor compile helper count for this phase."""
    cpu_count_ = os.cpu_count() or 1
    if phase == "warmup":
        cap = int(os.environ.get(
            "MHCFLURRY_TORCHINDUCTOR_WARMUP_COMPILE_THREADS_CAP",
            str(_INDUCTOR_WARMUP_THREAD_HARD_CAP),
        ))
        return max(1, min(cap, cpu_count_))
    cap = int(os.environ.get(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_CAP",
        str(_INDUCTOR_THREAD_HARD_CAP),
    ))
    return max(1, min(cap, cpu_count_ // max(int(num_jobs), 1)))


def _compile_threads_env_is_auto_owned():
    return (
        os.environ.get("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO") == "1"
        or os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") == "auto"
    )


def _set_auto_torchinductor_compile_threads(num_jobs, phase="production"):
    """Set ``TORCHINDUCTOR_COMPILE_THREADS`` to the auto value."""
    threads = _auto_torchinductor_compile_threads(num_jobs, phase=phase)
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(threads)
    os.environ["MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO"] = "1"
    return threads


def configure_cluster_worker_torch_compile_threads():
    """Auto-size Inductor helper threads inside one cluster worker process.

    Cluster parallelism submits each work item as its own process, often on
    different nodes. We therefore do not try to share a compile cache across
    the cluster. Each worker process still needs the same local policy: if
    compile is enabled and ``TORCHINDUCTOR_COMPILE_THREADS`` is unset or
    ``auto``, pick a numeric value on that machine before the first
    ``torch.compile`` call.

    If a scheduler packs several mhcflurry work items onto one node, set
    ``MHCFLURRY_CLUSTER_WORKERS_PER_NODE`` so the auto value is divided across
    those co-resident compiler processes. Otherwise the default assumes one
    work process owns its scheduler CPU allocation.
    """
    if not _torch_compile_enabled():
        return
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ and not (
            _compile_threads_env_is_auto_owned()):
        return
    workers_per_node = int(os.environ.get(
        "MHCFLURRY_CLUSTER_WORKERS_PER_NODE", "1"
    ))
    threads = _set_auto_torchinductor_compile_threads(
        num_jobs=max(workers_per_node, 1),
        phase="production",
    )
    print(
        "torch.compile: cluster worker auto-set "
        "TORCHINDUCTOR_COMPILE_THREADS=%d "
        "(MHCFLURRY_CLUSTER_WORKERS_PER_NODE=%d)" % (
            threads, workers_per_node,
        )
    )


def hoist_torchinductor_compile_threads(args, phase="production"):
    """Auto-size ``TORCHINDUCTOR_COMPILE_THREADS`` for local training.

    ``torch.compile`` (when enabled via ``MHCFLURRY_TORCH_COMPILE=1``)
    spins up an inductor compile worker pool that defaults to
    ``os.cpu_count()`` threads. With N fit() workers each running their own
    compile pool, that multiplies into an oversubscribed compile storm. The
    production phase uses an auto value derived from available cores and the
    worker count; the warmup phase uses a larger value because only one worker
    is compiling.

    The orchestrator owns "how many workers will exist", so it owns
    the env knob too: set once before forking, every worker inherits.
    Skips the hoist when the user has already pinned the value or when
    ``MHCFLURRY_TORCH_COMPILE`` isn't on. Cluster workers running on
    other hosts must size themselves locally; see
    ``configure_cluster_worker_torch_compile_threads``.

    Lives here (not in any one ``train_*_command`` module) so processing,
    allele-specific, and any future train command can call it the same
    way.
    """
    if not _torch_compile_enabled():
        # No compile = no compile pool to size; leave env untouched.
        return
    auto_owned = _compile_threads_env_is_auto_owned()
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ and not auto_owned:
        # User pinned explicitly; don't second-guess.
        print(
            "torch.compile: TORCHINDUCTOR_COMPILE_THREADS=%s "
            "(user-pinned, orchestrator hoist skipped)"
            % os.environ["TORCHINDUCTOR_COMPILE_THREADS"]
        )
        return
    num_jobs = max(int(getattr(args, "num_jobs", 0) or 0), 1)
    cpu_count_ = os.cpu_count() or 1
    threads = _auto_torchinductor_compile_threads(num_jobs, phase=phase)
    if (
            auto_owned
            and os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") == str(threads)):
        return
    _set_auto_torchinductor_compile_threads(num_jobs, phase=phase)
    print(
        "torch.compile: hoisted TORCHINDUCTOR_COMPILE_THREADS=%d "
        "(phase=%s, num_jobs=%d, cpu_count=%d)" % (
            threads, phase, num_jobs, cpu_count_,
        )
    )


# ---- Non-daemonic worker pool --------------------------------------------
#
# By default ``multiprocessing.Pool`` spawns daemon workers, and daemon
# processes cannot fork their own children. That's incompatible with the
# Phase 1 (#268) PyTorch DataLoader wrap in ``Class1NeuralNetwork.fit``,
# which uses ``num_workers>0`` to prefetch minibatches via per-DataLoader
# worker processes. Under the default Pool, the DataLoader init call
# raises ``AssertionError: daemonic processes are not allowed to have
# children`` the first time a training epoch tries to iterate.
#
# Phase 1 shipped a runtime downgrade (``_effective_num_workers`` in
# class1_neural_network.py) that silently forces ``num_workers=0`` when
# called from a daemon process — safe, but it effectively disables
# DataLoader prefetch in production training. The real fix is here:
# run Pool workers as NON-daemonic so they can spawn their own children.
#
# Non-daemon workers have one behavioral difference worth naming: if the
# parent process dies ungracefully (e.g. SIGKILL), the workers may
# linger as zombies rather than being auto-reaped by init. The
# training orchestrator's ``try/finally`` closes and joins the pool on
# clean exit, so this only matters under unusual fault modes.
class NonDaemonProcess(multiprocessing.Process):
    """A ``multiprocessing.Process`` whose ``daemon`` flag cannot be set.

    Reading ``.daemon`` always returns False; writes are no-ops. This
    lets us instantiate ``multiprocessing.pool.Pool`` with a worker
    class that declines to be a daemon, so the DataLoader inside each
    worker can spawn its own prefetch children.
    """

    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value) -> None:
        # Silently ignore; ``multiprocessing.Pool._repopulate_pool`` sets
        # daemon=True on every fresh worker, so we must tolerate the
        # assignment without raising.
        pass


class NonDaemonContext(type(multiprocessing.get_context())):
    """A multiprocessing context that hands out ``NonDaemonProcess`` workers.

    Subclasses the current default context so the start method (fork on
    Linux, spawn on macOS) is preserved — we only swap the Process
    class. The Pool uses ``self._ctx.Process(...)`` to create workers
    and will now get our non-daemonic variant.
    """

    Process = NonDaemonProcess


class NonDaemonPool(multiprocessing.pool.Pool):
    """A ``multiprocessing.Pool`` that runs non-daemonic workers.

    Pool's constructor takes a ``context`` kwarg — we thread a
    ``NonDaemonContext`` through so each worker is a
    ``NonDaemonProcess``. Everything else (apply_async, imap, etc.)
    inherits unchanged.
    """

    def __init__(self, *args, **kwargs):
        # Callers may pass their own context; if not, use our non-daemon one.
        kwargs.setdefault("context", NonDaemonContext())
        super().__init__(*args, **kwargs)


def add_local_parallelism_args(parser):
    """
    Add local parallelism arguments to the given argparse.ArgumentParser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    group = parser.add_argument_group("Local parallelism")

    group.add_argument(
        "--num-jobs",
        default="auto",
        type=_num_jobs_arg,
        metavar="N",
        help="Number of local processes to parallelize training over. "
             "Pass 'auto' (default) to derive from "
             "``--gpus * --max-workers-per-gpu`` once the latter is "
             "resolved (so workers never overflow to CPU silently). "
             "Pass 0 for serial run, or an int to pin.")
    group.add_argument(
        "--backend",
        choices=("auto", "default", "gpu", "mps", "cpu"),
        default="auto",
        help="Device backend. 'default' is a legacy alias for 'auto'. 'gpu' "
             "means CUDA. 'auto' (default) selects the "
             "best available device: GPU > MPS > CPU. When --gpus is set, "
             "GPU-assigned workers use CUDA and overflow workers are forced "
             "to CPU.")
    group.add_argument(
        "--gpus",
        type=int,
        metavar="N",
        help="Number of CUDA GPUs, starting at index 0, to assign across "
             "parallel workers. Requires --num-jobs > 0. Each assigned worker "
             "gets one GPU; workers beyond --gpus * --max-workers-per-gpu run "
             "on CPU.")
    group.add_argument(
        "--max-workers-per-gpu",
        type=_max_workers_per_gpu_arg,
        metavar="N",
        default="auto",
        help="Maximum number of workers to assign to a GPU. Pass 'auto' "
             "(default) to pick a value based on detected free VRAM, the "
             "per-worker VRAM upper bound, and a 4-worker hard cap "
             "(see ``auto_max_workers_per_gpu``). Pass an integer to pin. "
             "Workers beyond ``--gpus * --max-workers-per-gpu`` run on CPU.")
    group.add_argument(
        "--max-tasks-per-worker",
        type=int,
        metavar="N",
        default=None,
        help="Restart workers after N tasks. Workaround for memory "
             "leaks. Requires Python >=3.2.")
    group.add_argument(
        "--worker-log-dir",
        default=None,
        help="Write worker stdout and stderr logs to given directory.")
    group.add_argument(
        "--dataloader-num-workers",
        type=_dataloader_num_workers_arg,
        metavar="N",
        default="auto",
        help="Per-fit-worker DataLoader prefetch child count. Pass "
             "'auto' (default) to derive from box vCPUs / RAM / "
             "fit-worker plan via "
             "``mhcflurry.local_parallelism.auto_dataloader_num_workers`` "
             "(empirical hard cap = 4). Pass an integer to pin (0 disables "
             "prefetch and runs batch building in-process). Overrides any "
             "``dataloader_num_workers`` set in component-model "
             "hyperparameters when applicable; non-affinity train commands "
             "accept the flag for uniformity but currently no-op.")
    group.add_argument(
        "--random-negative-pool-epochs",
        type=_random_negative_pool_epochs_arg,
        metavar="N",
        default="auto",
        help="Number of consecutive epochs that share a pre-encoded "
             "random-negative pool. Pass 'auto' (default) to size from "
             "system RAM / fit-worker plan via "
             "``mhcflurry.local_parallelism.auto_random_negative_pool_epochs`` "
             "(hard cap = 10). Pass an integer to pin (1 reproduces "
             "pre-Phase-1 every-epoch-regen behavior). Overrides any "
             "``random_negative_pool_epochs`` set in component-model "
             "hyperparameters.")
    group.add_argument(
        "--torch-compile",
        choices=("auto", "0", "1"),
        default="auto",
        help="Enable torch.compile for forward kernels. '1' on, '0' off, "
             "'auto' (default) reads MHCFLURRY_TORCH_COMPILE env (off when "
             "unset). When on, the orchestrator also auto-sizes "
             "TORCHINDUCTOR_COMPILE_THREADS — see "
             "hoist_torchinductor_compile_threads.")
    group.add_argument(
        "--torch-compile-loss",
        choices=("auto", "0", "1"),
        default="auto",
        help="Enable torch.compile for training loss modules. 'auto' "
             "(default) reads MHCFLURRY_TORCH_COMPILE_LOSS env; when unset, "
             "loss compilation defaults on inside _maybe_compile_loss. CUDA "
             "workers run a one-op autograd warmup before compiling losses to "
             "avoid the PyTorch 2.4 / Triton invalid-device-context bug.")
    group.add_argument(
        "--matmul-precision",
        choices=("none", "highest", "high", "medium"),
        default="none",
        help="torch.set_float32_matmul_precision setting + cudnn.benchmark "
             "enable. 'highest' keeps full fp32 numerics with cudnn "
             "auto-tuning; 'high'/'medium' enable TF32 on Ampere+ (~2x "
             "matmul speedup, fp32 accumulation preserved, input-mantissa "
             "truncated). 'none' (default) leaves PyTorch's default "
             "untouched. CPU/MPS: no-op.")
    group.add_argument(
        "--enable-timing",
        action="store_true",
        default=False,
        help="Populate per-epoch timing arrays in fit_info "
             "(epoch_fetch_time, epoch_train_time, epoch_validation_time). "
             "Persisted in the model's config_json for post-hoc breakdown. "
             "No runtime cost beyond a few timestamp records per epoch.")


def add_prediction_parallelism_args(parser):
    """
    Add prediction-time local parallelism arguments to an argparse parser.

    This is the inference subset of ``add_local_parallelism_args``: the worker
    scheduler, backend selection, and torch forward-kernel knobs, without
    training-only DataLoader/random-negative options.
    """
    group = parser.add_argument_group("Prediction parallelism")

    group.add_argument(
        "--num-jobs",
        default="auto",
        type=_num_jobs_arg,
        metavar="N",
        help="Number of local prediction worker processes. Pass 'auto' "
             "(default) to use ``--gpus * --max-workers-per-gpu`` when "
             "CUDA GPUs are specified, otherwise run serially. Pass 0 for "
             "serial prediction.")
    group.add_argument(
        "--backend",
        choices=("auto", "default", "gpu", "mps", "cpu"),
        default="auto",
        help="Device backend. 'auto' (default) selects GPU > MPS > CPU. "
             "When --gpus is set, GPU-assigned workers use CUDA.")
    group.add_argument(
        "--gpus",
        type=int,
        metavar="N",
        help="Number of CUDA GPUs, starting at index 0, to assign across "
             "parallel prediction workers. Requires --num-jobs > 0.")
    group.add_argument(
        "--max-workers-per-gpu",
        type=_max_workers_per_gpu_arg,
        metavar="N",
        default="auto",
        help="Maximum prediction workers to assign to each CUDA GPU. Pass "
             "'auto' (default) to choose from detected free VRAM, or an int "
             "to pin.")
    group.add_argument(
        "--max-tasks-per-worker",
        type=int,
        metavar="N",
        default=None,
        help="Restart workers after N prediction chunks. Requires Python >=3.2.")
    group.add_argument(
        "--worker-log-dir",
        default=None,
        help="Write prediction worker stdout and stderr logs to this directory.")
    group.add_argument(
        "--torch-compile",
        choices=("auto", "0", "1"),
        default="auto",
        help="Enable torch.compile for forward kernels. 'auto' reads "
             "MHCFLURRY_TORCH_COMPILE.")
    group.add_argument(
        "--matmul-precision",
        choices=("none", "highest", "high", "medium"),
        default="none",
        help="torch.set_float32_matmul_precision setting. CPU/MPS: no-op.")


def chunk_ranges_for_local_parallelism(
        num_items, num_jobs=0, chunks_per_worker=4):
    """
    Split a row/sequence axis into stable contiguous chunks for local workers.

    Parameters
    ----------
    num_items : int
        Number of input items.
    num_jobs : int
        Number of worker processes. ``0`` yields one serial chunk.
    chunks_per_worker : int
        Target number of work chunks per worker for load balancing.

    Returns
    -------
    list of tuple
        ``(chunk_index, start, end)`` ranges.
    """
    num_items = int(num_items)
    if num_items <= 0:
        return []

    num_jobs = max(int(num_jobs or 0), 1)
    target_chunks = min(num_items, max(1, num_jobs * int(chunks_per_worker)))
    chunk_size = int(numpy.ceil(float(num_items) / target_chunks))
    return [
        (i, start, min(start + chunk_size, num_items))
        for (i, start) in enumerate(range(0, num_items, chunk_size))
    ]


def worker_pool_with_gpu_assignments_from_args(args):
    """
    Create a multiprocessing.Pool where each worker uses its own GPU.

    Uses commandline arguments. See `worker_pool_with_gpu_assignments`.
    Resolves ``args.max_workers_per_gpu="auto"`` to an int (mutating
    ``args`` so downstream consumers — e.g. inference batch sizing in
    calibrate — observe the same value).

    Parameters
    ----------
    args : argparse.ArgumentParser

    Returns
    -------
    multiprocessing.Pool
    """
    resolve_local_parallelism_args(args)

    return worker_pool_with_gpu_assignments(
        num_jobs=args.num_jobs,
        num_gpus=args.gpus,
        backend=args.backend,
        max_workers_per_gpu=args.max_workers_per_gpu,
        max_tasks_per_worker=args.max_tasks_per_worker,
        worker_log_dir=args.worker_log_dir,
    )


def worker_pool_uses_fork(worker_pool=None):
    """Return True when local Pool workers inherit parent globals by fork."""
    context = getattr(worker_pool, "_ctx", None)
    if context is not None:
        return context.get_start_method() == "fork"
    try:
        method = multiprocessing.get_start_method(allow_none=True)
        if method is None:
            method = multiprocessing.get_context().get_start_method()
        return method == "fork"
    except RuntimeError:
        return False


def attach_constant_data_to_work_items_if_needed(
        work_items,
        constant_data,
        worker_pool,
        *,
        log=None):
    """Attach constant data only when the Pool cannot inherit it by fork."""
    if log is None:
        log = print
    if worker_pool_uses_fork(worker_pool):
        log(
            "Local Pool uses fork; workers inherit GLOBAL_DATA without "
            "per-task pickle payloads."
        )
        return False
    log(
        "Local Pool does not use fork; attaching GLOBAL_DATA to each work "
        "item for worker delivery."
    )
    for item in work_items:
        item["constant_data"] = constant_data
    return True


_COMPILE_KEY_HYPERPARAMS = (
    "layer_sizes",
    "topology",
    "dropout_probability",
    "batch_normalization",
    "activation",
    "output_activation",
    "peptide_dense_layer_sizes",
    "allele_dense_layer_sizes",
    "peptide_allele_merge_method",
    "peptide_allele_merge_activation",
    "locally_connected_layers",
    "peptide_amino_acid_encoding",
    "peptide_amino_acid_encoding_torch",
    "peptide_encoding",
    "num_outputs",
    "loss",
    "convolutional_filters",
    "convolutional_kernel_size",
    "convolutional_activation",
    "post_convolutional_dense_layer_sizes",
    "convolutional_kernel_l1_l2",
    "n_flank_length",
    "c_flank_length",
)


def _arch_compile_key(hyperparameters):
    """Stable fingerprint for hyperparameters that change the compile graph.

    Two work items with the same key produce the same torch.compile graph
    and therefore share an on-disk compile cache entry; warming one warms
    both. Hyperparameters that only affect optimization or regularization
    (learning rate, L1/L2 reg, max_epochs, patience, ...) are excluded —
    they're applied outside the compiled forward / loss closure.
    """
    import json
    return json.dumps(
        {k: hyperparameters.get(k) for k in _COMPILE_KEY_HYPERPARAMS},
        sort_keys=True,
        default=str,
    )


def run_single_worker_torch_compile_warmup(
        args,
        work_items,
        work_function,
        constant_data=None):
    """Prime the torch.compile on-disk cache for every unique architecture.

    Walks ``work_items`` and groups them by architecture-compile fingerprint
    (``_arch_compile_key``). For each unique fingerprint, runs **one** work
    item in compile-warmup mode in a single non-daemon worker process —
    ``compile_warmup_only=True`` short-circuits the work function to one
    forward+backward pass after the network is constructed and compiled,
    skipping pretraining, validation, and full training. The same worker
    process handles every architecture sequentially so its CUDA context
    and Inductor cache are populated incrementally.

    Skipped entirely when ``MHCFLURRY_TORCH_COMPILE`` is off — there is
    no compile cache to warm.

    ``work_items`` is **not** mutated: every task still runs in the
    production pool. The trade-off is one extra ~1-batch fit per
    architecture (typically <1 sec each after compile codegen) for
    eliminating staggered first-compile costs in the production pool.

    Returns ``None`` when skipped, otherwise the number of unique
    architectures warmed.
    """
    if not work_items:
        return None
    resolve_local_parallelism_args(args)
    if getattr(args, "cluster_parallelism", False):
        return None
    if int(getattr(args, "num_jobs", 0) or 0) <= 1:
        return None
    if not _torch_compile_enabled():
        return None
    if os.environ.get("MHCFLURRY_TORCH_COMPILE_WARMUP", "1") == "0":
        return None

    backend = normalize_pytorch_backend(getattr(args, "backend", "auto") or "auto")
    if backend in ("cpu", "mps"):
        return None

    seen_keys = set()
    unique_warmup_items = []
    for item in work_items:
        hp = item.get("hyperparameters") or {}
        key = _arch_compile_key(hp)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_warmup_items.append(item)

    if not unique_warmup_items:
        return None

    print(
        "torch.compile warmup: priming on-disk cache for %d unique "
        "architecture(s) of %d work items "
        "(1 forward+backward per architecture, single-worker phase)" % (
            len(unique_warmup_items), len(work_items),
        )
    )

    explicit_threads = (
        "TORCHINDUCTOR_COMPILE_THREADS" in os.environ
        and not _compile_threads_env_is_auto_owned()
    )
    if not explicit_threads:
        threads = _set_auto_torchinductor_compile_threads(
            num_jobs=1, phase="warmup"
        )
        print(
            "torch.compile warmup: TORCHINDUCTOR_COMPILE_THREADS=%d "
            "(single-worker codegen phase)" % threads
        )

    warmup_pool = None
    try:
        warmup_pool = worker_pool_with_gpu_assignments(
            num_jobs=1,
            num_gpus=1 if int(getattr(args, "gpus", 0) or 0) > 0 else 0,
            backend=backend,
            max_workers_per_gpu=max(
                int(getattr(args, "max_workers_per_gpu", 1) or 1), 1,
            ),
            max_tasks_per_worker=len(unique_warmup_items) + 1,
            worker_log_dir=getattr(args, "worker_log_dir", None),
        )
        warmup_started_at = time.time()
        for warmup_item in unique_warmup_items:
            item_for_worker = dict(warmup_item)
            item_for_worker["compile_warmup_only"] = True
            if (
                    constant_data is not None
                    and not worker_pool_uses_fork(warmup_pool)
                    and "constant_data" not in item_for_worker):
                item_for_worker["constant_data"] = constant_data
            warmup_pool.apply(
                call_wrapped_kwargs, (work_function, item_for_worker)
            )
        warmup_pool.close()
        warmup_pool.join()
        warmup_pool = None
        print(
            "torch.compile warmup: completed %d architecture warmup(s) in "
            "%.1f sec." % (
                len(unique_warmup_items), time.time() - warmup_started_at,
            )
        )
    finally:
        if warmup_pool is not None:
            warmup_pool.terminate()
            warmup_pool.join()
        hoist_torchinductor_compile_threads(args, phase="production")

    return len(unique_warmup_items)


def worker_pool_with_gpu_assignments(
        num_jobs,
        num_gpus=0,
        backend=None,
        max_workers_per_gpu=1,
        max_tasks_per_worker=None,
        worker_log_dir=None):
    """
    Create a multiprocessing.Pool where each worker uses its own GPU.

    Parameters
    ----------
    num_jobs : int
        Number of worker processes.
    num_gpus : int
    backend : string
    max_workers_per_gpu : int
    max_tasks_per_worker : int
    worker_log_dir : string

    Returns
    -------
    multiprocessing.Pool
    """
    backend = normalize_pytorch_backend(backend or "auto")
    validate_worker_pool_args(
        num_jobs=num_jobs,
        num_gpus=num_gpus,
        backend=backend,
        max_workers_per_gpu=max_workers_per_gpu)

    if num_jobs == 0:
        configure_pytorch(backend=backend)
        return None

    worker_init_kwargs = worker_init_kwargs_for_scheduler(
        num_jobs=num_jobs,
        num_gpus=num_gpus,
        backend=backend,
        max_workers_per_gpu=max_workers_per_gpu)
    if num_gpus:
        print(
            "Assigning %d workers across %d CUDA GPUs (%d workers max per GPU). "
            "Overflow workers will run on CPU." % (
                num_jobs, num_gpus, max_workers_per_gpu))
        for (worker_num, kwargs) in enumerate(worker_init_kwargs):
            print(
                "Worker %d assigned backend=%s GPUs=%s" % (
                    worker_num,
                    kwargs["backend"],
                    kwargs.get("gpu_device_nums")))

    if worker_log_dir:
        for kwargs in worker_init_kwargs:
            kwargs["worker_log_dir"] = worker_log_dir

    worker_pool = make_worker_pool(
        processes=num_jobs,
        initializer=worker_init,
        initializer_kwargs_per_process=worker_init_kwargs,
        max_tasks_per_worker=max_tasks_per_worker)
    return worker_pool


def validate_worker_pool_args(
        num_jobs,
        num_gpus=0,
        backend="auto",
        max_workers_per_gpu=1):
    """
    Validate local worker scheduling arguments.

    ``--gpus`` controls CUDA worker assignment only. It does not select MPS
    devices and it does not distribute a single model across multiple GPUs.
    """
    backend = normalize_pytorch_backend(backend or "auto")
    if num_jobs < 0:
        raise ValueError("num_jobs must be >= 0")
    if num_gpus is None:
        num_gpus = 0
    if num_gpus < 0:
        raise ValueError("num_gpus must be >= 0")
    if max_workers_per_gpu < 1:
        raise ValueError("max_workers_per_gpu must be >= 1")
    if num_gpus:
        if num_jobs == 0:
            raise ValueError("num_gpus requires num_jobs > 0")
        if backend not in ("auto", "gpu"):
            raise ValueError(
                "num_gpus is only supported with backend 'auto' or 'gpu'")


def worker_init_kwargs_for_scheduler(
        num_jobs,
        num_gpus=0,
        backend="auto",
        max_workers_per_gpu=1):
    """
    Build per-worker init kwargs from the local scheduling configuration.

    When ``num_gpus`` is set, workers are assigned one CUDA GPU each in round
    robin order. Any additional workers are forced onto CPU by hiding CUDA and
    setting their backend to ``cpu``.
    """
    backend = normalize_pytorch_backend(backend or "auto")
    validate_worker_pool_args(
        num_jobs=num_jobs,
        num_gpus=num_gpus,
        backend=backend,
        max_workers_per_gpu=max_workers_per_gpu)

    if not num_gpus:
        return [
            {"backend": backend, "max_workers_per_gpu": max_workers_per_gpu}
            for _ in range(num_jobs)
        ]

    gpu_assignments = list(itertools.chain.from_iterable(
        range(num_gpus) for _ in range(max_workers_per_gpu)))

    worker_kwargs = []
    for worker_num in range(num_jobs):
        if worker_num < len(gpu_assignments):
            worker_kwargs.append({
                "backend": "gpu",
                "gpu_device_nums": [gpu_assignments[worker_num]],
                "max_workers_per_gpu": max_workers_per_gpu,
            })
        else:
            worker_kwargs.append({
                "backend": "cpu",
                "gpu_device_nums": [],
                "max_workers_per_gpu": max_workers_per_gpu,
            })
    return worker_kwargs


def make_worker_pool(
        processes=None,
        initializer=None,
        initializer_kwargs_per_process=None,
        max_tasks_per_worker=None):
    """
    Convenience wrapper to create a multiprocessing.Pool.

    This function adds support for per-worker initializer arguments, which are
    not natively supported by the multiprocessing module. The motivation for
    this feature is to support allocating each worker to a (different) GPU.

    IMPLEMENTATION NOTE:
        The per-worker initializer arguments are implemented using a Queue. Each
        worker reads its arguments from this queue when it starts. When it
        terminates, it adds its initializer arguments back to the queue, so a
        future process can initialize itself using these arguments.

        There is one issue with this approach, however. If a worker crashes, it
        never repopulates the queue of initializer arguments. This will prevent
        any future worker from re-using those arguments. To deal with this
        issue we add a second 'backup queue'. This queue always contains the
        full set of initializer arguments: whenever a worker reads from it, it
        always pushes the pop'd args back to the end of the queue immediately.
        If the primary arg queue is ever empty, then workers will read
        from this backup queue.

    Parameters
    ----------
    processes : int
        Number of workers. Default: num CPUs.

    initializer : function, optional
        Init function to call in each worker

    initializer_kwargs_per_process : list of dict, optional
        Arguments to pass to initializer function for each worker. Length of
        list must equal the number of workers.

    max_tasks_per_worker : int, optional
        Restart workers after this many tasks. Requires Python >=3.2.

    Returns
    -------
    multiprocessing.Pool
    """

    if not processes:
        processes = cpu_count()

    pool_kwargs = {
        'processes': processes,
    }
    if max_tasks_per_worker:
        pool_kwargs["maxtasksperchild"] = max_tasks_per_worker

    if initializer:
        if initializer_kwargs_per_process:
            assert len(initializer_kwargs_per_process) == processes
            kwargs_queue = Queue()
            kwargs_queue_backup = Queue()
            for kwargs in initializer_kwargs_per_process:
                kwargs_queue.put(kwargs)
                kwargs_queue_backup.put(kwargs)
            pool_kwargs["initializer"] = worker_init_entry_point
            pool_kwargs["initargs"] = (
                initializer, kwargs_queue, kwargs_queue_backup)
        else:
            pool_kwargs["initializer"] = initializer

    # Use a non-daemonic pool so workers can spawn DataLoader children.
    # See NonDaemonPool for the rationale and the Phase 1 (#268) crash
    # it unblocks.
    worker_pool = NonDaemonPool(**pool_kwargs)
    print("Started pool: %s" % str(worker_pool))
    pprint(pool_kwargs)
    return worker_pool


def worker_init_entry_point(
        init_function, arg_queue=None, backup_arg_queue=None):
    kwargs = {}
    if arg_queue:
        try:
            kwargs = arg_queue.get(block=False)
        except queue.Empty:
            print("Argument queue empty. Using round robin arg queue.")
            kwargs = backup_arg_queue.get(block=True)
            backup_arg_queue.put(kwargs)

        # On exit we add the init args back to the queue so restarted workers
        # (e.g. when when running with maxtasksperchild) will pickup init
        # arguments from a previously exited worker.
        Finalize(None, arg_queue.put, (kwargs,), exitpriority=1)

    print("Initializing worker: %s" % str(kwargs))
    init_function(**kwargs)


def worker_init(
        keras_backend=None, backend=None, gpu_device_nums=None,
        worker_log_dir=None, max_workers_per_gpu=None):
    del keras_backend  # legacy argument retained for API compatibility
    if worker_log_dir:
        # Line buffering (buffering=1) ensures every print/traceback flushes
        # to disk immediately. Without it, Python defaults to ~8 KB block
        # buffering on a regular file, which silently swallows worker
        # output if the worker is killed (OOM, signal) before the buffer
        # fills. That made debugging the openvax/mhcflurry#270 OOM nearly
        # impossible — workers spun at 99% CPU and the LOG files stayed
        # 0 bytes because no flush ever fired before the worker died.
        sys.stderr = sys.stdout = open(os.path.join(
            worker_log_dir,
            "LOG-worker.%d.%d.txt" % (os.getpid(), int(time.time()))),
            "w", buffering=1)

    # Each worker needs distinct random numbers
    numpy.random.seed()
    random.seed()
    if gpu_device_nums is not None:
        print("WORKER pid=%d assigned GPU devices: %s" % (
            os.getpid(), gpu_device_nums))
        configure_pytorch(backend=backend, gpu_device_nums=gpu_device_nums)
    else:
        configure_pytorch(backend=backend)
    # Propagate workers-per-GPU into the process env so auto-sized
    # batching (Class1NeuralNetwork.predict / fit's
    # check_training_batch_fits) can partition VRAM correctly when
    # multiple fit()/predict() calls are co-resident on one GPU.
    # See issue openvax/mhcflurry#272.
    if max_workers_per_gpu is not None:
        os.environ["MHCFLURRY_MAX_WORKERS_PER_GPU"] = str(int(max_workers_per_gpu))


# Solution suggested in https://bugs.python.org/issue13831
class WrapException(Exception):
    """
    Add traceback info to exception so exceptions raised in worker processes
    can still show traceback info when re-raised in the parent.
    """
    def __init__(self):
        exc_type, exc_value, exc_tb = sys.exc_info()
        self.exception = exc_value
        self.formatted = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    def __str__(self):
        return '%s\nOriginal traceback:\n%s' % (Exception.__str__(self), self.formatted)


def call_wrapped(function, *args, **kwargs):
    """
    Run function on args and kwargs and return result, wrapping any exception
    raised in a WrapException.

    Parameters
    ----------
    function : arbitrary function

    Any other arguments provided are passed to the function.

    Returns
    -------
    object
    """
    try:
        return function(*args, **kwargs)
    except Exception:
        raise WrapException()


def call_wrapped_kwargs(function, kwargs):
    """
    Invoke function on given kwargs and return result, wrapping any exception
    raised in a WrapException.

    Parameters
    ----------
    function : arbitrary function
    kwargs : dict

    Returns
    -------
    object

    result of calling function(**kwargs)

    """
    return call_wrapped(function, **kwargs)

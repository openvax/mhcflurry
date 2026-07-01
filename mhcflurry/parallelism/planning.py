# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware planning helpers for local parallelism."""

import logging
import os
import re
import subprocess
import sys

from ..common import normalize_pytorch_backend
from ..workload_planning import (
    HOST_RAM_PER_DATALOADER_CHILD_GB,
    WORKLOAD_GENERIC,
    capacity_warnings,
    env_float,
    env_int,
    plan_local_parallelism,
)


# Per-worker VRAM upper bound (gigabytes) used by ``auto_max_workers_per_gpu``
# to budget how many workers fit on each GPU. Live diagnostics from the
# 2026-04-28 release_exact run showed actual per-worker steady-state VRAM
# is 1.85-2.4 GB on the 4096-batch torch.compile path; the 16.0 GB historical
# value was set before the fixed-encoding + indices-on-device work landed
# and was producing only 2 workers/GPU on 80 GB cards (well below the
# hard_cap of 4). Lowered to 4.0 GB which keeps a 2x safety margin over
# observed and unlocks the 4-worker tier on 80 GB cards. 40 GB cards also
# reach the hard cap at full free VRAM (0.6 × 40 / 4 = 6, capped to 4); fewer
# only when free VRAM is already partly used. Override with
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
# ``dataloader_num_workers`` controls child processes for the streaming
# pretraining DataLoader. The standard affinity fit() loop is device-resident
# and forms minibatches by indexing tensors directly, so this knob does not
# control affinity fine-tuning.
#
# The pretraining trade-off:
#
# * 0 -> in-process batch building. No spawn cost; no parallel prefetch.
#   Acceptable on tiny/CPU-only configs.
# * 1 -> one prefetcher per fit() worker. Hides some CSV/encoding/collation
#   work behind the GPU step.
# * 2-4 -> multiple prefetchers feed batches in parallel. Past 4 we hit
#   diminishing returns on most boxes from PyTorch queue and process
#   scheduling overhead.
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

# Per-fit-worker DataLoader child cap for streaming pretraining. Past 4,
# PyTorch queue/process overhead hit diminishing returns on tested boxes
# (8xA100 / L40S / single-A100). Override with
# ``MHCFLURRY_AUTO_DATALOADER_HARD_CAP``.
_AUTO_DATALOADER_HARD_CAP_DEFAULT = 4

# Approximate RSS that one DataLoader child holds: torch + mhcflurry imports
# (~0.5 GB) plus a small per-batch buffer.
_AUTO_DATALOADER_RAM_PER_CHILD_GB = HOST_RAM_PER_DATALOADER_CHILD_GB

# Approximate RSS that the main fit() worker holds before any DataLoader
# children. Used by the RAM guard to avoid over-allocating prefetchers when
# RAM-per-fit is tight. ~2 GB covers torch, mhcflurry, the validation cache,
# and pretraining batch state.
_AUTO_DATALOADER_RAM_BASELINE_PER_FIT_GB = 2.0

# Effective cores per DL child for the CPU budget. Each prefetcher does
# fancy-indexing + collation in one process; with ~50% blocking on the work
# queue, two physical cores per child is a comfortable upper bound.
_AUTO_DATALOADER_CORES_PER_CHILD = 2

def cuda_visible_devices_from_env():
    """Return CUDA_VISIBLE_DEVICES entries, or ``None`` when unset."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    value = value.strip()
    if not value:
        return []
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices:
        return []
    if devices[0].lower() in ("-1", "none", "void", "nodevfiles"):
        return []
    return devices

def free_vram_per_gpu_override_gb(num_gpus):
    """Return env-pinned free VRAM in GB as a per-GPU list, or ``None``.

    ``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB`` accepts either one
    value applied to every GPU or a comma/space-separated list. It exists for
    tests and unusual launchers where ``nvidia-smi`` is hidden but the caller
    knows the device budget.
    """
    name = "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB"
    value = os.environ.get(name)
    if not value:
        return None
    parts = [p for p in value.replace(",", " ").split() if p]
    if not parts:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            "Environment variable %s=%r contains a non-float entry: %s"
            % (name, value, exc)
        ) from None
    if len(vals) == 1:
        # A single value applies to every GPU.
        vals = vals * max(int(num_gpus), 1)
    return vals[:max(int(num_gpus), 1)]


def free_vram_override_gb(num_gpus):
    """Minimum env-pinned free VRAM in GB, or ``None``."""
    vals = free_vram_per_gpu_override_gb(num_gpus)
    return min(vals) if vals else None


def detect_num_cuda_devices_no_torch():
    """Return the number of visible CUDA devices without importing torch.

    ``CUDA_VISIBLE_DEVICES`` is authoritative when set by a scheduler or
    container. Otherwise shell out so the orchestrator can size a fork-based
    worker pool without initializing CUDA in the parent process. Returns 0
    when nvidia-smi is unavailable or no GPUs are visible.
    """
    cuda_visible_devices = cuda_visible_devices_from_env()
    if cuda_visible_devices is not None:
        return len(cuda_visible_devices)

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


def _free_vram_per_gpu_from_nvidia_smi_gb(num_gpus):
    """Return a list of free VRAM (GB) per visible GPU using ``nvidia-smi``.

    Returns ``None`` if ``nvidia-smi`` is unavailable or returns nothing. The
    per-GPU list (not collapsed to a scalar) lets capacity diagnostics see
    heterogeneous / partially-occupied cards; the worker-count math separately
    takes the ``min`` via ``free_vram_from_nvidia_smi_gb``.

    This is intentionally a subprocess call instead of ``torch.cuda`` so the
    orchestrator can size a fork-based worker pool without initializing CUDA in
    the parent process.
    """
    cuda_visible_devices = cuda_visible_devices_from_env()
    if cuda_visible_devices is not None:
        cuda_visible_devices = cuda_visible_devices[:max(int(num_gpus), 1)]
        if not cuda_visible_devices:
            return None
        device_args = ["-i", ",".join(cuda_visible_devices)]
    else:
        device_args = []

    command = ["nvidia-smi"] + device_args + [
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
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
    return vals[:max(int(num_gpus), 1)]


def free_vram_from_nvidia_smi_gb(num_gpus):
    """Minimum free VRAM (GB) across visible GPUs, or ``None``."""
    vals = _free_vram_per_gpu_from_nvidia_smi_gb(num_gpus)
    return min(vals) if vals else None


def detect_free_vram_per_gpu_gb(num_gpus):
    """Per-GPU free VRAM (GB) as a list, or ``None`` if undetectable.

    Env override (``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB``) first,
    then ``nvidia-smi``. Unlike the scalar helpers used by the worker-count
    math, this preserves the per-GPU values so capacity warnings can flag
    small or uneven cards.
    """
    override = free_vram_per_gpu_override_gb(num_gpus)
    if override is not None:
        return override
    return _free_vram_per_gpu_from_nvidia_smi_gb(num_gpus)


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
    tensor, so the planner passes ``per_worker_gb`` explicitly — the affinity
    calibration workload profile's ``device_worker_gb`` of 24 GB — overriding
    the 4 GB train default. When given, the explicit hint wins over the env
    var. (No env var for it: the workload-specific knowledge belongs in the
    workload profile, not in a global env.)

    The result is logged so the chosen value is visible in the worker
    log alongside the reasoning.
    """
    if not num_gpus or num_gpus < 1 or backend == "cpu":
        return 1

    if per_worker_gb is None:
        per_worker_gb = env_float(
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB",
            _AUTO_MWPG_PER_WORKER_GB_DEFAULT,
            bounds=(0.0, None),
        )
    else:
        per_worker_gb = float(per_worker_gb)
    hard_cap = env_int(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP",
        _AUTO_MWPG_HARD_CAP_DEFAULT,
        bounds=(1, None),
    )

    # Honor an explicit env override even when it resolves to 0.0 (falsy);
    # only fall through to nvidia-smi when no override was set at all.
    free_vram_override = free_vram_override_gb(num_gpus)
    free_vram_gb = (
        free_vram_override
        if free_vram_override is not None
        else free_vram_from_nvidia_smi_gb(num_gpus)
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
    # 40% VRAM headroom (multiply by 0.6) absorbs three sources of
    # transient pressure that ``per_worker_gb`` does not model:
    #   (1) spawn-startup race — workers create their CUDA context and
    #       allocate model state in parallel, briefly holding more memory
    #       than steady-state;
    #   (2) activation peaks during torch.compile inductor codegen;
    #   (3) nvidia-smi's reported ``free`` lags actual contention when
    #       another process on the same card is also allocating.
    # The headroom is intentionally generous: an OOM during a multi-hour
    # training run costs far more than leaving a worker slot on the
    # table, and the cap is bounded above by ``hard_cap`` anyway.
    if per_worker_gb <= 0:
        # A non-positive per-worker budget (e.g. the env override set to 0)
        # disables the VRAM-based cap rather than dividing by zero; ``hard_cap``
        # still bounds the result. Mirrors host_memory_num_jobs_cap's guard.
        by_vram = hard_cap
    else:
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
        hard_cap = env_int(
            "MHCFLURRY_AUTO_DATALOADER_HARD_CAP",
            _AUTO_DATALOADER_HARD_CAP_DEFAULT,
            bounds=(0, None),
        )
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

    Returns an int >= 1. ``1`` means fresh random negatives every epoch.
    ``> 1`` amortizes random-negative generation + encoding across N epochs.

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
        safety_fraction = env_float(
            "MHCFLURRY_AUTO_RN_POOL_SAFETY_FRACTION", "0.5",
            bounds=(0.0, 1.0),
        )
    if per_pool_epoch_per_worker_bytes is None:
        per_pool_epoch_per_worker_bytes = env_float(
            "MHCFLURRY_AUTO_RN_POOL_PER_EPOCH_PER_WORKER_GB", "1.0",
            bounds=(0.0, None),
        ) * (1024 ** 3)
    if hard_cap is None:
        hard_cap = env_int(
            "MHCFLURRY_AUTO_RN_POOL_HARD_CAP", "10",
            bounds=(1, None),
        )
    available_bytes = float(ram_gb) * (1024 ** 3) * float(safety_fraction)
    per_worker_budget = available_bytes / max(int(num_workers), 1)
    by_memory = max(1, int(per_worker_budget / max(
        per_pool_epoch_per_worker_bytes, 1.0)))
    # The `num_random_negatives` and `peptide_max_length` inputs are
    # accepted for API symmetry / future extensions (e.g. variable-size
    # encodings); the empirical per-pool-epoch figure already covers them.
    del num_random_negatives, peptide_max_length
    return min(by_memory, max(1, int(hard_cap)))


def resolved_int(value, name):
    if isinstance(value, str) and value.strip().lower() == "auto":
        raise ValueError(
            "%s must be resolved to an int before use; got 'auto'" % name
        )
    return int(value)


def auto_num_jobs(num_gpus, max_workers_per_gpu):
    """Compute total fit-worker count from GPU plan.

    Returns ``num_gpus * max_workers_per_gpu`` for GPU runs, ``0`` when no
    GPUs are visible (caller decides serial vs explicit CPU pool). Treats
    ``"auto"`` ``max_workers_per_gpu`` as not-yet-resolved and raises;
    callers must resolve it first via ``auto_max_workers_per_gpu``.
    """
    if not num_gpus or int(num_gpus) <= 0:
        return 0
    max_workers_per_gpu = resolved_int(
        max_workers_per_gpu, "max_workers_per_gpu")
    return int(num_gpus) * max_workers_per_gpu


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


def num_workers_per_gpu_from_args(args):
    """Return resolved ``max_workers_per_gpu`` for model auto-sizing.

    Callers must run ``resolve_local_parallelism_args`` first so workload- and
    hardware-aware defaults have already converted the CLI sentinel ``"auto"``
    into an integer.
    """
    value = getattr(args, "max_workers_per_gpu", 1)
    if value is None:
        return 1
    if isinstance(value, str) and value.strip().lower() == "auto":
        raise ValueError(
            "max_workers_per_gpu is still 'auto'; call "
            "resolve_local_parallelism_args before using it as "
            "num_workers_per_gpu."
        )
    return resolved_int(value, "max_workers_per_gpu")


def resolve_local_parallelism_args(
        args,
        cap_auto_num_jobs=True,
        per_worker_gb=None,
        workload_name=WORKLOAD_GENERIC,
        workload_hints=None):
    """Resolve and normalize local parallelism arguments through the planner."""
    if getattr(args, "_local_parallelism_args_resolved", False):
        return args

    plan = plan_local_parallelism(
        args,
        workload_name=workload_name,
        workload_hints=workload_hints,
        per_worker_gb=per_worker_gb,
        cap_auto_num_jobs=cap_auto_num_jobs,
        normalize_backend=normalize_pytorch_backend,
        detect_num_cuda_devices=detect_num_cuda_devices_no_torch,
        auto_max_workers_per_gpu=auto_max_workers_per_gpu,
        auto_num_jobs=auto_num_jobs,
        resolve_dataloader_num_workers=resolve_dataloader_num_workers,
        auto_random_negative_pool_epochs=auto_random_negative_pool_epochs,
    )

    args.backend = plan.backend
    args.gpus = plan.gpus
    args.gpus_was_auto = plan.gpus_was_auto
    args.max_workers_per_gpu = plan.max_workers_per_gpu
    args.max_workers_per_gpu_was_auto = plan.max_workers_per_gpu_was_auto
    args.num_jobs = plan.num_jobs
    args.num_jobs_was_auto = plan.num_jobs_was_auto
    args.dataloader_num_workers = plan.dataloader_num_workers
    args.dataloader_num_workers_was_auto = plan.dataloader_num_workers_was_auto
    args.random_negative_pool_epochs = plan.random_negative_pool_epochs
    args.random_negative_pool_epochs_was_auto = (
        plan.random_negative_pool_epochs_was_auto
    )
    args.workload_plan = plan

    # Promote orchestrator-owned tuning knobs from CLI to env so the
    # existing call sites (pytorch_training.maybe_compile_network,
    # configure_matmul_precision, class1_training._timing_enabled,
    # hoist_torchinductor_compile_threads) read a single source of truth.
    # Auto -> leave env untouched (preserves backward-compat for env-only
    # deploys); explicit CLI value -> overrides env. Workers inherit
    # via the multiprocessing fork done after this resolver runs.
    if plan.torch_compile in ("0", "1"):
        os.environ["MHCFLURRY_TORCH_COMPILE"] = plan.torch_compile
    if plan.torch_compile_loss in ("0", "1"):
        os.environ["MHCFLURRY_TORCH_COMPILE_LOSS"] = plan.torch_compile_loss
    if plan.matmul_precision != "none":
        os.environ["MHCFLURRY_MATMUL_PRECISION"] = plan.matmul_precision
    if plan.enable_timing:
        os.environ["MHCFLURRY_ENABLE_TIMING"] = "1"

    # Preflight: compare the resolved plan against measured machine capacity
    # (per-GPU free VRAM, available RAM, CPU count) and warn when we are below
    # a safe operating range, so a future OOM/contention is diagnosable.
    preflight = capacity_warnings(
        workload_name=plan.workload_name,
        backend=plan.backend,
        gpus=plan.gpus,
        num_jobs=plan.num_jobs,
        per_gpu_free_vram_gb=detect_free_vram_per_gpu_gb(plan.gpus),
        device_worker_gb=plan.device_worker_gb,
        available_ram_gb=plan.host_memory_available_gb,
        host_worker_gb=plan.host_worker_gb,
        cpu_count=os.cpu_count(),
    )
    for warning in list(plan.warnings) + preflight:
        print("Local parallelism:", warning, file=sys.stderr)
    from .torch_compile import hoist_torchinductor_compile_threads
    hoist_torchinductor_compile_threads(args)
    print("Local workload plan:", plan, file=sys.stderr)
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


def apply_resolved_training_hyperparameters_to_work_items(
        work_items, args, *, log=None):
    """Inject resolved per-model training knobs into work item hyperparameters.

    ``resolve_local_parallelism_args`` owns hardware-dependent CLI resolution.
    Trainers should call this once after constructing work items so pan-allele,
    allele-specific, and future affinity trainers all persist the same resolved
    settings in component-model hyperparameters.
    """
    if getattr(args, "dataloader_num_workers", None) is not None:
        apply_dataloader_num_workers_to_work_items(
            work_items, int(args.dataloader_num_workers), log=log,
        )
    if getattr(args, "random_negative_pool_epochs", None) is not None:
        apply_random_negative_pool_epochs_to_work_items(
            work_items, int(args.random_negative_pool_epochs), log=log,
        )

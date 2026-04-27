"""
Infrastructure for "local" parallelism, i.e. multiprocess parallelism on one
compute node.
"""

import itertools
import logging
import multiprocessing
import multiprocessing.pool
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
# to budget how many workers fit on each GPU. The pan-allele MLP's steady-state
# VRAM is lower than this, but first-epoch compile / validation / allocator
# transients vary across hardware and torch versions. Use a conservative
# default that gives 2 workers on 80 GB cards and 1 worker on 40 GB cards.
# Override with ``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB`` if a
# different model family or batch size lands.
_AUTO_MWPG_PER_WORKER_GB_DEFAULT = 16.0

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


def auto_max_workers_per_gpu(num_jobs, num_gpus, backend="auto"):
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

    The result is logged so the chosen value is visible in the worker
    log alongside the reasoning.
    """
    if not num_gpus or num_gpus < 1 or backend == "cpu":
        return 1

    per_worker_gb = float(os.environ.get(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB",
        str(_AUTO_MWPG_PER_WORKER_GB_DEFAULT),
    ))
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

    by_jobs = max(1, int(num_jobs) // max(int(num_gpus), 1))
    by_vram = max(1, int(free_vram_gb_used * 0.6 / per_worker_gb))
    chosen = max(1, min(by_jobs, by_vram, hard_cap))

    logging.info(
        "auto_max_workers_per_gpu: chose %d "
        "(num_jobs=%d, num_gpus=%d, by_jobs=%d, by_vram=%d at "
        "free_vram=%.1f GB%s / %.1f GB/worker, hard_cap=%d)",
        chosen,
        int(num_jobs),
        int(num_gpus),
        by_jobs,
        by_vram,
        free_vram_gb_used,
        "" if free_vram_gb is not None else " (fallback)",
        per_worker_gb,
        hard_cap,
    )
    return chosen


def resolve_max_workers_per_gpu(args):
    """Resolve ``args.max_workers_per_gpu`` to an int, mutating ``args``.

    Accepts the literal string ``"auto"`` (the default) or an int. When
    ``"auto"``, calls ``auto_max_workers_per_gpu`` with the rest of the
    args' parallelism config to pick a value. Idempotent — calling
    twice on the same args is a no-op the second time.

    Returns the resolved int (also stored on ``args.max_workers_per_gpu``
    so subsequent consumers see the int).
    """
    value = getattr(args, "max_workers_per_gpu", None)
    if value is None:
        value = "auto"
    if isinstance(value, str) and value.lower() == "auto":
        resolved = auto_max_workers_per_gpu(
            num_jobs=getattr(args, "num_jobs", 0),
            num_gpus=getattr(args, "gpus", 0) or 0,
            backend=getattr(args, "backend", "auto"),
        )
    else:
        resolved = int(value)
    args.max_workers_per_gpu = resolved
    return resolved


def resolve_local_parallelism_args(args, cap_auto_num_jobs=True):
    """Resolve and normalize local parallelism arguments in one place.

    This is the single pre-fork normalization point for local worker pools:

    * resolves ``--max-workers-per-gpu=auto`` without touching CUDA;
    * when that value was auto, caps ``--num-jobs`` to the resolved GPU
      capacity so an oversized planning value does not silently create CPU
      overflow workers;
    * hoists torch.compile's worker-thread cap before the Pool forks.

    Explicit numeric ``--max-workers-per-gpu`` keeps the historical scheduler
    behavior: if ``num_jobs`` exceeds GPU capacity, overflow workers run on CPU.
    """
    if getattr(args, "_local_parallelism_args_resolved", False):
        return args

    original = getattr(args, "max_workers_per_gpu", None)
    was_auto = (
        original is None
        or (isinstance(original, str) and original.lower() == "auto")
    )
    resolved = resolve_max_workers_per_gpu(args)
    args.max_workers_per_gpu_was_auto = was_auto

    num_jobs = int(getattr(args, "num_jobs", 0) or 0)
    num_gpus = int(getattr(args, "gpus", 0) or 0)
    backend = normalize_pytorch_backend(getattr(args, "backend", "auto") or "auto")
    if (
            cap_auto_num_jobs
            and was_auto
            and num_jobs > 0
            and num_gpus > 0
            and backend in ("auto", "gpu")):
        gpu_capacity = num_gpus * int(resolved)
        if num_jobs > gpu_capacity:
            print(
                "Local parallelism: capping num_jobs from %d to %d because "
                "--max-workers-per-gpu=auto resolved to %d across %d GPU(s). "
                "Set MAX_WORKERS_PER_GPU=N / --max-workers-per-gpu N to allow "
                "explicit CPU overflow workers." % (
                    num_jobs, gpu_capacity, resolved, num_gpus,
                )
            )
            args.num_jobs = gpu_capacity

    hoist_torchinductor_compile_threads(args)
    args._local_parallelism_args_resolved = True
    return args


# Inductor's compile worker pool defaults to ``os.cpu_count()``; that's
# fine for one process but stacks badly when N fit() workers each spawn
# their own pool. Match the same SM-scheduler-style cap used for
# ``auto_max_workers_per_gpu``: 4 is enough compile parallelism to keep
# Inductor useful without amplifying jitter across N workers.
_INDUCTOR_THREAD_HARD_CAP = 4


# Estimated per-fit() Layer-2 SHM footprint. Empirically ~2–3 GB for the
# pan-allele MLP at the standard data scale (x_peptide + x_allele +
# y_encoded + sample_weights + random_negative buffers). We round up to
# 4 GB so the safety check has margin for outliers (longer max_length,
# bigger random-negative pool, future hyperparameter shifts).
_PER_FIT_SHM_FOOTPRINT_GB_DEFAULT = 4.0


def shm_free_gb(shm_dir="/dev/shm"):
    """Return free space (gigabytes) on ``shm_dir`` tmpfs, or ``None`` if unknown.

    Returns ``None`` (not 0) on platforms without ``/dev/shm`` so callers
    can distinguish "no /dev/shm" from "/dev/shm is full". macOS has no
    /dev/shm; container runtimes always do.
    """
    if not os.path.isdir(shm_dir):
        return None
    try:
        st = os.statvfs(shm_dir)
    except OSError:
        return None
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


def shm_total_gb(shm_dir="/dev/shm"):
    """Return total size (gigabytes) of ``shm_dir`` tmpfs, or ``None``."""
    if not os.path.isdir(shm_dir):
        return None
    try:
        st = os.statvfs(shm_dir)
    except OSError:
        return None
    return (st.f_blocks * st.f_frsize) / (1024 ** 3)


def fit_shm_capacity_check(num_workers, per_fit_gb=None, shm_dir="/dev/shm"):
    """Decide whether Layer-2 SHM is safe at this concurrency.

    Returns a dict::

        {"safe": bool,
         "shm_total_gb": float | None,
         "shm_free_gb": float | None,
         "estimated_required_gb": float,
         "message": str}

    Used by the orchestrator pre-fork (loud warning) and by fit()'s
    per-worker auto-detect (silent fallback). The threshold:
    ``num_workers * per_fit_gb * 1.5`` (50% margin for transient peaks,
    DataLoader semaphores, OpenMP per-process registrations).

    On platforms without /dev/shm (macOS), returns ``safe=True`` —
    PyTorch's file_descriptor sharing strategy bypasses the tmpfs so
    no capacity check applies.
    """
    if per_fit_gb is None:
        per_fit_gb = float(
            os.environ.get(
                "MHCFLURRY_PER_FIT_SHM_FOOTPRINT_GB",
                _PER_FIT_SHM_FOOTPRINT_GB_DEFAULT,
            )
        )
    free_gb = shm_free_gb(shm_dir)
    total_gb = shm_total_gb(shm_dir)
    required_gb = max(int(num_workers), 1) * per_fit_gb * 1.5
    if free_gb is None:
        # No /dev/shm visible; assume PyTorch's file_descriptor strategy
        # will handle sharing. Caller treats as safe.
        return {
            "safe": True,
            "shm_total_gb": total_gb,
            "shm_free_gb": free_gb,
            "estimated_required_gb": required_gb,
            "message": (
                "%s not present; skipping Layer-2 SHM capacity check"
                % shm_dir
            ),
        }
    safe = free_gb >= required_gb
    if safe:
        message = (
            "Layer-2 SHM capacity OK: %s has %.1f GB free / %.1f GB total, "
            "estimated need %.1f GB (%d workers × %.1f GB/fit × 1.5 margin)"
            % (
                shm_dir, free_gb, total_gb or 0.0, required_gb,
                num_workers, per_fit_gb,
            )
        )
    else:
        message = (
            "Layer-2 SHM capacity TIGHT: %s has %.1f GB free / %.1f GB "
            "total, estimated need %.1f GB (%d workers × %.1f GB/fit × "
            "1.5 margin). Workers will fall back to numpy DataLoader path "
            "(slower, no per-worker share). To re-enable: bump tmpfs "
            "(e.g. relaunch the container with --shm-size=64g) or reduce "
            "--num-jobs / --max-workers-per-gpu. Force-on with "
            "MHCFLURRY_FIT_DATALOADER_SHM=1 (will OOM mid-fit)."
            % (
                shm_dir, free_gb, total_gb or 0.0, required_gb,
                num_workers, per_fit_gb,
            )
        )
    return {
        "safe": safe,
        "shm_total_gb": total_gb,
        "shm_free_gb": free_gb,
        "estimated_required_gb": required_gb,
        "message": message,
    }


def configure_torch_sharing_strategy_for_capacity(
    num_workers,
    per_fit_gb=None,
    shm_dir="/dev/shm",
):
    """Switch torch's tensor-sharing strategy to ``file_descriptor`` when
    ``/dev/shm`` is too small for the default ``file_system`` strategy.

    Background: torch's default ``file_system`` strategy backs shared
    tensors with POSIX-shm files in ``/dev/shm``. With many fit() workers
    × multi-GB tensors, an 8 GB Docker-default ``/dev/shm`` runs out and
    ``share_memory_()`` crashes with ``OSError [Errno 28]``. The
    ``file_descriptor`` strategy passes anonymous FDs over Unix sockets
    instead — no ``/dev/shm`` use at all, just an FD per shared tensor.

    Returns one of:
      * ``"unchanged"`` — capacity is fine OR no torch available; default
        strategy retained.
      * ``"file_descriptor"`` — capacity tight, switched. Layer-2 SHM
        works without ``/dev/shm`` headroom.
      * ``"failed"`` — capacity tight but switch failed (e.g. torch
        already initialized sharing); caller should fall back to
        disabling Layer-2 SHM.

    Idempotent. Safe to call multiple times. Bumps ``RLIMIT_NOFILE`` to
    cover the FDs the file_descriptor strategy needs (typically a few
    hundred for 16-worker configs).

    Should be called BEFORE any worker spawns or any tensor is shared,
    typically from the orchestrator's pre-flight + from
    ``class1_neural_network`` at module import time so fit() workers
    that don't go through the orchestrator hook (tests, allele-specific
    training, etc.) still benefit.
    """
    result = fit_shm_capacity_check(num_workers, per_fit_gb, shm_dir)
    if result["safe"]:
        return "unchanged"

    try:
        import torch.multiprocessing as torch_mp
    except ImportError:
        return "unchanged"

    try:
        current = torch_mp.get_sharing_strategy()
    except RuntimeError:
        # Torch unable to query; bail.
        return "unchanged"

    if current == "file_descriptor":
        # Already switched (idempotent path).
        return "file_descriptor"

    try:
        torch_mp.set_sharing_strategy("file_descriptor")
    except (RuntimeError, ValueError, AssertionError):
        # Strategy may be locked once any tensor has been shared, or the
        # platform may not support file_descriptor (e.g. macOS torch
        # only ships file_system). Fall through to caller's fallback.
        return "failed"

    # file_descriptor needs ~num_workers × num_dataloader_workers × N FDs
    # per fit. Default ulimit -n is 1024 on most Linux distros, which is
    # close to the line for 16-worker configs. Raise to 16384 if the
    # hard cap allows.
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(16384, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError, ImportError):
        # macOS sometimes refuses RLIMIT_NOFILE bumps; that's fine,
        # macOS doesn't have /dev/shm and won't hit this path anyway.
        pass

    print(
        "torch.multiprocessing: switched sharing strategy to "
        "'file_descriptor' (was '%s'). Layer-2 SHM tensors will now use "
        "anonymous FDs instead of /dev/shm; the capacity guard above is "
        "no longer load-bearing." % current
    )
    return "file_descriptor"


def hoist_torchinductor_compile_threads(args):
    """Cap ``TORCHINDUCTOR_COMPILE_THREADS`` based on parallel-worker count.

    ``torch.compile`` (when enabled via ``MHCFLURRY_TORCH_COMPILE=1``)
    spins up an inductor compile worker pool that defaults to
    ``os.cpu_count()`` threads. With N fit() workers each running
    their own compile pool, an 8-job × 64-core box would spawn 512
    compile threads — orders of magnitude over-subscribed.

    The orchestrator owns "how many workers will exist", so it owns
    the env knob too: set once before forking, every worker inherits.
    Skips the hoist when the user has already pinned the value or when
    ``MHCFLURRY_TORCH_COMPILE`` isn't on. Cluster workers running on
    other hosts inherit nothing — but they share the same kernel cache
    via inductor's content-addressed FX cache, so per-worker compile
    counts there are bounded by cache hits, not by env.

    Lives here (not in any one ``train_*_command`` module) so processing,
    allele-specific, and any future train command can call it the same
    way.
    """
    if os.environ.get("MHCFLURRY_TORCH_COMPILE", "0") != "1":
        # No compile = no compile pool to size; leave env untouched.
        return
    auto_owned = (
        os.environ.get("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO") == "1"
    )
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
    threads = max(1, min(_INDUCTOR_THREAD_HARD_CAP, cpu_count_ // num_jobs))
    if auto_owned and os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") == str(threads):
        return
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(threads)
    os.environ["MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO"] = "1"
    print(
        "torch.compile: hoisted TORCHINDUCTOR_COMPILE_THREADS=%d "
        "(num_jobs=%d, cpu_count=%d)" % (threads, num_jobs, cpu_count_)
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
        default=0,
        type=int,
        metavar="N",
        help="Number of local processes to parallelize training over. "
             "Set to 0 for serial run. Default: %(default)s.")
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

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
from multiprocessing import Queue, cpu_count
from multiprocessing.util import Finalize
from pprint import pprint
import random

import numpy

from .common import configure_pytorch, normalize_pytorch_backend


# Per-worker VRAM upper bound (gigabytes) used by ``auto_max_workers_per_gpu``
# to budget how many workers fit on each GPU. The pan-allele MLP at the
# default minibatch=4096 + RMSprop optimizer state + activations was measured
# at ~5.5 GB on the 2026-04-25 8xA100 run; rounded up to 8 GB to leave
# headroom for the largest architecture in the sweep + per-batch peaks.
# Override with ``MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB`` if a
# different model family or batch size lands.
_AUTO_MWPG_PER_WORKER_GB_DEFAULT = 8.0

# SM-scheduler ceiling. Beyond ~4 workers/GPU the kernel queue serializes
# behind a single SM scheduler, so per-worker throughput drops faster than
# you gain from more parallelism. Tunable via env, but the empirical sweet
# spot is 2-4 for small MLP workloads.
_AUTO_MWPG_HARD_CAP_DEFAULT = 4

# Fallback free-VRAM estimate per GPU (gigabytes) when ``torch.cuda.mem_get_info``
# isn't callable yet (e.g. we're picking the pool size before any worker has
# initialized CUDA). 16 GB is the common-denominator across A100-40GB / V100 /
# L40S / A100-80GB at full freshness — conservative; real runs will see more.
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

    Free VRAM is read live from ``torch.cuda.mem_get_info`` per GPU when
    CUDA is initialized; falls back to a conservative 16 GB estimate
    otherwise. Per-worker VRAM upper bound and the hard cap are both
    overridable via the env vars
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

    free_vram_gb = None
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            mins = []
            for i in range(min(int(num_gpus), torch.cuda.device_count())):
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                    mins.append(free / (1024 ** 3))
                except (RuntimeError, AssertionError):
                    # mem_get_info can raise before the device is selected.
                    pass
            if mins:
                free_vram_gb = min(mins)
    except Exception:  # pragma: no cover — diagnostic fallback
        free_vram_gb = None

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


# Inductor's compile worker pool defaults to ``os.cpu_count()``; that's
# fine for one process but stacks badly when N fit() workers each spawn
# their own pool. Match the same SM-scheduler-style cap used for
# ``auto_max_workers_per_gpu``: 4 is enough compile parallelism to keep
# Inductor useful without amplifying jitter across N workers.
_INDUCTOR_THREAD_HARD_CAP = 4


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
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
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
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(threads)
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
    resolve_max_workers_per_gpu(args)

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

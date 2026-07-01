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

"""Multiprocessing worker-pool scheduling helpers."""

import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import sys
from multiprocessing import cpu_count
from pprint import pprint

import numpy

from ..common import configure_pytorch, normalize_pytorch_backend
from ..workload_planning import WORKLOAD_GENERIC
from .planning import _cuda_visible_devices_from_env, resolve_local_parallelism_args
from .worker_runtime import worker_init, worker_init_entry_point


# ---- Non-daemonic worker pool --------------------------------------------
#
# By default ``multiprocessing.Pool`` spawns daemon workers, and daemon
# processes cannot fork their own children. Streaming pretraining can use
# PyTorch DataLoader workers to prefetch batches, so mhcflurry's outer training
# pool must use non-daemonic workers. The runtime fallback in
# ``class1_training._effective_num_workers`` still downgrades to
# ``num_workers=0`` if an external caller uses a daemon process, but production
# training should keep prefetch available by using this pool.
#
# Non-daemon workers have one behavioral difference worth naming: if the
# parent process dies ungracefully (e.g. SIGKILL), the workers may
# linger as zombies rather than being auto-reaped by init. The
# training orchestrator's ``try/finally`` closes and joins the pool on
# clean exit, so this only matters under unusual fault modes.
class _NonDaemonProcessMixin:
    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value) -> None:
        # Silently ignore; ``multiprocessing.Pool._repopulate_pool`` sets
        # daemon=True on every fresh worker, so we must tolerate the
        # assignment without raising.
        pass


class NonDaemonProcess(_NonDaemonProcessMixin, multiprocessing.Process):
    """A ``multiprocessing.Process`` whose ``daemon`` flag cannot be set.

    Reading ``.daemon`` always returns False; writes are no-ops. This
    lets us instantiate ``multiprocessing.pool.Pool`` with a worker
    class that declines to be a daemon, so the DataLoader inside each
    worker can spawn its own prefetch children.
    """


class NonDaemonContext(type(multiprocessing.get_context())):
    """A multiprocessing context that hands out ``NonDaemonProcess`` workers.

    Subclasses the current default context so the start method (fork on
    Linux, spawn on macOS) is preserved — we only swap the Process
    class. The Pool uses ``self._ctx.Process(...)`` to create workers
    and will now get our non-daemonic variant.
    """

    Process = NonDaemonProcess


class NonDaemonSpawnProcess(
        _NonDaemonProcessMixin, multiprocessing.context.SpawnProcess):
    pass


class NonDaemonSpawnContext(multiprocessing.context.SpawnContext):
    Process = NonDaemonSpawnProcess


if hasattr(multiprocessing.context, "ForkProcess"):
    class NonDaemonForkProcess(
            _NonDaemonProcessMixin, multiprocessing.context.ForkProcess):
        pass


    class NonDaemonForkContext(multiprocessing.context.ForkContext):
        Process = NonDaemonForkProcess
else:
    NonDaemonForkContext = None


if hasattr(multiprocessing.context, "ForkServerProcess"):
    class NonDaemonForkServerProcess(
            _NonDaemonProcessMixin,
            multiprocessing.context.ForkServerProcess):
        pass


    class NonDaemonForkServerContext(
            multiprocessing.context.ForkServerContext):
        Process = NonDaemonForkServerProcess
else:
    NonDaemonForkServerContext = None


_NON_DAEMON_CONTEXT_BY_START_METHOD = {
    "spawn": NonDaemonSpawnContext,
}
if NonDaemonForkContext is not None:
    _NON_DAEMON_CONTEXT_BY_START_METHOD["fork"] = NonDaemonForkContext
if NonDaemonForkServerContext is not None:
    _NON_DAEMON_CONTEXT_BY_START_METHOD["forkserver"] = (
        NonDaemonForkServerContext)


def non_daemon_context(start_method=None):
    """Return a multiprocessing context whose workers are non-daemonic."""
    if start_method is None:
        return NonDaemonContext()
    try:
        context_class = _NON_DAEMON_CONTEXT_BY_START_METHOD[start_method]
    except KeyError:
        raise ValueError(
            "Unsupported multiprocessing start_method: %s" % (
                start_method,)) from None
    return context_class()


class NonDaemonPool(multiprocessing.pool.Pool):
    """A ``multiprocessing.Pool`` that runs non-daemonic workers.

    Pool's constructor takes a ``context`` kwarg — we thread a
    ``NonDaemonContext`` through so each worker is a
    ``NonDaemonProcess``. Everything else (apply_async, imap, etc.)
    inherits unchanged.
    """

    def __init__(self, *args, **kwargs):
        start_method = kwargs.pop("start_method", None)
        if start_method is not None and "context" in kwargs:
            raise ValueError("Pass either context or start_method, not both")
        # Callers may pass their own context; if not, use our non-daemon one.
        kwargs.setdefault("context", non_daemon_context(start_method))
        super().__init__(*args, **kwargs)

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


def worker_pool_with_gpu_assignments_from_args(
        args,
        workload_name=WORKLOAD_GENERIC,
        workload_hints=None,
        start_method=None):
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
    resolve_local_parallelism_args(
        args,
        workload_name=workload_name,
        workload_hints=workload_hints,
    )

    # --gpus only takes effect when there are worker processes to assign it to.
    # A serial run (num_jobs == 0) ignores it. Warn when the user *explicitly*
    # asked for GPUs but ends up serial, so "--gpus 4 --num-jobs 0" doesn't
    # silently drop the GPU request. (The auto path legitimately resolves gpus
    # alongside num_jobs == 0 on CPU-only boxes, so don't warn there.)
    if (args.num_jobs == 0
            and args.gpus
            and not getattr(args, "gpus_was_auto", False)):
        print(
            "Warning: --gpus %d is ignored because num_jobs resolved to 0 "
            "(serial run). Pass --num-jobs > 0 to fan out across the "
            "requested GPUs." % args.gpus,
            file=sys.stderr)

    return worker_pool_with_gpu_assignments(
        num_jobs=args.num_jobs,
        num_gpus=args.gpus,
        backend=args.backend,
        max_workers_per_gpu=args.max_workers_per_gpu,
        max_tasks_per_worker=args.max_tasks_per_worker,
        worker_log_dir=args.worker_log_dir,
        start_method=start_method,
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

def worker_pool_with_gpu_assignments(
        num_jobs,
        num_gpus=0,
        backend=None,
        max_workers_per_gpu=1,
        max_tasks_per_worker=None,
        worker_log_dir=None,
        start_method=None):
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
    start_method : string
        Optional multiprocessing start method, e.g. ``"spawn"`` when workers
        must not inherit PyTorch state from the parent process.

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
                num_jobs, num_gpus, max_workers_per_gpu),
            file=sys.stderr)
        for (worker_num, kwargs) in enumerate(worker_init_kwargs):
            print(
                "Worker %d assigned backend=%s GPUs=%s" % (
                    worker_num,
                    kwargs["backend"],
                    kwargs.get("gpu_device_nums")),
                file=sys.stderr)

    if worker_log_dir:
        os.makedirs(worker_log_dir, exist_ok=True)
        for kwargs in worker_init_kwargs:
            kwargs["worker_log_dir"] = worker_log_dir

    worker_pool = make_worker_pool(
        processes=num_jobs,
        initializer=worker_init,
        initializer_kwargs_per_process=worker_init_kwargs,
        max_tasks_per_worker=max_tasks_per_worker,
        start_method=start_method)
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
    if num_gpus and num_jobs > 0:
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

    cuda_visible_devices = _cuda_visible_devices_from_env()
    if cuda_visible_devices is None:
        gpu_device_nums = list(range(num_gpus))
    else:
        gpu_device_nums = cuda_visible_devices[:num_gpus]
        if len(gpu_device_nums) < num_gpus:
            logging.warning(
                "num_gpus=%d exceeds CUDA_VISIBLE_DEVICES=%r; assigning "
                "only the %d scheduler-visible GPU(s)",
                num_gpus,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
                len(gpu_device_nums),
            )

    gpu_assignments = list(itertools.chain.from_iterable(
        gpu_device_nums for _ in range(max_workers_per_gpu)))

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
        max_tasks_per_worker=None,
        start_method=None):
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
        Restart workers after this many tasks.

    start_method : string, optional
        Multiprocessing start method to use for the worker pool.

    Returns
    -------
    multiprocessing.Pool
    """

    if not processes:
        processes = cpu_count()

    pool_context = non_daemon_context(start_method) if start_method else None
    pool_kwargs = {
        'processes': processes,
    }
    if max_tasks_per_worker:
        pool_kwargs["maxtasksperchild"] = max_tasks_per_worker
    if start_method:
        pool_kwargs["context"] = pool_context

    if initializer:
        if initializer_kwargs_per_process:
            assert len(initializer_kwargs_per_process) == processes
            queue_context = pool_context or multiprocessing.get_context()
            kwargs_queue = queue_context.Queue()
            kwargs_queue_backup = queue_context.Queue()
            for kwargs in initializer_kwargs_per_process:
                kwargs_queue.put(kwargs)
                kwargs_queue_backup.put(kwargs)
            pool_kwargs["initializer"] = worker_init_entry_point
            pool_kwargs["initargs"] = (
                initializer, kwargs_queue, kwargs_queue_backup)
        else:
            pool_kwargs["initializer"] = initializer

    # Use a non-daemonic pool so workers can spawn DataLoader children.
    # See NonDaemonPool for the rationale.
    worker_pool = NonDaemonPool(**pool_kwargs)
    print("Started pool: %s" % str(worker_pool), file=sys.stderr)
    pprint(pool_kwargs, stream=sys.stderr)
    return worker_pool

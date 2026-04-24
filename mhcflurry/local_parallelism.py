"""
Infrastructure for "local" parallelism, i.e. multiprocess parallelism on one
compute node.
"""

import itertools
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
        type=int,
        metavar="N",
        default=1000,
        help="Maximum number of workers to assign to a GPU. Additional tasks will "
             "run on CPU.")
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

    Parameters
    ----------
    args : argparse.ArgumentParser

    Returns
    -------
    multiprocessing.Pool
    """

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
        sys.stderr = sys.stdout = open(os.path.join(
            worker_log_dir,
            "LOG-worker.%d.%d.txt" % (os.getpid(), int(time.time()))), "w")

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

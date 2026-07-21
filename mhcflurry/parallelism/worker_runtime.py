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

"""Worker initialization and error-wrapping helpers."""

import importlib
import os
import queue
import random
import sys
import time
import traceback
from multiprocessing.util import Finalize

import numpy
from threadpoolctl import threadpool_limits

from ..common import configure_pytorch
from .planning import resolved_int


_WORKER_THREADPOOL_LIMITER = None


def install_worker_context(module_name, data):
    """Install one process-wide constant-data mapping for worker functions."""
    module = importlib.import_module(module_name)
    target = getattr(module, "WORKER_CONTEXT", None)
    if not isinstance(target, dict):
        raise ValueError("%s.WORKER_CONTEXT must be a dict" % module_name)
    if data is None:
        raise ValueError(
            "worker_context_data is required with worker_context_module"
        )
    # Under fork this may be the exact inherited dictionary. Under spawn it is
    # a separately unpickled object that must be installed in the module-level
    # dictionary captured by worker function defaults.
    if target is not data:
        target.clear()
        target.update(data)
    return target


def configure_worker_cpu_threads(num_threads, auto_owned=True):
    """Apply a fit worker's CPU-thread budget to loaded native runtimes.

    Forked workers inherit already-initialized BLAS/OpenMP pools, so setting
    only environment variables in the parent is insufficient. Keep the
    environment synchronized for libraries loaded later and use threadpoolctl
    to resize the native pools already present in this process. Caller-owned
    OMP/MKL/OpenBLAS settings are authoritative and must not be replaced by a
    uniform auto-derived limit.
    """
    global _WORKER_THREADPOOL_LIMITER
    if num_threads is None or not auto_owned:
        return None
    num_threads = resolved_int(num_threads, "cpu_threads_per_worker")
    if num_threads < 1:
        raise ValueError("cpu_threads_per_worker must be at least 1")
    for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(num_threads)
    _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=num_threads)
    return num_threads


def worker_init_entry_point(
        init_function, arg_queue=None, backup_arg_queue=None,
        shared_kwargs=None):
    kwargs = {}
    if arg_queue:
        try:
            kwargs = arg_queue.get(block=False)
        except queue.Empty:
            print(
                "Argument queue empty. Using round robin arg queue.",
                file=sys.stderr)
            kwargs = backup_arg_queue.get(block=True)
            backup_arg_queue.put(kwargs)

        # On exit we add the init args back to the queue so restarted workers
        # (e.g. when when running with maxtasksperchild) will pickup init
        # arguments from a previously exited worker.
        # Keep a separate small mapping for worker replacement. ``kwargs`` is
        # extended with shared context below; registering the same mutable
        # object would make every exiting worker enqueue the entire dataset.
        # With no reader during pool shutdown, that fills the pipe and hangs
        # join() indefinitely.
        restart_kwargs = dict(kwargs)
        Finalize(None, arg_queue.put, (restart_kwargs,), exitpriority=1)

    print("Initializing worker: %s" % str(kwargs), file=sys.stderr)
    if shared_kwargs:
        overlap = set(kwargs).intersection(shared_kwargs)
        if overlap:
            raise ValueError(
                "Worker initializer arguments supplied as both shared and "
                "per-process values: %s" % ", ".join(sorted(overlap))
            )
        kwargs.update(shared_kwargs)
    init_function(**kwargs)


def worker_init(
        keras_backend=None, backend=None, gpu_device_nums=None,
        worker_log_dir=None, max_workers_per_gpu=None,
        cpu_threads_per_worker=None,
        cpu_threads_per_worker_was_auto=True,
        worker_context_module=None, worker_context_data=None):
    del keras_backend  # legacy argument retained for API compatibility
    if worker_log_dir:
        os.makedirs(worker_log_dir, exist_ok=True)
        # Line buffering (buffering=1) ensures every print/traceback flushes
        # to disk immediately. Without it, Python defaults to ~8 KB block
        # buffering on a regular file, which silently swallows worker
        # output if the worker is killed (OOM, signal) before the buffer
        # fills. With block buffering, a worker spinning at 99% CPU and
        # then OOM-killed leaves a 0-byte LOG file because no flush
        # ever fires before the worker dies — line buffering avoids that.
        sys.stderr = sys.stdout = open(os.path.join(
            worker_log_dir,
            "LOG-worker.%d.%d.txt" % (os.getpid(), int(time.time()))),
            "w", buffering=1)

    # Each worker needs distinct random numbers
    numpy.random.seed()
    random.seed()
    cpu_threads_per_worker = configure_worker_cpu_threads(
        cpu_threads_per_worker,
        auto_owned=cpu_threads_per_worker_was_auto,
    )
    configure_kwargs = {"backend": backend}
    if gpu_device_nums is not None:
        configure_kwargs["gpu_device_nums"] = gpu_device_nums
    if cpu_threads_per_worker is not None:
        configure_kwargs["num_threads"] = cpu_threads_per_worker
    if gpu_device_nums is not None:
        print("WORKER pid=%d assigned GPU devices: %s" % (
            os.getpid(), gpu_device_nums), file=sys.stderr)
    configure_pytorch(**configure_kwargs)
    # Reseed torch's global RNG too (numpy/random above don't touch it).
    # Torch RNG drives weight init and any tensor op without an explicit
    # generator; without this, forked workers would share a stream. Seed
    # from the freshly-reseeded numpy RNG so the value is per-worker
    # distinct. Done after configure_pytorch so any visible CUDA device is
    # seeded; a no-op for CUDA generators when the worker is CPU-pinned.
    import torch
    torch.manual_seed(int(numpy.random.randint(0, 2 ** 31 - 1)))
    # Install model/data context only after the worker's runtime and GPU
    # assignment are configured. This keeps context initialization from
    # accidentally creating runtime state against the parent's device view.
    if worker_context_module is not None:
        install_worker_context(worker_context_module, worker_context_data)
    # Propagate workers-per-GPU into the process env so auto-sized
    # batching (Class1NeuralNetwork.predict / fit's
    # check_training_batch_fits) can partition VRAM correctly when
    # multiple fit()/predict() calls are co-resident on one GPU.
    if max_workers_per_gpu is not None:
        os.environ["MHCFLURRY_MAX_WORKERS_PER_GPU"] = str(
            resolved_int(max_workers_per_gpu, "max_workers_per_gpu"))


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

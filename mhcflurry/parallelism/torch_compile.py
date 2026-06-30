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

"""Torch compile environment and cache-warmup helpers."""

import os
import sys
import time

from ..common import normalize_pytorch_backend
from ..workload_planning import env_int
from .planning import (
    num_workers_per_gpu_from_args,
    resolve_local_parallelism_args,
)
from .worker_pool import worker_pool_uses_fork, worker_pool_with_gpu_assignments
from .worker_runtime import call_wrapped_kwargs


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
        cap = env_int(
            "MHCFLURRY_TORCHINDUCTOR_WARMUP_COMPILE_THREADS_CAP",
            _INDUCTOR_WARMUP_THREAD_HARD_CAP,
            bounds=(1, None),
        )
        return max(1, min(cap, cpu_count_))
    cap = env_int(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_CAP",
        _INDUCTOR_THREAD_HARD_CAP,
        bounds=(1, None),
    )
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


def _compile_threads_num_jobs(num_jobs):
    if num_jobs is None:
        return 1
    if isinstance(num_jobs, str) and num_jobs.strip().lower() == "auto":
        return 1
    return max(int(num_jobs), 1)


def resolve_torchinductor_compile_threads_env(num_jobs=1, phase="production"):
    """Ensure ``TORCHINDUCTOR_COMPILE_THREADS`` is parseable by PyTorch.

    MHCflurry launchers accept ``TORCHINDUCTOR_COMPILE_THREADS=auto`` as an
    orchestrator-owned sentinel so each command can size the Inductor compiler
    helper pool after it knows its local worker count. PyTorch itself does not
    accept that sentinel: Inductor parses the env var with ``int(...)``.

    This helper is for entry points that do not run the full local-parallelism
    resolver before importing or spawning PyTorch work. It leaves unset and
    user-pinned integer values alone, resolves orchestrator-owned ``auto`` to a
    numeric value, and fails early on other invalid values with a clearer
    message than the downstream Inductor traceback.
    """
    value = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if value in (None, ""):
        return None
    auto_owned = (
        value == "auto"
        or os.environ.get("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO") == "1"
    )
    if auto_owned:
        return _set_auto_torchinductor_compile_threads(
            num_jobs=_compile_threads_num_jobs(num_jobs),
            phase=phase,
        )
    try:
        int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "TORCHINDUCTOR_COMPILE_THREADS must be an integer or the "
            "MHCflurry-owned sentinel 'auto'; got %r" % (value,)
        )
    return int(value)


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
        resolve_torchinductor_compile_threads_env(num_jobs=1)
        return
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ and not (
            _compile_threads_env_is_auto_owned()):
        resolve_torchinductor_compile_threads_env(num_jobs=1)
        return
    workers_per_node = env_int(
        "MHCFLURRY_CLUSTER_WORKERS_PER_NODE", "1",
        bounds=(1, None),
    )
    threads = _set_auto_torchinductor_compile_threads(
        num_jobs=max(workers_per_node, 1),
        phase="production",
    )
    print(
        "torch.compile: cluster worker auto-set "
        "TORCHINDUCTOR_COMPILE_THREADS=%d "
        "(MHCFLURRY_CLUSTER_WORKERS_PER_NODE=%d)" % (
            threads, workers_per_node,
        ),
        file=sys.stderr,
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
        # No compile = no compile pool to size, but still normalize any
        # inherited MHCflurry-owned sentinel before a worker imports PyTorch.
        resolve_torchinductor_compile_threads_env(
            num_jobs=getattr(args, "num_jobs", 1),
            phase=phase,
        )
        return
    auto_owned = _compile_threads_env_is_auto_owned()
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ and not auto_owned:
        # User pinned explicitly; don't second-guess.
        resolve_torchinductor_compile_threads_env(
            num_jobs=getattr(args, "num_jobs", 1),
            phase=phase,
        )
        print(
            "torch.compile: TORCHINDUCTOR_COMPILE_THREADS=%s "
            "(user-pinned, orchestrator hoist skipped)"
            % os.environ["TORCHINDUCTOR_COMPILE_THREADS"],
            file=sys.stderr,
        )
        return
    num_jobs = _compile_threads_num_jobs(getattr(args, "num_jobs", 1))
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
        ),
        file=sys.stderr,
    )

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
        ),
        file=sys.stderr,
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
            "(single-worker codegen phase)" % threads,
            file=sys.stderr,
        )

    warmup_pool = None
    try:
        warmup_pool = worker_pool_with_gpu_assignments(
            num_jobs=1,
            num_gpus=1 if int(getattr(args, "gpus", 0) or 0) > 0 else 0,
            backend=backend,
            max_workers_per_gpu=max(num_workers_per_gpu_from_args(args), 1),
            max_tasks_per_worker=len(unique_warmup_items) + 1,
            worker_log_dir=getattr(args, "worker_log_dir", None),
            # Spawn (not fork) for parity with the production pools below;
            # forked CUDA workers break if the parent has touched CUDA.
            start_method="spawn",
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
            ),
            file=sys.stderr,
        )
    finally:
        if warmup_pool is not None:
            warmup_pool.terminate()
            warmup_pool.join()
        hoist_torchinductor_compile_threads(args, phase="production")

    return len(unique_warmup_items)

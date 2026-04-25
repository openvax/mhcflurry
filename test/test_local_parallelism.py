from argparse import ArgumentParser

import multiprocessing
import pytest

from mhcflurry.local_parallelism import (
    NonDaemonPool,
    NonDaemonProcess,
    add_local_parallelism_args,
    validate_worker_pool_args,
    worker_init_kwargs_for_scheduler,
)


def test_worker_init_kwargs_round_robin_across_gpus():
    assert worker_init_kwargs_for_scheduler(
        num_jobs=5,
        num_gpus=2,
        backend="auto",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "gpu", "gpu_device_nums": [0], "max_workers_per_gpu": 2},
        {"backend": "gpu", "gpu_device_nums": [1], "max_workers_per_gpu": 2},
        {"backend": "gpu", "gpu_device_nums": [0], "max_workers_per_gpu": 2},
        {"backend": "gpu", "gpu_device_nums": [1], "max_workers_per_gpu": 2},
        {"backend": "cpu", "gpu_device_nums": [], "max_workers_per_gpu": 2},
    ]


def test_worker_init_kwargs_without_gpu_scheduling_uses_backend():
    assert worker_init_kwargs_for_scheduler(
        num_jobs=3,
        num_gpus=0,
        backend="mps",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "mps", "max_workers_per_gpu": 2},
        {"backend": "mps", "max_workers_per_gpu": 2},
        {"backend": "mps", "max_workers_per_gpu": 2},
    ]


def test_worker_init_kwargs_normalizes_default_backend_alias():
    assert worker_init_kwargs_for_scheduler(
        num_jobs=2,
        num_gpus=0,
        backend="default",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "auto", "max_workers_per_gpu": 2},
        {"backend": "auto", "max_workers_per_gpu": 2},
    ]


def test_worker_init_kwargs_with_gpus_normalizes_default_backend_alias():
    assert worker_init_kwargs_for_scheduler(
        num_jobs=3,
        num_gpus=1,
        backend="default",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "gpu", "gpu_device_nums": [0], "max_workers_per_gpu": 2},
        {"backend": "gpu", "gpu_device_nums": [0], "max_workers_per_gpu": 2},
        {"backend": "cpu", "gpu_device_nums": [], "max_workers_per_gpu": 2},
    ]


def test_backend_default_alias_parses():
    parser = ArgumentParser()
    add_local_parallelism_args(parser)
    args = parser.parse_args(["--backend", "default"])
    assert args.backend == "default"


def test_validate_worker_pool_args_requires_parallelism_for_gpus():
    with pytest.raises(ValueError, match="num_jobs > 0"):
        validate_worker_pool_args(
            num_jobs=0,
            num_gpus=1,
            backend="auto",
            max_workers_per_gpu=1,
        )


def test_validate_worker_pool_args_rejects_non_cuda_backends_for_gpus():
    with pytest.raises(ValueError, match="backend 'auto' or 'gpu'"):
        validate_worker_pool_args(
            num_jobs=2,
            num_gpus=1,
            backend="mps",
            max_workers_per_gpu=1,
        )


def test_validate_worker_pool_args_rejects_invalid_backend():
    with pytest.raises(ValueError, match="Invalid backend"):
        validate_worker_pool_args(
            num_jobs=2,
            num_gpus=0,
            backend="gpuu",
            max_workers_per_gpu=1,
        )


# ---- Non-daemonic pool regression tests ----------------------------------
#
# Phase 1 (#268) needs Pool workers to be non-daemonic so the PyTorch
# DataLoader inside a training worker can spawn its own prefetch
# children. The default multiprocessing.Pool ships daemon workers,
# which makes DataLoader(num_workers>0) raise AssertionError. These
# tests lock down the new NonDaemonPool / NonDaemonProcess behavior.


def _is_daemon_in_pool_worker(_):
    """Run inside a pool worker; returns whether the current process is a daemon."""
    return multiprocessing.current_process().daemon


def test_nondaemonprocess_reports_not_daemon():
    """NonDaemonProcess.daemon must always read as False regardless of writes."""
    p = NonDaemonProcess(target=lambda: None)
    assert p.daemon is False
    # The Pool internals WILL assign .daemon = True on every fresh worker —
    # tolerate the assignment silently without raising or flipping the flag.
    p.daemon = True
    assert p.daemon is False, "daemon setter should have been a no-op"


def test_nondaemonpool_workers_are_not_daemonic():
    """The whole point: Pool workers must report daemon=False."""
    with NonDaemonPool(processes=2) as pool:
        results = pool.map(_is_daemon_in_pool_worker, [None, None, None, None])
    assert all(r is False for r in results), (
        f"expected all workers daemon=False, got {results}. If True values "
        f"appear, the custom NonDaemonContext isn't threading through to "
        f"Pool._repopulate_pool, so DataLoader(num_workers>0) will still "
        f"detonate inside mhcflurry Pool workers."
    )


def _grandchild_entry():
    """Minimal module-level child entry (pickle-able across spawn)."""
    return None


def _spawn_child_from_pool_worker(_):
    """Spawn a grandchild process from within a Pool worker.

    The AssertionError from the Phase 1 (#268) A100 crash reproduces here
    if the Pool worker is daemonic: the inner multiprocessing.Process
    .start() call raises "daemonic processes are not allowed to have
    children". With a non-daemonic worker this succeeds.
    """
    child = multiprocessing.Process(target=_grandchild_entry)
    child.start()
    child.join(timeout=10)
    return child.exitcode


def test_nondaemonpool_worker_can_spawn_children():
    """Non-daemon pool workers must be able to spawn multiprocessing children.

    This is the regression test for the Phase 1 (#268) A100 crash
    where DataLoader(num_workers>0) called from a mhcflurry Pool worker
    raised AssertionError("daemonic processes are not allowed to have
    children"). With NonDaemonPool, the Pool workers are not daemonic,
    so spawning a child succeeds.
    """
    with NonDaemonPool(processes=2) as pool:
        # Without the fix, this would raise from deep in the worker's
        # own multiprocessing machinery with the AssertionError above.
        exit_codes = pool.map(_spawn_child_from_pool_worker, [None, None])
    assert all(code == 0 for code in exit_codes), (
        f"children spawned from pool workers did not exit cleanly: "
        f"{exit_codes}. If this fails, Pool workers have regressed to "
        f"daemonic and the DataLoader prefetch optimization is broken "
        f"in production."
    )

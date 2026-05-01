from argparse import ArgumentParser, ArgumentTypeError, Namespace

import multiprocessing
import builtins
import pytest

from mhcflurry.local_parallelism import (
    NonDaemonPool,
    NonDaemonProcess,
    _max_workers_per_gpu_arg,
    add_local_parallelism_args,
    add_prediction_parallelism_args,
    auto_max_workers_per_gpu,
    chunk_ranges_for_local_parallelism,
    resolve_local_parallelism_args,
    resolve_max_workers_per_gpu,
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


def test_max_workers_per_gpu_arg_parses_auto_and_int():
    assert _max_workers_per_gpu_arg("auto") == "auto"
    assert _max_workers_per_gpu_arg("AUTO") == "auto"
    assert _max_workers_per_gpu_arg("3") == 3
    assert _max_workers_per_gpu_arg(5) == 5


def test_max_workers_per_gpu_arg_rejects_garbage():
    with pytest.raises(ArgumentTypeError):
        _max_workers_per_gpu_arg("hello")
    with pytest.raises(ArgumentTypeError):
        _max_workers_per_gpu_arg("0")
    with pytest.raises(ArgumentTypeError):
        _max_workers_per_gpu_arg("-1")


def test_add_local_parallelism_args_default_is_auto():
    parser = ArgumentParser()
    add_local_parallelism_args(parser)
    args = parser.parse_args([])
    assert args.max_workers_per_gpu == "auto"
    args = parser.parse_args(["--max-workers-per-gpu", "3"])
    assert args.max_workers_per_gpu == 3


def test_add_prediction_parallelism_args_omits_training_only_flags():
    parser = ArgumentParser()
    add_prediction_parallelism_args(parser)
    args = parser.parse_args([])
    assert args.num_jobs == "auto"
    assert args.max_workers_per_gpu == "auto"
    assert not hasattr(args, "dataloader_num_workers")
    assert not hasattr(args, "random_negative_pool_epochs")


def test_chunk_ranges_for_local_parallelism_are_contiguous():
    ranges = chunk_ranges_for_local_parallelism(
        num_items=10, num_jobs=2, chunks_per_worker=2)
    assert ranges == [
        (0, 0, 3),
        (1, 3, 6),
        (2, 6, 9),
        (3, 9, 10),
    ]


def test_chunk_ranges_for_local_parallelism_handles_empty():
    assert chunk_ranges_for_local_parallelism(0, num_jobs=4) == []


def test_auto_max_workers_per_gpu_cpu_only_returns_one():
    """No GPUs → 1 worker (CPU)."""
    assert auto_max_workers_per_gpu(num_jobs=8, num_gpus=0) == 1
    assert auto_max_workers_per_gpu(num_jobs=8, num_gpus=4, backend="cpu") == 1


def test_auto_max_workers_per_gpu_caps_at_jobs_per_gpu(monkeypatch):
    """``num_jobs // num_gpus`` is the natural ceiling per GPU."""
    # 16 jobs, 8 GPUs, abundant VRAM → 2 (jobs/gpus, not VRAM-bound).
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", "1")
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    assert auto_max_workers_per_gpu(num_jobs=16, num_gpus=8) == 2
    # 32 jobs, 8 GPUs → 4 (hits the hard cap, not jobs/gpus).
    assert auto_max_workers_per_gpu(num_jobs=32, num_gpus=8) == 4


def test_auto_max_workers_per_gpu_caps_at_vram(monkeypatch):
    """Per-worker VRAM upper bound caps when jobs/gpus would oversubscribe."""
    # 32 jobs, 8 GPUs, large per-worker VRAM, low free-VRAM fallback (16 GB)
    # → by_jobs=4, by_vram=floor(16*0.6/16)=0 → max(1,...) = 1.
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", "16")
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "16")
    assert auto_max_workers_per_gpu(num_jobs=32, num_gpus=8) == 1


def test_auto_max_workers_per_gpu_respects_hard_cap(monkeypatch):
    """Hard cap clamps even when jobs/gpus and VRAM allow more."""
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", "0.1")
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP", "3")
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    # 32 jobs/8 gpus = 4 by_jobs, vram unbounded → hard_cap clamps to 3.
    assert auto_max_workers_per_gpu(num_jobs=32, num_gpus=8) == 3


def test_auto_max_workers_per_gpu_does_not_import_torch(monkeypatch):
    """Resolving local Pool size must not initialize CUDA in the parent."""
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("auto_max_workers_per_gpu imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert auto_max_workers_per_gpu(num_jobs=16, num_gpus=8) == 2


def test_resolve_max_workers_per_gpu_passes_through_int():
    args = Namespace(max_workers_per_gpu=4, num_jobs=16, gpus=8, backend="auto")
    assert resolve_max_workers_per_gpu(args) == 4
    assert args.max_workers_per_gpu == 4


def test_resolve_max_workers_per_gpu_resolves_auto(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", "1")
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    args = Namespace(
        max_workers_per_gpu="auto", num_jobs=16, gpus=8, backend="auto"
    )
    resolved = resolve_max_workers_per_gpu(args)
    assert isinstance(resolved, int)
    assert resolved >= 1
    assert args.max_workers_per_gpu == resolved


def test_resolve_max_workers_per_gpu_is_idempotent(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", "1")
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    args = Namespace(
        max_workers_per_gpu="auto", num_jobs=16, gpus=8, backend="auto"
    )
    first = resolve_max_workers_per_gpu(args)
    second = resolve_max_workers_per_gpu(args)
    assert first == second


def test_resolve_local_parallelism_args_caps_auto_num_jobs(monkeypatch):
    # Force a VRAM cap by pinning per-worker GB high so by_vram=1 and the
    # auto MWPG resolves below by_jobs (which would otherwise pin to 2 here).
    # 40 GB free / 0.6 / 24 GB/worker = 1 worker. by_jobs=16/8=2.
    # min(1, 2, 4) = 1 -> num_jobs capped to 8.
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "40")
    monkeypatch.setenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", "24"
    )
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu="auto", num_jobs=16, gpus=8, backend="auto"
    )
    resolve_local_parallelism_args(args)
    assert args.max_workers_per_gpu == 1
    assert args.num_jobs == 8
    assert args.max_workers_per_gpu_was_auto is True


def test_resolve_local_parallelism_args_unlocks_4_per_gpu_on_80gb(
    monkeypatch,
):
    # The post-2026-04-28 default (per_worker=4 GB) lets 80 GB cards
    # resolve to the hard_cap of 4 workers/GPU once num_jobs is also auto:
    # by_vram = floor(0.6 * 80 / 4) = 12, by_jobs skipped, hard_cap=4 wins.
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.delenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", raising=False
    )
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs=0,  # auto -> by_jobs clamp skipped
        gpus=8,
        backend="auto",
    )
    resolve_local_parallelism_args(args)
    assert args.max_workers_per_gpu == 4
    # num_jobs="auto" (here num_jobs=0 in the older signature treated as
    # auto-equivalent for the resolver's CPU-only branch — but with the
    # new resolver, num_jobs=0 is taken at face value). For the new auto
    # path see ``test_resolve_local_parallelism_args_num_jobs_auto``.
    assert args.max_workers_per_gpu_was_auto is True


def test_auto_max_workers_per_gpu_pins_to_by_jobs_when_num_jobs_explicit(
    monkeypatch,
):
    # Production today passes --num-jobs 16. With 8 GPUs and the new
    # 4 GB/worker default + 80 GB free, by_vram=12 and hard_cap=4 — but
    # by_jobs=16//8=2 still wins. This is intentional: explicit num_jobs
    # is a user contract.
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.delenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", raising=False
    )
    chosen = auto_max_workers_per_gpu(num_jobs=16, num_gpus=8, backend="auto")
    assert chosen == 2


def test_resolve_local_parallelism_args_caps_explicit_num_jobs_to_capacity(
    monkeypatch,
):
    # Even with an explicit numeric ``--max-workers-per-gpu``, an
    # oversized ``--num-jobs`` is now capped to GPU capacity rather than
    # silently spilling to CPU. The CPU-overflow workers used to chew on
    # GPU-shaped work items at CPU speed and starve the GPU workers'
    # imap_unordered queue (see release-2.3.0 sweep mb_1024 calibrate
    # hang at 91% on 24 CPU stragglers) so we drop them.
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu=1, num_jobs=16, gpus=8, backend="auto"
    )
    resolve_local_parallelism_args(args)
    assert args.max_workers_per_gpu == 1
    assert args.num_jobs == 8  # 8 GPUs × 1 worker/GPU
    assert args.max_workers_per_gpu_was_auto is False


def test_resolve_local_parallelism_args_num_jobs_auto_resolves_to_capacity(
    monkeypatch,
):
    # ``num_jobs="auto"`` (the new default) resolves to gpus × MWPG so
    # production no longer has to hand-pick a value that may not match
    # the resolver's MWPG choice.
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.delenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", raising=False
    )
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=8,
        backend="auto",
    )
    resolve_local_parallelism_args(args)
    assert args.max_workers_per_gpu == 4  # by_vram=12, hard_cap=4
    assert args.num_jobs == 32  # 8 × 4
    assert args.num_jobs_was_auto is True


def test_resolve_local_parallelism_args_num_jobs_auto_cpu_only(
    monkeypatch,
):
    # CPU-only / no GPUs: num_jobs="auto" resolves to 0 (serial) — caller
    # decides whether to spawn an explicit CPU pool.
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=0,
        backend="cpu",
    )
    resolve_local_parallelism_args(args)
    assert args.num_jobs == 0
    assert args.num_jobs_was_auto is True


def test_resolve_local_parallelism_args_auto_detects_gpus(monkeypatch):
    """When --gpus is unset and backend allows GPU, the resolver
    auto-detects the visible CUDA device count via nvidia-smi -L
    (no torch import in the parent). Without this, parallel prediction
    with --num-jobs N and no --gpus piles every worker onto GPU 0."""
    from mhcflurry import local_parallelism

    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.delenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", raising=False
    )
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP", "4")
    monkeypatch.setattr(
        local_parallelism, "_detect_num_cuda_devices_no_torch", lambda: 4)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=None,        # unset — should be auto-filled
        backend="auto",
    )
    resolve_local_parallelism_args(args)
    assert args.gpus == 4
    assert args.gpus_was_auto is True
    assert args.max_workers_per_gpu == 4
    assert args.num_jobs == 16


def test_resolve_local_parallelism_args_detects_gpus_before_explicit_jobs(
        monkeypatch):
    """Auto MWPG must see auto-detected GPUs before explicit num_jobs is
    capacity-checked; otherwise 16 jobs on 8 GPUs is under-capped to 8."""
    from mhcflurry import local_parallelism

    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.delenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", raising=False
    )
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP", "4")
    monkeypatch.setattr(
        local_parallelism, "_detect_num_cuda_devices_no_torch", lambda: 8)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs=16,
        gpus=None,
        backend="auto",
    )
    resolve_local_parallelism_args(args)
    assert args.gpus == 8
    assert args.max_workers_per_gpu == 2
    assert args.num_jobs == 16


def test_resolve_local_parallelism_args_skips_auto_detect_with_explicit_gpus(
        monkeypatch):
    """An explicit --gpus value isn't overridden by auto-detect."""
    from mhcflurry import local_parallelism

    monkeypatch.setattr(
        local_parallelism, "_detect_num_cuda_devices_no_torch",
        lambda: 8)  # would lie if called
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=2,            # explicit
        backend="auto",
    )
    resolve_local_parallelism_args(args)
    assert args.gpus == 2
    assert args.gpus_was_auto is False


def test_resolve_local_parallelism_args_no_auto_detect_for_cpu_backend(
        monkeypatch):
    """backend=cpu must not auto-detect GPUs (we don't want CUDA-pinned
    workers when the user explicitly asked for CPU inference)."""
    from mhcflurry import local_parallelism

    monkeypatch.setattr(
        local_parallelism, "_detect_num_cuda_devices_no_torch",
        lambda: 8)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=None,
        backend="cpu",
    )
    resolve_local_parallelism_args(args)
    assert (args.gpus or 0) == 0
    assert args.gpus_was_auto is False


def test_detect_num_cuda_devices_parses_nvidia_smi_l(monkeypatch):
    """Counts only ``GPU N:``-prefixed lines, not MIG sub-devices or any
    diagnostic lines that happen to start with ``GPU ``."""
    from mhcflurry import local_parallelism

    sample = (
        b"GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-aaa)\n"
        b"  MIG 1g.10gb     Device  0: (UUID: MIG-bbb)\n"
        b"GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-ccc)\n"
        b"GPU debug: this should not be counted\n"
    )
    monkeypatch.setattr(
        local_parallelism.subprocess, "check_output", lambda *a, **kw: sample)
    assert local_parallelism._detect_num_cuda_devices_no_torch() == 2


def test_detect_num_cuda_devices_returns_zero_when_smi_missing(monkeypatch):
    """OSError (e.g. nvidia-smi not on PATH) must be swallowed and return 0."""
    from mhcflurry import local_parallelism

    def boom(*a, **kw):
        raise OSError("nvidia-smi: not found")
    monkeypatch.setattr(local_parallelism.subprocess, "check_output", boom)
    assert local_parallelism._detect_num_cuda_devices_no_torch() == 0


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

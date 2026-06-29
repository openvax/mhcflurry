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

from argparse import ArgumentParser, ArgumentTypeError, Namespace

import multiprocessing
import builtins
import os
import pytest

from mhcflurry.local_parallelism import (
    NonDaemonPool,
    NonDaemonProcess,
    _max_workers_per_gpu_arg,
    add_local_parallelism_args,
    add_prediction_parallelism_args,
    auto_max_workers_per_gpu,
    chunk_ranges_for_local_parallelism,
    non_daemon_context,
    num_workers_per_gpu_from_args,
    resolve_local_parallelism_args,
    resolve_max_workers_per_gpu,
    validate_worker_pool_args,
    worker_pool_with_gpu_assignments,
    worker_pool_with_gpu_assignments_from_args,
    worker_init_kwargs_for_scheduler,
)
from mhcflurry.workload_planning import (
    GIB,
    HOST_RAM_PER_DATALOADER_CHILD_GB,
    WORKLOAD_PRESENTATION_CALIBRATION,
    WORKLOAD_PROCESSING_TRAINING,
    estimate_workload_memory,
    host_memory_num_jobs_cap,
    system_memory_info_gb,
)


@pytest.fixture(autouse=True)
def clear_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


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


def test_worker_init_kwargs_honors_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert worker_init_kwargs_for_scheduler(
        num_jobs=5,
        num_gpus=2,
        backend="auto",
        max_workers_per_gpu=2,
    ) == [
        {"backend": "gpu", "gpu_device_nums": ["2"], "max_workers_per_gpu": 2},
        {"backend": "gpu", "gpu_device_nums": ["3"], "max_workers_per_gpu": 2},
        {"backend": "gpu", "gpu_device_nums": ["2"], "max_workers_per_gpu": 2},
        {"backend": "gpu", "gpu_device_nums": ["3"], "max_workers_per_gpu": 2},
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


def test_validate_worker_pool_args_allows_serial_mode_with_gpus():
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
# Pool workers must be non-daemonic so the PyTorch DataLoader inside a
# training worker can spawn its own prefetch children. The default
# multiprocessing.Pool ships daemon workers, which makes
# DataLoader(num_workers>0) raise AssertionError. These tests lock down
# the NonDaemonPool / NonDaemonProcess behavior.


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


def test_explicit_spawn_context_keeps_workers_non_daemonic():
    context = non_daemon_context("spawn")
    process = context.Process(target=_grandchild_entry)

    assert context.get_start_method() == "spawn"
    assert process.daemon is False
    process.daemon = True
    assert process.daemon is False


@pytest.mark.slow
@pytest.mark.integration
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

    If the Pool worker is daemonic, the inner multiprocessing.Process
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


def test_num_workers_per_gpu_from_args_requires_resolved_value():
    args = Namespace(max_workers_per_gpu="auto")
    with pytest.raises(ValueError, match="resolve_local_parallelism_args"):
        num_workers_per_gpu_from_args(args)

    args.max_workers_per_gpu = 3
    assert num_workers_per_gpu_from_args(args) == 3


def test_resolve_local_parallelism_args_caps_auto_num_jobs(monkeypatch):
    # Force a VRAM cap by pinning per-worker GB high so by_vram=1 and the
    # auto MWPG resolves to 1; auto num_jobs follows GPU capacity.
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "40")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "512")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "512")
    monkeypatch.setenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", "24"
    )
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu="auto", num_jobs="auto", gpus=8, backend="auto"
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
    assert args.num_jobs == 0
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


def test_resolve_local_parallelism_args_honors_explicit_num_jobs_override(
    monkeypatch,
):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "512")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "512")
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu=1, num_jobs=16, gpus=8, backend="auto"
    )
    resolve_local_parallelism_args(args)
    assert args.max_workers_per_gpu == 1
    assert args.num_jobs == 16
    assert args.max_workers_per_gpu_was_auto is False
    assert "num_jobs" in args.workload_plan.cli_overrides
    assert any("honoring CLI override" in w for w in args.workload_plan.warnings)


def test_resolve_local_parallelism_args_num_jobs_auto_resolves_to_capacity(
    monkeypatch,
):
    # ``num_jobs="auto"`` (the new default) resolves to gpus × MWPG so
    # production no longer has to hand-pick a value that may not match
    # the resolver's MWPG choice.
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "512")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "512")
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


def test_resolve_local_parallelism_args_keeps_stdout_clean(monkeypatch, capsys):
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "512")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "512")
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=1,
        backend="auto",
    )
    resolve_local_parallelism_args(args)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Local workload plan:" in captured.err


def test_resolve_local_parallelism_args_uses_workload_profile(monkeypatch):
    # Presentation calibration keeps a full presentation predictor stack
    # resident, so its auto plan should be much more conservative than the
    # generic training fallback on 40 GB GPUs.
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "40")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "256")
    monkeypatch.delenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB", raising=False
    )
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=4,
        backend="auto",
    )
    resolve_local_parallelism_args(
        args,
        workload_name=WORKLOAD_PRESENTATION_CALIBRATION,
        workload_hints={"prediction_rows": 40_000_000},
    )
    assert args.max_workers_per_gpu == 1
    assert args.num_jobs == 4
    assert args.workload_plan.workload_name == WORKLOAD_PRESENTATION_CALIBRATION
    assert args.workload_plan.device_worker_gb == 24.0


def test_resolve_local_parallelism_args_caps_auto_jobs_by_available_host_memory(
    monkeypatch,
):
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "32")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "10")
    monkeypatch.setenv("MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION", "0.70")
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=4,
        backend="auto",
    )
    resolve_local_parallelism_args(
        args,
        workload_name=WORKLOAD_PRESENTATION_CALIBRATION,
    )
    assert args.num_jobs == 1
    assert args.workload_plan.host_memory_available_gb == 10.0
    assert args.workload_plan.host_memory_num_jobs_cap == 1
    assert args.workload_plan.host_worker_gb == (
        6.0 + args.dataloader_num_workers * HOST_RAM_PER_DATALOADER_CHILD_GB
    )


def test_system_memory_info_uses_env_override(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "32")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "9.5")
    info = system_memory_info_gb()
    assert info["total_gb"] == 32.0
    assert info["available_gb"] == 9.5
    assert info["source"] == "env"


def test_host_memory_num_jobs_cap_uses_available_memory(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION", "0.50")
    sizing = estimate_workload_memory(WORKLOAD_PRESENTATION_CALIBRATION)
    cap = host_memory_num_jobs_cap(
        {"total_gb": 32.0, "available_gb": 10.0, "source": "test"},
        sizing["host_worker_gb"],
    )
    assert cap == 1


def test_workload_memory_accepts_plain_hint_dict():
    small = estimate_workload_memory(
        WORKLOAD_PROCESSING_TRAINING,
        {"data_bytes": int(1 * GIB)},
    )
    large = estimate_workload_memory(
        WORKLOAD_PROCESSING_TRAINING,
        {"data_bytes": int(20 * GIB)},
    )
    assert small["device_worker_gb"] == 8.0
    assert large["device_worker_gb"] > small["device_worker_gb"]
    assert large["data_pressure_gb"] > 0


def test_explicit_cli_flags_flow_through_workload_plan(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "512")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "512")
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE_LOSS", raising=False)
    monkeypatch.delenv("MHCFLURRY_MATMUL_PRECISION", raising=False)
    args = Namespace(
        max_workers_per_gpu=2,
        num_jobs=8,
        gpus=4,
        backend="auto",
        dataloader_num_workers=3,
        random_negative_pool_epochs=5,
        torch_compile="1",
        torch_compile_loss="0",
        matmul_precision="high",
        enable_timing=True,
    )

    resolve_local_parallelism_args(args)

    assert args.num_jobs == 8
    assert args.max_workers_per_gpu == 2
    assert args.dataloader_num_workers == 3
    assert args.random_negative_pool_epochs == 5
    for name in (
        "num_jobs",
        "max_workers_per_gpu",
        "dataloader_num_workers",
        "random_negative_pool_epochs",
        "torch_compile",
        "torch_compile_loss",
        "matmul_precision",
        "enable_timing",
    ):
        assert name in args.workload_plan.cli_overrides
    assert os.environ["MHCFLURRY_TORCH_COMPILE"] == "1"
    assert os.environ["MHCFLURRY_TORCH_COMPILE_LOSS"] == "0"
    assert os.environ["MHCFLURRY_MATMUL_PRECISION"] == "high"


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
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "512")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "512")
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


def test_resolve_local_parallelism_args_counts_cuda_visible_devices(monkeypatch):
    """Scheduler CUDA masks are authoritative for GPU auto-detection."""
    from mhcflurry import local_parallelism

    def boom(*args, **kwargs):
        raise AssertionError("nvidia-smi should not be called when masked")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "512")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "512")
    monkeypatch.setattr(local_parallelism.subprocess, "check_output", boom)
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=None,
        backend="auto",
    )
    resolve_local_parallelism_args(args)
    assert args.gpus == 2
    assert args.gpus_was_auto is True
    assert args.num_jobs == 8


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


def test_worker_pool_from_args_preserves_serial_mode_with_auto_gpus(
        monkeypatch):
    """Explicit ``--num-jobs 0`` must not become invalid after GPU detect."""
    from mhcflurry import local_parallelism

    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    monkeypatch.setattr(
        local_parallelism, "_detect_num_cuda_devices_no_torch", lambda: 2)
    configured = []
    monkeypatch.setattr(
        local_parallelism, "configure_pytorch",
        lambda backend: configured.append(backend))
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs=0,
        gpus=None,
        backend="auto",
        max_tasks_per_worker=None,
        worker_log_dir=None,
    )

    worker_pool = worker_pool_with_gpu_assignments_from_args(args)

    assert worker_pool is None
    assert args.gpus == 2
    assert args.num_jobs == 0
    assert configured == ["auto"]


def test_worker_pool_creates_worker_log_dir(tmp_path, monkeypatch):
    from mhcflurry import local_parallelism

    captured = {}

    def fake_make_worker_pool(**kwargs):
        captured.update(kwargs)
        return "pool"

    monkeypatch.setattr(local_parallelism, "make_worker_pool", fake_make_worker_pool)
    worker_log_dir = tmp_path / "missing" / "worker-logs"

    worker_pool = worker_pool_with_gpu_assignments(
        num_jobs=1,
        num_gpus=0,
        backend="cpu",
        max_workers_per_gpu=1,
        worker_log_dir=str(worker_log_dir),
    )

    assert worker_pool == "pool"
    assert worker_log_dir.is_dir()
    assert (
        captured["initializer_kwargs_per_process"][0]["worker_log_dir"]
        == str(worker_log_dir)
    )


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


def test_detect_num_cuda_devices_uses_empty_cuda_visible_devices(monkeypatch):
    """An empty CUDA mask means no visible GPUs, even if nvidia-smi exists."""
    from mhcflurry import local_parallelism

    def boom(*args, **kwargs):
        raise AssertionError("nvidia-smi should not be called when masked")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(local_parallelism.subprocess, "check_output", boom)
    assert local_parallelism._detect_num_cuda_devices_no_torch() == 0


def test_free_vram_from_nvidia_smi_uses_cuda_visible_devices(monkeypatch):
    """Free-VRAM sizing should query scheduler-visible physical GPUs."""
    from mhcflurry import local_parallelism

    calls = []

    def fake_check_output(args, **kwargs):
        calls.append(args)
        return b"1024\n2048\n"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(
        local_parallelism.subprocess, "check_output", fake_check_output)
    assert local_parallelism._free_vram_from_nvidia_smi_gb(2) == 1.0
    assert calls == [[
        "nvidia-smi",
        "-i",
        "2,3",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]]


def test_detect_free_vram_per_gpu_preserves_heterogeneity(monkeypatch):
    """Per-GPU detection returns the vector; the scalar helper still mins it."""
    from mhcflurry import local_parallelism

    monkeypatch.delenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", raising=False)
    monkeypatch.setattr(
        local_parallelism.subprocess, "check_output",
        lambda args, **kw: b"8192\n71680\n")  # 8 GB, 70 GB
    assert local_parallelism.detect_free_vram_per_gpu_gb(2) == [8.0, 70.0]
    assert local_parallelism._free_vram_from_nvidia_smi_gb(2) == 8.0


def test_free_vram_env_override_single_value_broadcasts(monkeypatch):
    from mhcflurry import local_parallelism

    monkeypatch.setenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "12")
    assert local_parallelism._free_vram_per_gpu_override_gb(4) == [12.0] * 4
    assert local_parallelism._free_vram_override_gb(4) == 12.0


def test_capacity_warnings_flags_below_safe_range():
    from mhcflurry.workload_planning import capacity_warnings

    common = dict(host_worker_gb=4.0, cpu_count=32)

    # Small GPU vs a heavy per-worker estimate.
    small = capacity_warnings(
        workload_name="affinity_calibration", backend="gpu", gpus=1,
        num_jobs=1, per_gpu_free_vram_gb=[8.0], device_worker_gb=24.0,
        available_ram_gb=64.0, **common)
    assert any("below the per-worker estimate" in m for m in small)

    # Uneven GPUs.
    hetero = capacity_warnings(
        workload_name="affinity_training", backend="gpu", gpus=2, num_jobs=2,
        per_gpu_free_vram_gb=[8.0, 70.0], device_worker_gb=4.0,
        available_ram_gb=256.0, **common)
    assert any("uneven across GPUs" in m for m in hetero)

    # Undetectable VRAM.
    blind = capacity_warnings(
        workload_name="affinity_training", backend="gpu", gpus=1, num_jobs=1,
        per_gpu_free_vram_gb=None, device_worker_gb=4.0,
        available_ram_gb=64.0, **common)
    assert any("could not detect free GPU VRAM" in m for m in blind)

    # Host RAM below the per-worker estimate (4 workers x 4 GB > 8 GB).
    low_ram = capacity_warnings(
        workload_name="affinity_training", backend="gpu", gpus=2, num_jobs=4,
        per_gpu_free_vram_gb=[80.0, 80.0], device_worker_gb=4.0,
        available_ram_gb=8.0, **common)
    assert any("available host RAM" in m for m in low_ram)

    # More workers than CPUs.
    cpu_short = capacity_warnings(
        workload_name="affinity_training", backend="gpu", gpus=8, num_jobs=8,
        per_gpu_free_vram_gb=[80.0] * 8, device_worker_gb=4.0,
        available_ram_gb=512.0, host_worker_gb=3.0, cpu_count=4)
    assert any("exceed detected CPUs" in m for m in cpu_short)


def test_capacity_warnings_silent_on_healthy_machine():
    from mhcflurry.workload_planning import capacity_warnings

    # Ample 8x80GB box: nothing below safe range.
    assert capacity_warnings(
        workload_name="affinity_calibration", backend="gpu", gpus=8,
        num_jobs=16, per_gpu_free_vram_gb=[78.0] * 8, device_worker_gb=24.0,
        available_ram_gb=900.0, host_worker_gb=4.0, cpu_count=176) == []

    # CPU backend skips all GPU checks.
    assert capacity_warnings(
        workload_name="affinity_training", backend="cpu", gpus=0, num_jobs=0,
        per_gpu_free_vram_gb=None, device_worker_gb=4.0, available_ram_gb=8.0,
        host_worker_gb=3.0, cpu_count=2) == []


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


@pytest.mark.slow
@pytest.mark.integration
def test_nondaemonpool_worker_can_spawn_children():
    """Non-daemon pool workers must be able to spawn multiprocessing children.

    Regression test for the case where DataLoader(num_workers>0) called
    from a mhcflurry Pool worker raised
    AssertionError("daemonic processes are not allowed to have children").
    With NonDaemonPool the Pool workers are not daemonic, so spawning a
    child succeeds.
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

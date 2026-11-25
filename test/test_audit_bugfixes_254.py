"""Regression tests for 2.5.4 parallelism, sizing, and cluster bugfixes (#580-#587)."""

import argparse
import multiprocessing
import os
import pickle
from types import SimpleNamespace

import pytest
import torch

from mhcflurry import common
from mhcflurry.affinity import calibration_sizing
from mhcflurry.cluster_parallelism import cluster_results
from mhcflurry.parallelism import (
    add_local_parallelism_args,
    add_prediction_parallelism_args,
    auto_max_workers_per_gpu,
    chunk_ranges_for_local_parallelism,
    non_daemon_context,
    refine_local_parallelism_from_spawn_context,
    refine_local_parallelism_from_warmup,
    resolve_local_parallelism_args,
    torch_compile,
    worker_init_kwargs_for_scheduler,
    worker_pool,
)
from mhcflurry.pytorch_sizing import (
    compute_prediction_batch_size,
    estimate_peak_bytes_per_row,
)


def test_580_serial_runs_honor_backend_flag(monkeypatch):
    old_backend = common._pytorch_backend
    try:
        # Even when native thread env vars are user-pinned (auto_owned=False),
        # serial resolution must still configure the requested PyTorch backend.
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        monkeypatch.delenv("MHCFLURRY_OMP_NUM_THREADS_AUTO", raising=False)
        common.configure_pytorch(backend="auto")

        parser = argparse.ArgumentParser()
        add_local_parallelism_args(parser)
        args = parser.parse_args(["--backend", "cpu"])
        resolve_local_parallelism_args(args)
        assert args.num_jobs == 0
        assert common._pytorch_backend == "cpu"

        # Spawn-context refinement dropping to num_jobs=0 also configures backend.
        common.configure_pytorch(backend="auto")
        monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
        monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "64")
        monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "64")
        args2 = argparse.Namespace(
            max_workers_per_gpu="auto",
            num_jobs="auto",
            gpus=2,
            backend="gpu",
        )
        resolve_local_parallelism_args(args2)
        assert args2.num_jobs > 0
        assert common._pytorch_backend == "auto"

        refine_local_parallelism_from_spawn_context(
            args2,
            context_bytes=8 * (1 << 30),
            memory={"total_gb": 64.0, "available_gb": 10.0, "source": "test"},
        )
        assert args2.num_jobs == 0
        assert common._pytorch_backend == "gpu"
    finally:
        common.configure_pytorch(backend=old_backend)


def test_581_non_daemon_context_follows_runtime_start_method(monkeypatch):
    monkeypatch.setattr(
        multiprocessing, "get_start_method", lambda allow_none=True: "spawn"
    )
    ctx_spawn = non_daemon_context(None)
    assert ctx_spawn.get_start_method() == "spawn"
    assert isinstance(ctx_spawn, worker_pool.NonDaemonSpawnContext)

    if worker_pool.NonDaemonForkContext is not None:
        monkeypatch.setattr(
            multiprocessing, "get_start_method", lambda allow_none=True: "fork"
        )
        ctx_fork = non_daemon_context(None)
        assert ctx_fork.get_start_method() == "fork"
        assert isinstance(ctx_fork, worker_pool.NonDaemonForkContext)
        fake_pool = SimpleNamespace(_ctx=ctx_fork)
        assert worker_pool.worker_pool_uses_fork(fake_pool) is True


def test_582_auto_max_workers_per_gpu_uses_ceiling_division(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    mwpg = auto_max_workers_per_gpu(
        num_jobs=3, num_gpus=2, backend="auto", per_worker_gb=4.0
    )
    assert mwpg == 2
    kwargs = worker_init_kwargs_for_scheduler(
        num_jobs=3, num_gpus=2, backend="auto", max_workers_per_gpu=mwpg
    )
    assert [k["backend"] for k in kwargs] == ["gpu", "gpu", "gpu"]

    assert auto_max_workers_per_gpu(
        num_jobs=7, num_gpus=4, backend="auto", per_worker_gb=4.0
    ) == 2


def test_583_torchinductor_compile_threads_edge_cases(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.delenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False
    )
    monkeypatch.delenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO_VALUE", raising=False
    )

    # 1. Empty string is cleared when compile is disabled, and replaced when enabled.
    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "0")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "")
    assert torch_compile.resolve_torchinductor_compile_threads_env() is None
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ

    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "")
    torch_compile.hoist_torchinductor_compile_threads(
        argparse.Namespace(num_jobs=4)
    )
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "4"

    # 2. User override after an earlier auto hoist is respected (not clobbered).
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "2")
    torch_compile.hoist_torchinductor_compile_threads(
        argparse.Namespace(num_jobs=1)
    )
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "2"

    # Also respected when only legacy AUTO=1 is inherited alongside a user pin.
    monkeypatch.delenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO_VALUE", raising=False
    )
    monkeypatch.setenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "2")
    torch_compile.hoist_torchinductor_compile_threads(
        argparse.Namespace(num_jobs=1)
    )
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "2"

    # 3. Warmup probe re-hoists AFTER refine_local_parallelism_from_warmup
    # updates args.num_jobs.
    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.delenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False
    )
    monkeypatch.delenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO_VALUE", raising=False
    )
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "64")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "64")

    class _FakePool:
        def apply(self, fn, fn_args):
            return {"host_peak_rss_bytes": 24 * (1 << 30)}

        def close(self):
            pass

        def join(self):
            pass

        def terminate(self):
            pass

    monkeypatch.setattr(
        torch_compile,
        "worker_pool_with_gpu_assignments",
        lambda **_kwargs: _FakePool(),
    )
    args = argparse.Namespace(
        num_jobs="auto", gpus=4, backend="auto", max_workers_per_gpu="auto"
    )
    resolve_local_parallelism_args(args)
    assert args.num_jobs > 2
    torch_compile.run_single_worker_resource_probe(
        args,
        work_items=[{"hyperparameters": {"layer_sizes": [32]}}],
        work_function=lambda **_kw: None,
    )
    assert args.num_jobs == 2
    # 16 CPUs // 2 refined workers == 8 compile threads (not stale pre-warmup count).
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "8"


def test_584_warmup_host_memory_cap_uses_total_gb_when_available_is_none(
        monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "80")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "16")
    monkeypatch.delenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", raising=False)
    args = argparse.Namespace(
        num_jobs="auto", gpus=4, backend="auto", max_workers_per_gpu="auto"
    )
    resolve_local_parallelism_args(args)
    assert args.workload_plan.host_memory_available_gb is None
    assert args.workload_plan.host_memory_total_gb == 16.0
    assert args.num_jobs > 1

    refine_local_parallelism_from_warmup(
        args, [{"host_peak_rss_bytes": 10 * (1 << 30)}]
    )
    assert args.num_jobs == 1


def test_585_num_jobs_none_negative_gpus_and_serial_chunking(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "40")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "64")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "64")
    args = argparse.Namespace(
        num_jobs=None, gpus=2, backend="auto", max_workers_per_gpu="auto"
    )
    resolve_local_parallelism_args(args)
    assert args.num_jobs > 0
    assert args.max_workers_per_gpu >= 1

    for add_fn in (add_local_parallelism_args, add_prediction_parallelism_args):
        parser = argparse.ArgumentParser()
        add_fn(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--gpus", "-2"])

    with pytest.raises(ValueError, match="gpus must be >= 0"):
        resolve_local_parallelism_args(
            argparse.Namespace(
                num_jobs=0, gpus=-2, backend="auto", max_workers_per_gpu=1
            )
        )

    assert chunk_ranges_for_local_parallelism(100, num_jobs=0) == [(0, 0, 100)]


def test_586_cluster_results_retry_preserves_stderr_and_cleans_stale_result(
        tmp_path, monkeypatch):
    import inspect
    from mhcflurry.cluster_parallelism import worker_entry_point

    assert inspect.signature(worker_entry_point).parameters["argv"].default is None

    results_workdir = tmp_path / "cluster_runs"
    launches = []

    def fake_check_call(cmd, shell=False):
        launches.append(cmd)
        # Extract work_dir from "... < /path/to/item_workdir/run.0.sh"
        script_path = cmd.split("<", 1)[1].strip().strip("'\"")
        work_dir = os.path.dirname(script_path)
        stderr_path = os.path.join(work_dir, "STDERR")
        error_path = os.path.join(work_dir, "error.pkl")
        result_path = os.path.join(work_dir, "result")
        complete_dir = os.path.join(work_dir, "COMPLETE")
        if len(launches) == 1:
            # First attempt crashed without creating COMPLETE dir, leaving
            # STDERR, error.pkl, and a stale partial result behind.
            with open(stderr_path, "w") as fd:
                fd.write("oom traceback details")
            with open(error_path, "wb") as fd:
                pickle.dump(RuntimeError("boom"), fd)
            with open(result_path, "wb") as fd:
                pickle.dump("stale", fd)
        else:
            # Retry sees clean work_dir (no stale result/error/STDERR) and succeeds.
            assert not os.path.exists(result_path)
            assert not os.path.exists(error_path)
            assert not os.path.exists(stderr_path)
            with open(result_path, "wb") as fd:
                pickle.dump("fresh_ok", fd)
            os.mkdir(complete_dir)

    monkeypatch.setattr("subprocess.check_call", fake_check_call)

    gen = cluster_results(
        work_function=abs,
        work_items=[{"x": 1}],
        results_workdir=str(results_workdir),
        submit_command="echo submit",
        max_retries=1,
        additional_complete_file="STDERR",
    )
    assert list(gen) == ["fresh_ok"]
    assert len(launches) == 2

    script_path = launches[0].split("<", 1)[1].strip().strip("'\"")
    work_dir = os.path.dirname(script_path)
    attempt_dir = os.path.join(work_dir, "attempt.0")
    assert os.path.isdir(attempt_dir)
    with open(os.path.join(attempt_dir, "STDERR")) as fd:
        assert fd.read() == "oom traceback details"
    assert os.path.exists(os.path.join(attempt_dir, "error.pkl"))
    assert os.path.exists(os.path.join(attempt_dir, "result"))


def test_587_empty_networks_peak_bytes_and_dual_pinned_calibration():
    empty_merged = SimpleNamespace(networks=[])
    peak = estimate_peak_bytes_per_row(empty_merged)
    assert peak > 0
    rows = compute_prediction_batch_size(
        torch.device("cpu"), model=empty_merged, total_rows=100
    )
    assert rows >= 1

    shape = calibration_sizing.choose_calibration_batch_shape(
        total_rows=100_000,
        n_peptides=50_000,
        n_alleles=100,
        min_peptide_batch=2000,
        fixed_peptide_batch=5000,
        fixed_allele_batch=10,
    )
    assert shape == (5000, 10)

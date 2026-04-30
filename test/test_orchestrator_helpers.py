"""Unit tests for orchestrator-side helpers added in 2.3.0.

Covers:
* ``mhcflurry.common.filter_canonicalizable_alleles`` and its
  back-compat alias.
* ``mhcflurry.encoding_cache.prebuild_encoding_caches`` and
  ``deterministic_unique_peptide_list``.
* ``mhcflurry.train_pan_allele_models_command._hoist_torchinductor_compile_threads``
* Confirmation that the select commands' source actually invokes the
  shared filter (cheap regression for the same class of bug as the
  calibrate pseudogene crash).

All tests are pure-Python; no GPU, no subprocess, no models on disk.
"""

import argparse
import inspect
import os
import pathlib
import subprocess

import pytest


def test_hoist_helper_lives_in_local_parallelism():
    """``hoist_torchinductor_compile_threads`` belongs in the parallelism
    module so processing/allele-specific commands can use it without
    importing from train_pan_allele."""
    from mhcflurry.local_parallelism import hoist_torchinductor_compile_threads
    from mhcflurry.train_pan_allele_models_command import (
        _hoist_torchinductor_compile_threads,
    )
    assert _hoist_torchinductor_compile_threads is hoist_torchinductor_compile_threads


def test_filter_helper_lives_in_common():
    from mhcflurry.common import filter_canonicalizable_alleles
    from mhcflurry.calibrate_percentile_ranks_command import (
        _filter_canonicalizable_alleles,
    )
    # back-compat alias preserved for any external import path.
    assert _filter_canonicalizable_alleles is filter_canonicalizable_alleles


def test_filter_select_call_sites():
    """Both select commands that iterate predictor.supported_alleles
    must invoke the canonicalization filter. Without coverage at
    these sites, a pseudogene allele in the input predictor would
    crash the parallel selection mid-fold the same way calibrate
    used to."""
    from mhcflurry import select_pan_allele_models_command as pan_mod
    from mhcflurry import select_allele_specific_models_command as as_mod

    pan_src = inspect.getsource(pan_mod.run)
    as_src = inspect.getsource(as_mod.run)
    assert "filter_canonicalizable_alleles" in pan_src, (
        "select_pan_allele must filter supported_alleles"
    )
    assert "filter_canonicalizable_alleles" in as_src, (
        "select_allele_specific must filter supported_alleles"
    )


def test_deterministic_unique_peptide_list_preserves_first_seen_order():
    from mhcflurry.encoding_cache import deterministic_unique_peptide_list
    raw = ["ABC", "DEF", "ABC", "GHI", "DEF", "JKL"]
    assert deterministic_unique_peptide_list(raw) == ["ABC", "DEF", "GHI", "JKL"]


def test_prebuild_encoding_caches_idempotent(tmp_path):
    """Calling prebuild twice should be a cheap cache-hit second time."""
    from mhcflurry.encoding_cache import prebuild_encoding_caches

    peptides = ["SIINFEKL", "GILGFVFTL", "NLVPMVATV"]
    cfg = [{}]  # default EncodingParams

    log_lines = []
    prebuild_encoding_caches(
        cache_dir=str(tmp_path),
        peptides=peptides,
        encoding_configs=cfg,
        label="unit",
        log=log_lines.append,
    )
    # Second call should hit cache for every config.
    log_lines.clear()
    prebuild_encoding_caches(
        cache_dir=str(tmp_path),
        peptides=peptides,
        encoding_configs=cfg,
        label="unit",
        log=log_lines.append,
    )
    assert any("hit" in line for line in log_lines), log_lines


def test_hoist_torchinductor_no_op_when_compile_disabled(monkeypatch):
    """If MHCFLURRY_TORCH_COMPILE!=1 the hoist must not touch the env."""
    from mhcflurry.local_parallelism import (
        hoist_torchinductor_compile_threads as _hoist_torchinductor_compile_threads,
    )
    monkeypatch.delenv("MHCFLURRY_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.delenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    args = argparse.Namespace(num_jobs=8)
    _hoist_torchinductor_compile_threads(args)
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ


def test_hoist_torchinductor_respects_user_pin(monkeypatch):
    from mhcflurry.local_parallelism import (
        hoist_torchinductor_compile_threads as _hoist_torchinductor_compile_threads,
    )
    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "12")
    monkeypatch.delenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    args = argparse.Namespace(num_jobs=8)
    _hoist_torchinductor_compile_threads(args)
    # User-pinned value must survive.
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "12"


def test_hoist_torchinductor_sizes_against_num_jobs(monkeypatch):
    from mhcflurry.local_parallelism import (
        hoist_torchinductor_compile_threads as _hoist_torchinductor_compile_threads,
    )
    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.delenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    # 8 jobs on a 64-core box → 64/8=8.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    args = argparse.Namespace(num_jobs=8)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "8"

    # 1 job on 4-core box → 4/1=4, hits cap.
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.delenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    args = argparse.Namespace(num_jobs=1)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "4"

    # 16 jobs on 32-core box → 32/16=2.
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.delenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    args = argparse.Namespace(num_jobs=16)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "2"

    # Always at least 1 thread even with absurd over-subscription.
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.delenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    args = argparse.Namespace(num_jobs=64)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


def test_hoist_torchinductor_recomputes_auto_owned_value(monkeypatch):
    from mhcflurry.local_parallelism import (
        hoist_torchinductor_compile_threads as _hoist_torchinductor_compile_threads,
    )
    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.setenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    args = argparse.Namespace(num_jobs=4)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "4"


def test_hoist_torchinductor_accepts_auto_env(monkeypatch):
    from mhcflurry.local_parallelism import (
        hoist_torchinductor_compile_threads as _hoist_torchinductor_compile_threads,
    )

    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "auto")
    monkeypatch.delenv("MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    args = argparse.Namespace(num_jobs=4)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "16"
    assert os.environ["MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO"] == "1"


def test_hoist_torchinductor_warmup_uses_larger_single_worker_budget(monkeypatch):
    from mhcflurry.local_parallelism import (
        hoist_torchinductor_compile_threads as _hoist_torchinductor_compile_threads,
    )

    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "auto")
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    args = argparse.Namespace(num_jobs=1)
    _hoist_torchinductor_compile_threads(args, phase="warmup")
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "64"


def test_cluster_worker_compile_threads_auto(monkeypatch):
    from mhcflurry.local_parallelism import configure_cluster_worker_torch_compile_threads

    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "auto")
    monkeypatch.setenv("MHCFLURRY_CLUSTER_WORKERS_PER_NODE", "4")
    monkeypatch.setattr(os, "cpu_count", lambda: 64)

    configure_cluster_worker_torch_compile_threads()

    # Production cap=16, 64/4=16.
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "16"


def test_attach_constant_data_skips_fork_pool():
    from mhcflurry.local_parallelism import attach_constant_data_to_work_items_if_needed

    class Ctx:
        def get_start_method(self):
            return "fork"

    pool = argparse.Namespace(_ctx=Ctx())
    work_items = [{"a": 1}, {"b": 2}]
    constant_data = {"large": object()}
    logs = []

    attached = attach_constant_data_to_work_items_if_needed(
        work_items, constant_data, pool, log=logs.append
    )

    assert attached is False
    assert all("constant_data" not in item for item in work_items)
    assert any("inherit GLOBAL_DATA" in line for line in logs)


def test_attach_constant_data_attaches_for_non_fork_pool():
    from mhcflurry.local_parallelism import attach_constant_data_to_work_items_if_needed

    class Ctx:
        def get_start_method(self):
            return "spawn"

    pool = argparse.Namespace(_ctx=Ctx())
    work_items = [{"a": 1}, {"b": 2}]
    constant_data = {"large": object()}

    attached = attach_constant_data_to_work_items_if_needed(
        work_items, constant_data, pool, log=lambda _: None
    )

    assert attached is True
    assert [item["constant_data"] for item in work_items] == [
        constant_data,
        constant_data,
    ]


def test_set_cpu_threads_accepts_auto_when_num_jobs_is_provided():
    script = pathlib.Path(__file__).resolve().parents[1] / (
        "scripts/training/set_cpu_threads.sh"
    )
    command = (
        "source %s; "
        "NUM_JOBS=16 GPUS=8 MAX_WORKERS_PER_GPU=auto "
        "DATALOADER_NUM_WORKERS=1 set_cpu_threads"
    ) % script
    result = subprocess.run(
        ["bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "total_workers=16" in result.stderr


def test_auto_dataloader_num_workers_hardware_tiers():
    """Cross-check the heuristic against the production hardware tiers.

    These are the configs the recipe currently runs on or has been
    measured against. Regressions here will silently change runtime
    parallelism on the next run, so the table is fixed.
    """
    from mhcflurry.local_parallelism import auto_dataloader_num_workers

    cases = [
        # (label, vcpus, ram_gb, num_fit_workers, expected)
        ("Verda 8xA100-80GB", 176, 400, 16, 4),
        ("8xA100-40GB (1 worker/GPU)", 176, 400, 8, 4),
        ("8xL40S sweep box (96v)", 96, 200, 16, 3),
        ("Single A100-80G Lambda (30v)", 30, 200, 2, 4),
        ("Single A100-80G tight (16v)", 16, 64, 2, 4),
        ("Single T4 / RTX (8v / 16G)", 8, 16, 1, 4),
        ("Tight cluster node (32v / 16fit)", 32, 64, 16, 1),
        # Edge cases that exercise the floors / caps:
        ("Tiny cluster node (8v / 4fit)", 8, 16, 4, 1),
        ("Very tight (8v / 8fit)", 8, 16, 8, 0),  # cpu_per_fit=1, cpu_cap=0 → 0
        ("Serial run (no fit workers)", 64, 200, 0, 0),
        ("Single-fit / generous", 64, 200, 1, 4),
    ]
    for label, vcpus, ram, fit, want in cases:
        got = auto_dataloader_num_workers(
            num_fit_workers=fit, vcpus=vcpus, ram_gb=ram
        )
        assert got == want, (
            "%s: vcpus=%d ram_gb=%s fit=%d → got %d, want %d"
            % (label, vcpus, ram, fit, got, want)
        )


def test_auto_dataloader_num_workers_ram_constrained():
    """RAM-tight boxes step the dataloader count down or to zero.

    Each DL child is ~500 MB RSS over the main fit-worker baseline (~2 GB).
    When ram_per_fit < 2 GB, no children should be spawned (in-process).
    """
    from mhcflurry.local_parallelism import auto_dataloader_num_workers

    cases = [
        # (label, vcpus, ram_gb, num_fit_workers, expected)
        ("Generous CPU + tight RAM (32G/16fit)", 176, 32, 16, 0),  # ram_per_fit=2 → 0 dl
        ("Verda CPU + adequate RAM (64G/16fit)", 176, 64, 16, 4),  # ram_per_fit=4 → cap=4
        ("Verda CPU + very tight RAM (16G/16fit)", 176, 16, 16, 0),  # baseline alone exhausts
        ("Generous RAM + few fit (200G/2fit)", 30, 200, 2, 4),  # ram_per_fit=100 → no constraint
    ]
    for label, vcpus, ram, fit, want in cases:
        got = auto_dataloader_num_workers(
            num_fit_workers=fit, vcpus=vcpus, ram_gb=ram
        )
        assert got == want, (
            "%s: vcpus=%d ram_gb=%s fit=%d → got %d, want %d"
            % (label, vcpus, ram, fit, got, want)
        )


def test_auto_dataloader_num_workers_no_ram_input_means_cpu_only_decision():
    """When ram_gb is None, only CPU/fit-worker math drives the choice."""
    from mhcflurry.local_parallelism import auto_dataloader_num_workers

    # 176v / 16fit → cpu_per_fit=11, cpu_cap=5, hard_cap=4 → 4 regardless of RAM.
    assert auto_dataloader_num_workers(num_fit_workers=16, vcpus=176) == 4
    # 8v / 8fit → cpu_per_fit=1, cpu_cap=0 → 0 regardless of RAM.
    assert auto_dataloader_num_workers(num_fit_workers=8, vcpus=8) == 0


def test_auto_dataloader_num_workers_hard_cap_env_override(monkeypatch):
    """``MHCFLURRY_AUTO_DATALOADER_HARD_CAP`` shifts the ceiling."""
    from mhcflurry.local_parallelism import auto_dataloader_num_workers

    # Default cap = 4; with 176v/16fit we'd get 4.
    monkeypatch.setenv("MHCFLURRY_AUTO_DATALOADER_HARD_CAP", "2")
    assert auto_dataloader_num_workers(
        num_fit_workers=16, vcpus=176, ram_gb=400) == 2
    monkeypatch.setenv("MHCFLURRY_AUTO_DATALOADER_HARD_CAP", "0")
    assert auto_dataloader_num_workers(
        num_fit_workers=16, vcpus=176, ram_gb=400) == 0
    # Increasing cap above default — picks min(cpu_cap, hard_cap).
    monkeypatch.setenv("MHCFLURRY_AUTO_DATALOADER_HARD_CAP", "8")
    # 176/16 = 11 cpu_per_fit, cpu_cap=5, hard_cap=8 → min(5,8) = 5.
    assert auto_dataloader_num_workers(
        num_fit_workers=16, vcpus=176, ram_gb=400) == 5


def test_auto_random_negative_pool_epochs_hardware_tiers():
    """Cross-check the RN-pool-epoch heuristic across production hardware tiers.

    With default safety_fraction=0.5 and per-pool-epoch=1 GB/worker, the
    pool epochs scale inversely with ``num_workers / ram_gb``. The hard
    cap of 10 prevents marginal-return values from running away.
    """
    from mhcflurry.local_parallelism import auto_random_negative_pool_epochs

    cases = [
        # (label, ram_gb, num_workers, expected)
        # 8x80GB Verda (32 fit-workers, 400 GB): 400*0.5/32 = 6.25 → 6
        ("Verda 8xA100-80GB (32 workers / 400G)", 400, 32, 6),
        # 8x40GB (16 fit-workers, 400 GB): 400*0.5/16 = 12.5 → cap=10
        ("8xA100-40GB (16 workers / 400G)", 400, 16, 10),
        # 8xL40S (16 fit-workers, 200 GB): 200*0.5/16 = 6.25 → 6
        ("8xL40S (16 workers / 200G)", 200, 16, 6),
        # Single A100-80G Lambda (2 fit-workers, 200 GB): 200*0.5/2 = 50 → cap=10
        ("Single A100-80G Lambda (2/200G)", 200, 2, 10),
        # Tight cluster (16 fit-workers, 64 GB): 64*0.5/16 = 2 → 2
        ("Tight cluster (16/64G)", 64, 16, 2),
        # RAM-starved (16 fit-workers, 32 GB): 32*0.5/16 = 1 → 1 (legacy)
        ("RAM-starved (16/32G)", 32, 16, 1),
        # Single T4 (1 fit-worker, 16 GB): 16*0.5/1 = 8 → 8
        ("Single T4 (1/16G)", 16, 1, 8),
    ]
    for label, ram, workers, want in cases:
        got = auto_random_negative_pool_epochs(
            num_random_negatives=1_500_000,
            peptide_max_length=15,
            num_workers=workers,
            ram_gb=ram,
        )
        assert got == want, (
            "%s: ram_gb=%s workers=%d → got %d, want %d"
            % (label, ram, workers, got, want)
        )


def test_auto_random_negative_pool_epochs_serial_or_no_ram_returns_one():
    """Serial run or unknown-RAM box → conservative pool_epochs=1."""
    from mhcflurry.local_parallelism import auto_random_negative_pool_epochs

    # No fit-workers: serial run, no need to amortize.
    assert auto_random_negative_pool_epochs(
        num_random_negatives=1_500_000,
        peptide_max_length=15,
        num_workers=0,
        ram_gb=400,
    ) == 1
    # Unknown RAM (e.g. macOS dev machine with no /proc/meminfo): keep
    # legacy regen-every-epoch behavior since we don't know the budget.
    assert auto_random_negative_pool_epochs(
        num_random_negatives=1_500_000,
        peptide_max_length=15,
        num_workers=8,
        ram_gb=None,
    ) == 1


def test_auto_random_negative_pool_epochs_env_overrides(monkeypatch):
    """Env knobs let operators tighten or loosen the heuristic."""
    from mhcflurry.local_parallelism import auto_random_negative_pool_epochs

    # Verda baseline: 6.
    assert auto_random_negative_pool_epochs(
        num_random_negatives=1_500_000, peptide_max_length=15,
        num_workers=32, ram_gb=400) == 6

    # Halve the safety fraction → halves the budget.
    monkeypatch.setenv("MHCFLURRY_AUTO_RN_POOL_SAFETY_FRACTION", "0.25")
    assert auto_random_negative_pool_epochs(
        num_random_negatives=1_500_000, peptide_max_length=15,
        num_workers=32, ram_gb=400) == 3
    monkeypatch.delenv("MHCFLURRY_AUTO_RN_POOL_SAFETY_FRACTION")

    # Lower the per-pool cost estimate → higher pool epochs.
    monkeypatch.setenv("MHCFLURRY_AUTO_RN_POOL_PER_EPOCH_PER_WORKER_GB", "0.25")
    # 400*0.5/32/0.25 = 25 → cap=10.
    assert auto_random_negative_pool_epochs(
        num_random_negatives=1_500_000, peptide_max_length=15,
        num_workers=32, ram_gb=400) == 10
    monkeypatch.delenv("MHCFLURRY_AUTO_RN_POOL_PER_EPOCH_PER_WORKER_GB")

    # Lower the hard cap.
    monkeypatch.setenv("MHCFLURRY_AUTO_RN_POOL_HARD_CAP", "3")
    # 400*0.5/32/1 = 6.25 → cap=3.
    assert auto_random_negative_pool_epochs(
        num_random_negatives=1_500_000, peptide_max_length=15,
        num_workers=32, ram_gb=400) == 3


def test_auto_dataloader_num_workers_handles_none_and_zero_fit_workers():
    """Serial / no-fit-worker case should return 0 (in-process)."""
    from mhcflurry.local_parallelism import auto_dataloader_num_workers

    assert auto_dataloader_num_workers(num_fit_workers=None) == 0
    assert auto_dataloader_num_workers(num_fit_workers=0) == 0
    assert auto_dataloader_num_workers(num_fit_workers=-1) == 0


def test_resolve_dataloader_num_workers_passthrough_int():
    """Pinned int is a passthrough; only "auto" or None invokes the heuristic."""
    from mhcflurry.local_parallelism import resolve_dataloader_num_workers

    assert resolve_dataloader_num_workers(0, num_fit_workers=16) == 0
    assert resolve_dataloader_num_workers(2, num_fit_workers=16) == 2
    assert resolve_dataloader_num_workers("3", num_fit_workers=16) == 3


def test_resolve_dataloader_num_workers_auto_calls_heuristic():
    from mhcflurry.local_parallelism import resolve_dataloader_num_workers

    # Verda-like config; "auto" → 4.
    got = resolve_dataloader_num_workers(
        "auto", num_fit_workers=16, vcpus=176, ram_gb=400
    )
    assert got == 4
    # None is the same as "auto".
    got = resolve_dataloader_num_workers(
        None, num_fit_workers=16, vcpus=176, ram_gb=400
    )
    assert got == 4


def test_resolve_dataloader_num_workers_rejects_negative():
    from mhcflurry.local_parallelism import resolve_dataloader_num_workers

    with pytest.raises(ValueError, match=">= 0"):
        resolve_dataloader_num_workers(-1, num_fit_workers=16)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_dataloader_num_workers("not-a-number", num_fit_workers=16)


def test_auto_num_jobs_basic():
    from mhcflurry.local_parallelism import auto_num_jobs

    assert auto_num_jobs(num_gpus=8, max_workers_per_gpu=2) == 16
    assert auto_num_jobs(num_gpus=1, max_workers_per_gpu=2) == 2
    assert auto_num_jobs(num_gpus=0, max_workers_per_gpu=2) == 0
    assert auto_num_jobs(num_gpus=None, max_workers_per_gpu=2) == 0


def test_auto_num_jobs_rejects_unresolved_auto():
    """Caller must resolve max_workers_per_gpu first."""
    from mhcflurry.local_parallelism import auto_num_jobs

    with pytest.raises(ValueError, match="must be resolved"):
        auto_num_jobs(num_gpus=8, max_workers_per_gpu="auto")


def test_apply_dataloader_num_workers_to_work_items_overrides_existing():
    """Applies the resolved value uniformly; counts overrides for the log."""
    from mhcflurry.local_parallelism import apply_dataloader_num_workers_to_work_items

    work_items = [
        {"hyperparameters": {"dataloader_num_workers": 1, "minibatch_size": 4096}},
        {"hyperparameters": {"dataloader_num_workers": 0}},
        {"hyperparameters": {}},  # missing key → set fresh
        {"hyperparameters": {"dataloader_num_workers": 4}},  # already correct
    ]
    log_lines = []
    apply_dataloader_num_workers_to_work_items(
        work_items, num_workers=4, log=log_lines.append
    )
    for item in work_items:
        assert item["hyperparameters"]["dataloader_num_workers"] == 4
    # 3 items had a non-4 / missing value before; the 4th was already 4.
    msg = log_lines[0]
    assert "4/4 items" in msg
    assert "dataloader_num_workers=4" in msg
    assert "overridden=3" in msg


def test_apply_dataloader_num_workers_skips_items_without_hyperparameters():
    """Defensive: items lacking the 'hyperparameters' key are left alone."""
    from mhcflurry.local_parallelism import apply_dataloader_num_workers_to_work_items

    work_items = [{"work_item_name": "no-hp"}, {"hyperparameters": {}}]
    apply_dataloader_num_workers_to_work_items(
        work_items, num_workers=2, log=lambda *a: None
    )
    assert "hyperparameters" not in work_items[0]
    assert work_items[1]["hyperparameters"]["dataloader_num_workers"] == 2


def test_data_size_growth_does_not_change_dataloader_count():
    """Dataset growth scales SHM bytes, but DL worker count is hardware-bound.

    The auto resolver is intentionally hardware-only — total dataset rows
    affect the per-fit-worker SHM footprint (covered by the SHM
    estimator), not the per-fit-worker prefetch count. Per-batch CPU work
    is bounded by minibatch_size. This regression test pins that
    independence: for the same Verda-like box, the resolver should pick
    the same value whether train_rows is 700K (today) or 7M (10× growth).
    """
    from mhcflurry.local_parallelism import auto_dataloader_num_workers

    big_box = dict(num_fit_workers=16, vcpus=176, ram_gb=400)
    chosen = auto_dataloader_num_workers(**big_box)
    assert chosen == 4
    # Heuristic doesn't accept train_rows — explicit by design, not omission.


def test_cluster_l1shm_warns(monkeypatch, capsys):
    """When --cluster-parallelism + --random-negative-shared-pool-dir
    are both passed, the orchestrator must loud-warn that the dir
    must be NFS-shared. The message names the absolute pool path so
    the user can verify it."""
    from mhcflurry import train_pan_allele_models_command as mod

    src = inspect.getsource(mod.train_models)
    assert "WARNING" in src and "cluster_parallelism" in src and (
        "random-negative-shared-pool-dir" in src
        or "random_negative_shared_pool_dir" in src
    ), (
        "train_models must warn when cluster + run mmap cache are "
        "combined; see docs/orchestrator.md"
    )

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
    args = argparse.Namespace(num_jobs=8)
    _hoist_torchinductor_compile_threads(args)
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ


def test_hoist_torchinductor_respects_user_pin(monkeypatch):
    from mhcflurry.local_parallelism import (
        hoist_torchinductor_compile_threads as _hoist_torchinductor_compile_threads,
    )
    monkeypatch.setenv("MHCFLURRY_TORCH_COMPILE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "12")
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
    # 8 jobs on a 64-core box → 64/8=8, capped at 4.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    args = argparse.Namespace(num_jobs=8)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "4"

    # 1 job on 4-core box → 4/1=4, hits cap.
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    args = argparse.Namespace(num_jobs=1)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "4"

    # 16 jobs on 32-core box → 32/16=2.
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    args = argparse.Namespace(num_jobs=16)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "2"

    # Always at least 1 thread even with absurd over-subscription.
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    args = argparse.Namespace(num_jobs=64)
    _hoist_torchinductor_compile_threads(args)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


def test_shm_capacity_check_safe_when_plenty_free(monkeypatch):
    from mhcflurry import local_parallelism as lp

    # Fake /dev/shm: 256 GB total, 250 GB free.
    monkeypatch.setattr(lp, "shm_free_gb", lambda *a, **k: 250.0)
    monkeypatch.setattr(lp, "shm_total_gb", lambda *a, **k: 256.0)
    result = lp.fit_shm_capacity_check(num_workers=16, per_fit_gb=4.0)
    # 16 * 4 * 1.5 = 96 GB needed, 250 free.
    assert result["safe"] is True
    assert result["estimated_required_gb"] == 96.0


def test_shm_capacity_check_unsafe_when_tmpfs_tight(monkeypatch):
    from mhcflurry import local_parallelism as lp

    # Docker default: 8 GB total, ~7.9 free.
    monkeypatch.setattr(lp, "shm_free_gb", lambda *a, **k: 7.9)
    monkeypatch.setattr(lp, "shm_total_gb", lambda *a, **k: 8.0)
    result = lp.fit_shm_capacity_check(num_workers=16, per_fit_gb=4.0)
    assert result["safe"] is False
    assert "TIGHT" in result["message"]
    assert "shm-size" in result["message"]


def test_shm_capacity_check_no_shm_dir_returns_safe(monkeypatch):
    """macOS / no-/dev/shm platforms should return safe=True."""
    from mhcflurry import local_parallelism as lp

    monkeypatch.setattr(lp, "shm_free_gb", lambda *a, **k: None)
    monkeypatch.setattr(lp, "shm_total_gb", lambda *a, **k: None)
    result = lp.fit_shm_capacity_check(num_workers=8, per_fit_gb=4.0)
    assert result["safe"] is True


def test_preflight_shm_auto_disables_on_tight_tmpfs(monkeypatch, capsys):
    """When /dev/shm is too small, orchestrator should set
    MHCFLURRY_FIT_DATALOADER_SHM=0 unless the user has pinned it."""
    from mhcflurry import train_pan_allele_models_command as cmd
    from mhcflurry import local_parallelism as lp

    monkeypatch.setattr(lp, "shm_free_gb", lambda *a, **k: 7.0)
    monkeypatch.setattr(lp, "shm_total_gb", lambda *a, **k: 8.0)
    monkeypatch.delenv("MHCFLURRY_FIT_DATALOADER_SHM", raising=False)

    args = argparse.Namespace(num_jobs=16)
    cmd._preflight_shm_capacity(args)
    assert os.environ.get("MHCFLURRY_FIT_DATALOADER_SHM") == "0"
    out = capsys.readouterr().out
    assert "TIGHT" in out and "Auto-disabling Layer-2 SHM" in out


def test_preflight_shm_respects_force_pinned_off(monkeypatch, capsys):
    from mhcflurry import train_pan_allele_models_command as cmd
    from mhcflurry import local_parallelism as lp

    monkeypatch.setattr(lp, "shm_free_gb", lambda *a, **k: 7.0)
    monkeypatch.setattr(lp, "shm_total_gb", lambda *a, **k: 8.0)
    monkeypatch.setenv("MHCFLURRY_FIT_DATALOADER_SHM", "0")

    args = argparse.Namespace(num_jobs=16)
    cmd._preflight_shm_capacity(args)
    # User pinned off; orchestrator should leave it.
    assert os.environ["MHCFLURRY_FIT_DATALOADER_SHM"] == "0"


def test_preflight_shm_warns_loud_on_force_pinned_on(monkeypatch, capsys):
    from mhcflurry import train_pan_allele_models_command as cmd
    from mhcflurry import local_parallelism as lp

    monkeypatch.setattr(lp, "shm_free_gb", lambda *a, **k: 7.0)
    monkeypatch.setattr(lp, "shm_total_gb", lambda *a, **k: 8.0)
    monkeypatch.setenv("MHCFLURRY_FIT_DATALOADER_SHM", "1")

    args = argparse.Namespace(num_jobs=16)
    cmd._preflight_shm_capacity(args)
    # User pinned on despite tight tmpfs — env unchanged but loud warn.
    assert os.environ["MHCFLURRY_FIT_DATALOADER_SHM"] == "1"
    out = capsys.readouterr().out
    assert "WARNING" in out and "force-pinned" in out


def test_preflight_shm_skipped_for_serial_run(monkeypatch):
    from mhcflurry import train_pan_allele_models_command as cmd

    monkeypatch.delenv("MHCFLURRY_FIT_DATALOADER_SHM", raising=False)
    args = argparse.Namespace(num_jobs=0)  # serial
    cmd._preflight_shm_capacity(args)
    # Serial run: orchestrator should not touch the env.
    assert "MHCFLURRY_FIT_DATALOADER_SHM" not in os.environ


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
        "train_models must warn when cluster + L1-SHM are combined; "
        "see docs/orchestrator.md"
    )

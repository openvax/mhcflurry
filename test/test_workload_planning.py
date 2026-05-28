"""Unit tests for the workload-planning auto-tuner.

``mhcflurry.workload_planning`` decides num_jobs / max_workers_per_gpu /
dataloader_num_workers from hardware facts and CLI overrides. These tests
inject fakes for the hardware/heuristic callbacks so the orchestrator's
own arithmetic is exercised without touching real GPUs.
"""

import argparse
import pytest

from mhcflurry import workload_planning as wp


# ---------------------------------------------------------------------------
# env_float / env_int
# ---------------------------------------------------------------------------


def test_env_float_default_when_unset(monkeypatch):
    monkeypatch.delenv("MHCFLURRY_TEST_X", raising=False)
    assert wp.env_float("MHCFLURRY_TEST_X", 0.5) == 0.5


def test_env_float_default_when_empty_string(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "")
    assert wp.env_float("MHCFLURRY_TEST_X", 0.5) == 0.5


def test_env_float_reads_string_value(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "2.5")
    assert wp.env_float("MHCFLURRY_TEST_X", 0.0) == 2.5


def test_env_float_bad_value_names_variable(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "nope")
    with pytest.raises(ValueError) as exc_info:
        wp.env_float("MHCFLURRY_TEST_X", 0.0)
    assert "MHCFLURRY_TEST_X" in str(exc_info.value)
    assert "nope" in str(exc_info.value)


def test_env_float_bounds_lower(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "-1.0")
    with pytest.raises(ValueError):
        wp.env_float("MHCFLURRY_TEST_X", 0.5, bounds=(0.0, 1.0))


def test_env_float_bounds_upper(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "5.0")
    with pytest.raises(ValueError):
        wp.env_float("MHCFLURRY_TEST_X", 0.5, bounds=(0.0, 1.0))


def test_env_float_bounds_endpoints_inclusive(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "1.0")
    assert wp.env_float("MHCFLURRY_TEST_X", 0.0, bounds=(0.0, 1.0)) == 1.0
    monkeypatch.setenv("MHCFLURRY_TEST_X", "0.0")
    assert wp.env_float("MHCFLURRY_TEST_X", 0.0, bounds=(0.0, 1.0)) == 0.0


def test_env_int_default_when_unset(monkeypatch):
    monkeypatch.delenv("MHCFLURRY_TEST_X", raising=False)
    assert wp.env_int("MHCFLURRY_TEST_X", 7) == 7


def test_env_int_reads_string_value(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "12")
    assert wp.env_int("MHCFLURRY_TEST_X", 0) == 12


def test_env_int_rejects_floats(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "3.5")
    with pytest.raises(ValueError) as exc_info:
        wp.env_int("MHCFLURRY_TEST_X", 0)
    assert "MHCFLURRY_TEST_X" in str(exc_info.value)


def test_env_int_bounds(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_TEST_X", "0")
    with pytest.raises(ValueError):
        wp.env_int("MHCFLURRY_TEST_X", 1, bounds=(1, None))


# ---------------------------------------------------------------------------
# get_workload_profile
# ---------------------------------------------------------------------------


def test_get_workload_profile_known_name():
    profile = wp.get_workload_profile(wp.WORKLOAD_AFFINITY_INFERENCE)
    assert profile.name == wp.WORKLOAD_AFFINITY_INFERENCE
    assert profile.device_worker_gb == 4.0


def test_get_workload_profile_unknown_falls_back_to_generic():
    profile = wp.get_workload_profile("does-not-exist")
    assert profile.name == wp.WORKLOAD_GENERIC


def test_get_workload_profile_all_workload_constants_have_profiles():
    """Every WORKLOAD_* constant must map to a real profile, not generic fallback."""
    constants = [
        getattr(wp, name) for name in dir(wp)
        if name.startswith("WORKLOAD_") and isinstance(getattr(wp, name), str)
    ]
    for workload in constants:
        profile = wp.get_workload_profile(workload)
        assert profile.name == workload, (workload, profile.name)


# ---------------------------------------------------------------------------
# path_size_bytes
# ---------------------------------------------------------------------------


def test_path_size_bytes_none_for_empty(tmp_path):
    assert wp.path_size_bytes(None) is None
    assert wp.path_size_bytes("") is None


def test_path_size_bytes_returns_size(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert wp.path_size_bytes(str(p)) == 11


def test_path_size_bytes_returns_none_for_missing(tmp_path):
    assert wp.path_size_bytes(str(tmp_path / "absent.bin")) is None


# ---------------------------------------------------------------------------
# Memory probes
# ---------------------------------------------------------------------------


def test_memory_env_override_neither_set(monkeypatch):
    monkeypatch.delenv("MHCFLURRY_SYSTEM_RAM_GB", raising=False)
    monkeypatch.delenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", raising=False)
    assert wp._memory_env_override() is None


def test_memory_env_override_both_set(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "128.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "100.0")
    info = wp._memory_env_override()
    assert info == {
        "total_gb": 128.0,
        "available_gb": 100.0,
        "source": "env",
    }


def test_memory_env_override_only_total(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "64.0")
    monkeypatch.delenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", raising=False)
    info = wp._memory_env_override()
    assert info["total_gb"] == 64.0
    assert info["available_gb"] is None
    assert info["source"] == "env"


def test_memory_env_override_bad_value_names_var(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "xx")
    with pytest.raises(ValueError) as exc_info:
        wp._memory_env_override()
    assert "MHCFLURRY_SYSTEM_RAM_GB" in str(exc_info.value)


def test_system_memory_info_gb_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "200.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "150.0")
    info = wp.system_memory_info_gb()
    assert info["source"] == "env"
    assert info["total_gb"] == 200.0


def test_system_memory_info_gb_falls_through_to_platform(monkeypatch):
    """When env override is absent, return whatever the OS probe yields."""
    monkeypatch.delenv("MHCFLURRY_SYSTEM_RAM_GB", raising=False)
    monkeypatch.delenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", raising=False)
    info = wp.system_memory_info_gb()
    # Will be /proc/meminfo on linux, vm_stat on mac, or the empty dict on
    # platforms with neither — all three are valid shapes here.
    assert info["source"] in ("/proc/meminfo", "vm_stat", "unknown")


# ---------------------------------------------------------------------------
# _normalize_hints
# ---------------------------------------------------------------------------


def test_normalize_hints_empty():
    assert wp._normalize_hints() == {}


def test_normalize_hints_dict_only():
    assert wp._normalize_hints({"foo": 1, "bar": 2}) == {"foo": 1, "bar": 2}


def test_normalize_hints_kwargs_drop_none():
    result = wp._normalize_hints({"foo": 1}, per_worker_gb=4.0, missing=None)
    assert result == {"foo": 1, "per_worker_gb": 4.0}


def test_normalize_hints_kwargs_override_dict():
    result = wp._normalize_hints({"foo": 1}, foo=99)
    assert result == {"foo": 99}


# ---------------------------------------------------------------------------
# estimate_workload_memory
# ---------------------------------------------------------------------------


def test_estimate_workload_memory_uses_profile_default(monkeypatch):
    monkeypatch.delenv(
        "MHCFLURRY_WORKLOAD_AFFINITY_INFERENCE_PER_WORKER_GB", raising=False)
    result = wp.estimate_workload_memory(wp.WORKLOAD_AFFINITY_INFERENCE)
    assert result["device_worker_gb"] == 4.0
    assert "profile default" in result["notes"]


def test_estimate_workload_memory_env_override_wins(monkeypatch):
    monkeypatch.setenv(
        "MHCFLURRY_WORKLOAD_AFFINITY_INFERENCE_PER_WORKER_GB", "12.0")
    result = wp.estimate_workload_memory(
        wp.WORKLOAD_AFFINITY_INFERENCE,
        hints={"per_worker_gb": 8.0},
    )
    assert result["device_worker_gb"] == 12.0
    assert "env override" in result["notes"]


def test_estimate_workload_memory_command_hint_used_when_no_env(monkeypatch):
    monkeypatch.delenv(
        "MHCFLURRY_WORKLOAD_AFFINITY_INFERENCE_PER_WORKER_GB", raising=False)
    result = wp.estimate_workload_memory(
        wp.WORKLOAD_AFFINITY_INFERENCE,
        hints={"per_worker_gb": 7.5},
    )
    assert result["device_worker_gb"] == 7.5
    assert "command estimate" in result["notes"]


def test_estimate_workload_memory_data_pressure_below_start_is_zero(monkeypatch):
    """data_bytes under start_gb should not add pressure."""
    monkeypatch.delenv(
        "MHCFLURRY_WORKLOAD_AFFINITY_INFERENCE_PER_WORKER_GB", raising=False)
    result = wp.estimate_workload_memory(
        wp.WORKLOAD_AFFINITY_INFERENCE,
        hints={"data_bytes": int(1.0 * wp.GIB)},  # under 2 GB start
    )
    assert result["data_pressure_gb"] == 0.0


def test_estimate_workload_memory_data_pressure_scales_then_caps(monkeypatch):
    monkeypatch.delenv(
        "MHCFLURRY_WORKLOAD_AFFINITY_INFERENCE_PER_WORKER_GB", raising=False)
    profile = wp.get_workload_profile(wp.WORKLOAD_AFFINITY_INFERENCE)
    # Below cap: 50 GB data → (50 - 2) * 0.02 = 0.96 GB pressure.
    moderate = wp.estimate_workload_memory(
        wp.WORKLOAD_AFFINITY_INFERENCE,
        hints={"data_bytes": int(50.0 * wp.GIB)},
    )
    assert moderate["data_pressure_gb"] == pytest.approx(0.96, abs=1e-6)
    # Above cap: 10 TB data → would be huge, clamps to data_pressure_cap_gb.
    huge = wp.estimate_workload_memory(
        wp.WORKLOAD_AFFINITY_INFERENCE,
        hints={"data_bytes": int(10000.0 * wp.GIB)},
    )
    assert huge["data_pressure_gb"] == profile.data_pressure_cap_gb


def test_estimate_workload_memory_no_data_pressure_when_device_gb_unknown(
        monkeypatch):
    """Profiles with device_worker_gb=None don't accumulate data pressure."""
    monkeypatch.delenv(
        "MHCFLURRY_WORKLOAD_GENERIC_PER_WORKER_GB", raising=False)
    result = wp.estimate_workload_memory(
        wp.WORKLOAD_GENERIC,
        hints={"data_bytes": int(50.0 * wp.GIB)},
    )
    assert result["device_worker_gb"] is None
    assert result["data_pressure_gb"] == 0.0


def test_estimate_workload_memory_host_pressure_caps(monkeypatch):
    """host_data_cap_gb bounds the host-memory adder."""
    monkeypatch.delenv(
        "MHCFLURRY_WORKLOAD_PROCESSING_TRAINING_PER_WORKER_GB", raising=False)
    profile = wp.get_workload_profile(wp.WORKLOAD_PROCESSING_TRAINING)
    result = wp.estimate_workload_memory(
        wp.WORKLOAD_PROCESSING_TRAINING,
        hints={"data_bytes": int(10000.0 * wp.GIB)},
    )
    # Without cap this would be 10000 * 0.15 = 1500 GB.
    assert result["host_worker_gb"] == (
        profile.host_worker_gb + profile.host_data_cap_gb
    )


# ---------------------------------------------------------------------------
# host_memory_num_jobs_cap
# ---------------------------------------------------------------------------


def test_host_memory_num_jobs_cap_returns_none_when_memory_unknown():
    cap = wp.host_memory_num_jobs_cap(
        memory={"total_gb": None, "available_gb": None},
        host_worker_gb=4.0,
    )
    assert cap is None


def test_host_memory_num_jobs_cap_uses_available_over_total():
    cap = wp.host_memory_num_jobs_cap(
        memory={"available_gb": 50.0, "total_gb": 128.0, "source": "env"},
        host_worker_gb=4.0,
        safety_fraction=1.0,
    )
    assert cap == 12  # 50 / 4 = 12.5 → 12


def test_host_memory_num_jobs_cap_falls_back_to_total():
    cap = wp.host_memory_num_jobs_cap(
        memory={"available_gb": None, "total_gb": 50.0, "source": "env"},
        host_worker_gb=4.0,
        safety_fraction=1.0,
    )
    assert cap == 12


def test_host_memory_num_jobs_cap_applies_safety_fraction():
    cap = wp.host_memory_num_jobs_cap(
        memory={"available_gb": 100.0, "source": "env"},
        host_worker_gb=4.0,
        safety_fraction=0.5,
    )
    assert cap == 12  # 100 * 0.5 / 4 = 12.5 → 12


def test_host_memory_num_jobs_cap_includes_dataloader_overhead():
    """Each dataloader child adds HOST_RAM_PER_DATALOADER_CHILD_GB to the cost."""
    cap = wp.host_memory_num_jobs_cap(
        memory={"available_gb": 100.0, "source": "env"},
        host_worker_gb=4.0,
        dataloader_num_workers=4,
        safety_fraction=1.0,
    )
    # 4 + 4*0.5 = 6 GB/worker → 100/6 = 16
    assert cap == 16


def test_host_memory_num_jobs_cap_min_floor_is_one():
    """Even RAM-starved boxes get at least 1 worker."""
    cap = wp.host_memory_num_jobs_cap(
        memory={"available_gb": 1.0, "source": "env"},
        host_worker_gb=100.0,
        safety_fraction=0.01,
    )
    assert cap == 1


def test_host_memory_num_jobs_cap_reads_safety_env(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION", "0.50")
    cap = wp.host_memory_num_jobs_cap(
        memory={"available_gb": 100.0, "source": "env"},
        host_worker_gb=4.0,
    )
    assert cap == 12  # 100 * 0.5 / 4 = 12


def test_host_memory_num_jobs_cap_rejects_bad_safety_env(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION", "2.0")
    with pytest.raises(ValueError):
        wp.host_memory_num_jobs_cap(
            memory={"available_gb": 100.0, "source": "env"},
            host_worker_gb=4.0,
        )


# ---------------------------------------------------------------------------
# _is_auto
# ---------------------------------------------------------------------------


def test_is_auto_none():
    assert wp._is_auto(None)


def test_is_auto_string():
    assert wp._is_auto("auto")
    assert wp._is_auto("AUTO")
    assert wp._is_auto("Auto")


def test_is_auto_int_is_not_auto():
    assert not wp._is_auto(0)
    assert not wp._is_auto(8)


def test_is_auto_other_string_not_auto():
    assert not wp._is_auto("manual")
    assert not wp._is_auto("")


# ---------------------------------------------------------------------------
# plan_local_parallelism
# ---------------------------------------------------------------------------


def _identity_normalize_backend(b):
    return b or "auto"


def _planner_fakes(
        *,
        num_gpus=1,
        mwpg_value=2,
        num_jobs_capacity=lambda gpus, mwpg: gpus * mwpg,
        dataloader_workers=0,
        rn_pool_epochs=1):
    """Build a kwargs dict of injected callables for plan_local_parallelism."""

    def auto_max_workers_per_gpu(num_jobs, num_gpus, backend, per_worker_gb):
        return mwpg_value

    def auto_num_jobs(gpus, mwpg):
        return num_jobs_capacity(gpus, mwpg)

    def resolve_dataloader_num_workers(raw, num_fit_workers, ram_gb):
        return dataloader_workers

    def auto_random_negative_pool_epochs(
            num_random_negatives, peptide_max_length, num_workers, ram_gb):
        return rn_pool_epochs

    return {
        "normalize_backend": _identity_normalize_backend,
        "detect_num_cuda_devices": lambda: num_gpus,
        "auto_max_workers_per_gpu": auto_max_workers_per_gpu,
        "auto_num_jobs": auto_num_jobs,
        "resolve_dataloader_num_workers": resolve_dataloader_num_workers,
        "auto_random_negative_pool_epochs": auto_random_negative_pool_epochs,
    }


def _args(**overrides):
    """Build an argparse.Namespace with auto defaults that match the CLI."""
    defaults = dict(
        backend="auto",
        gpus=None,
        max_workers_per_gpu="auto",
        num_jobs="auto",
        dataloader_num_workers="auto",
        random_negative_pool_epochs="auto",
        torch_compile="auto",
        torch_compile_loss="auto",
        matmul_precision="none",
        enable_timing=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_plan_cpu_only_box(monkeypatch):
    """0 GPUs → num_jobs=0, capacity=0, no cli overrides."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "32.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "16.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=0, mwpg_value=1, num_jobs_capacity=lambda g, m: 0),
    )
    assert plan.gpus == 0
    assert plan.gpus_was_auto
    assert plan.num_jobs == 0
    assert plan.capacity == 0
    assert plan.cli_overrides == ()


def test_plan_single_gpu_resolved(monkeypatch):
    """1 GPU, MWPG=2 → capacity=2, num_jobs=2 under auto."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "200.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=1, mwpg_value=2),
    )
    assert plan.gpus == 1
    assert plan.max_workers_per_gpu == 2
    assert plan.num_jobs == 2
    assert plan.capacity == 2
    assert plan.num_jobs_was_auto


def test_plan_8x_a100_resolved(monkeypatch):
    """8×A100-style box with MWPG=2 → 16 workers."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "1024.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "900.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=8, mwpg_value=2),
    )
    assert plan.num_jobs == 16
    assert plan.capacity == 16


def test_plan_explicit_num_jobs_records_override(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "200.0")
    plan = wp.plan_local_parallelism(
        _args(num_jobs=4),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=1, mwpg_value=2),
    )
    assert plan.num_jobs == 4
    assert not plan.num_jobs_was_auto
    assert "num_jobs" in plan.cli_overrides


def test_plan_explicit_num_jobs_above_capacity_warns(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "200.0")
    plan = wp.plan_local_parallelism(
        _args(num_jobs=99),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=1, mwpg_value=2),
    )
    assert plan.num_jobs == 99
    assert any("exceeds GPU capacity" in w for w in plan.warnings), plan.warnings


def test_plan_auto_num_jobs_clipped_by_host_memory(monkeypatch):
    """8 GPUs × MWPG 2 = 16 capacity, but only 8 GB RAM forces fewer workers."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "8.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "8.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,  # host_worker_gb=3.0
        **_planner_fakes(num_gpus=8, mwpg_value=2),
    )
    # 8 GB * 0.70 / 3 GB ≈ 1.87 → 1 worker
    assert plan.num_jobs == 1
    assert plan.host_memory_num_jobs_cap == 1
    assert any("capped from" in w for w in plan.warnings), plan.warnings


def test_plan_auto_num_jobs_reclipped_after_dataloader_sizing(monkeypatch):
    """Auto jobs must include per-DataLoader-child RAM in the final cap."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "16.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "16.0")
    monkeypatch.setenv("MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION", "1.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=8, mwpg_value=2, dataloader_workers=4),
    )
    assert plan.dataloader_num_workers == 4
    assert plan.host_worker_gb == 5.0
    assert plan.host_memory_num_jobs_cap == 3
    assert plan.num_jobs == 3
    assert plan.num_jobs <= plan.host_memory_num_jobs_cap
    assert any("capped from 5 to 3" in w for w in plan.warnings), plan.warnings


def test_plan_explicit_num_jobs_above_host_memory_warns_but_honors(monkeypatch):
    """User CLI override is respected; the planner just records a warning."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "8.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "8.0")
    plan = wp.plan_local_parallelism(
        _args(num_jobs=8),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=8, mwpg_value=2),
    )
    assert plan.num_jobs == 8  # CLI override respected
    assert any("host-memory estimate" in w for w in plan.warnings), plan.warnings


def test_plan_explicit_num_jobs_warning_uses_dataloader_inflated_cap(monkeypatch):
    """Explicit-num_jobs warning cap must include per-DataLoader-child RAM.

    Without the in-loop reclip the explicit path would compare against the
    raw host_worker_gb cap and silently undercount memory pressure from
    DataLoader children. The cap reported in the warning must match the
    one returned in the plan (both inflated by dataloader_num_workers).
    """
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "16.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "16.0")
    monkeypatch.setenv("MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION", "1.0")
    plan = wp.plan_local_parallelism(
        _args(num_jobs=8),  # explicit, above the inflated cap
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,  # host_worker_gb=3.0
        **_planner_fakes(num_gpus=8, mwpg_value=2, dataloader_workers=4),
    )
    # 16 GB * 1.0 / (3.0 + 4*0.5) = 16/5 = 3.2 → cap=3.
    # Without the dataloader inflation it would be 16/3 = 5.33 → cap=5,
    # so the cap reported in the warning must be 3, not 5.
    assert plan.num_jobs == 8  # CLI override honored
    assert plan.host_memory_num_jobs_cap == 3
    assert plan.host_worker_gb == 5.0
    cap_warnings = [w for w in plan.warnings if "host-memory estimate" in w]
    assert len(cap_warnings) == 1, plan.warnings
    assert "exceeds host-memory estimate 3" in cap_warnings[0], cap_warnings[0]
    assert "exceeds host-memory estimate 5" not in cap_warnings[0], (
        cap_warnings[0])


def test_plan_cap_auto_num_jobs_false_disables_clip(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "8.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "8.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        cap_auto_num_jobs=False,
        **_planner_fakes(num_gpus=8, mwpg_value=2),
    )
    assert plan.num_jobs == 16  # unclipped capacity
    assert not any("capped from" in w for w in plan.warnings)


def test_plan_clip_rebalances_mwpg_when_mwpg_was_auto(monkeypatch):
    """Clipping num_jobs while MWPG was auto should redistribute workers."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "16.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "16.0")
    plan = wp.plan_local_parallelism(
        _args(),  # MWPG="auto", num_jobs="auto"
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,  # host_worker_gb=3
        **_planner_fakes(num_gpus=8, mwpg_value=2),
    )
    # 16 GB * 0.70 / 3 ≈ 3 workers across 8 GPUs → MWPG must be ceil(3/8)=1
    assert plan.num_jobs == 3
    assert plan.max_workers_per_gpu == 1
    assert plan.max_workers_per_gpu_was_auto


def test_plan_clip_leaves_mwpg_when_explicit(monkeypatch):
    """An explicit MWPG should survive a clip — only auto MWPG gets rebalanced."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "16.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "16.0")
    plan = wp.plan_local_parallelism(
        _args(max_workers_per_gpu=4),  # explicit MWPG
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=8, mwpg_value=4),
    )
    assert plan.max_workers_per_gpu == 4
    assert "max_workers_per_gpu" in plan.cli_overrides


def test_plan_explicit_gpus_recorded(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "200.0")
    plan = wp.plan_local_parallelism(
        _args(gpus=2),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=8, mwpg_value=2),  # detector says 8
    )
    assert plan.gpus == 2  # CLI wins
    assert not plan.gpus_was_auto
    assert "gpus" in plan.cli_overrides


def test_plan_torch_compile_override_recorded(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "32.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "16.0")
    plan = wp.plan_local_parallelism(
        _args(torch_compile="on", torch_compile_loss="on", matmul_precision="high"),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=1, mwpg_value=1),
    )
    assert plan.torch_compile == "on"
    assert plan.torch_compile_loss == "on"
    assert plan.matmul_precision == "high"
    assert "torch_compile" in plan.cli_overrides
    assert "torch_compile_loss" in plan.cli_overrides
    assert "matmul_precision" in plan.cli_overrides


def test_plan_cpu_backend_zero_capacity(monkeypatch):
    """backend=cpu → capacity=0 regardless of detected GPUs."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "200.0")
    plan = wp.plan_local_parallelism(
        _args(backend="cpu"),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=4, mwpg_value=2),
    )
    assert plan.backend == "cpu"
    assert plan.capacity == 0
    assert "backend" in plan.cli_overrides


def test_plan_records_memory_provenance(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "200.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=1, mwpg_value=1),
    )
    assert plan.host_memory_total_gb == 256.0
    assert plan.host_memory_available_gb == 200.0
    assert plan.host_memory_source == "env"


def test_plan_dataloader_workers_added_to_host_worker_gb(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "200.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,  # host_worker_gb=3.0
        **_planner_fakes(num_gpus=1, mwpg_value=1, dataloader_workers=4),
    )
    # 3 + 4 * 0.5 = 5
    assert plan.host_worker_gb == 5.0


def test_plan_str_renders_summary(monkeypatch):
    """LocalParallelismPlan.__str__ surfaces all the orchestration knobs."""
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "32.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "16.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        **_planner_fakes(num_gpus=1, mwpg_value=2),
    )
    rendered = str(plan)
    assert "workload=affinity_inference" in rendered
    assert "gpus=1" in rendered
    assert "workers/gpu=2" in rendered


def test_plan_hints_pass_through_to_estimator(monkeypatch):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "256.0")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "200.0")
    plan = wp.plan_local_parallelism(
        _args(),
        workload_name=wp.WORKLOAD_AFFINITY_INFERENCE,
        workload_hints={"data_bytes": int(50.0 * wp.GIB)},
        **_planner_fakes(num_gpus=1, mwpg_value=1),
    )
    # data_pressure_gb computed from the hint
    assert plan.data_pressure_gb > 0
    assert ("data_bytes", int(50.0 * wp.GIB)) in plan.hints

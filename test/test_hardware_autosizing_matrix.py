# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Characteristic-driven hardware matrix for automatic worker sizing."""

from argparse import Namespace

import pytest

from mhcflurry.memory_budget import GIB, usable_memory_bytes
from mhcflurry.parallelism import (
    refine_local_parallelism_from_warmup,
    resolve_local_parallelism_args,
)
from mhcflurry.parallelism import planning
from mhcflurry.parallelism import worker_runtime
from mhcflurry.workload_planning import (
    WORKLOAD_AFFINITY_INFERENCE,
    WORKLOAD_AFFINITY_TRAINING,
    WORKLOAD_PRESENTATION_INFERENCE,
    WORKLOAD_PROCESSING_INFERENCE,
)


# Maximums observed by the seven-architecture full-residency affinity probe on
# the 4xA100-40GB release host. CUDA's process-level estimate is intentionally
# larger than the allocator counters because it includes the CUDA context and
# non-PyTorch allocations.
AFFINITY_A100_PROBE_REPORT = {
    "cuda_process_peak_estimate_bytes": 38_003_998_720,
    "cuda_peak_allocated_bytes": 35_649_949_184,
    "cuda_peak_reserved_bytes": 37_459_329_024,
    "host_peak_rss_bytes": 12_021_784_576,
}


HARDWARE_CASES = (
    pytest.param(
        {
            "gpus": 0,
            "free_vram_gb": None,
            "cpus": 8,
            "total_ram_gb": 16.0,
            "available_ram_gb": 12.0,
            "initial": (0, 1, 0, 6),
            "final": None,
        },
        id="cpu-only-8cpu-16gb",
    ),
    pytest.param(
        {
            "backend": "cpu",
            "gpus": 4,
            "free_vram_gb": 80.0,
            "cpus": 8,
            "total_ram_gb": 16.0,
            "available_ram_gb": 12.0,
            "initial": (0, 1, 0, 6),
            "final": None,
        },
        id="cpu-selected-with-4-gpus-visible",
    ),
    pytest.param(
        {
            "gpus": 1,
            "free_vram_gb": 32.0,
            "cpus": 32,
            "total_ram_gb": 64.0,
            "available_ram_gb": 60.0,
            "initial": (11, 11, 1, 1),
            "final": (1, 1, 1, 29),
            "undersized": True,
        },
        id="rtx-5090-32gb-workstation",
    ),
    pytest.param(
        {
            "gpus": 1,
            "free_vram_gb": 40_442 / 1024.0,
            "cpus": 12,
            "total_ram_gb": 85.0,
            "available_ram_gb": 80.0,
            "initial": (12, 12, 0, 1),
            "final": (1, 1, 0, 10),
        },
        id="a100-40gb-single",
    ),
    pytest.param(
        {
            "gpus": 4,
            "free_vram_gb": 40_442 / 1024.0,
            "cpus": 48,
            "total_ram_gb": 340.0,
            "available_ram_gb": 329.3,
            "initial": (48, 12, 0, 1),
            "final": (4, 1, 0, 11),
            "golden": True,
        },
        id="a100-40gb-4x-gcp-observed",
    ),
    pytest.param(
        {
            "gpus": 4,
            "free_vram_gb": 80.0,
            "cpus": 96,
            "total_ram_gb": 640.0,
            "available_ram_gb": 600.0,
            "initial": (96, 24, 0, 1),
            "final": (4, 1, 0, 23),
        },
        id="h100-80gb-4x",
    ),
    pytest.param(
        {
            "gpus": 8,
            "free_vram_gb": 141.0,
            "cpus": 112,
            "total_ram_gb": 2048.0,
            "available_ram_gb": 1900.0,
            "initial": (112, 14, 0, 1),
            "final": (24, 3, 0, 4),
        },
        id="h200-141gb-8x",
    ),
    pytest.param(
        {
            "gpus": 8,
            "free_vram_gb": 141.0,
            "cpus": 16,
            "total_ram_gb": 2048.0,
            "available_ram_gb": 1900.0,
            "initial": (16, 2, 0, 1),
            "final": (16, 2, 0, 1),
        },
        id="h200-141gb-8x-cpu-constrained",
    ),
    pytest.param(
        {
            "gpus": 4,
            "free_vram_gb": 80.0,
            "cpus": 96,
            "total_ram_gb": 40.0,
            "available_ram_gb": 32.0,
            "initial": (6, 2, 3, 12),
            "final": (2, 1, 3, 44),
        },
        id="h100-80gb-4x-ram-constrained",
    ),
)


def _resolve_case(monkeypatch, case):
    """Resolve one matrix row without depending on the test runner's host."""
    for name in (
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP",
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB",
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_VRAM_FRACTION",
            "MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION",
            "MHCFLURRY_AUTO_DATALOADER_HARD_CAP",
            "MHCFLURRY_TORCH_COMPILE",
            "TORCHINDUCTOR_COMPILE_THREADS",
            "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MHCFLURRY_OMP_NUM_THREADS_AUTO",
            "MHCFLURRY_MKL_NUM_THREADS_AUTO",
            "MHCFLURRY_OPENBLAS_NUM_THREADS_AUTO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", str(case["total_ram_gb"]))
    monkeypatch.setenv(
        "MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB",
        str(case["available_ram_gb"]),
    )
    if case["free_vram_gb"] is None:
        monkeypatch.delenv(
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB",
            str(case["free_vram_gb"]),
        )
    monkeypatch.setattr(planning.os, "cpu_count", lambda: case["cpus"])
    monkeypatch.setattr(planning, "configure_pytorch", lambda **_kwargs: None)
    monkeypatch.setattr(
        worker_runtime,
        "configure_worker_cpu_threads",
        lambda num_threads, auto_owned=True: num_threads,
    )
    args = Namespace(
        backend=case.get("backend", "auto"),
        gpus=case["gpus"],
        max_workers_per_gpu="auto",
        num_jobs="auto",
        dataloader_num_workers="auto",
        random_negative_pool_epochs="auto",
        torch_compile="auto",
        torch_compile_loss="auto",
        matmul_precision="none",
        enable_timing=False,
        cluster_parallelism=False,
    )
    resolve_local_parallelism_args(
        args,
        workload_name=WORKLOAD_AFFINITY_TRAINING,
        per_worker_gb=2.5,
    )
    return args


def _plan_shape(plan):
    return (
        plan.num_jobs,
        plan.max_workers_per_gpu,
        plan.dataloader_num_workers,
        plan.cpu_threads_per_worker,
    )


@pytest.mark.parametrize("case", HARDWARE_CASES)
def test_affinity_autosizer_hardware_matrix(monkeypatch, case):
    """Capacity follows resource facts and a measured workload envelope.

    Accelerator labels document representative machines; the autosizer sees
    only GPU count/free memory, CPUs, RAM, and the workload measurements. Only
    the 4xA100 row is an empirical golden result. Other accelerator rows test
    how the same measured envelope maps onto published memory capacities.
    """
    args = _resolve_case(monkeypatch, case)
    initial = args.workload_plan
    assert _plan_shape(initial) == case["initial"]
    if initial.gpus:
        assert initial.capacity <= (
            initial.gpus * initial.max_workers_per_gpu
        )
    else:
        assert initial.capacity == 0
    assert initial.num_jobs <= case["cpus"]
    if initial.host_memory_num_jobs_cap is not None and initial.num_jobs:
        assert initial.num_jobs <= initial.host_memory_num_jobs_cap

    if case["final"] is None:
        assert initial.backend == "cpu" or initial.gpus == 0
        assert args.device_memory_budget_bytes is None
        return

    refine_local_parallelism_from_warmup(
        args,
        [AFFINITY_A100_PROBE_REPORT],
    )
    final = args.workload_plan
    assert _plan_shape(final) == case["final"]
    assert final.num_jobs <= initial.num_jobs
    assert final.max_workers_per_gpu <= initial.max_workers_per_gpu
    assert final.capacity <= final.gpus * final.max_workers_per_gpu
    assert final.num_jobs <= case["cpus"]
    if final.host_memory_num_jobs_cap is not None and final.num_jobs:
        assert final.num_jobs <= final.host_memory_num_jobs_cap

    usable_device_gb = usable_memory_bytes(
        case["free_vram_gb"] * GIB,
    ) / GIB
    assert (
        final.device_memory_budget_gb * final.max_workers_per_gpu
        <= usable_device_gb + 1e-9
    )
    if case.get("undersized"):
        assert final.device_worker_gb > final.device_memory_budget_gb
        assert any(
            "observed device peak" in warning
            for warning in final.warnings
        )
    else:
        assert final.warmup_device_peak_gb <= final.device_memory_budget_gb

    if case.get("golden"):
        assert initial.device_memory_budget_gb == pytest.approx(2.9621, abs=1e-4)
        assert final.warmup_device_peak_gb == pytest.approx(35.3940, abs=1e-4)
        assert final.device_worker_gb == pytest.approx(40.7031, abs=1e-4)
        assert final.warmup_host_peak_gb == pytest.approx(11.1962, abs=1e-4)
        assert final.host_worker_gb == pytest.approx(12.3158, abs=1e-4)
        assert final.device_memory_budget_gb == pytest.approx(35.5447, abs=1e-4)


INFERENCE_HARDWARE_CASES = (
    pytest.param(1, 32.0, 64, 512.0, (7, 2, 1), id="rtx-5090-32gb"),
    pytest.param(
        1, 40_442 / 1024.0, 64, 512.0, (8, 3, 2), id="a100-40gb"),
    pytest.param(
        4, 40_442 / 1024.0, 48, 340.0, (32, 12, 8), id="4x-a100-40gb"),
    pytest.param(4, 80.0, 96, 640.0, (72, 28, 16), id="4x-h100-80gb"),
    pytest.param(8, 141.0, 112, 2048.0, (112, 96, 56), id="8x-h200-141gb"),
)


@pytest.mark.parametrize(
    ("gpus", "free_vram_gb", "cpus", "ram_gb", "expected_jobs"),
    INFERENCE_HARDWARE_CASES,
)
def test_inference_autosizer_hardware_matrix(
        monkeypatch, gpus, free_vram_gb, cpus, ram_gb, expected_jobs):
    """Elastic batches never erase complete resident-workload capacity."""
    monkeypatch.setenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB",
        str(free_vram_gb),
    )
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", str(ram_gb))
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", str(ram_gb))
    monkeypatch.setattr(planning.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(planning, "configure_pytorch", lambda **_kwargs: None)
    monkeypatch.setattr(
        worker_runtime,
        "configure_worker_cpu_threads",
        lambda num_threads, auto_owned=True: num_threads,
    )
    for name in (
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP",
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_PER_WORKER_GB",
            "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_VRAM_FRACTION",
            "MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION"):
        monkeypatch.delenv(name, raising=False)

    workloads = (
        WORKLOAD_AFFINITY_INFERENCE,
        WORKLOAD_PROCESSING_INFERENCE,
        WORKLOAD_PRESENTATION_INFERENCE,
    )
    for workload, expected in zip(workloads, expected_jobs):
        args = Namespace(
            backend="auto",
            gpus=gpus,
            max_workers_per_gpu="auto",
            num_jobs="auto",
            dataloader_num_workers="auto",
            random_negative_pool_epochs="auto",
            torch_compile="auto",
            torch_compile_loss="auto",
            matmul_precision="none",
            enable_timing=False,
            cluster_parallelism=False,
        )
        resolve_local_parallelism_args(
            args,
            workload_name=workload,
            workload_hints={
                # Matches the order of magnitude of the 2.3 release archives;
                # it must not replace the complete resident-workload floor.
                "model_bytes": 200 * (1 << 20),
                "elastic_batch": True,
                "prediction_rows": 2_054_263,
            },
        )
        assert args.num_jobs == expected
        assert args.workload_plan.device_worker_gb == {
            WORKLOAD_AFFINITY_INFERENCE: 4.0,
            WORKLOAD_PROCESSING_INFERENCE: 10.0,
            WORKLOAD_PRESENTATION_INFERENCE: 16.0,
        }[workload]

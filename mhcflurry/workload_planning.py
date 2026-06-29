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

"""Central workload planning for local GPU/CPU orchestration."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple


GIB = float(1 << 30)
HOST_RAM_PER_DATALOADER_CHILD_GB = 0.5


def env_float(name, default, bounds=None):
    """Read ``name`` from the environment as float, falling back to ``default``.

    ``bounds`` is an optional ``(low, high)`` tuple; either side may be
    ``None``. Both endpoints are inclusive. Bad values are reported with the
    offending variable name so users don't get a bare ``ValueError``.
    """
    raw = os.environ.get(name)
    if raw in (None, ""):
        value = float(default)
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                "Environment variable %s=%r is not a valid float: %s"
                % (name, raw, exc)
            ) from None
    if bounds is not None:
        low, high = bounds
        if (low is not None and value < low) or (
                high is not None and value > high):
            raise ValueError(
                "Environment variable %s=%r is outside allowed range %s"
                % (name, raw if raw not in (None, "") else value, bounds)
            )
    return value


def env_int(name, default, bounds=None):
    """Read ``name`` from the environment as int, falling back to ``default``.

    Bad values surface with the offending variable name. ``bounds`` works the
    same as :func:`env_float`.
    """
    raw = os.environ.get(name)
    if raw in (None, ""):
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "Environment variable %s=%r is not a valid int: %s"
                % (name, raw, exc)
            ) from None
    if bounds is not None:
        low, high = bounds
        if (low is not None and value < low) or (
                high is not None and value > high):
            raise ValueError(
                "Environment variable %s=%r is outside allowed range %s"
                % (name, raw if raw not in (None, "") else value, bounds)
            )
    return value

WORKLOAD_GENERIC = "generic"
WORKLOAD_AFFINITY_TRAINING = "affinity_training"
WORKLOAD_AFFINITY_INFERENCE = "affinity_inference"
WORKLOAD_AFFINITY_SELECTION = "affinity_selection"
WORKLOAD_AFFINITY_CALIBRATION = "affinity_calibration"
WORKLOAD_PROCESSING_DATA = "processing_data"
WORKLOAD_PROCESSING_TRAINING = "processing_training"
WORKLOAD_PROCESSING_INFERENCE = "processing_inference"
WORKLOAD_PROCESSING_SELECTION = "processing_selection"
WORKLOAD_PRESENTATION_TRAINING = "presentation_training"
WORKLOAD_PRESENTATION_CALIBRATION = "presentation_calibration"
WORKLOAD_PRESENTATION_INFERENCE = "presentation_inference"


@dataclass(frozen=True)
class WorkloadProfile:
    """Policy defaults for one workload family."""

    name: str
    device_worker_gb: Optional[float]
    data_pressure_start_gb: float = 0.0
    data_pressure_factor: float = 0.0
    data_pressure_cap_gb: float = 0.0
    host_worker_gb: float = 2.0
    host_data_multiplier: float = 0.0
    host_data_cap_gb: float = 32.0
    description: str = ""


@dataclass(frozen=True)
class LocalParallelismPlan:
    """Resolved local orchestration plan.

    This is the single object readers need to understand: it records the
    workload, hardware facts, CLI overrides, planner estimates, and final
    values that get written back onto argparse ``args``.
    """

    workload_name: str
    backend: str
    gpus: int
    gpus_was_auto: bool
    max_workers_per_gpu: int
    max_workers_per_gpu_was_auto: bool
    num_jobs: int
    num_jobs_was_auto: bool
    dataloader_num_workers: int
    dataloader_num_workers_was_auto: bool
    random_negative_pool_epochs: Optional[int]
    random_negative_pool_epochs_was_auto: bool
    torch_compile: str
    torch_compile_loss: str
    matmul_precision: str
    enable_timing: bool
    capacity: int
    device_worker_gb: Optional[float]
    data_pressure_gb: float
    host_worker_gb: float
    host_memory_total_gb: Optional[float]
    host_memory_available_gb: Optional[float]
    host_memory_source: str
    host_memory_num_jobs_cap: Optional[int]
    cli_overrides: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    hints: Tuple[Tuple[str, object], ...] = ()

    def __str__(self):
        device_worker = (
            "%.1f" % self.device_worker_gb
            if self.device_worker_gb is not None
            else "default"
        )
        host_available = (
            "%.1f GB" % self.host_memory_available_gb
            if self.host_memory_available_gb is not None
            else "unknown"
        )
        return (
            "LocalParallelismPlan("
            "workload=%s, backend=%s, jobs=%d, gpus=%d, workers/gpu=%d, "
            "dataloader_workers=%d, device_worker_gb=%s, "
            "host_worker_gb=%.1f, host_available=%s from %s)"
            % (
                self.workload_name,
                self.backend,
                self.num_jobs,
                self.gpus,
                self.max_workers_per_gpu,
                self.dataloader_num_workers,
                device_worker,
                self.host_worker_gb,
                host_available,
                self.host_memory_source,
            )
        )


WORKLOAD_PROFILES = {
    WORKLOAD_GENERIC: WorkloadProfile(
        name=WORKLOAD_GENERIC,
        device_worker_gb=None,
        host_worker_gb=2.0,
        description="Fallback plan; preserves global auto_max_workers defaults.",
    ),
    WORKLOAD_AFFINITY_TRAINING: WorkloadProfile(
        name=WORKLOAD_AFFINITY_TRAINING,
        device_worker_gb=None,
        data_pressure_start_gb=4.0,
        data_pressure_factor=0.02,
        data_pressure_cap_gb=2.0,
        host_worker_gb=3.0,
        host_data_multiplier=0.05,
        description="Pan-allele affinity training and fine-tuning.",
    ),
    WORKLOAD_AFFINITY_INFERENCE: WorkloadProfile(
        name=WORKLOAD_AFFINITY_INFERENCE,
        device_worker_gb=4.0,
        data_pressure_start_gb=2.0,
        data_pressure_factor=0.02,
        data_pressure_cap_gb=2.0,
        host_worker_gb=3.0,
        host_data_multiplier=0.02,
        description="Affinity predictor inference.",
    ),
    WORKLOAD_AFFINITY_SELECTION: WorkloadProfile(
        name=WORKLOAD_AFFINITY_SELECTION,
        device_worker_gb=None,
        host_worker_gb=3.0,
        host_data_multiplier=0.02,
        description="Affinity model selection / held-out scoring.",
    ),
    WORKLOAD_AFFINITY_CALIBRATION: WorkloadProfile(
        name=WORKLOAD_AFFINITY_CALIBRATION,
        device_worker_gb=24.0,
        host_worker_gb=4.0,
        description="Affinity percentile-rank calibration.",
    ),
    WORKLOAD_PROCESSING_DATA: WorkloadProfile(
        name=WORKLOAD_PROCESSING_DATA,
        device_worker_gb=8.0,
        data_pressure_start_gb=2.0,
        data_pressure_factor=0.05,
        data_pressure_cap_gb=4.0,
        host_worker_gb=4.0,
        host_data_multiplier=0.10,
        description="Processing training-data generation.",
    ),
    WORKLOAD_PROCESSING_TRAINING: WorkloadProfile(
        name=WORKLOAD_PROCESSING_TRAINING,
        device_worker_gb=8.0,
        data_pressure_start_gb=2.0,
        data_pressure_factor=0.10,
        data_pressure_cap_gb=8.0,
        host_worker_gb=4.0,
        host_data_multiplier=0.15,
        description="Processing Conv1d training.",
    ),
    WORKLOAD_PROCESSING_INFERENCE: WorkloadProfile(
        name=WORKLOAD_PROCESSING_INFERENCE,
        device_worker_gb=8.0,
        data_pressure_start_gb=2.0,
        data_pressure_factor=0.03,
        data_pressure_cap_gb=4.0,
        host_worker_gb=4.0,
        host_data_multiplier=0.03,
        description="Processing predictor inference.",
    ),
    WORKLOAD_PROCESSING_SELECTION: WorkloadProfile(
        name=WORKLOAD_PROCESSING_SELECTION,
        device_worker_gb=8.0,
        data_pressure_start_gb=2.0,
        data_pressure_factor=0.05,
        data_pressure_cap_gb=4.0,
        host_worker_gb=4.0,
        host_data_multiplier=0.05,
        description="Processing model selection / AUC scoring.",
    ),
    WORKLOAD_PRESENTATION_TRAINING: WorkloadProfile(
        name=WORKLOAD_PRESENTATION_TRAINING,
        device_worker_gb=16.0,
        data_pressure_start_gb=2.0,
        data_pressure_factor=0.05,
        data_pressure_cap_gb=6.0,
        host_worker_gb=6.0,
        host_data_multiplier=0.05,
        description="Presentation feature generation and model fitting.",
    ),
    WORKLOAD_PRESENTATION_CALIBRATION: WorkloadProfile(
        name=WORKLOAD_PRESENTATION_CALIBRATION,
        device_worker_gb=24.0,
        data_pressure_start_gb=2.0,
        data_pressure_factor=0.05,
        data_pressure_cap_gb=6.0,
        host_worker_gb=6.0,
        host_data_multiplier=0.02,
        description="Presentation percentile-rank calibration.",
    ),
    WORKLOAD_PRESENTATION_INFERENCE: WorkloadProfile(
        name=WORKLOAD_PRESENTATION_INFERENCE,
        device_worker_gb=16.0,
        data_pressure_start_gb=2.0,
        data_pressure_factor=0.05,
        data_pressure_cap_gb=6.0,
        host_worker_gb=6.0,
        host_data_multiplier=0.05,
        description="Presentation predictor inference.",
    ),
}


def get_workload_profile(workload_name):
    """Return a workload profile, falling back to generic for unknown names."""
    return WORKLOAD_PROFILES.get(workload_name, WORKLOAD_PROFILES[WORKLOAD_GENERIC])


def path_size_bytes(path):
    """Best-effort file size lookup for planner hints."""
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _memory_info(total_gb=None, available_gb=None, source="unknown"):
    return {
        "total_gb": total_gb,
        "available_gb": available_gb,
        "source": source,
    }


def _memory_env_override():
    total_raw = os.environ.get("MHCFLURRY_SYSTEM_RAM_GB")
    available_raw = os.environ.get("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB")
    if total_raw in (None, "") and available_raw in (None, ""):
        return None
    return _memory_info(
        total_gb=env_float(
            "MHCFLURRY_SYSTEM_RAM_GB", 0.0, bounds=(0.0, None))
            if total_raw not in (None, "") else None,
        available_gb=env_float(
            "MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", 0.0, bounds=(0.0, None))
            if available_raw not in (None, "") else None,
        source="env",
    )


def _linux_memory_info_gb():
    values = {}
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                key, rest = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    parts = rest.split()
                    values[key] = float(parts[0]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None
    if not values:
        return None
    return _memory_info(
        total_gb=values.get("MemTotal"),
        available_gb=values.get("MemAvailable"),
        source="/proc/meminfo",
    )


def _macos_memory_info_gb():
    try:
        total_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8").strip())
    except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError):
        total_bytes = None

    try:
        output = subprocess.check_output(
            ["vm_stat"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="ignore")
    except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        output = ""

    page_size = 4096
    first_line = output.splitlines()[0] if output else ""
    match = re.search(r"page size of (\d+) bytes", first_line)
    if match:
        page_size = int(match.group(1))

    pages = {}
    for line in output.splitlines()[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            pages[key.strip()] = int(value.strip().rstrip("."))
        except ValueError:
            continue
    available_pages = sum(
        pages.get(name, 0)
        for name in ("Pages free", "Pages inactive", "Pages speculative")
    )
    available_bytes = (
        available_pages * page_size if available_pages else None
    )
    if total_bytes is None and available_bytes is None:
        return None
    return _memory_info(
        total_gb=total_bytes / GIB if total_bytes is not None else None,
        available_gb=(
            available_bytes / GIB if available_bytes is not None else None
        ),
        source="vm_stat",
    )


def system_memory_info_gb():
    """Return host memory totals without importing heavy optional packages."""
    return (
        _memory_env_override()
        or _linux_memory_info_gb()
        or _macos_memory_info_gb()
        or _memory_info()
    )


def _workload_env_name(workload_name):
    return "MHCFLURRY_WORKLOAD_%s_PER_WORKER_GB" % (
        workload_name.upper().replace("-", "_"),
    )


def _normalize_hints(hints=None, **extra_hints):
    values = {}
    if hints:
        values.update(dict(hints))
    values.update({k: v for (k, v) in extra_hints.items() if v is not None})
    return values


def estimate_workload_memory(workload_name=WORKLOAD_GENERIC, hints=None):
    """Estimate device and host memory for a workload."""
    hints = _normalize_hints(hints)
    profile = get_workload_profile(workload_name)
    notes = []
    data_gb = (hints.get("data_bytes") or 0) / GIB

    workload_env = _workload_env_name(profile.name)
    if os.environ.get(workload_env) not in (None, ""):
        device_worker_gb = env_float(workload_env, 0.0, bounds=(0.0, None))
        notes.append("env override")
    elif hints.get("per_worker_gb") is not None:
        device_worker_gb = float(hints["per_worker_gb"])
        notes.append("command estimate")
    else:
        device_worker_gb = profile.device_worker_gb
        if device_worker_gb is not None:
            notes.append("profile default")

    data_pressure_gb = 0.0
    if hints.get("data_bytes") and device_worker_gb is not None:
        if data_gb > profile.data_pressure_start_gb:
            data_pressure_gb = min(
                profile.data_pressure_cap_gb,
                (data_gb - profile.data_pressure_start_gb)
                * profile.data_pressure_factor,
            )
            if data_pressure_gb:
                device_worker_gb += data_pressure_gb
                notes.append("data pressure %.2f GB" % data_pressure_gb)

    host_worker_gb = profile.host_worker_gb
    if data_gb and profile.host_data_multiplier:
        host_worker_gb += min(
            profile.host_data_cap_gb,
            data_gb * profile.host_data_multiplier,
        )

    return {
        "workload_name": profile.name,
        "device_worker_gb": device_worker_gb,
        "host_worker_gb": host_worker_gb,
        "data_pressure_gb": data_pressure_gb,
        "hints": tuple(sorted(hints.items())),
        "notes": tuple(notes),
    }


def host_memory_num_jobs_cap(
        memory,
        host_worker_gb,
        dataloader_num_workers=0,
        safety_fraction=None):
    """Return worker cap from currently available host memory."""
    available_gb = memory.get("available_gb") or memory.get("total_gb")
    if available_gb is None:
        return None
    if safety_fraction is None:
        safety_fraction = env_float(
            "MHCFLURRY_AUTO_HOST_MEMORY_SAFETY_FRACTION",
            "0.70",
            bounds=(0.0, 1.0),
        )
    worker_gb = (
        float(host_worker_gb)
        + int(dataloader_num_workers) * HOST_RAM_PER_DATALOADER_CHILD_GB
    )
    if worker_gb <= 0:
        return None
    return max(1, int(float(available_gb) * float(safety_fraction) / worker_gb))


def _is_auto(value):
    return value is None or (isinstance(value, str) and value.lower() == "auto")


def _clip_auto_num_jobs_to_host_memory(
        num_jobs, max_workers_per_gpu, num_gpus, was_auto_mwpg,
        memory, memory_plan, warnings, dataloader_num_workers=0):
    """Clamp an auto-resolved ``num_jobs`` to what host RAM can support.

    Returns ``(num_jobs, max_workers_per_gpu, host_cap)``. ``host_cap`` is
    ``None`` when no clamp was applied (the input fit within memory); when a
    clamp happens it is the host-memory worker cap, which equals the returned
    ``num_jobs``. Callers use a non-None ``host_cap`` to report capacity as the
    clamped worker count rather than re-deriving it from the (ceil-rounded)
    ``max_workers_per_gpu``.
    """
    cap = host_memory_num_jobs_cap(
        memory,
        memory_plan["host_worker_gb"],
        dataloader_num_workers=dataloader_num_workers,
    )
    if cap is None or int(num_jobs) <= cap:
        return num_jobs, max_workers_per_gpu, None
    worker_gb = (
        memory_plan["host_worker_gb"]
        + int(dataloader_num_workers) * HOST_RAM_PER_DATALOADER_CHILD_GB
    )
    warnings.append(
        "auto num_jobs capped from %d to %d by available host memory "
        "(%.1f GB from %s, workload=%s, %.1f GB/worker)" % (
            int(num_jobs),
            cap,
            memory.get("available_gb") or memory.get("total_gb"),
            memory.get("source"),
            memory_plan["workload_name"],
            worker_gb,
        )
    )
    num_jobs = cap
    if was_auto_mwpg and num_gpus > 0:
        max_workers_per_gpu = max(1, (int(num_jobs) + int(num_gpus) - 1) // int(num_gpus))
    return num_jobs, max_workers_per_gpu, cap


# Free VRAM below this multiple of the per-worker estimate is flagged as
# "way below safe operating range" — the auto batch-sizer can still shrink to
# fit, but the margin for a worker's minimum working set is thin.
_CAPACITY_VRAM_SAFE_FRACTION = 1.0
# Free VRAM this many times larger on the biggest GPU than the smallest is
# flagged as heterogeneous (the plan sizes everything to the smallest card).
_CAPACITY_VRAM_HETEROGENEITY_RATIO = 1.5


def capacity_warnings(
        *,
        workload_name,
        backend,
        gpus,
        num_jobs,
        per_gpu_free_vram_gb,
        device_worker_gb,
        available_ram_gb,
        host_worker_gb,
        cpu_count):
    """Compare a resolved plan against measured machine capacity.

    Returns a list of human-readable warning strings for resources that are
    below a safe operating range (small / uneven / undetectable GPU VRAM,
    host RAM below the per-worker estimate, more workers than CPUs). Pure: no
    I/O and no side effects — the caller decides how to surface the strings.
    All measured inputs may be ``None`` when detection failed; each check is
    skipped when its inputs are missing.
    """
    out = []
    gpu_backend = backend in ("auto", "gpu") and gpus and int(gpus) > 0

    if gpu_backend:
        if per_gpu_free_vram_gb is None:
            out.append(
                "could not detect free GPU VRAM (nvidia-smi unavailable and no "
                "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB override); "
                "worker and batch sizing fall back to conservative defaults — "
                "pin --max-workers-per-gpu / the batch-size flags if you OOM.")
        else:
            free = [float(g) for g in per_gpu_free_vram_gb if g is not None]
            if free and device_worker_gb:
                min_free = min(free)
                if min_free < _CAPACITY_VRAM_SAFE_FRACTION * float(device_worker_gb):
                    out.append(
                        "free GPU VRAM is below the per-worker estimate for "
                        "workload %r: smallest free %.1f GB vs ~%.1f GB/worker. "
                        "The auto batch-sizer shrinks batches to fit, but "
                        "throughput drops and a worker can OOM if its minimum "
                        "working set exceeds free VRAM — reduce the batch-size "
                        "flags or use a larger GPU."
                        % (workload_name, min_free, float(device_worker_gb)))
                if (len(free) > 1 and min_free > 0
                        and max(free) / min_free >= _CAPACITY_VRAM_HETEROGENEITY_RATIO):
                    out.append(
                        "free GPU VRAM is uneven across GPUs (%s GB); the plan "
                        "sizes every GPU to the smallest (%.1f GB), "
                        "underutilizing the larger card(s)."
                        % (", ".join("%.0f" % g for g in free), min_free))

    if (num_jobs and int(num_jobs) > 0 and available_ram_gb is not None
            and host_worker_gb):
        needed = float(host_worker_gb) * int(num_jobs)
        if float(available_ram_gb) < needed:
            out.append(
                "available host RAM (%.0f GB) is below the ~%.0f GB estimate "
                "for %d workers (~%.1f GB each); workers may be OOM-killed "
                "under memory pressure — reduce --num-jobs."
                % (float(available_ram_gb), needed, int(num_jobs),
                   float(host_worker_gb)))

    if num_jobs and cpu_count and int(num_jobs) > int(cpu_count):
        out.append(
            "planned %d worker processes exceed detected CPUs (%d); workers "
            "will contend for cores — consider --num-jobs <= %d."
            % (int(num_jobs), int(cpu_count), int(cpu_count)))

    return out


def plan_local_parallelism(
        args,
        *,
        workload_name=WORKLOAD_GENERIC,
        workload_hints=None,
        per_worker_gb=None,
        cap_auto_num_jobs=True,
        normalize_backend,
        detect_num_cuda_devices,
        auto_max_workers_per_gpu,
        auto_num_jobs,
        resolve_dataloader_num_workers,
        auto_random_negative_pool_epochs):
    """Resolve all local-parallelism CLI flags into a plan.

    Explicit CLI values override planner estimates. For explicit values that
    exceed estimated capacity, the planner records a warning and still returns
    the explicit value.
    """
    hints = _normalize_hints(workload_hints, per_worker_gb=per_worker_gb)
    memory_plan = estimate_workload_memory(workload_name, hints)
    memory = system_memory_info_gb()
    warnings = []
    cli_overrides = []

    backend_raw = getattr(args, "backend", "auto") or "auto"
    backend = normalize_backend(backend_raw)
    if backend_raw not in ("auto", None):
        cli_overrides.append("backend")

    gpus_raw = getattr(args, "gpus", None)
    gpus_was_auto = gpus_raw is None and backend in ("auto", "gpu")
    if gpus_was_auto:
        gpus = int(detect_num_cuda_devices())
    else:
        gpus = int(gpus_raw or 0)
        if gpus_raw is not None:
            cli_overrides.append("gpus")

    mwpg_raw = getattr(args, "max_workers_per_gpu", None)
    mwpg_was_auto = _is_auto(mwpg_raw)
    if mwpg_was_auto:
        max_workers_per_gpu = int(auto_max_workers_per_gpu(
            num_jobs=getattr(args, "num_jobs", 0),
            num_gpus=gpus,
            backend=backend,
            per_worker_gb=memory_plan["device_worker_gb"],
        ))
    else:
        max_workers_per_gpu = int(mwpg_raw)
        cli_overrides.append("max_workers_per_gpu")

    capacity = (
        int(auto_num_jobs(gpus, max_workers_per_gpu))
        if backend in ("auto", "gpu") else 0
    )
    num_jobs_raw = getattr(args, "num_jobs", "auto")
    num_jobs_was_auto = _is_auto(num_jobs_raw)
    if num_jobs_was_auto:
        num_jobs = capacity
        if cap_auto_num_jobs and num_jobs > 0:
            num_jobs, max_workers_per_gpu, host_cap = (
                _clip_auto_num_jobs_to_host_memory(
                    num_jobs,
                    max_workers_per_gpu,
                    gpus,
                    mwpg_was_auto,
                    memory,
                    memory_plan,
                    warnings,
                )
            )
            if host_cap is not None and backend in ("auto", "gpu"):
                # Host memory is now the binding constraint, so report the
                # clamped worker count as capacity. Recomputing
                # gpus * ceil(num_jobs / gpus) here would over-report (e.g.
                # num_jobs=5 on 2 GPUs -> mwpg=3 -> capacity=6 > num_jobs).
                capacity = int(num_jobs)
    else:
        num_jobs = int(num_jobs_raw)
        cli_overrides.append("num_jobs")
        if cap_auto_num_jobs and capacity > 0 and num_jobs > capacity:
            warnings.append(
                "explicit num_jobs=%d exceeds GPU capacity estimate %d; "
                "honoring CLI override" % (num_jobs, capacity)
            )

    effective_fit_workers = max(1, int(num_jobs))
    if int(num_jobs) <= 0 and gpus > 0:
        effective_fit_workers = 1

    dl_raw = getattr(args, "dataloader_num_workers", "auto")
    dl_was_auto = _is_auto(dl_raw)
    if dl_was_auto:
        dataloader_num_workers = int(resolve_dataloader_num_workers(
            dl_raw,
            num_fit_workers=effective_fit_workers,
            ram_gb=memory.get("available_gb") or memory.get("total_gb"),
        ))
    else:
        dataloader_num_workers = int(dl_raw)
        cli_overrides.append("dataloader_num_workers")

    torch_compile = getattr(args, "torch_compile", "auto")
    if torch_compile != "auto":
        cli_overrides.append("torch_compile")
    torch_compile_loss = getattr(args, "torch_compile_loss", "auto")
    if torch_compile_loss != "auto":
        cli_overrides.append("torch_compile_loss")
    matmul_precision = getattr(args, "matmul_precision", "none")
    if matmul_precision != "none":
        cli_overrides.append("matmul_precision")
    enable_timing = bool(getattr(args, "enable_timing", False))
    if enable_timing:
        cli_overrides.append("enable_timing")

    host_worker_gb = (
        memory_plan["host_worker_gb"]
        + int(dataloader_num_workers) * HOST_RAM_PER_DATALOADER_CHILD_GB
    )
    host_memory_cap = host_memory_num_jobs_cap(
        memory,
        memory_plan["host_worker_gb"],
        dataloader_num_workers=dataloader_num_workers,
    )
    if host_memory_cap is not None:
        if (
                cap_auto_num_jobs
                and num_jobs_was_auto
                and int(num_jobs) > host_memory_cap):
            num_jobs, max_workers_per_gpu, host_cap = (
                _clip_auto_num_jobs_to_host_memory(
                    num_jobs,
                    max_workers_per_gpu,
                    gpus,
                    mwpg_was_auto,
                    memory,
                    memory_plan,
                    warnings,
                    dataloader_num_workers=dataloader_num_workers,
                )
            )
            if host_cap is not None and backend in ("auto", "gpu"):
                # Host memory is now the binding constraint, so report the
                # clamped worker count as capacity. Recomputing
                # gpus * ceil(num_jobs / gpus) here would over-report (e.g.
                # num_jobs=5 on 2 GPUs -> mwpg=3 -> capacity=6 > num_jobs).
                capacity = int(num_jobs)
            effective_fit_workers = max(1, int(num_jobs))
            if int(num_jobs) <= 0 and gpus > 0:
                effective_fit_workers = 1
            host_memory_cap = host_memory_num_jobs_cap(
                memory,
                memory_plan["host_worker_gb"],
                dataloader_num_workers=dataloader_num_workers,
            )
        elif (not num_jobs_was_auto) and int(num_jobs) > host_memory_cap:
            warnings.append(
                "explicit num_jobs=%d exceeds host-memory estimate %d; "
                "honoring CLI override" % (num_jobs, host_memory_cap)
            )

    rn_raw = getattr(args, "random_negative_pool_epochs", "auto")
    rn_was_auto = _is_auto(rn_raw)
    if rn_was_auto:
        random_negative_pool_epochs = int(auto_random_negative_pool_epochs(
            num_random_negatives=None,
            peptide_max_length=None,
            num_workers=effective_fit_workers,
            ram_gb=memory.get("available_gb") or memory.get("total_gb"),
        ))
    else:
        random_negative_pool_epochs = int(rn_raw)
        cli_overrides.append("random_negative_pool_epochs")

    return LocalParallelismPlan(
        workload_name=memory_plan["workload_name"],
        backend=backend,
        gpus=int(gpus),
        gpus_was_auto=gpus_was_auto,
        max_workers_per_gpu=int(max_workers_per_gpu),
        max_workers_per_gpu_was_auto=mwpg_was_auto,
        num_jobs=int(num_jobs),
        num_jobs_was_auto=num_jobs_was_auto,
        dataloader_num_workers=int(dataloader_num_workers),
        dataloader_num_workers_was_auto=dl_was_auto,
        random_negative_pool_epochs=random_negative_pool_epochs,
        random_negative_pool_epochs_was_auto=rn_was_auto,
        torch_compile=torch_compile,
        torch_compile_loss=torch_compile_loss,
        matmul_precision=matmul_precision,
        enable_timing=enable_timing,
        capacity=int(capacity),
        device_worker_gb=memory_plan["device_worker_gb"],
        data_pressure_gb=memory_plan["data_pressure_gb"],
        host_worker_gb=host_worker_gb,
        host_memory_total_gb=memory.get("total_gb"),
        host_memory_available_gb=memory.get("available_gb"),
        host_memory_source=memory.get("source", "unknown"),
        host_memory_num_jobs_cap=host_memory_cap,
        cli_overrides=tuple(cli_overrides),
        warnings=tuple(warnings),
        hints=memory_plan["hints"],
    )

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

"""Prediction and training batch-size helpers."""

import gc
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass

from .memory_budget import (
    DEVICE_MIN_RESERVE_BYTES,
    MEMORY_RESERVE_FRACTION,
    memory_reserve_bytes,
    usable_memory_bytes,
)

DEFAULT_PREDICT_BATCH_SIZE = "auto"
AUTO_BATCH_MAX_ROWS = sys.maxsize
AUTO_BATCH_MIN_ROWS = 1024  # floor: avoid pathologically tiny batches
AUTO_BATCH_CPU_FALLBACK = 32_768  # CPU: large batches thrash L3; stay modest
AUTO_BATCH_FREE_FRACTION = None
AUTO_BATCH_CALIBRATION_PROBE_ROWS = 4096
AUTO_BATCH_CALIBRATION_SAFETY_MULTIPLIER = 2.0
_MPS_PSUTIL_WARNED = False  # one-shot warning if psutil is missing on MPS
CUDA_FREE_BEFORE_CONTEXT_ENV = (
    "MHCFLURRY_CUDA_FREE_BEFORE_CONTEXT_BYTES"
)
if os.environ.get("MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"):
    raw_default_batch_size = os.environ["MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"]
    try:
        DEFAULT_PREDICT_BATCH_SIZE = int(raw_default_batch_size)
    except ValueError:
        raise ValueError(
            "MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE must be a positive integer; "
            "got %r" % raw_default_batch_size
        ) from None
    if DEFAULT_PREDICT_BATCH_SIZE < 1:
        raise ValueError(
            "MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE must be a positive integer; "
            "got %r" % raw_default_batch_size
        )
    logging.info(
        "Configured default predict batch size: %s" % DEFAULT_PREDICT_BATCH_SIZE
    )


def default_prediction_batch_is_auto():
    """Whether the effective default prediction batch can shrink on OOM."""
    return DEFAULT_PREDICT_BATCH_SIZE in (None, "auto")


def begin_peak_memory_measurement():
    """Reset CUDA peak counters and return an opaque measurement token."""
    token = {"started": True}
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            token["cuda_device"] = int(device)
    except Exception as exc:
        token["cuda_error"] = str(exc)
    return token


def _process_namespace_pids():
    """Return host/container PID aliases reported for this process."""
    result = {os.getpid()}
    try:
        with open("/proc/self/status") as status_fd:
            for line in status_fd:
                if not line.startswith("NSpid:"):
                    continue
                for value in line.split()[1:]:
                    result.add(int(value))
                break
    except (OSError, ValueError):
        pass
    return result


def cuda_process_memory_bytes(pid=None):
    """Return this process's CUDA memory from ``nvidia-smi``, if available.

    PyTorch allocator counters omit the CUDA context and other non-PyTorch
    allocations. Those bytes matter when many worker processes share a GPU.
    Querying the driver gives resource probes a complete steady-state working
    set without initializing CUDA in the parent process.
    """
    pids = _process_namespace_pids() if pid is None else {int(pid)}
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return None
    total_mib = 0.0
    found = False
    for line in output.decode("utf-8", errors="ignore").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            row_pid = int(fields[0])
            used_mib = float(fields[1].split()[0])
        except ValueError:
            continue
        if row_pid in pids:
            total_mib += used_mib
            found = True
    return int(total_mib * (1 << 20)) if found else None


def cuda_free_memory_before_context_bytes(device_id):
    """Query one physical CUDA device's free memory without importing torch."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=%s" % device_id,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        value = float(output.decode("utf-8", errors="ignore").splitlines()[0])
    except (
            IndexError,
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError):
        return None
    return int(value * (1 << 20))


def _nonnegative_env_bytes(name):
    """Read an optional non-negative byte count from the environment."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            "%s must be a non-negative integer; got %r" % (name, raw)
        ) from None
    if value < 0:
        raise ValueError(
            "%s must be a non-negative integer; got %r" % (name, raw)
        )
    return value


def _cuda_process_bytes_with_baseline(
        free_bytes, measured_process_bytes, *, allow_baseline=True):
    """Include context usage when driver PIDs differ across namespaces."""
    candidates = [max(int(measured_process_bytes or 0), 0)]
    baseline_free = _nonnegative_env_bytes(CUDA_FREE_BEFORE_CONTEXT_ENV)
    if allow_baseline and baseline_free is not None:
        candidates.append(max(baseline_free - max(int(free_bytes), 0), 0))
    return max(candidates)


def _process_peak_rss_bytes():
    """Best-effort process peak RSS in bytes on Linux and macOS."""
    try:
        import resource
        import sys
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except Exception:
        return None


def end_peak_memory_measurement(token):
    """Finish a measurement begun by :func:`begin_peak_memory_measurement`."""
    result = {"host_peak_rss_bytes": _process_peak_rss_bytes()}
    device = token.get("cuda_device") if token else None
    if device is not None:
        try:
            import torch
            torch.cuda.synchronize(device)
            result.update({
                "cuda_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)),
            })
            current_reserved = int(torch.cuda.memory_reserved(device))
            current_free, _ = torch.cuda.mem_get_info(device)
            measured_process_bytes = cuda_process_memory_bytes()
            process_bytes = _cuda_process_bytes_with_baseline(
                current_free,
                measured_process_bytes,
            )
            if measured_process_bytes is not None or os.environ.get(
                    CUDA_FREE_BEFORE_CONTEXT_ENV) not in (None, ""):
                # nvidia-smi is a current whole-process measurement; PyTorch's
                # peak is an allocator high-water mark. A before-context free
                # memory baseline covers PID-namespaced containers where the
                # nvidia-smi process PID cannot match os.getpid(). Preserve the
                # observed non-PyTorch component and add allocator growth that
                # was released before the probe ended.
                non_torch_bytes = max(process_bytes - current_reserved, 0)
                result["cuda_process_memory_bytes"] = process_bytes
                result["cuda_process_peak_estimate_bytes"] = (
                    non_torch_bytes
                    + result["cuda_peak_reserved_bytes"]
                )
        except Exception as exc:
            result["cuda_error"] = str(exc)
    return result


def is_device_out_of_memory_error(exc):
    """Whether ``exc`` is a CUDA/MPS allocator out-of-memory error."""
    message = str(exc).lower()
    return (
        "out of memory" in message
        and ("cuda" in message or "mps" in message or "allocator" in message)
    )


def release_device_memory_after_oom(device):
    """Release cached allocator blocks before retrying a smaller auto batch."""
    try:
        import torch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def synchronize_device(device):
    """Wait for accelerator work so deferred errors surface at their source.

    CUDA and MPS kernels are asynchronous. Without an explicit synchronization
    at the end of an elastic prediction attempt, an OOM from a network forward
    can be reported by an unrelated later tensor transfer, outside the retry
    loop that can safely reduce the batch size.
    """
    try:
        import torch
    except ImportError:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def release_unused_torch_memory():
    """Collect dead tensors and return unused accelerator cache memory.

    Resource probes deliberately reuse one worker for several architectures.
    Python reference cycles can otherwise keep the previous fit's tensors
    alive until a later, unrelated collection, while allocator cache blocks
    continue to hide global device headroom. Collect cycles before emptying
    the backend cache so the next probe observes only its own residency.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return
    except RuntimeError:
        pass

    try:
        if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "empty_cache")):
            torch.mps.empty_cache()
    except RuntimeError:
        pass


def estimate_peak_bytes_per_row(model):
    """Worst-case peak activation bytes per sample during a forward.

    Walks the model's configured layers and returns the maximum hidden-
    layer width (in fp32 bytes) × 2 (one input + one output of the
    current layer stay live under torch's eval-time no_grad reuse). A
    4× multiplier covers framework overhead, cuDNN scratch buffers,
    and Python-side tensor bookkeeping. Used by ``compute_prediction_batch_size``.
    """
    if model is None:
        return 32 * 1024  # conservative 32 KB/row fallback
    # Class1ProcessingModel is convolutional over the full
    # flank+peptide sequence. Its peak activation is not represented by
    # the affinity model's peptide_dense_layers/dense_layers attributes,
    # so compute it from the Conv1d sequence width. This keeps prediction
    # autosizing sensitive to future flank length / filter count changes
    # instead of falling back to the generic 1024-width estimate.
    try:
        conv1 = model.conv1
        seq_len = (
            int(model.n_flank_length)
            + int(model.peptide_max_length)
            + int(model.c_flank_length)
        )
        widths = [
            seq_len * int(conv1.in_channels),
            seq_len * int(conv1.out_channels),
        ]
        for convs_name in ("n_flank_post_convs", "c_flank_post_convs"):
            for layer in getattr(model, convs_name, []):
                widths.append(seq_len * int(layer.out_channels))
        if getattr(model, "flanking_averages", False):
            widths.append(seq_len * int(conv1.out_channels))
        peak = max(widths)
        return int(peak * 4 * 2 * 4)  # fp32 × 2 buffers × 4x safety
    except (AttributeError, TypeError, ValueError):
        pass
    # MergedClass1NeuralNetwork wraps N sub-networks and runs each one's
    # forward independently in a list comprehension, then combines the
    # outputs. All N sub-networks' peak intermediates are alive
    # simultaneously, so the per-row peak is the SUM of per-sub-network
    # peaks, not the max. Without this, the merged ensemble's auto-sized
    # batch overshoots VRAM by Nx (~8x for the production 8-network
    # release ensemble) and OOMs in calibrate's cartesian forward.
    sub_networks = getattr(model, "networks", None)
    if sub_networks is not None and not hasattr(model, "peptide_encoding_shape"):
        try:
            return int(sum(
                estimate_peak_bytes_per_row(net) for net in sub_networks
            ))
        except (AttributeError, TypeError) as exc:
            logging.warning(
                "Could not estimate peak per-row bytes for merged ensemble; "
                "falling back to per-network walk: %s",
                exc,
            )
    widths = []
    try:
        lc_out_len = int(model.peptide_encoding_shape[0])
        lc_out_ch = int(model.peptide_encoding_shape[1])
        for lc_layer in model.lc_layers:
            lc_out_ch = getattr(lc_layer, "out_channels", lc_out_ch)
            try:
                lc_out_len = int(lc_layer.output_length)
            except AttributeError:
                pass
        widths.append(lc_out_len * lc_out_ch)
        for layer in model.peptide_dense_layers:
            widths.append(int(layer.out_features))
    except AttributeError:
        widths.append(1024)
    try:
        allele_out = None
        if getattr(model, "allele_embedding", None) is not None:
            allele_out = int(model.allele_embedding.weight.shape[1])
        for layer in getattr(model, "allele_dense_layers", []):
            allele_out = layer.out_features
        if allele_out is not None:
            widths.append(allele_out)
    except AttributeError:
        pass
    try:
        for layer in model.dense_layers:
            widths.append(int(layer.out_features))
    except AttributeError:
        widths.append(1024)
    peak = max(widths) if widths else 1024
    return int(peak * 4 * 2 * 4)  # fp32 × 2 buffers × 4x safety


def free_device_memory_bytes(device):
    """Best-effort free-memory query. Returns a conservative value when
    the device doesn't expose a direct free-memory API.

    CUDA: ``torch.cuda.mem_get_info`` (exposed free + reserved tracking).
    MPS: Apple's ``recommended_max_memory`` on unified memory,
        minus whatever the MPS driver has already handed us. Cap by
        ``psutil`` available RAM when present so other apps aren't
        evicted. Falls back to 4 GB if neither API is reachable.
    CPU / unknown: 2 GB conservative budget (the helper short-circuits
        for CPU anyway, but keep a sensible value in case callers pass
        a foreign device).
    """
    import torch
    if device.type == "cuda":
        try:
            free, _ = torch.cuda.mem_get_info(device)
            return int(free)
        except Exception:
            props = torch.cuda.get_device_properties(device)
            reserved = torch.cuda.memory_reserved(device)
            return max(int(props.total_memory) - int(reserved), 0)
    if device.type == "mps":
        # Apple Silicon: unified memory, so "free VRAM" is better
        # estimated from the MPS driver's recommended ceiling minus
        # whatever it's already allocated. Ceiling is typically
        # ~75-80% of total system RAM on M-series chips.
        try:
            ceiling = int(torch.mps.recommended_max_memory())
        except Exception:
            ceiling = 4 * (1 << 30)
        allocated = 0
        try:
            allocated = int(torch.mps.driver_allocated_memory())
        except Exception:
            pass
        free = max(ceiling - allocated, 0)
        # Don't evict other apps: also cap by the OS-reported free
        # RAM when psutil is available. This gets us a realistic
        # "what's safe to claim right now" rather than the MPS
        # driver's peak permission.
        try:
            import psutil
            free = min(free, int(psutil.virtual_memory().available))
        except ImportError:
            try:
                from .workload_planning import system_memory_info_gb
                available_gb = system_memory_info_gb().get("available_gb")
                if available_gb is not None:
                    free = min(free, int(available_gb * (1 << 30)))
            except Exception:
                # psutil isn't a hard dep. Log once per process so the
                # skip is visible rather than silent — without this cap
                # the MPS driver's "recommended max" can exceed what's
                # actually safe to claim alongside other apps.
                global _MPS_PSUTIL_WARNED
                if not _MPS_PSUTIL_WARNED:
                    logging.warning(
                        "psutil not available and OS memory fallback failed; "
                        "MPS free-memory estimate will use "
                        "torch.mps.recommended_max_memory alone, which may "
                        "overshoot actual available RAM."
                    )
                    _MPS_PSUTIL_WARNED = True
        except Exception:
            # Any other psutil failure (broken install, etc.) — skip
            # the cap but don't fail the whole batch-size query.
            pass
        # Zero is a valid and important measurement: on unified-memory Macs it
        # means the process has no safe headroom. Replacing it with the 4 GiB
        # API fallback would make auto-sizing attempt another large allocation
        # under peak memory pressure. The caller turns a zero budget into its
        # one-row minimum batch.
        return free
    return 2 * (1 << 30)


@dataclass(frozen=True)
class DeviceMemoryBudget:
    """Stable per-worker device-memory entitlement and remaining headroom."""

    free_bytes: int
    total_bytes: int
    reserve_bytes: int
    worker_entitlement_bytes: int
    process_bytes: int
    available_bytes: int


def device_memory_budget(
        device,
        num_workers_per_gpu=1,
        free_memory_fraction=None,
        reserve_fraction=MEMORY_RESERVE_FRACTION,
        reserve_min_bytes=DEVICE_MIN_RESERVE_BYTES):
    """Return a race-free memory budget for one co-resident worker.

    The old calculation divided *live free memory* by the declared worker
    count. Live free memory already reflects allocations made by workers that
    initialized earlier, so that calculation double-discounted memory and made
    the resulting batch size depend on startup order. Here every worker gets a
    fixed entitlement captured from launch-time free capacity (falling back to
    total device capacity for direct API calls), subtracts its own resident
    working set, and finally caps the result by live global headroom. Managed
    workers therefore cannot collectively claim more than the shared launch
    budget.
    """
    workers = int(num_workers_per_gpu)
    if workers < 1:
        raise ValueError("num_workers_per_gpu must be at least 1")
    if free_memory_fraction is not None:
        free_memory_fraction = float(free_memory_fraction)
        if (
                not math.isfinite(free_memory_fraction)
                or not 0 < free_memory_fraction <= 1):
            raise ValueError("free_memory_fraction must be in (0, 1]")

    free = int(free_device_memory_bytes(device))
    total = free
    process_bytes = 0
    if device.type == "cuda":
        try:
            import torch
            free, total = (
                int(value) for value in torch.cuda.mem_get_info(device)
            )
            process_bytes = cuda_process_memory_bytes()
            if process_bytes is None:
                process_bytes = int(torch.cuda.memory_reserved(device))
            process_bytes = _cuda_process_bytes_with_baseline(
                free,
                process_bytes,
                # A launch baseline is isolated only when this is the sole
                # resident worker. With peers it can include allocations made
                # by workers that initialized after this process.
                allow_baseline=(workers == 1),
            )
        except Exception:
            try:
                import torch
                total = int(torch.cuda.get_device_properties(device).total_memory)
                process_bytes = int(torch.cuda.memory_reserved(device))
            except Exception:
                total = free
                process_bytes = 0
    elif device.type == "mps":
        try:
            import torch
            total = int(torch.mps.recommended_max_memory())
            process_bytes = int(torch.mps.driver_allocated_memory())
        except Exception:
            total = free
            process_bytes = 0

    reserve = memory_reserve_bytes(
        total,
        min_reserve_bytes=reserve_min_bytes,
        reserve_fraction=reserve_fraction,
    )
    spendable = usable_memory_bytes(
        total,
        min_reserve_bytes=reserve_min_bytes,
        reserve_fraction=reserve_fraction,
    )
    if free_memory_fraction is not None:
        spendable = min(
            spendable,
            int(total * free_memory_fraction),
        )
    entitlement = spendable // workers
    launch_budget = _nonnegative_env_bytes(
        "MHCFLURRY_DEVICE_MEMORY_BUDGET_BYTES")
    if launch_budget is not None:
        entitlement = min(entitlement, launch_budget)
    available = min(
        max(free, 0),
        max(entitlement - max(process_bytes, 0), 0),
    )
    return DeviceMemoryBudget(
        free_bytes=max(free, 0),
        total_bytes=max(total, 0),
        reserve_bytes=max(reserve, 0),
        worker_entitlement_bytes=max(entitlement, 0),
        process_bytes=max(process_bytes, 0),
        available_bytes=max(available, 0),
    )


def compute_prediction_batch_size(
        device,
        model=None,
        num_workers_per_gpu=1,
        free_memory_fraction=AUTO_BATCH_FREE_FRACTION,
        max_rows=AUTO_BATCH_MAX_ROWS,
        min_rows=AUTO_BATCH_MIN_ROWS,
        cpu_fallback=AUTO_BATCH_CPU_FALLBACK,
        total_rows=None):
    """Auto-size a prediction batch for ``device`` and ``model``.

    Uses the worker's remaining fixed device-memory entitlement and the model's
    per-row peak activation estimate, capped by current global headroom.
    ``max_rows`` is only an optional caller limit; the default auto path has no
    hardwired batch cap.

    CPU: returns ``cpu_fallback`` — large batches on CPU thrash L3
    cache and rarely help for the small networks mhcflurry trains.
    """
    workers = int(num_workers_per_gpu)
    if workers < 1:
        raise ValueError("num_workers_per_gpu must be at least 1")
    if max_rows is not None and int(max_rows) < 1:
        raise ValueError("max_rows must be at least 1")
    if total_rows is not None and int(total_rows) < 0:
        raise ValueError("total_rows must be non-negative")
    if free_memory_fraction is not None:
        free_memory_fraction = float(free_memory_fraction)
        if (
                not math.isfinite(free_memory_fraction)
                or not 0 < free_memory_fraction <= 1):
            raise ValueError("free_memory_fraction must be in (0, 1]")
    if device.type == "cpu":
        rows = int(cpu_fallback)
        if rows < 1:
            raise ValueError("cpu_fallback must be at least 1")
        return (
            min(rows, max(int(total_rows), 1))
            if total_rows is not None else rows
        )
    peak_bytes = estimate_peak_bytes_per_row(model)
    memory = device_memory_budget(
        device,
        num_workers_per_gpu=workers,
        free_memory_fraction=free_memory_fraction,
    )
    free = memory.free_bytes
    budget = memory.available_bytes
    rows = max(1, budget // peak_bytes)
    if rows < min_rows:
        logging.warning(
            "Auto-sized prediction batch below the normal floor: %d rows "
            "(free=%.2f GB, workers/GPU=%d, peak=%.1f KB/row).",
            rows,
            free / float(1 << 30),
            workers,
            peak_bytes / 1024.0,
        )
    if max_rows is not None:
        rows = min(rows, int(max_rows))
    if total_rows is not None:
        rows = min(rows, max(int(total_rows), 1))
    return int(rows)


def env_workers_per_gpu(default=1):
    """Read the ``MHCFLURRY_MAX_WORKERS_PER_GPU`` env var.

    The local parallelism pool sets this in each training worker so
    auto-sized batching + training-memory checks can partition VRAM
    across co-resident workers without the caller wiring it explicitly.
    """
    value = os.environ.get("MHCFLURRY_MAX_WORKERS_PER_GPU")
    if value:
        try:
            return max(int(value), 1)
        except ValueError:
            pass
    return default


def resolve_prediction_batch_size(
        value, device, model=None, num_workers_per_gpu=1, total_rows=None):
    """Resolve an explicit int or ``"auto"`` to a concrete batch size.

    Accepts ``None`` as a synonym for ``"auto"``. Propagates an
    explicit int through unchanged so callers can always pin the size
    when they know better than the heuristic.
    """
    if value in (None, "auto"):
        return compute_prediction_batch_size(
            device,
            model=model,
            num_workers_per_gpu=num_workers_per_gpu,
            total_rows=total_rows,
        )
    result = int(value)
    if result < 1:
        raise ValueError("prediction batch size must be at least 1")
    return result


def calibrate_prediction_batch_size(
        batch_size,
        device,
        model,
        inputs,
        num_workers_per_gpu=1,
        total_rows=None,
        probe_rows=AUTO_BATCH_CALIBRATION_PROBE_ROWS,
        safety_multiplier=AUTO_BATCH_CALIBRATION_SAFETY_MULTIPLIER,
        transfer_inputs_to_device=False):
    """Tighten an automatic CUDA prediction batch from a real forward probe.

    The analytic estimator cannot know the exact allocator and convolution
    workspace chosen for every model shape and CUDA runtime. This function
    measures the incremental peak of the loaded model with its real resident
    input tensors, then recomputes the batch from the worker's remaining fixed
    device-memory entitlement. The result never exceeds either ``batch_size``
    or the largest batch exercised successfully by the probe. CUDA convolution
    workspaces are not reliably linear in batch size, especially while peer
    workers initialize, so extrapolating above a measured batch is unsafe.

    CPU and MPS return ``batch_size`` unchanged. MPS does not expose an
    equivalent resettable peak allocator counter, so it retains the analytic
    estimate and the caller's elastic OOM retry. Explicit batches should not be
    passed to this function; callers retain authority over pinned values.

    Parameters
    ----------
    batch_size : int
        Analytically selected automatic batch size.
    device : torch.device
        Device on which ``model`` and ``inputs`` reside.
    model : callable
        Eval-mode model accepting the sliced ``inputs`` mapping.
    inputs : mapping
        Tensor inputs with a shared leading row dimension. They may remain on
        the host when ``transfer_inputs_to_device`` is true.
    num_workers_per_gpu : int
        Co-resident workers sharing the device entitlement.
    total_rows : int, optional
        Number of available rows. Inferred from ``inputs`` when omitted.
    probe_rows : int
        Maximum rows in the measurement forward. The actual probe is also
        capped at one eighth of the analytic entitlement so measurement cannot
        consume the batch it is meant to protect.
    safety_multiplier : float
        Margin applied to the measured per-row peak when deciding whether the
        verified probe batch must be tightened further.
    transfer_inputs_to_device : bool
        Move each probe slice to ``device`` inside the protected measurement.
        This models streaming inference and makes transfer OOMs recoverable.

    Returns
    -------
    int
        A calibrated batch no larger than ``batch_size`` or the successfully
        exercised probe batch.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    workers = int(num_workers_per_gpu)
    if workers < 1:
        raise ValueError("num_workers_per_gpu must be at least 1")
    probe_rows = int(probe_rows)
    if probe_rows < 1:
        raise ValueError("probe_rows must be at least 1")
    safety_multiplier = float(safety_multiplier)
    if (
            not math.isfinite(safety_multiplier)
            or safety_multiplier < 1):
        raise ValueError("safety_multiplier must be finite and at least 1")
    if device.type != "cuda" or batch_size == 1:
        return batch_size
    if not inputs:
        raise ValueError("inputs must contain at least one tensor")
    if total_rows is None:
        total_rows = len(next(iter(inputs.values())))
    total_rows = int(total_rows)
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if total_rows <= 1:
        return min(batch_size, max(total_rows, 1))

    analytic_bytes_per_row = max(estimate_peak_bytes_per_row(model), 1)
    memory_before = device_memory_budget(
        device,
        num_workers_per_gpu=workers,
    )
    # Keep the probe safely inside the analytic entitlement. A one-eighth
    # slice is large enough to exercise release processing convolutions while
    # leaving room for co-resident workers probing at the same time.
    entitlement_probe_rows = max(
        memory_before.available_bytes // (analytic_bytes_per_row * 8),
        1,
    )
    actual_probe_rows = min(
        batch_size,
        total_rows,
        probe_rows,
        entitlement_probe_rows,
    )

    try:
        import torch
    except Exception as exc:
        logging.warning(
            "CUDA prediction batch calibration unavailable before any "
            "successful probe; forcing batch 1: %s",
            exc,
        )
        return 1

    def run_probe(rows):
        probe_inputs = {
            name: (
                value[:rows].to(device)
                if transfer_inputs_to_device else value[:rows]
            )
            for name, value in inputs.items()
        }
        with torch.no_grad():
            return model(probe_inputs)

    # Trigger lazy CUDA context, kernel, and optional compile setup before
    # resetting counters. The measured pass then represents repeatable
    # inference allocation rather than one-time initialization.
    warmup_output = None
    try:
        warmup_output = run_probe(1)
        torch.cuda.synchronize(device)
    except Exception as exc:
        if is_device_out_of_memory_error(exc):
            release_device_memory_after_oom(device)
        logging.warning(
            "CUDA prediction batch calibration failed before any successful "
            "probe; forcing batch 1: %s",
            exc,
        )
        return 1
    finally:
        del warmup_output

    measurement_error = None
    while True:
        try:
            allocated_before = int(torch.cuda.memory_allocated(device))
            reserved_before = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        except Exception as exc:
            measurement_error = exc

        probe_output = None
        try:
            probe_output = run_probe(actual_probe_rows)
            torch.cuda.synchronize(device)
        except Exception as exc:
            if is_device_out_of_memory_error(exc) and actual_probe_rows > 1:
                previous_probe_rows = actual_probe_rows
                actual_probe_rows = max(1, actual_probe_rows // 2)
                release_device_memory_after_oom(device)
                logging.warning(
                    "CUDA prediction batch probe OOM at %d rows; retrying "
                    "calibration at %d.",
                    previous_probe_rows,
                    actual_probe_rows,
                )
                continue
            if is_device_out_of_memory_error(exc):
                release_device_memory_after_oom(device)
            logging.warning(
                "CUDA prediction batch calibration failed above the verified "
                "one-row warmup; forcing batch 1: %s",
                exc,
            )
            return 1
        finally:
            del probe_output
        break

    if measurement_error is not None:
        logging.warning(
            "CUDA prediction batch counters unavailable; using verified "
            "probe batch %d: %s",
            actual_probe_rows,
            measurement_error,
        )
        return min(batch_size, actual_probe_rows, total_rows)

    try:
        peak_allocated = max(
            int(torch.cuda.max_memory_allocated(device)) - allocated_before,
            0,
        )
        peak_reserved = max(
            int(torch.cuda.max_memory_reserved(device)) - reserved_before,
            0,
        )
    except Exception as exc:
        logging.warning(
            "CUDA prediction batch peak counters unavailable; using verified "
            "probe batch %d: %s",
            actual_probe_rows,
            exc,
        )
        return min(batch_size, actual_probe_rows, total_rows)

    measured_peak_bytes = max(peak_allocated, peak_reserved)
    if measured_peak_bytes <= 0:
        logging.warning(
            "CUDA prediction batch calibration observed no incremental peak; "
            "using verified probe batch %d.",
            actual_probe_rows,
        )
        return min(batch_size, actual_probe_rows, total_rows)
    measured_bytes_per_row = int(math.ceil(
        measured_peak_bytes / float(actual_probe_rows)
    ))
    effective_bytes_per_row = max(
        analytic_bytes_per_row,
        int(math.ceil(measured_bytes_per_row * safety_multiplier)),
    )
    memory_after = device_memory_budget(
        device,
        num_workers_per_gpu=workers,
    )
    calibrated = max(
        memory_after.available_bytes // effective_bytes_per_row,
        1,
    )
    # A real forward proves that ``actual_probe_rows`` fits this architecture;
    # it does not prove that a larger convolution uses the same CUDA algorithm
    # or workspace. In particular, co-resident workers can cross those
    # allocator thresholds at slightly different times. Never extrapolate an
    # automatic batch above the largest shape that actually ran successfully.
    calibrated = min(
        calibrated,
        actual_probe_rows,
        batch_size,
        total_rows,
    )
    logging.info(
        "Calibrated CUDA prediction batch: %d -> %d rows "
        "(probe=%d, analytic=%.1f KB/row, measured=%.1f KB/row, "
        "effective=%.1f KB/row, remaining entitlement=%.2f GB, "
        "verified ceiling=%d).",
        batch_size,
        calibrated,
        actual_probe_rows,
        analytic_bytes_per_row / 1024.0,
        measured_bytes_per_row / 1024.0,
        effective_bytes_per_row / 1024.0,
        memory_after.available_bytes / float(1 << 30),
        actual_probe_rows,
    )
    return int(calibrated)


# Inference keeps only activations of the current layer alive (input +
# output). Training keeps the whole forward-pass activation stack for backward
# plus gradients and optimizer state. RMSProp/Adam each store 1-2x weights in
# moving averages on top, so 4x the inference peak is a conservative floor that
# leaves headroom for cuDNN workspace and Python-side torch overhead.
TRAINING_PEAK_MULTIPLIER = 4


def check_training_batch_fits(
        requested_batch_size,
        device,
        model,
        num_workers_per_gpu=1,
        free_memory_fraction=None,
        min_batch=64,
        logger=None):
    """Verify that ``requested_batch_size`` will fit on ``device``.

    Training peak memory = activations kept alive across the forward
    pass (for backward), plus gradients, plus optimizer state. That's
    roughly ``4 × estimate_peak_bytes_per_row`` (inference peak).

    Returns ``(effective_batch_size, shrunk: bool)``. When the
    requested batch is too large for the available VRAM — partitioned
    across co-resident workers — the batch is shrunk to the largest
    power-of-two that fits. ``min_batch`` is a normal-performance floor, not
    a memory-safety floor: under severe pressure the result may be smaller so
    the guard does not knowingly force an OOM. A loud warning is emitted via
    ``logger`` / stderr explaining that the
    training dynamics (BN running stats, gradient noise scale) now
    differ from what the caller configured.

    CPU short-circuits — no OOM risk there that a size-based heuristic
    can catch. Returns ``(requested_batch_size, False)`` in that case.
    """
    import sys
    requested_batch_size = int(requested_batch_size)
    if requested_batch_size < 1:
        raise ValueError("requested training batch size must be at least 1")
    if device.type == "cpu":
        return requested_batch_size, False
    workers = int(num_workers_per_gpu)
    if workers < 1:
        raise ValueError("num_workers_per_gpu must be at least 1")
    if free_memory_fraction is not None:
        free_memory_fraction = float(free_memory_fraction)
        if (
                not math.isfinite(free_memory_fraction)
                or not 0 < free_memory_fraction <= 1):
            raise ValueError("free_memory_fraction must be in (0, 1]")
    peak_bytes = estimate_peak_bytes_per_row(model) * TRAINING_PEAK_MULTIPLIER
    memory = device_memory_budget(
        device,
        num_workers_per_gpu=workers,
        free_memory_fraction=free_memory_fraction,
    )
    free = memory.free_bytes
    budget = memory.available_bytes
    max_rows = max(budget // peak_bytes, 1)
    if requested_batch_size <= max_rows:
        return requested_batch_size, False
    shrunk = 1
    while shrunk * 2 <= max_rows:
        shrunk *= 2
    below_normal_floor = shrunk < int(min_batch)
    message = (
        "!!! TRAINING BATCH WILL NOT FIT !!!  "
        "Requested minibatch_size=%d on %s with %d worker(s)/GPU. "
        "Estimated need ~%.1f GB of %.1f GB free VRAM (per-worker budget "
        "~%.1f GB). Shrinking to %d.  "
        "This CHANGES TRAINING DYNAMICS: batch-norm running stats, "
        "gradient noise scale, and effective learning-rate schedule "
        "all depend on batch size. Re-check convergence before "
        "trusting the trained model. To pin an explicit size and "
        "silence this guard, set a minibatch_size the caller knows "
        "fits.%s" % (
            requested_batch_size, device, workers,
            requested_batch_size * peak_bytes / 1e9,
            free / 1e9,
            budget / 1e9,
            shrunk,
            (
                " Available memory requires going below the normal "
                "performance floor of %d." % int(min_batch)
                if below_normal_floor else ""
            ),
        )
    )
    if logger is not None:
        logger.warning(message)
    else:
        logging.warning(message)
    # Also scream to stderr so it's loud in the job log regardless of
    # which logger config the caller uses.
    print("\n" + message + "\n", file=sys.stderr, flush=True)
    if os.environ.get("MHCFLURRY_FAIL_ON_TRAINING_BATCH_SHRINK") == "1":
        raise RuntimeError(
            message
            + " Automatic shrink is forbidden by "
            "MHCFLURRY_FAIL_ON_TRAINING_BATCH_SHRINK=1."
        )
    return int(shrunk), True

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

import logging
import os
import sys

from .memory_budget import per_worker_memory_budget_bytes

DEFAULT_PREDICT_BATCH_SIZE = "auto"
AUTO_BATCH_MAX_ROWS = sys.maxsize
AUTO_BATCH_MIN_ROWS = 1024  # floor: avoid pathologically tiny batches
AUTO_BATCH_CPU_FALLBACK = 32_768  # CPU: large batches thrash L3; stay modest
AUTO_BATCH_FREE_FRACTION = None
_MPS_PSUTIL_WARNED = False  # one-shot warning if psutil is missing on MPS
if os.environ.get("MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"):
    DEFAULT_PREDICT_BATCH_SIZE = int(os.environ["MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"])
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
        return free if free > 0 else 4 * (1 << 30)
    return 2 * (1 << 30)


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

    Divides live free VRAM (after one shared reserve) by co-resident workers
    and the model's per-row peak activation estimate. ``max_rows`` is only an
    optional caller limit; the default auto path has no hardwired batch cap.

    CPU: returns ``cpu_fallback`` — large batches on CPU thrash L3
    cache and rarely help for the small networks mhcflurry trains.
    """
    if device.type == "cpu":
        rows = int(cpu_fallback)
        return min(rows, int(total_rows)) if total_rows is not None else rows
    peak_bytes = estimate_peak_bytes_per_row(model)
    free = free_device_memory_bytes(device)
    workers = max(int(num_workers_per_gpu), 1)
    if free_memory_fraction is None:
        budget = per_worker_memory_budget_bytes(free, workers)
    else:
        # Backward-compatible expert API: an explicit fraction replaces the
        # default shared reserve calculation.
        budget = int(free * float(free_memory_fraction) / workers)
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
    return int(value)


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
    power-of-two that fits (floored at ``min_batch``) and a loud
    warning is emitted via ``logger`` / stderr explaining that the
    training dynamics (BN running stats, gradient noise scale) now
    differ from what the caller configured.

    CPU short-circuits — no OOM risk there that a size-based heuristic
    can catch. Returns ``(requested_batch_size, False)`` in that case.
    """
    import sys
    if device.type == "cpu" or requested_batch_size <= min_batch:
        return int(requested_batch_size), False
    peak_bytes = estimate_peak_bytes_per_row(model) * TRAINING_PEAK_MULTIPLIER
    free = free_device_memory_bytes(device)
    workers = max(int(num_workers_per_gpu), 1)
    if free_memory_fraction is None:
        budget = per_worker_memory_budget_bytes(free, workers)
    else:
        budget = int(free * float(free_memory_fraction) / workers)
    max_rows = max(budget // peak_bytes, min_batch)
    requested_batch_size = int(requested_batch_size)
    if requested_batch_size <= max_rows:
        return requested_batch_size, False
    shrunk = 1
    while shrunk * 2 <= max_rows:
        shrunk *= 2
    shrunk = max(shrunk, min_batch)
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
        "fits." % (
            requested_batch_size, device, workers,
            requested_batch_size * peak_bytes / 1e9,
            free / 1e9,
            budget / 1e9,
            shrunk,
        )
    )
    if logger is not None:
        logger.warning(message)
    else:
        logging.warning(message)
    # Also scream to stderr so it's loud in the job log regardless of
    # which logger config the caller uses.
    print("\n" + message + "\n", file=sys.stderr, flush=True)
    return int(shrunk), True

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

"""Calibration sizing and cache helpers for class I affinity predictors."""

import hashlib
import logging

import numpy


def peptide_sequences_fingerprint(sequences):
    """Order-and-content-sensitive SHA-256 of a peptide list.

    Length-prefixes each peptide so e.g. ``["AB", "C"]`` and ``["A", "BC"]``
    cannot collide by concatenation. Used as the cache key for the fast
    calibration peptide-stage cache; collisions there silently reuse the
    wrong tensors and produce wrong PercentRankTransforms.
    """
    h = hashlib.sha256()
    for peptide in sequences:
        b = str(peptide).encode("utf8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()

class CalibrationFastCache(object):
    """Per-predictor-instance state for ``calibrate_percentile_ranks_fast``.

    Holds the device-resident peptide-stage tensors and the motif-summary
    helper state that survive across calibrate tasks within one worker.
    Centralizing both fields here makes the cache lifecycle visible —
    it used to live behind dynamic ``setattr`` of private names.
    """

    __slots__ = (
        "stage_signature",
        "cached_stages",
        "motif_signature",
        "motif_state",
    )

    def __init__(self):
        self.stage_signature = None
        self.cached_stages = None
        self.motif_signature = None
        self.motif_state = None

    def clear(self):
        self.stage_signature = None
        self.cached_stages = None
        self.motif_signature = None
        self.motif_state = None


def peptide_encoding_feature_dim(model):
    """Return the raw encoded peptide feature width for ``model`` if known."""
    enc_shape = getattr(model, "peptide_encoding_shape", None)
    if enc_shape is None:
        return None
    return int(enc_shape[0]) * int(enc_shape[1])


def resolve_peptide_feature_dim(model, peptide_feature_dim, peak_bytes):
    """Resolve the width of cached peptide features used during calibration.

    ``peptide_feature_dim`` is the actual width of
    ``forward_peptide_stage(...)`` when the caller has probed it. Without a
    probe, fall back to shape-derived estimates and then to the generic peak
    memory estimate.
    """
    if peptide_feature_dim is not None:
        return int(peptide_feature_dim)

    sub_networks = getattr(model, "networks", None)
    if sub_networks is not None and not hasattr(model, "peptide_encoding_shape"):
        try:
            sub_dims = []
            for net in sub_networks:
                sub_dim = peptide_encoding_feature_dim(net)
                sub_dims.append(1024 if sub_dim is None else sub_dim)
            return int(sum(sub_dims))
        except (AttributeError, TypeError, ValueError, IndexError):
            pass

    try:
        feature_dim = peptide_encoding_feature_dim(model)
    except (AttributeError, TypeError, ValueError, IndexError):
        feature_dim = None
    if feature_dim is not None:
        return feature_dim
    return int(peak_bytes // 32) if peak_bytes else 1024


def auto_size_calibration_batches(
        model, device, n_peptides, n_alleles,
        num_workers_per_gpu=1,
        free_memory_fraction=0.85,
        num_cached_networks=1,
        peptide_feature_dim=None,
        num_sub_networks=None,
        cuda_overhead_bytes=2 * (1 << 30),
        safety_multiplier=1.3,
        fixed_peptide_batch=None,
        fixed_allele_batch=None):
    """Split the auto-sized batch budget between peptide and allele axes.

    ``fixed_peptide_batch`` / ``fixed_allele_batch`` pin one axis to a
    user-supplied value (mixed pin/auto mode). The pinned axis is held
    constant and only the other axis is sized against the VRAM budget, so
    the auto axis shrinks to keep ``allele_batch × peptide_batch`` within
    the per-worker budget instead of being sized as if the pinned axis were
    also auto (which would underestimate peak VRAM).

    Models the per-worker VRAM peak as:

        peak = cuda_overhead
             + cache_bytes                # peptide-stage cache,
                                          # ``num_cached_networks ×
                                          # n_peptides × peptide_feature_dim × 4``
             + cartesian_intermediate     # transient forward
                                          # ``a_size × p_batch ×
                                          # peak_bytes_per_row``
             + small_state                # log-IC50 acc, ic50_unique,
                                          # motif state (~1 GB)
        with explicit free-memory headroom plus a safety factor on
        CUDA/runtime scratch allocations to absorb fragmentation that
        ``mem_get_info`` can't see.

    ``peak_bytes_per_row`` is calibrated for the cartesian fast path.
    For a merged ensemble it uses one sub-network's hidden activation
    peak plus the small retained per-sub-network outputs, matching
    ``MergedClass1NeuralNetwork.forward_cartesian_from_peptide_stage``
    which runs each sub-network to completion before starting the next.
    The cartesian-intermediate term scales with ``a_size × p_batch``,
    which is exactly the rate the budget carves out — preserving the existing
    ``total_rows = forward_budget // peak_bytes`` math.

    Returns the chosen ``(peptide_batch, allele_batch)``.
    """
    from ..pytorch_sizing import (
        AUTO_BATCH_MAX_ROWS,
        AUTO_BATCH_MIN_ROWS,
        compute_prediction_batch_size,
        free_device_memory_bytes,
    )
    import torch
    from ..workload_planning import env_float

    if n_peptides == 0 or n_alleles == 0:
        return max(n_peptides, 1), max(n_alleles, 1)
    free_memory_fraction = env_float(
        "MHCFLURRY_CALIBRATE_AUTO_FREE_MEMORY_FRACTION",
        free_memory_fraction,
        bounds=(0.0, 1.0),
    )
    reserve_fraction = env_float(
        "MHCFLURRY_CALIBRATE_AUTO_RESERVE_FRACTION", 0.10,
        bounds=(0.0, 1.0))
    reserve_min_bytes = int(
        env_float(
            "MHCFLURRY_CALIBRATE_AUTO_RESERVE_GB", 2.0,
            bounds=(0.0, None))
        * (1 << 30)
    )
    fixed_safety_multiplier = env_float(
        "MHCFLURRY_CALIBRATE_AUTO_FIXED_SAFETY_MULTIPLIER",
        safety_multiplier,
        bounds=(0.0, None),
    )
    if device.type != "cuda":
        total_rows = compute_prediction_batch_size(
            device,
            model=model,
            num_workers_per_gpu=num_workers_per_gpu,
            free_memory_fraction=free_memory_fraction,
        )
    else:
        workers = max(int(num_workers_per_gpu), 1)
        free = free_device_memory_bytes(device)
        total_memory = free
        try:
            props = torch.cuda.get_device_properties(device)
            total_memory = int(props.total_memory)
        except (
                AssertionError, AttributeError, RuntimeError,
                TypeError, ValueError):
            # Tests and nonstandard CUDA wrappers may not expose device
            # properties. Fall back to free memory; the explicit reserve
            # still keeps the budget bounded.
            pass
        peak_bytes = estimate_calibration_peak_bytes_per_row(model)
        reserve_bytes = max(
            reserve_min_bytes,
            int(total_memory * reserve_fraction),
        )
        fraction_budget = int(
            free * float(free_memory_fraction) / workers)
        reserved_headroom_budget = int(
            max(free - reserve_bytes, 0) / workers)
        per_worker_budget = min(fraction_budget, reserved_headroom_budget)
        sub_networks = getattr(model, "networks", None)
        if num_sub_networks is None:
            num_sub_networks = (
                len(sub_networks) if sub_networks is not None else 1
            )
        peptide_feature_dim = resolve_peptide_feature_dim(
            model,
            peptide_feature_dim,
            peak_bytes,
        )
        cache_bytes = (
            int(num_cached_networks)
            * int(n_peptides)
            * int(peptide_feature_dim)
            * 4
        )
        # Small-state pad: log-IC50 accumulator, ic50_unique view,
        # motif-summary state, PercentRankTransform.fit_batch_torch
        # buffers, allele_idx tensor, etc. Scales with a_size and
        # n_peptides; a 1 GB constant covers the production ranges
        # without further accounting. Cheap insurance vs OOM.
        small_state_bytes = 1 * (1 << 30)
        # The peptide-stage cache estimate is shape-derived and
        # persistent, so multiplying it by a fragmentation safety factor
        # unnecessarily strands several GB on A100-40GB. Keep explicit
        # global headroom, and safety-pad only CUDA/runtime scratch and
        # small state whose allocation behavior is less predictable.
        scratch_state_bytes = int(cuda_overhead_bytes) + small_state_bytes
        guarded_fixed_overhead = (
            cache_bytes
            + int(scratch_state_bytes * fixed_safety_multiplier)
        )
        forward_budget = (
            per_worker_budget - guarded_fixed_overhead
        )
        log_fields = {
            "free_gb": free / 1e9,
            "workers": workers,
            "total_gb": total_memory / 1e9,
            "reserve_gb": reserve_bytes / 1e9,
            "fraction_budget_gb": fraction_budget / 1e9,
            "per_worker_budget_gb": per_worker_budget / 1e9,
            "peptide_feature_dim": peptide_feature_dim,
            "num_sub_networks": num_sub_networks,
            "num_cached_networks": num_cached_networks,
            "cache_gb": cache_bytes / 1e9,
            "cuda_overhead_gb": cuda_overhead_bytes / 1e9,
            "small_state_gb": small_state_bytes / 1e9,
            "scratch_safety": fixed_safety_multiplier,
            "guarded_fixed_gb": guarded_fixed_overhead / 1e9,
            "forward_budget_gb": forward_budget / 1e9,
            "peak_bytes_per_row": peak_bytes,
            "peak_kb_per_row": peak_bytes / 1024,
        }
        logging.info(
            "calibrate auto-sizer: free=%(free_gb).2f GB, "
            "workers=%(workers)d, total=%(total_gb).2f GB, "
            "reserve=%(reserve_gb).2f GB, "
            "fraction_budget=%(fraction_budget_gb).2f GB, "
            "per_worker_budget=%(per_worker_budget_gb).2f GB, "
            "peptide_feature_dim=%(peptide_feature_dim)d "
            "(sub_networks=%(num_sub_networks)d, "
            "num_cached=%(num_cached_networks)d), "
            "cache=%(cache_gb).2f GB, "
            "cuda_overhead=%(cuda_overhead_gb).2f GB, "
            "small_state=%(small_state_gb).2f GB, "
            "scratch_safety=%(scratch_safety).2fx, "
            "guarded_fixed=%(guarded_fixed_gb).2f GB "
            "-> forward_budget=%(forward_budget_gb).2f GB, "
            "peak_bytes_per_row=%(peak_bytes_per_row)d "
            "(≈%(peak_kb_per_row).2f KB)",
            log_fields,
        )
        if forward_budget < peak_bytes * AUTO_BATCH_MIN_ROWS:
            logging.warning(
                "calibrate auto-sizer: fixed overhead "
                "(cache %.2f GB + scratch/state %.2f GB × safety %.2fx) "
                "exceeds per-worker budget %.2f GB "
                "(%.2f GB free, %.2f GB reserved, %d workers). "
                "Falling back to minimum batch; reduce "
                "--max-workers-per-gpu or lower the peptide universe "
                "size.",
                cache_bytes / 1e9,
                scratch_state_bytes / 1e9,
                fixed_safety_multiplier,
                per_worker_budget / 1e9,
                free / 1e9,
                reserve_bytes / 1e9,
                workers,
            )
            forward_budget = peak_bytes * AUTO_BATCH_MIN_ROWS
        total_rows = max(
            AUTO_BATCH_MIN_ROWS,
            min(forward_budget // peak_bytes, AUTO_BATCH_MAX_ROWS),
        )
    return choose_calibration_batch_shape(
        total_rows,
        n_peptides=n_peptides,
        n_alleles=n_alleles,
        min_peptide_batch=max(AUTO_BATCH_MIN_ROWS, 2_000),
        fixed_peptide_batch=fixed_peptide_batch,
        fixed_allele_batch=fixed_allele_batch,
    )

def estimate_calibration_peak_bytes_per_row(model):
    """Estimate cartesian calibration forward peak bytes per row.

    The generic prediction estimator is deliberately conservative for
    arbitrary merged forwards. Calibration has a more specific execution
    shape: ``MergedClass1NeuralNetwork.forward_cartesian_from_peptide_stage``
    evaluates sub-networks serially and retains only their final outputs
    before combining them. Hidden-layer peak memory is therefore the max
    sub-network peak, not the sum of every sub-network peak.
    """
    from ..pytorch_sizing import estimate_peak_bytes_per_row

    if model is None:
        return estimate_peak_bytes_per_row(model)

    sub_networks = getattr(model, "networks", None)
    if sub_networks is None or hasattr(model, "peptide_encoding_shape"):
        return estimate_peak_bytes_per_row(model)

    try:
        sub_peaks = [
            estimate_peak_bytes_per_row(net)
            for net in sub_networks
        ]
        if not sub_peaks:
            return estimate_peak_bytes_per_row(model)
        output_channels = 0
        for net in sub_networks:
            output_layer = getattr(net, "output_layer", None)
            output_channels += int(getattr(output_layer, "out_features", 1))
        # Retained outputs are small compared with hidden activations but
        # include them, padded for stack/cat/log-IC50 temporary tensors.
        retained_output_bytes = max(output_channels, 1) * 8 * 4
        return int(max(sub_peaks) + retained_output_bytes)
    except (AttributeError, TypeError, ValueError) as exc:
        logging.warning(
            "Could not estimate calibration peak for merged ensemble; "
            "falling back to generic estimator: %s",
            exc,
        )
        return estimate_peak_bytes_per_row(model)

def choose_calibration_batch_shape(
        total_rows,
        n_peptides,
        n_alleles,
        min_peptide_batch,
        max_allele_batch=256,
        fixed_peptide_batch=None,
        fixed_allele_batch=None):
    """Choose ``(peptide_batch, allele_batch)`` under a row budget.

    Minimize the number of cartesian forward chunks rather than filling
    one axis greedily. This matters for the production shape
    (tens of alleles × tens of thousands of peptides), where many
    equivalent row budgets can differ by 20-40% in Python-level loop
    count and kernel launches.

    ``fixed_peptide_batch`` / ``fixed_allele_batch`` pin one axis (mixed
    pin/auto mode). The pinned axis is held at the user value and the other
    axis is sized as ``total_rows // pinned`` so the product stays within
    the budget; the pinned axis is *not* capped by ``max_allele_batch``
    (the user asked for it explicitly).
    """
    total_rows = max(int(total_rows), 1)
    n_peptides = max(int(n_peptides), 1)
    n_alleles = max(int(n_alleles), 1)
    min_peptide_batch = max(int(min_peptide_batch), 1)
    max_allele_batch = max(int(max_allele_batch), 1)

    # Mixed pin/auto: hold the pinned axis fixed and size the other axis
    # against the same row budget so peak VRAM isn't underestimated.
    if fixed_allele_batch is not None and fixed_peptide_batch is None:
        allele_batch = min(max(int(fixed_allele_batch), 1), n_alleles)
        peptide_batch = max(total_rows // allele_batch, 1)
        peptide_batch = min(n_peptides, peptide_batch)
        return int(peptide_batch), int(allele_batch)
    if fixed_peptide_batch is not None and fixed_allele_batch is None:
        peptide_batch = min(max(int(fixed_peptide_batch), 1), n_peptides)
        allele_batch = max(total_rows // peptide_batch, 1)
        allele_batch = min(n_alleles, max_allele_batch, allele_batch)
        return int(peptide_batch), int(allele_batch)

    max_a = min(n_alleles, max_allele_batch, max(total_rows, 1))
    best = None
    for allele_batch in range(1, max_a + 1):
        peptide_batch = total_rows // allele_batch
        if peptide_batch < min_peptide_batch:
            continue
        peptide_batch = min(n_peptides, max(min_peptide_batch, peptide_batch))
        chunk_count = (
            int(numpy.ceil(n_alleles / float(allele_batch)))
            * int(numpy.ceil(n_peptides / float(peptide_batch)))
        )
        used_rows = allele_batch * peptide_batch
        candidate = (
            chunk_count,
            -used_rows,
            -allele_batch,
            -peptide_batch,
            peptide_batch,
            allele_batch,
        )
        if best is None or candidate < best:
            best = candidate

    if best is None:
        allele_batch = min(n_alleles, max_a)
        peptide_batch = min(n_peptides, max(min_peptide_batch, total_rows))
        while allele_batch * peptide_batch > total_rows and allele_batch > 1:
            allele_batch -= 1
        while (
                allele_batch * peptide_batch > total_rows
                and peptide_batch > min_peptide_batch):
            peptide_batch = max(min_peptide_batch, peptide_batch // 2)
        return int(peptide_batch), int(allele_batch)

    return int(best[4]), int(best[5])

def calibration_stage_cache_signature(encoded_peptides, networks, device):
    """Return the key for a reusable peptide-stage calibration cache.

    Keyed on the peptide-set fingerprint, the network object identities
    (``id``), and the device -- NOT on weight *content*. Adding, removing,
    or replacing ensemble models changes the ``networks`` list (new
    objects -> new ids) and so invalidates the cache, but mutating an
    existing network's weights *in place* (e.g. a re-fit) does not. A
    weight-content fingerprint is deliberately not used here:
    ``borrow_cached_network`` serves architecturally-identical networks
    from a single process-wide ``MODELS_CACHE`` module, so the underlying
    torch parameter storage is shared across ensemble members and is not a
    reliable per-network weight signal. Callers that mutate weights in
    place between fast-calibrate calls on the same predictor instance must
    therefore call ``clear_calibration_fast_cache()`` first (see
    ``calibrate_percentile_ranks_fast``).
    """
    return (
        peptide_sequences_fingerprint(encoded_peptides.sequences),
        tuple(id(net) for net in networks),
        str(device),
    )

def probe_peptide_feature_dim(net_obj, encoded_peptides, device):
    """Run a 1-row forward through ``forward_peptide_stage`` to
    record the actual width of the peptide feature tensor.

    The auto-sizer's cache estimate depends on this and
    the encoding-shape heuristic under-counts when the model
    configures ``peptide_dense_layer_sizes`` or LC layers.

    Returns ``None`` on probe failure so callers can fall back
    to the heuristic.
    """
    import torch
    from ..encodable_sequences import EncodableSequences
    try:
        seqs = list(encoded_peptides.sequences)
        if not seqs:
            return None
        probe_seqs = EncodableSequences.create([seqs[0]])
        probe_input = net_obj.peptides_to_network_input(probe_seqs)
        probe_is_int = net_obj.uses_peptide_torch_encoding()
        if (
            probe_is_int
            and probe_input.ndim == 2
            and numpy.issubdtype(probe_input.dtype, numpy.integer)
        ):
            probe_tensor = torch.from_numpy(probe_input).to(device)
        else:
            probe_tensor = torch.from_numpy(
                numpy.asarray(probe_input, dtype=numpy.float32)
            ).to(device)
        model = net_obj.network(borrow=True)
        model.eval()
        with torch.no_grad():
            peptide_features = model.forward_peptide_stage(probe_tensor)
        return int(peptide_features.shape[-1])
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logging.warning(
            "calibrate auto-sizer: peptide_feature_dim probe failed "
            "(%s); falling back to encoding-shape heuristic", exc,
        )
        return None

def calibration_fast_cache(self):
    """Return (creating if needed) the per-instance fast-calibrate cache.

    See ``CalibrationFastCache``. Lazy so that predictors loaded for
    prediction never pay the allocation cost.
    """
    cache = getattr(self, "_calibration_fast_cache_state", None)
    if cache is None:
        cache = CalibrationFastCache()
        self._calibration_fast_cache_state = cache
    return cache

def clear_calibration_fast_cache(self):
    """Drop any cached fast-calibrate state on this predictor.

    Long-lived workers that finish calibrate but stay alive for
    prediction can reclaim the (potentially many GB) of device-
    resident peptide-stage tensors via this hook.
    """
    cache = getattr(self, "_calibration_fast_cache_state", None)
    if cache is not None:
        cache.clear()
        del self._calibration_fast_cache_state

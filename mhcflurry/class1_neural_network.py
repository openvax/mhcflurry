"""
Class1NeuralNetwork - PyTorch implementation for MHC class I binding prediction.
"""

import gc
import time
import collections
import json
import multiprocessing
import weakref
import itertools
import os
import logging

import numpy
import pandas
import torch
import torch.nn as nn

from .hyperparameters import HyperparameterDefaults
from .encodable_sequences import EncodableSequences, EncodingError
from .allele_encoding import AlleleEncoding
from .regression_target import to_ic50, from_ic50
from .common import get_pytorch_device
from .pytorch_layers import LocallyConnected1D, get_activation
from .pytorch_losses import get_pytorch_loss
from .data_dependent_weights_initialization import lsuv_init
from .random_negative_peptides import RandomNegativePeptides, RandomNegativesPool


DEFAULT_PREDICT_BATCH_SIZE = "auto"
_AUTO_BATCH_MAX_ROWS = 1_000_000  # cap past which kernel-launch savings flatten
_AUTO_BATCH_MIN_ROWS = 1024       # floor: avoid pathologically tiny batches
_AUTO_BATCH_CPU_FALLBACK = 32_768 # CPU: large batches thrash L3 — stay modest
_AUTO_BATCH_FREE_FRACTION = 0.5   # half of free VRAM is the working-set budget
_MPS_PSUTIL_WARNED = False        # one-shot warning if psutil is missing on MPS
if os.environ.get("MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"):
    DEFAULT_PREDICT_BATCH_SIZE = int(os.environ["MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE"])
    logging.info(
        "Configured default predict batch size: %s" % DEFAULT_PREDICT_BATCH_SIZE
    )


def _estimate_peak_bytes_per_row(model):
    """Worst-case peak activation bytes per sample during a forward.

    Walks the model's configured layers and returns the maximum hidden-
    layer width (in fp32 bytes) × 2 (one input + one output of the
    current layer stay live under torch's eval-time no_grad reuse). A
    4× multiplier covers framework overhead, cuDNN scratch buffers,
    and Python-side tensor bookkeeping. Used by ``compute_prediction_batch_size``.
    """
    if model is None:
        return 32 * 1024  # conservative 32 KB/row fallback
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
        prev = lc_out_len * lc_out_ch
        for layer in model.peptide_dense_layers:
            prev = layer.out_features
            widths.append(prev)
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


def _free_device_memory_bytes(device):
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
            # psutil isn't a hard dep. Log once per process so the
            # skip is visible rather than silent — without this cap
            # the MPS driver's "recommended max" can exceed what's
            # actually safe to claim alongside other apps.
            global _MPS_PSUTIL_WARNED
            if not _MPS_PSUTIL_WARNED:
                logging.warning(
                    "psutil not available; MPS free-memory estimate "
                    "will use torch.mps.recommended_max_memory alone, "
                    "which may overshoot actual available RAM. "
                    "`pip install psutil` to enable the OS-level cap."
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
        free_memory_fraction=_AUTO_BATCH_FREE_FRACTION,
        max_rows=_AUTO_BATCH_MAX_ROWS,
        min_rows=_AUTO_BATCH_MIN_ROWS,
        cpu_fallback=_AUTO_BATCH_CPU_FALLBACK):
    """Auto-size a prediction batch for ``device`` and ``model``.

    Divides free VRAM by the per-row peak activation estimate for the
    model, scales by 1/``num_workers_per_gpu`` so co-resident workers
    don't step on each other's budget, caps at ``max_rows`` (past which
    kernel-launch savings flatten out), floors at ``min_rows``.

    CPU: returns ``cpu_fallback`` — large batches on CPU thrash L3
    cache and rarely help for the small networks mhcflurry trains.
    """
    if device.type == "cpu":
        return cpu_fallback
    peak_bytes = _estimate_peak_bytes_per_row(model)
    free = _free_device_memory_bytes(device)
    workers = max(int(num_workers_per_gpu), 1)
    budget = int(free * float(free_memory_fraction) / workers)
    budget = max(budget, peak_bytes * min_rows)
    rows = budget // peak_bytes
    return int(max(min_rows, min(rows, max_rows)))


def _env_workers_per_gpu(default=1):
    """Read the ``MHCFLURRY_MAX_WORKERS_PER_GPU`` env var.

    The local_parallelism pool sets this in each training worker so
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
        value, device, model=None, num_workers_per_gpu=1):
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
        )
    return int(value)


# Training-time memory multiplier: inference keeps only activations of
# the current layer alive (input + output), training keeps the whole
# forward-pass activation stack for backward plus gradients plus
# optimizer state. RMSProp/Adam each store 1-2× weights in moving
# averages on top. 4× the inference peak is a conservative floor that
# leaves headroom for cuDNN workspace + Python-side torch overhead.
_TRAINING_PEAK_MULTIPLIER = 4
_FIT_DATALOADER_SPAWN_COPY_LIMIT_BYTES = int(
    float(os.environ.get("MHCFLURRY_FIT_DATALOADER_MAX_PICKLE_MB", "128"))
    * 1024
    * 1024
)
_FIT_DATALOADER_DOWNGRADE_WARNED = False


def check_training_batch_fits(
        requested_batch_size,
        device,
        model,
        num_workers_per_gpu=1,
        free_memory_fraction=0.5,
        min_batch=64,
        logger=None):
    """Verify that ``requested_batch_size`` will fit on ``device``.

    Training peak memory = activations kept alive across the forward
    pass (for backward), plus gradients, plus optimizer state. That's
    roughly ``4 × _estimate_peak_bytes_per_row`` (inference peak).

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
    peak_bytes = _estimate_peak_bytes_per_row(model) * _TRAINING_PEAK_MULTIPLIER
    free = _free_device_memory_bytes(device)
    workers = max(int(num_workers_per_gpu), 1)
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


KERAS_BATCH_NORM_EPSILON = 1e-3
# Keras uses moving = moving * 0.99 + batch * 0.01. PyTorch's momentum is the
# new-batch coefficient, so the equivalent value is 0.01.
KERAS_BATCH_NORM_MOMENTUM = 0.01


class _FitBatchDataset(torch.utils.data.Dataset):
    """Map-style Dataset backing fit()'s inner batch loop.

    Wraps the already-built per-epoch arrays (x_peptide, x_allele,
    y_encoded, sample_weights_with_negatives) and the shuffled
    train_indices. __getitem__ does a single-row fancy-index into each
    array. With ``shuffle=False`` on the DataLoader, iteration order
    matches the current (non-DataLoader) code path exactly, which is
    what the bit-identical integration test requires.

    See issue openvax/mhcflurry#268 for motivation and semantic-
    preservation rationale.
    """

    def __init__(
            self,
            x_peptide,
            x_allele,
            y_encoded,
            sample_weights_with_negatives,
            train_indices,
            *,
            random_negative_x_peptide=None,
            random_negative_x_allele=None,
            num_random_negatives=None):
        self.x_peptide = x_peptide
        self.x_allele = x_allele
        self.random_negative_x_peptide = random_negative_x_peptide
        self.random_negative_x_allele = random_negative_x_allele
        self.num_random_negatives = (
            None if num_random_negatives is None else int(num_random_negatives)
        )
        # Pre-cast y to float32 once at dataset construction rather than
        # per ``__getitem__`` call. ``y_encoded`` is a fresh per-epoch
        # numpy array so this is a single allocation of the full
        # (train+neg) vector (~2x the training-set size × 4 bytes).
        # Without this pre-cast, fit() paid a tiny per-batch astype() —
        # millions of one-element allocs per epoch on large pan-allele
        # training sets.
        if y_encoded.dtype != numpy.float32:
            y_encoded = y_encoded.astype(numpy.float32)
        self.y_encoded = y_encoded
        self.sample_weights = sample_weights_with_negatives
        self.train_indices = train_indices

        if self.x_peptide is None and self.random_negative_x_peptide is None:
            raise ValueError("FitBatchDataset requires peptide features")

    def _lookup_feature_row(self, idx, *, base_array, negative_array):
        if self.num_random_negatives is None:
            return base_array[idx]
        if idx < self.num_random_negatives:
            return negative_array[idx]
        return base_array[idx - self.num_random_negatives]

    def _gather_feature_rows(self, indices, *, base_array, negative_array):
        if base_array is None and negative_array is None:
            return None
        indices = numpy.asarray(indices, dtype=numpy.int64)
        if self.num_random_negatives is None:
            return base_array[indices]
        result = numpy.empty(
            (len(indices),) + base_array.shape[1:],
            dtype=base_array.dtype,
        )
        negative_mask = indices < self.num_random_negatives
        if negative_mask.any():
            result[negative_mask] = negative_array[indices[negative_mask]]
        if (~negative_mask).any():
            result[~negative_mask] = base_array[
                indices[~negative_mask] - self.num_random_negatives
            ]
        return result

    def __len__(self):
        return len(self.train_indices)

    def __getitem__(self, i):
        idx = self.train_indices[i]
        sample = {
            "peptide": self._lookup_feature_row(
                idx,
                base_array=self.x_peptide,
                negative_array=self.random_negative_x_peptide,
            ),
            "y": self.y_encoded[idx],
        }
        if self.x_allele is not None or self.random_negative_x_allele is not None:
            sample["allele"] = self._lookup_feature_row(
                idx,
                base_array=self.x_allele,
                negative_array=self.random_negative_x_allele,
            )
        if self.sample_weights is not None:
            sample["weight"] = self.sample_weights[idx]
        return sample

    def batch_for_indices(self, indices):
        """Return a batch dict for the given global row indices."""
        indices = numpy.asarray(indices, dtype=numpy.int64)
        batch = {
            "peptide": self._gather_feature_rows(
                indices,
                base_array=self.x_peptide,
                negative_array=self.random_negative_x_peptide,
            ),
            "y": self.y_encoded[indices],
        }
        allele_rows = self._gather_feature_rows(
            indices,
            base_array=self.x_allele,
            negative_array=self.random_negative_x_allele,
        )
        if allele_rows is not None:
            batch["allele"] = allele_rows
        if self.sample_weights is not None:
            batch["weight"] = self.sample_weights[indices]
        return batch


def _numpy_batch_collate(batch):
    """Collate a list of sample dicts into numpy arrays.

    On some platforms / restricted environments, PyTorch's default
    worker-side collate path converts numpy arrays to tensors and then
    tries to stand up ``torch_shm_manager`` to share those tensors back
    to the parent process. That can fail with ``Operation not
    permitted`` even though plain worker processes themselves are fine.

    Returning numpy arrays keeps inter-process transport on the standard
    Python pickling path. The parent process then does the numpy->torch
    conversion immediately before the H2D copy in fit().
    """
    return {
        key: numpy.stack([sample[key] for sample in batch], axis=0)
        for key in batch[0]
    }


def _identity_collate(sample):
    """Return a pre-batched sample unchanged."""
    return sample


def _materialize_repeated_peptide_batch(batch):
    """Return ``batch`` with compact repeated-peptide input expanded.

    Pretrain batches can carry each unique peptide encoding once plus a
    ``peptide_repeat_count`` telling the training forward path to form the
    peptide × allele batch on device. LSUV/data-dependent initialization still
    expects ordinary row-aligned numpy arrays, so materialize that rare
    one-batch path here.
    """
    repeat_count = batch.get("peptide_repeat_count")
    if repeat_count is None:
        return batch
    result = dict(batch)
    result["peptide"] = numpy.repeat(
        result["peptide"],
        int(repeat_count),
        axis=0,
    )
    del result["peptide_repeat_count"]
    return result


class _FitGeneratorBatchIterableDataset(torch.utils.data.IterableDataset):
    """IterableDataset backing ``fit_generator``.

    Supports two input formats:
    - Legacy raw tuples: ``(alleles, peptides, affinities)``
    - Pre-encoded tuples: ``(x_dict, y)``

    The pre-encoded path is what allows worker-side prefetch in
    pretraining: the dataset can be pickled into spawned workers without
    serializing the full ``Class1NeuralNetwork`` instance.

    Picklability contract
    ---------------------
    When ``num_workers>0``, PyTorch's DataLoader pickles the dataset via
    ``ForkingPickler`` to ship it to each spawned worker. Two classes of
    attributes would break that:

    - Live generator objects (from ``generator=iter(...)``). Generators
      are intrinsically unpicklable.
    - Bound methods of ``Class1NeuralNetwork`` (``peptides_to_network_input``,
      ``allele_encoding_to_network_input``). Pickling a bound method
      pickles ``self``, i.e. the entire network.

    We sidestep both by dropping those references whenever they're
    redundant: the factory path never reads ``self.generator`` (always
    fresh-instantiates per worker), and the pre-encoded path never reads
    the ``*_to_input`` callbacks. The live-generator / raw-tuple path is
    single-process by construction — the downgrade lives in
    ``fit_generator`` — so zeroing these out doesn't remove any
    functionality.
    """

    def __init__(
        self,
        *,
        generator,
        generator_factory=None,
        source_batches_are_encoded=False,
        allele_encoding_to_input=None,
        peptides_to_network_input=None,
    ):
        # Factory takes precedence: every call to ``_iter_source`` will
        # invoke the factory and ignore ``self.generator``, so don't
        # retain the live (unpicklable) generator object.
        self.generator = None if generator_factory is not None else generator
        self.generator_factory = generator_factory
        self.source_batches_are_encoded = source_batches_are_encoded
        # Pre-encoded batches never consume these callbacks. Drop them
        # to keep ``ForkingPickler`` from dragging the whole
        # ``Class1NeuralNetwork`` instance into every spawned worker.
        if source_batches_are_encoded:
            self.allele_encoding_to_input = None
            self.peptides_to_network_input = None
        else:
            self.allele_encoding_to_input = allele_encoding_to_input
            self.peptides_to_network_input = peptides_to_network_input

    def _iter_source(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.generator_factory is not None:
            worker_id = worker_info.id if worker_info is not None else 0
            num_workers = (
                worker_info.num_workers if worker_info is not None else 1
            )
            return self.generator_factory(
                worker_id=worker_id,
                num_workers=num_workers,
            )
        if worker_info is not None:
            raise ValueError(
                "fit_generator with DataLoader workers requires a "
                "picklable generator_factory."
            )
        return self.generator

    def _normalize_item(self, item):
        if self.source_batches_are_encoded:
            x_dict, y = item
        else:
            if self.peptides_to_network_input is None:
                raise ValueError(
                    "peptides_to_network_input is required for raw "
                    "fit_generator batches."
                )
            alleles, peptides, affinities = item
            allele_input = None
            if self.allele_encoding_to_input is not None:
                allele_input, _ = self.allele_encoding_to_input(alleles)
            x_dict = {
                "peptide": self.peptides_to_network_input(peptides),
            }
            if allele_input is not None:
                x_dict["allele"] = allele_input
            y = from_ic50(affinities)

        result = dict(x_dict)
        if not isinstance(y, numpy.ndarray):
            y = numpy.asarray(y)
        if y.dtype != numpy.float32:
            y = y.astype(numpy.float32)
        result["y"] = y
        return result

    def __iter__(self):
        for item in self._iter_source():
            yield self._normalize_item(item)


def _configure_matmul_precision(device):
    """Optionally enable TF32 + cuDNN benchmark on CUDA Ampere+.

    Both are runtime settings with no JIT/startup overhead, but TF32 changes
    CUDA matmul numerics. Leave PyTorch's default behavior untouched unless
    the caller explicitly opts in with ``MHCFLURRY_MATMUL_PRECISION``.

    **TF32 (``torch.set_float32_matmul_precision``)** — changes matmul
    kernel selection. On Ampere+ (A100/H100/L40S/...) it can be ~2×
    faster for matmul-heavy paths, with fp32 accumulation preserved but
    input-mantissa truncation.

    **cuDNN benchmark** — tells cuDNN to search for the fastest
    algorithm for each (shape, dtype, stride) tuple on first call and
    cache the result. Useful when input shapes are stable across
    iterations (our fit-generator + fit paths guarantee this via
    drop_last + fixed-size pretrain chunks). One-time cost of
    benchmarking on the first forward pass (~1-2 s), then every
    subsequent call hits the cached best algorithm. mhcflurry's MLP
    doesn't use Conv layers in the default architecture, so the gain
    is modest here — but it's free and lets convolutional variants
    (if ever configured via ``locally_connected_layers``) benefit.

    Backend interactions:
    - CUDA Ampere+: TF32 enabled + cudnn.benchmark on.
    - CUDA pre-Ampere (V100/T4): TF32 is no-op; cudnn.benchmark still helps.
    - CPU: both are no-ops.
    - MPS: both are no-ops.

    Opt in via ``MHCFLURRY_MATMUL_PRECISION={highest,high,medium}``.
    ``highest`` keeps full fp32 precision while still enabling
    ``cudnn.benchmark``.
    """
    if device.type != "cuda":
        return
    precision = os.environ.get("MHCFLURRY_MATMUL_PRECISION")
    if not precision:
        return
    torch.set_float32_matmul_precision(precision)
    # cuDNN benchmark is cheap to enable and has no effect if the
    # workload never triggers a cuDNN kernel (plain Linear + RMSprop
    # MLP). Guarded against environments that disabled cuDNN entirely.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def _maybe_compile_network(network, device):
    """Wrap ``network`` with ``torch.compile`` when the env asks for it.

    Gated on ``MHCFLURRY_TORCH_COMPILE=1`` and a CUDA device.
    ``torch.compile`` is heavy: JIT graph capture + kernel fusion + on-
    disk cache + recompile-on-shape-change. The TF32 knob is cheaper
    and independent — see ``_configure_matmul_precision``.

    ``MHCFLURRY_TORCH_COMPILE_MODE`` picks the ``mode=`` kwarg (default,
    reduce-overhead, max-autotune). Default is "default" — codegen time
    is already heavy without max-autotune, and our shape-stable batching
    is what unlocks the big wins regardless of mode.

    Returns the compiled module (an ``OptimizedModule`` that forwards
    ``.train()``, ``.eval()``, ``.state_dict()``, ``.parameters()`` to
    the wrapped original) so call sites can swap it in without
    threading a second reference through the training loop.

    Compilation cost: first forward pass triggers graph capture +
    codegen (typically 30 s – 2 min on a 2-layer MLP like ours).
    Subsequent calls with the same input shapes hit the in-process
    cache; subsequent *processes* hit the on-disk cache
    (``~/.cache/torch``) as long as the graph matches.
    """
    if device.type != "cuda":
        return network
    if os.environ.get("MHCFLURRY_TORCH_COMPILE", "0") != "1":
        return network
    # Idempotent: if ``network`` is already an OptimizedModule (i.e.
    # we've been called before on the same instance, from inside an
    # epoch loop), return it unchanged.
    if hasattr(network, "_orig_mod"):
        return network
    mode = os.environ.get("MHCFLURRY_TORCH_COMPILE_MODE", "default")
    # ``dynamic=True`` tells dynamo to generate one shape-polymorphic
    # graph instead of specializing on every batch shape it sees.
    # mhcflurry's forward is called with at least three distinct row
    # counts per work item — pretrain (64 rows), finetune (128), and
    # validation (4× finetune = 512) — so dynamic=False triggers a
    # recompile storm (8+ specializations observed in stderr with
    # [0/8] from torch._dynamo.convert_frame). Each recompile is a
    # 10-30 s codegen pass, defeating the point. Dynamic mode costs a
    # few % on the individual kernel but avoids paying the storm.
    # Override with MHCFLURRY_TORCH_COMPILE_DYNAMIC=0 for static mode
    # if a caller can guarantee single-shape input.
    dynamic = os.environ.get("MHCFLURRY_TORCH_COMPILE_DYNAMIC", "1") != "0"
    logging.info("torch.compile enabled (mode=%s, dynamic=%s)", mode, dynamic)
    return torch.compile(network, mode=mode, dynamic=dynamic)


def _batch_value_to_device(value, device, *, non_blocking, cast_float):
    """Move a batch value (numpy array or tensor) to ``device``.

    When ``cast_float=True`` and the source is float64, cast to float32 on
    the CPU *before* transfer. MPS refuses to host float64 tensors, and
    transferring fp64 just to narrow it on-device wastes bandwidth anyway.
    int8 / float32 inputs skip the CPU-side cast so the Phase 4a (#268)
    pattern of shipping int8 and widening on GPU stays intact.

    When ``cast_float=False`` the value is shipped through in its source
    dtype — used by the Phase 2 on-device BLOSUM62 path (#268) where
    peptide tensors are (N, L) int indices that get widened to (N, L, 21)
    fp32 via embedding lookup inside the network's forward pass, not via
    an integer→float cast here.
    """
    if isinstance(value, numpy.ndarray):
        value = torch.from_numpy(value)
    if cast_float and value.dtype == torch.float64:
        value = value.float()
    value = value.to(device, non_blocking=non_blocking)
    if cast_float:
        value = value.float()
    return value


# BLOSUM62 weight table cached per device — one copy per CUDA GPU. Used
# by the Phase 2 on-device embedding path below (#268). Populated lazily
# on first use because the model can span multiple CUDA contexts in
# multi-GPU setups even though a single fit() call runs on one.
_BLOSUM62_DEVICE_CACHE = {}


def _blosum62_table_for_device(device):
    """Return a (21, 21) float32 torch tensor of BLOSUM62 on ``device``.

    The values are integers in [-4, +11], so fp32 is lossless. Indexed by
    ``amino_acid.AMINO_ACID_INDEX`` (same ordering as ``AMINO_ACIDS``).
    Cached per device so it's materialized exactly once per process.
    """
    # device can be torch.device or str; build a stable dict key.
    key = (str(device.type), getattr(device, "index", None))
    if key not in _BLOSUM62_DEVICE_CACHE:
        from .amino_acid import BLOSUM62_MATRIX
        table = BLOSUM62_MATRIX.to_numpy().astype(numpy.float32)
        _BLOSUM62_DEVICE_CACHE[key] = torch.from_numpy(table).to(device)
    return _BLOSUM62_DEVICE_CACHE[key]


def _timing_enabled():
    """Return True when fine-grained training timing is enabled."""
    return os.environ.get("MHCFLURRY_ENABLE_TIMING", "0") == "1"


def _timing_synchronize(device):
    """Synchronize asynchronous device work for accurate wall timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _timing_start(device, enabled):
    if enabled:
        _timing_synchronize(device)
    return time.perf_counter()


def _timing_stop(start, device, enabled):
    if enabled:
        _timing_synchronize(device)
    return time.perf_counter() - start


def _uncompiled_network(network):
    """Return the eager module behind ``network``."""
    return network._orig_mod if hasattr(network, "_orig_mod") else network


def _maybe_compile_loss(loss_obj, device):
    """Wrap a loss module with ``torch.compile`` when the env asks for it.

    Gated on ``MHCFLURRY_TORCH_COMPILE=1`` and a CUDA device — same
    criteria as ``_maybe_compile_network``. ``MSEWithInequalities``
    issues ~10 small elementwise kernels per step in eager mode
    (reshape → subtract → compare → cast → multiply → compare → cast →
    multiply → square → sum), each with its own launch overhead that
    adds up to meaningful wall-clock on A100 with a sub-ms compute
    budget. Dynamo fuses those into a couple of kernels and cuts the
    loss's share of step time to near-zero.

    Dispatch via ``MHCFLURRY_TORCH_COMPILE_LOSS_MODE`` (falls back to
    ``MHCFLURRY_TORCH_COMPILE_MODE``) so the loss's compile mode can
    be tuned independently of the network's — loss ops are tiny and
    "reduce-overhead" makes less sense than on the full forward pass.

    Idempotent: a second call on an already-wrapped loss returns it
    unchanged.
    """
    if device.type != "cuda":
        return loss_obj
    # Gated on a SEPARATE env var from the network compile. Setting
    # ``MHCFLURRY_TORCH_COMPILE=1`` no longer implicitly compiles the
    # loss — opt in with ``MHCFLURRY_TORCH_COMPILE_LOSS=1``. Caught on
    # the 2026-04-24 release_exact launch: compiling ``MSEWithInequalities``
    # on the 4096-batch CUDA path crashed every training worker with
    # ``RuntimeError: Triton Error [CUDA]: invalid device context`` inside
    # the fused backward kernel (``triton_poi_fused__to_copy_add_ge_gt_le_lt_mul_pow_sub``).
    # Likely a torch.compile x loss-module x dynamic-batch interaction;
    # turning the loss compile off keeps the much bigger network-compile
    # win intact. Revisit once the upstream Triton / torch bug is known
    # or after we build a smoke-test harness to catch regressions.
    if os.environ.get("MHCFLURRY_TORCH_COMPILE_LOSS", "0") != "1":
        return loss_obj
    if hasattr(loss_obj, "_orig_mod"):
        return loss_obj
    mode = os.environ.get(
        "MHCFLURRY_TORCH_COMPILE_LOSS_MODE",
        os.environ.get("MHCFLURRY_TORCH_COMPILE_MODE", "default"),
    )
    # Loss takes (y_pred, y_true) and optionally sample_weights with
    # dynamic-batch shapes that mirror the network's forward. Match the
    # network's dynamic/static policy via the same env knob.
    dynamic = os.environ.get("MHCFLURRY_TORCH_COMPILE_DYNAMIC", "1") != "0"
    logging.info("torch.compile applied to loss (mode=%s, dynamic=%s)", mode, dynamic)
    return torch.compile(loss_obj, mode=mode, dynamic=dynamic)


def _effective_validation_batch_size(
        device, configured_batch_size, minibatch_size,
        model=None, num_workers_per_gpu=1):
    """Return the validation batch size to use for the current device.

    Static heuristic. MUST be deterministic across calls — fit() /
    fit_generator() call this per-epoch, and torch.compile caches
    specializations by input shape. A validation batch that varies
    with live free-VRAM (the auto-sized approach) forces the
    compiled graph to re-codegen every epoch and with 16 training
    workers × 32 inductor compile workers on a 128-vCPU box pins the
    CPU at hundreds of concurrent compile jobs — observed to stall
    training indefinitely. The auto-sized prediction batch size in
    ``compute_prediction_batch_size`` is fine for mhcflurry-predict
    where each call is a single forward; training-time validation is
    not that shape.

    ``model`` and ``num_workers_per_gpu`` kwargs are retained for API
    compatibility with the call sites but are intentionally unused.
    """
    del model, num_workers_per_gpu
    if configured_batch_size:
        return int(configured_batch_size)
    if device.type == "cuda":
        # Validation is forward-only and the networks are tiny relative
        # to modern GPU memory. A much larger default batch dramatically
        # cuts kernel-launch / Python-loop overhead versus 4 *
        # minibatch_size, while staying deterministic across epochs.
        return max(4 * minibatch_size, 4096)
    return 4 * minibatch_size


def _move_fit_batch_to_device(batch, device, *, non_blocking):
    """Move a fit()/fit_generator batch dict to ``device``.

    When ``batch["peptide"]`` is a 2D integer array the Phase 2 (#268)
    on-device BLOSUM62 path is active — keep it as int so the network's
    embedding lookup can consume it. A 3D integer array is the Phase 4a
    (#268) int8 BLOSUM-encoded cache payload; still widen that to fp32
    on device. Float arrays flow through as before.
    """
    peptide_source = batch["peptide"]
    peptide_repeat_count = batch.get("peptide_repeat_count")
    if isinstance(peptide_source, numpy.ndarray):
        _peptide_is_indices = (
            peptide_source.ndim == 2
            and numpy.issubdtype(peptide_source.dtype, numpy.integer)
        )
    else:
        _peptide_is_indices = (
            hasattr(peptide_source, "dim")
            and peptide_source.dim() == 2
            and peptide_source.dtype in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
            )
        )
    peptide_batch = _batch_value_to_device(
        peptide_source,
        device,
        non_blocking=non_blocking,
        cast_float=not _peptide_is_indices,
    )
    y_batch = _batch_value_to_device(
        batch["y"],
        device,
        non_blocking=non_blocking,
        cast_float=False,
    )
    inputs = {"peptide": peptide_batch}
    if peptide_repeat_count is not None:
        inputs["peptide_repeat_count"] = int(peptide_repeat_count)
    if "allele" in batch:
        inputs["allele"] = _batch_value_to_device(
            batch["allele"],
            device,
            non_blocking=non_blocking,
            cast_float=True,
        )
    weights_batch = None
    if "weight" in batch:
        weights_batch = _batch_value_to_device(
            batch["weight"],
            device,
            non_blocking=non_blocking,
            cast_float=True,
        )
    return (inputs, y_batch, weights_batch)


def _iterate_tensor_slice_batches(
    *,
    peptide_device,
    allele_device,
    y_device,
    weights_device,
    batch_size,
    shuffle_permutation,
    drop_last=True,
):
    """**PARKED** — primitive for issue openvax/mhcflurry#268 Phase 4.

    Intentionally NOT used by production training at head. The in-
    ``fit()`` DataLoader replacement that would call this helper was
    deferred during the #270 review: after Phase 1 (negative pool) +
    #272 auto-batching landed, most of the Python-dispatch savings
    this primitive would deliver are already captured upstream, so
    the implementation risk of rewiring ``fit()``'s inner loop
    (interacting with torch.compile shape stability, val-tensor
    hoist, tail-batch handling, training OOM guard) wasn't worth
    the marginal remaining win.

    Preserved here, with its own test coverage, so that if the
    broader hyperparameter sweep in #271 exposes a regime where
    the remaining per-step overhead matters, the integration diff
    is small (caller-side only, no helper-API churn).

    Yields ``(inputs, y_batch, weights_batch)`` tuples by index-
    slicing pre-placed device tensors under a caller-supplied
    permutation. Zero copy, no collate, no gather. When
    ``drop_last=True`` (the default) the tail that doesn't fill a
    batch is discarded — the caller handles it eagerly, matching
    the current DataLoader path where the tail triggers an eager-
    network forward.

    This helper does NOT run the forward/backward/optimizer step;
    any integrator pairs it with ``_run_training_batch`` so loss /
    regularization / logging logic stays single-sourced with the
    DataLoader path.
    """
    n = int(shuffle_permutation.shape[0])
    full_batches = n // batch_size
    for step in range(full_batches):
        idx = shuffle_permutation[step * batch_size : (step + 1) * batch_size]
        inputs = {"peptide": peptide_device.index_select(0, idx)}
        if allele_device is not None:
            inputs["allele"] = allele_device.index_select(0, idx)
        y_batch = y_device.index_select(0, idx)
        weights_batch = None
        if weights_device is not None:
            weights_batch = weights_device.index_select(0, idx)
        yield inputs, y_batch, weights_batch
    if not drop_last and n % batch_size != 0:
        tail_idx = shuffle_permutation[full_batches * batch_size :]
        inputs = {"peptide": peptide_device.index_select(0, tail_idx)}
        if allele_device is not None:
            inputs["allele"] = allele_device.index_select(0, tail_idx)
        y_batch = y_device.index_select(0, tail_idx)
        weights_batch = None
        if weights_device is not None:
            weights_batch = weights_device.index_select(0, tail_idx)
        yield inputs, y_batch, weights_batch


def _run_training_batch(
    *,
    network,
    optimizer,
    loss_obj,
    regularization_parameters,
    l1_reg,
    l2_reg,
    inputs,
    y_batch,
    weights_batch=None,
):
    """Run one optimizer step and return the detached loss tensor."""
    optimizer.zero_grad()
    predictions = network(inputs)
    loss = loss_obj(predictions, y_batch, sample_weights=weights_batch)
    regularization_penalty = Class1NeuralNetwork._regularization_penalty(
        regularization_parameters,
        l1=l1_reg,
        l2=l2_reg,
    )
    if regularization_penalty is not None:
        loss = loss + regularization_penalty
    loss.backward()
    optimizer.step()
    return loss.detach()


def _build_epoch_input_arrays(
    random_negative_peptides_encoding,
    x_dict_without_random_negatives,
    *,
    random_negatives_allele_encoding,
    allele_encoding_to_input,
):
    """Build the per-epoch ``(x_peptide, x_allele)`` arrays for fit().

    Extracted from fit()'s epoch loop so that the previous epoch's
    arrays go out of scope naturally on the caller's reassignment of
    x_peptide / x_allele — no explicit ``del`` or ``= None`` needed
    before the concat to keep peak RAM at 1× rather than 2×. For a
    1.85M-row × 45-wide BLOSUM62 training set that's an extra ~7 GB
    briefly held per worker per epoch; on A100-40GB it's the
    difference between ``MAX_WORKERS_PER_GPU=1`` and 2.

    When there are no random negatives, returns the un-concatenated
    training arrays directly (no copy). When there are random
    negatives but no allele encoding, returns x_allele=None.
    """
    no_random_negs = random_negatives_allele_encoding is None and len(
        random_negative_peptides_encoding
    ) == 0
    if no_random_negs:
        return (
            x_dict_without_random_negatives["peptide"],
            x_dict_without_random_negatives.get("allele"),
        )

    x_peptide = numpy.concatenate([
        random_negative_peptides_encoding,
        x_dict_without_random_negatives["peptide"],
    ])
    if "allele" in x_dict_without_random_negatives:
        x_allele = numpy.concatenate([
            allele_encoding_to_input(random_negatives_allele_encoding)[0],
            x_dict_without_random_negatives["allele"],
        ])
    else:
        x_allele = None
    return x_peptide, x_allele


def _effective_num_workers(num_workers):
    """Downgrade ``num_workers`` to 0 when running inside a daemon process.

    Safety net for DataLoader's ``num_workers>0`` path, which requires
    spawning child processes — forbidden from daemon parents. mhcflurry's
    training orchestrator switched to ``NonDaemonPool`` (see
    ``local_parallelism.py``) so in production Pool workers are now
    non-daemonic and this downgrade does NOT fire. It remains as
    defense-in-depth for:

      - External callers that wrap ``Class1NeuralNetwork.fit`` in a
        daemonic process of their own (e.g. ``ProcessPoolExecutor`` is
        non-daemon by default but ``multiprocessing.Pool`` without the
        NonDaemonPool subclass is daemonic).
      - Test suites that explicitly construct daemon subprocesses (our
        own ``test_dataloader_num_workers_downgrades_in_daemon_context``
        exercises this path).

    Emits a ``DeprecationWarning`` — it's worth knowing if this fires,
    since it means the caller is running in a daemon context and
    DataLoader prefetch is silently disabled. Use ``warnings.filterwarnings``
    to silence if intentional.
    """
    if num_workers > 0 and multiprocessing.current_process().daemon:
        import warnings
        warnings.warn(
            f"dataloader_num_workers={num_workers} requested from a daemon "
            f"process; downgrading to 0 because daemon processes cannot "
            f"spawn children. If this is mhcflurry's Pool worker, switch "
            f"to NonDaemonPool (mhcflurry/local_parallelism.py); if it's "
            f"your own caller, use a non-daemonic executor or accept the "
            f"downgrade (set dataloader_num_workers=0 explicitly to silence).",
            DeprecationWarning,
            stacklevel=3,
        )
        return 0
    return num_workers


def _make_fit_dataloader(
    dataset,
    batch_size,
    num_workers,
    use_pinned_memory,
    drop_last=False,
):
    """Construct a DataLoader for fit()'s inner batch loop.

    ``num_workers=0`` path runs everything in the main process.
    ``num_workers>0`` spawns worker processes (via 'spawn' to avoid
    CUDA-context-fork hazards) that prefetch numpy minibatches, overlapping
    CPU slicing with GPU compute while avoiding PyTorch CPU tensor allocator
    growth in the workers.

    ``drop_last=True`` makes every yielded batch exactly ``batch_size``
    rows. This is what torch.compile (Phase 4c scope) and bf16 autocast
    want: one shape means one compiled kernel. The cost is that the
    final ``n_train % batch_size`` rows are not seen in a given epoch,
    but because fit() reshuffles ``train_indices`` each epoch, every row
    eventually cycles through. For default minibatch_size=512 and a
    1.85M-row training set that's ~0.028% rows dropped per epoch —
    negligible relative to stochastic-gradient noise. See Phase 4b of
    openvax/mhcflurry#268.

    Caller is responsible for ensuring ``len(dataset) >= batch_size``
    when setting ``drop_last=True``; otherwise the DataLoader yields
    zero batches.

    If called from a daemon process (i.e. inside a
    ``multiprocessing.Pool`` worker without the NonDaemonPool override),
    ``num_workers`` is forced to 0 and ``pin_memory`` to False because
    daemon processes cannot spawn their own children. See
    ``_effective_num_workers``.

    Note: PyTorch's ``persistent_workers`` flag isn't exposed here
    because ``fit()`` rebuilds the DataLoader per epoch (the x_peptide
    arrays change with each epoch's random negatives) — persistent
    workers have no DataLoader instance to persist across. If a future
    refactor hoists DataLoader construction out of the epoch loop, add
    the flag back then.
    """
    effective_workers = _effective_num_workers(num_workers)
    # Keep worker-produced batches as numpy arrays on every platform. PyTorch's
    # default worker collate allocates fresh CPU tensors (and, with
    # ``pin_memory=True``, pinned staging buffers) for every batch; on the
    # pan-allele release job those allocator caches were the dominant anonymous
    # RSS growth. Parent-side numpy->torch conversion is a little less fancy
    # than pinned prefetch, but it has bounded host memory and still overlaps
    # worker numpy slicing with GPU compute.
    effective_pin = False
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset already reflects shuffled train_indices.
        num_workers=effective_workers,
        pin_memory=effective_pin,
        drop_last=drop_last,
    )
    # Always use the numpy collate path. PyTorch's default_collate calls
    # ``torch.tensor(numpy_array)`` for every batch, allocating new CPU
    # tensors through ``c10::CPUAllocator`` on every step. That allocator
    # caches in libc malloc and on a 16-worker pan-allele run accumulates
    # ~16 GB per fit() per worker over a 30-epoch run, which is a major
    # contributor to the OOM tracked in openvax/mhcflurry#270. Returning
    # numpy arrays keeps the H2D conversion in ``_move_fit_batch_to_device``
    # where ``torch.from_numpy`` shares the buffer (no extra alloc) and the
    # ``.to(device)`` copy releases the host side immediately afterward.
    kwargs["collate_fn"] = _numpy_batch_collate
    if effective_workers > 0:
        # CUDA contexts aren't fork-safe. Using 'spawn' avoids copying the
        # training process's CUDA state into workers, at a ~200-500 ms
        # per-worker startup cost. Safe default.
        kwargs["multiprocessing_context"] = "spawn"
        # prefetch_factor is only valid when num_workers > 0.
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(**kwargs)


def _dataset_backing_nbytes(dataset):
    """Approximate bytes that spawn-based DataLoader workers would pickle."""
    total = 0
    seen = set()
    for value in (
        dataset.x_peptide,
        dataset.x_allele,
        dataset.random_negative_x_peptide,
        dataset.random_negative_x_allele,
        dataset.y_encoded,
        dataset.sample_weights,
        dataset.train_indices,
    ):
        if value is None:
            continue
        ident = id(value)
        if ident in seen:
            continue
        seen.add(ident)
        total += int(getattr(value, "nbytes", 0))
    return total


def _effective_fit_dataloader_num_workers(num_workers, dataset):
    """Avoid spawn-pickling huge fit() arrays into DataLoader workers."""
    global _FIT_DATALOADER_DOWNGRADE_WARNED
    if num_workers <= 0:
        return int(num_workers), None
    if os.environ.get("MHCFLURRY_FORCE_FIT_DATALOADER_WORKERS", "0") == "1":
        return int(num_workers), None

    backing_bytes = _dataset_backing_nbytes(dataset)
    if backing_bytes <= _FIT_DATALOADER_SPAWN_COPY_LIMIT_BYTES:
        return int(num_workers), None

    reason = (
        "fit() DataLoader worker prefetch disabled: requested "
        "dataloader_num_workers=%d, but the dataset backing arrays are %.1f MB "
        "and spawn workers would pickle a copy into each worker every epoch "
        "(limit %.1f MB). Set MHCFLURRY_FORCE_FIT_DATALOADER_WORKERS=1 to "
        "force the old behavior."
        % (
            num_workers,
            backing_bytes / (1024 * 1024),
            _FIT_DATALOADER_SPAWN_COPY_LIMIT_BYTES / (1024 * 1024),
        )
    )
    if not _FIT_DATALOADER_DOWNGRADE_WARNED:
        logging.warning(reason)
        _FIT_DATALOADER_DOWNGRADE_WARNED = True
    return 0, reason


def _make_fit_generator_dataloader(
    dataset,
    num_workers,
):
    """Construct a DataLoader for ``fit_generator``.

    The dataset already yields complete batches, so automatic batching is
    disabled (``batch_size=None``) and we keep the payload as numpy
    arrays via ``_identity_collate``. That avoids worker-side tensor
    materialization and keeps the worker path picklable for spawned
    subprocesses.
    """
    effective_workers = _effective_num_workers(num_workers)
    kwargs = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=effective_workers,
        collate_fn=_identity_collate,
    )
    if effective_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(**kwargs)


def _batched_validation_loss(
    *,
    network,
    eager_network,
    val_peptide,
    val_allele,
    val_y,
    val_weights,
    loss_obj,
    batch_size,
):
    """Compute validation loss with a compiled fast path and eager tail path.

    Every ``network(inputs)`` call inside this helper sees the same
    shape (``batch_size``) so torch.compile / CUDA graphs / bf16
    autocast can specialize once and reuse. That matters for H100 and
    A100 training where single-shot forward passes over the full
    validation set (up to ~185K rows for pan-allele training) were
    eating 15+ GB of VRAM per worker and defeating any shape-stable
    optimization.

    Semantics: iterate ``val_peptide`` in ``batch_size``-row chunks and
    return the exact mean validation loss across all rows. Full-size
    batches run through ``network`` (typically compiled); any short tail
    batch runs through ``eager_network`` to avoid recompiling on an odd
    shape.

    Weighted validation and cross-example losses fall back to the
    original single-shot computation to preserve exact semantics:
    averaging per-batch weighted means is not the same as taking the
    weighted mean over the whole validation set, and losses like
    ``MultiallelicMassSpecLoss`` couple samples within a batch.

    Returns a Python float (already detached from autograd).
    """
    n_val = val_peptide.shape[0]
    if n_val == 0:
        return float("nan")

    if (
        n_val < batch_size
        or val_weights is not None
        or not getattr(loss_obj, "supports_independent_samples", True)
    ):
        # Fallback for tiny val sets (typical in unit tests) and for
        # weighted / cross-example losses whose reductions are not
        # decomposable batchwise.
        inputs = {"peptide": val_peptide}
        if val_allele is not None:
            inputs["allele"] = val_allele
        preds = eager_network(inputs)
        batch_loss = loss_obj(preds, val_y, sample_weights=val_weights)
        return batch_loss.item()

    n_full = (n_val // batch_size) * batch_size
    loss_accum = 0.0
    weight_accum = 0.0
    for start in range(0, n_full, batch_size):
        end = start + batch_size
        inputs = {"peptide": val_peptide[start:end]}
        if val_allele is not None:
            inputs["allele"] = val_allele[start:end]
        preds = network(inputs)
        weights_slice = (
            val_weights[start:end] if val_weights is not None else None
        )
        batch_loss = loss_obj(
            preds, val_y[start:end], sample_weights=weights_slice
        )
        loss_accum += batch_loss.item() * batch_size
        weight_accum += batch_size
    if n_full < n_val:
        inputs = {"peptide": val_peptide[n_full:n_val]}
        if val_allele is not None:
            inputs["allele"] = val_allele[n_full:n_val]
        tail_loss = loss_obj(
            eager_network(inputs),
            val_y[n_full:n_val],
            sample_weights=None,
        )
        tail_size = n_val - n_full
        loss_accum += tail_loss.item() * tail_size
        weight_accum += tail_size
    return loss_accum / weight_accum


class Class1NeuralNetworkModel(nn.Module):
    """
    PyTorch module for Class1 neural network.
    """

    def __init__(
            self,
            peptide_encoding_shape,
            allele_representations=None,
            locally_connected_layers=None,
            peptide_dense_layer_sizes=None,
            allele_dense_layer_sizes=None,
            layer_sizes=None,
            peptide_allele_merge_method="multiply",
            peptide_allele_merge_activation="",
            activation="tanh",
            output_activation="sigmoid",
            dropout_probability=0.0,
            batch_normalization=False,
            dense_layer_l1_regularization=0.001,
            dense_layer_l2_regularization=0.0,
            topology="feedforward",
            num_outputs=1,
            init="glorot_uniform",
            peptide_input_is_indices=False):
        super(Class1NeuralNetworkModel, self).__init__()

        self.peptide_encoding_shape = peptide_encoding_shape
        self.peptide_input_is_indices = peptide_input_is_indices
        # Phase 2 (#268) on-device BLOSUM62: when enabled, peptide input is
        # (N, L) int indices and the first step of ``forward`` widens to
        # (N, L, 21) fp32 via embedding. Ship the table as a registered
        # buffer so it moves with ``.to(device)`` and is serialized by
        # ``state_dict`` like a normal non-learnable tensor.
        if peptide_input_is_indices:
            from .amino_acid import BLOSUM62_MATRIX
            self.register_buffer(
                "blosum62_table",
                torch.from_numpy(
                    BLOSUM62_MATRIX.to_numpy().astype(numpy.float32)
                ),
                persistent=False,
            )
        self.has_allele = allele_representations is not None
        self.peptide_allele_merge_method = peptide_allele_merge_method
        self.peptide_allele_merge_activation = peptide_allele_merge_activation
        self.dropout_probability = dropout_probability
        self.topology = topology
        self.num_outputs = num_outputs
        self.activation_name = activation
        self.output_activation_name = output_activation

        if locally_connected_layers is None:
            locally_connected_layers = []
        if peptide_dense_layer_sizes is None:
            peptide_dense_layer_sizes = []
        if allele_dense_layer_sizes is None:
            allele_dense_layer_sizes = []
        if layer_sizes is None:
            layer_sizes = [32]

        # Build locally connected layers
        self.lc_layers = nn.ModuleList()
        input_length = peptide_encoding_shape[0]
        in_channels = peptide_encoding_shape[1]

        for i, lc_params in enumerate(locally_connected_layers):
            filters = lc_params.get('filters', 8)
            kernel_size = lc_params.get('kernel_size', 3)
            lc_activation = lc_params.get('activation', 'tanh')

            lc_layer = LocallyConnected1D(
                in_channels=in_channels,
                out_channels=filters,
                input_length=input_length,
                kernel_size=kernel_size,
                activation=lc_activation
            )
            self.lc_layers.append(lc_layer)
            in_channels = filters
            input_length = lc_layer.output_length

        # Flattened size after locally connected layers
        self.flatten_size = input_length * in_channels

        # Peptide dense layers
        self.peptide_dense_layers = nn.ModuleList()
        peptide_layer_input = self.flatten_size
        for i, size in enumerate(peptide_dense_layer_sizes):
            layer = nn.Linear(peptide_layer_input, size)
            self.peptide_dense_layers.append(layer)
            peptide_layer_input = size

        # Batch normalization after peptide processing (early)
        self.batch_norm_early = None
        if batch_normalization:
            self.batch_norm_early = nn.BatchNorm1d(
                peptide_layer_input,
                eps=KERAS_BATCH_NORM_EPSILON,
                momentum=KERAS_BATCH_NORM_MOMENTUM,
            )

        # Allele embedding and processing
        self.allele_embedding = None
        self.allele_dense_layers = nn.ModuleList()
        allele_output_size = 0

        if self.has_allele:
            num_alleles = allele_representations.shape[0]
            embedding_dim = numpy.prod(allele_representations.shape[1:])

            self.allele_embedding = nn.Embedding(
                num_embeddings=num_alleles,
                embedding_dim=embedding_dim
            )
            # Set embedding weights and freeze
            self.allele_embedding.weight.data = torch.from_numpy(
                allele_representations.reshape(num_alleles, -1).astype(numpy.float32)
            )
            self.allele_embedding.weight.requires_grad = False

            allele_layer_input = embedding_dim
            for i, size in enumerate(allele_dense_layer_sizes):
                layer = nn.Linear(allele_layer_input, size)
                self.allele_dense_layers.append(layer)
                allele_layer_input = size
            allele_output_size = allele_layer_input

        # Compute merged size
        if self.has_allele:
            if peptide_allele_merge_method == "concatenate":
                merged_size = peptide_layer_input + allele_output_size
            elif peptide_allele_merge_method == "multiply":
                # Both must have the same size for multiply
                merged_size = peptide_layer_input
            else:
                raise ValueError(f"Unknown merge method: {peptide_allele_merge_method}")
        else:
            merged_size = peptide_layer_input

        # Merge activation
        self.merge_activation = get_activation(peptide_allele_merge_activation)

        # Main dense layers
        self.dense_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # For DenseNet topology, track input sizes for skip connections
        self.merged_size = merged_size
        current_size = merged_size
        prev_sizes = []  # Track previous layer output sizes for skip connections

        for i, size in enumerate(layer_sizes):
            # For DenseNet topology (with-skip-connections):
            # - Layer 0: input = merged_size
            # - Layer 1: input = merged_size + layer_sizes[0] (skip from input)
            # - Layer 2+: input = layer_sizes[i-2] + layer_sizes[i-1] (skip from 2 layers back)
            if topology == "with-skip-connections" and i > 0:
                if i == 1:
                    # Skip from original merged input
                    current_size = merged_size + prev_sizes[-1]
                else:
                    # Skip from 2 layers back
                    current_size = prev_sizes[-2] + prev_sizes[-1]

            layer = nn.Linear(current_size, size)
            self.dense_layers.append(layer)

            if batch_normalization:
                self.batch_norms.append(nn.BatchNorm1d(
                    size,
                    eps=KERAS_BATCH_NORM_EPSILON,
                    momentum=KERAS_BATCH_NORM_MOMENTUM,
                ))
            else:
                self.batch_norms.append(None)

            if dropout_probability > 0:
                # Dropout probability in MHCflurry hyperparameters is keep-probability.
                drop_prob = max(0.0, 1.0 - dropout_probability)
                if drop_prob > 0:
                    self.dropouts.append(nn.Dropout(p=drop_prob))
                else:
                    self.dropouts.append(None)
            else:
                self.dropouts.append(None)

            prev_sizes.append(size)
            current_size = size

        # Note: For DenseNet topology, output layer receives only the last hidden layer output
        # (skip connections are only between hidden layers, not to the output layer)

        # Output layer
        self.output_layer = nn.Linear(current_size, num_outputs)

        # Activation functions
        self.activation = get_activation(activation)
        self.output_activation = get_activation(output_activation)

        # Initialize weights
        self._initialize_weights(init)

    def _initialize_weights(self, init):
        """Initialize layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init == "glorot_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init == "glorot_normal":
                    nn.init.xavier_normal_(module.weight)
                elif init == "he_uniform":
                    nn.init.kaiming_uniform_(module.weight)
                elif init == "he_normal":
                    nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward_peptide_stage_before_early_batch_norm(self, peptide):
        if (
            self.peptide_input_is_indices
            and peptide.dim() == 2
        ):
            peptide = torch.nn.functional.embedding(
                peptide.long(), self.blosum62_table
            )
        x = peptide
        for lc_layer in self.lc_layers:
            x = lc_layer(x)
        x = x.reshape(x.size(0), -1)
        for layer in self.peptide_dense_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        return x

    def forward_peptide_stage(self, peptide):
        """Run only the peptide-side of the network.

        Ends at the point where the allele information enters — just
        before the merge step. Used by the calibration fast path
        (#272) to compute the peptide-dependent activations once and
        reuse them across thousands of alleles.

        Parameters
        ----------
        peptide : torch.Tensor
            (N, L, 21) fp32 BLOSUM62-encoded, or (N, L) int indices
            when ``peptide_input_is_indices`` is True.

        Returns
        -------
        torch.Tensor of shape (N, peptide_representation_dim) — the
        input the ``forward_from_peptide_stage`` fast path expects.
        """
        x = self._forward_peptide_stage_before_early_batch_norm(peptide)
        if self.batch_norm_early is not None:
            x = self.batch_norm_early(x)
        return x

    def _forward_allele_stage(self, allele_idx):
        allele_idx = allele_idx.long()
        if allele_idx.dim() > 1:
            allele_idx = allele_idx.squeeze(-1)
        allele_embed = self.allele_embedding(allele_idx)
        for layer in self.allele_dense_layers:
            allele_embed = layer(allele_embed)
            if self.activation is not None:
                allele_embed = self.activation(allele_embed)
        return allele_embed.reshape(allele_embed.size(0), -1)

    def forward_from_peptide_stage(self, peptide_stage, allele_idx):
        """Run the allele-merge + main dense path from cached peptide reps.

        ``peptide_stage`` must be the output of
        ``forward_peptide_stage``. ``allele_idx`` has shape matching
        ``peptide_stage``'s batch dim — typical calibration usage
        tiles one allele across many peptides or the cross-product
        of (peptide_chunk, allele_chunk).

        This path skips all peptide-side ops, which for pan-allele
        calibration lets a single precomputed activation be reused
        across tens of thousands of allele forwards.
        """
        if not self.has_allele:
            raise RuntimeError(
                "forward_from_peptide_stage called on a has_allele=False "
                "model — just call forward() directly"
            )
        x = peptide_stage
        allele_embed = self._forward_allele_stage(allele_idx)
        if self.peptide_allele_merge_method == "concatenate":
            x = torch.cat([x, allele_embed], dim=-1)
        elif self.peptide_allele_merge_method == "multiply":
            x = x * allele_embed
        if self.merge_activation is not None:
            x = self.merge_activation(x)
        prev_outputs = []
        merged_input = x
        for i, layer in enumerate(self.dense_layers):
            if self.topology == "with-skip-connections" and i > 0:
                if i == 1:
                    x = torch.cat([merged_input, prev_outputs[-1]], dim=-1)
                else:
                    x = torch.cat([prev_outputs[-2], prev_outputs[-1]], dim=-1)
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)
            prev_outputs.append(x)
        output = self.output_layer(x)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

    def _first_linear_from_cartesian_stages(self, peptide_stage, allele_stage, layer):
        if self.peptide_allele_merge_method == "concatenate":
            peptide_width = peptide_stage.shape[-1]
            peptide_weight = layer.weight[:, :peptide_width]
            allele_weight = layer.weight[:, peptide_width:]
            x = (
                peptide_stage.matmul(peptide_weight.t()).unsqueeze(1)
                + allele_stage.matmul(allele_weight.t()).unsqueeze(0)
            )
        elif self.peptide_allele_merge_method == "multiply":
            x = torch.einsum(
                "pd,ad,hd->pah",
                peptide_stage,
                allele_stage,
                layer.weight,
            )
        else:
            raise ValueError(
                f"Unknown merge method: {self.peptide_allele_merge_method}"
            )
        return x + layer.bias

    def _forward_compact_cartesian(self, peptide, allele_idx, repeat_count):
        """Forward a compact peptide × allele batch without raw peptide repeat."""
        if not self.has_allele:
            raise RuntimeError(
                "compact peptide-repeat batches require a pan-allele model"
            )
        if repeat_count <= 0:
            raise ValueError("peptide_repeat_count must be positive")

        allele_idx = allele_idx.long()
        if allele_idx.dim() > 1:
            allele_idx = allele_idx.squeeze(-1)
        peptide_count = peptide.shape[0]
        allele_count = allele_idx.shape[0]
        if (
            isinstance(peptide_count, int)
            and isinstance(allele_count, int)
            and allele_count != peptide_count * repeat_count
        ):
            raise ValueError(
                "compact peptide-repeat batch has %d peptides, repeat_count=%d, "
                "and %d allele rows"
                % (peptide_count, repeat_count, allele_count)
            )

        peptide_stage = self._forward_peptide_stage_before_early_batch_norm(peptide)
        if self.batch_norm_early is not None:
            # BatchNorm's running variance depends on the batch size. Repeat
            # before early BN so compact training is numerically equivalent to
            # the historical fully repeated peptide batch.
            peptide_stage = peptide_stage.repeat_interleave(repeat_count, dim=0)
            peptide_stage = self.batch_norm_early(peptide_stage)
            peptide_stage = peptide_stage.reshape(
                peptide_count, repeat_count, peptide_stage.shape[-1]
            )[:, 0, :]

        allele_stage = self._forward_allele_stage(allele_idx[:repeat_count])
        can_factorize_first_layer = (
            self.merge_activation is None
            and self.topology != "with-skip-connections"
        )
        if not can_factorize_first_layer:
            peptide_stage = peptide_stage.repeat_interleave(repeat_count, dim=0)
            return self.forward_from_peptide_stage(peptide_stage, allele_idx)

        if self.dense_layers:
            first_layer = self.dense_layers[0]
        else:
            first_layer = self.output_layer
        x = self._first_linear_from_cartesian_stages(
            peptide_stage,
            allele_stage,
            first_layer,
        )
        x = x.reshape(peptide_count * repeat_count, x.shape[-1])

        if self.dense_layers:
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[0] is not None:
                x = self.batch_norms[0](x)
            if self.dropouts[0] is not None:
                x = self.dropouts[0](x)

            # ``enumerate(seq, start=N)`` triggers a torch._dynamo graph
            # break on PyTorch <=2.4 (``call_enumerate`` rejects the
            # ``start`` kwarg in builtin.py:775) — every traced forward
            # then falls back to eager for the layer-stack loop, losing
            # the compile speedup. Iterate by index instead so the loop
            # body stays inside the compiled graph. Issue #270 perf note.
            for offset, layer in enumerate(self.dense_layers[1:]):
                i = offset + 1
                x = layer(x)
                if self.activation is not None:
                    x = self.activation(x)
                if self.batch_norms[i] is not None:
                    x = self.batch_norms[i](x)
                if self.dropouts[i] is not None:
                    x = self.dropouts[i](x)
            output = self.output_layer(x)
        else:
            output = x

        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

    def forward_cartesian_from_peptide_stage(self, peptide_stage, allele_idx):
        """Forward every peptide-stage row against every allele index.

        Returns predictions in allele-major order with shape
        ``(num_alleles, num_peptides, num_outputs)``. When the model's
        merge + first linear layer can be factored, this avoids materializing
        the larger ``num_alleles * num_peptides * peptide_stage_dim`` repeated
        peptide-stage tensor used by ``forward_from_peptide_stage``.
        """
        if not self.has_allele:
            raise RuntimeError(
                "forward_cartesian_from_peptide_stage called on a "
                "has_allele=False model"
            )
        allele_idx = allele_idx.long()
        if allele_idx.dim() > 1:
            allele_idx = allele_idx.squeeze(-1)
        num_peptides = peptide_stage.shape[0]
        num_alleles = allele_idx.shape[0]

        can_factorize_first_layer = (
            self.merge_activation is None
            and self.topology != "with-skip-connections"
        )
        if not can_factorize_first_layer:
            peptide_width = peptide_stage.shape[-1]
            expanded = peptide_stage.unsqueeze(0).expand(
                num_alleles, num_peptides, peptide_width
            ).reshape(num_alleles * num_peptides, peptide_width)
            expanded_alleles = allele_idx.unsqueeze(1).expand(
                num_alleles, num_peptides
            ).reshape(-1)
            return self.forward_from_peptide_stage(
                expanded,
                expanded_alleles,
            ).reshape(num_alleles, num_peptides, -1)

        allele_stage = self._forward_allele_stage(allele_idx)
        if self.dense_layers:
            first_layer = self.dense_layers[0]
        else:
            first_layer = self.output_layer
        x = self._first_linear_from_cartesian_stages(
            peptide_stage,
            allele_stage,
            first_layer,
        )
        x = x.transpose(0, 1).reshape(num_alleles * num_peptides, x.shape[-1])

        if self.dense_layers:
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[0] is not None:
                x = self.batch_norms[0](x)
            if self.dropouts[0] is not None:
                x = self.dropouts[0](x)

            # ``enumerate(seq, start=N)`` triggers a torch._dynamo graph
            # break on PyTorch <=2.4 (``call_enumerate`` rejects the
            # ``start`` kwarg in builtin.py:775) — every traced forward
            # then falls back to eager for the layer-stack loop, losing
            # the compile speedup. Iterate by index instead so the loop
            # body stays inside the compiled graph. Issue #270 perf note.
            for offset, layer in enumerate(self.dense_layers[1:]):
                i = offset + 1
                x = layer(x)
                if self.activation is not None:
                    x = self.activation(x)
                if self.batch_norms[i] is not None:
                    x = self.batch_norms[i](x)
                if self.dropouts[i] is not None:
                    x = self.dropouts[i](x)
            output = self.output_layer(x)
        else:
            output = x

        if self.output_activation is not None:
            output = self.output_activation(output)
        return output.reshape(num_alleles, num_peptides, -1)

    def forward(self, inputs):
        """
        Forward pass.

        Parameters
        ----------
        inputs : dict
            Dictionary with 'peptide' and optionally 'allele' keys

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, num_outputs)
        """
        if self.has_allele != ('allele' in inputs):
            raise ValueError(
                "Class1NeuralNetworkModel input mismatch: "
                f"network has_allele={self.has_allele} but "
                f"received allele key={('allele' in inputs)}"
            )

        peptide = inputs['peptide']
        peptide_repeat_count = inputs.get("peptide_repeat_count")
        if peptide_repeat_count is not None:
            if "allele" not in inputs:
                raise ValueError(
                    "compact peptide-repeat batch requires an allele input"
                )
            return self._forward_compact_cartesian(
                peptide,
                inputs["allele"],
                int(peptide_repeat_count),
            )

        # Phase 2 (#268) on-device BLOSUM62: (N, L) int indices → (N, L, 21)
        # fp32. Skipped when the input is already the BLOSUM-encoded 3D
        # tensor. ``torch.nn.functional.embedding`` is a pure gather so
        # the op cost is comparable to the int8→fp32 widening cast it
        # replaces; the saving is 21× less H2D and 21× less cache size.
        if self.peptide_input_is_indices and peptide.dim() == 2:
            peptide = torch.nn.functional.embedding(
                peptide.long(), self.blosum62_table
            )

        # Locally connected layers
        x = peptide
        for lc_layer in self.lc_layers:
            x = lc_layer(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Peptide dense layers
        for layer in self.peptide_dense_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)

        # Early batch normalization
        if self.batch_norm_early is not None:
            x = self.batch_norm_early(x)

        # Allele processing and merge
        if self.has_allele:
            allele_idx = inputs['allele'].long()
            # Handle case where input might be (batch,) or (batch, 1)
            if allele_idx.dim() > 1:
                allele_idx = allele_idx.squeeze(-1)
            allele_embed = self.allele_embedding(allele_idx)

            # Allele dense layers
            for layer in self.allele_dense_layers:
                allele_embed = layer(allele_embed)
                if self.activation is not None:
                    allele_embed = self.activation(allele_embed)

            # Flatten allele embedding
            allele_embed = allele_embed.reshape(allele_embed.size(0), -1)

            # Merge
            if self.peptide_allele_merge_method == "concatenate":
                x = torch.cat([x, allele_embed], dim=-1)
            elif self.peptide_allele_merge_method == "multiply":
                x = x * allele_embed

            # Merge activation
            if self.merge_activation is not None:
                x = self.merge_activation(x)

        # Main dense layers (with optional skip connections for DenseNet topology)
        prev_outputs = []  # Track outputs for skip connections
        merged_input = x  # Save for DenseNet skip connections

        for i, layer in enumerate(self.dense_layers):
            # For DenseNet topology, concatenate skip connections
            if self.topology == "with-skip-connections" and i > 0:
                if i == 1:
                    # Skip from original merged input
                    x = torch.cat([merged_input, prev_outputs[-1]], dim=-1)
                else:
                    # Skip from 2 layers back
                    x = torch.cat([prev_outputs[-2], prev_outputs[-1]], dim=-1)

            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
            if self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)

            prev_outputs.append(x)

        # Note: For DenseNet topology, output layer receives only the last hidden layer output
        # (skip connections are only between hidden layers, not to the output layer)

        # Output
        output = self.output_layer(x)
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output

    def get_weights_list(self):
        """
        Get weights as a list of numpy arrays (for compatibility with NPZ format).

        Returns
        -------
        list of numpy.ndarray
        """
        weights = []
        for name, param in self.named_parameters():
            weights.append(param.detach().cpu().numpy())
        # Also include buffers (running mean/var for batch norm)
        for name, buffer in self.named_buffers():
            weights.append(buffer.detach().cpu().numpy())
        return weights

    def set_weights_list(self, weights, auto_convert_keras=True):
        """
        Set weights from a list of numpy arrays.

        Supports automatic detection and conversion of Keras-format weights
        to PyTorch format for backward compatibility with pre-trained models.

        Parameters
        ----------
        weights : list of numpy.ndarray
        auto_convert_keras : bool
            If True, automatically detect and convert Keras-format weights
        """
        if auto_convert_keras and getattr(self, "_keras_config", None):
            keras_layers = self._keras_config.get("config", {}).get("layers", [])
            idx = 0

            def assign_dense(layer, w, b):
                w = w.astype(numpy.float32)
                b = b.astype(numpy.float32)
                if w.shape == layer.weight.shape[::-1] or (
                    w.shape == layer.weight.shape and w.shape[0] == w.shape[1]
                ):
                    w = w.T
                if w.shape != layer.weight.shape:
                    raise ValueError(
                        f"Weight shape mismatch for {layer}: got {w.shape}, "
                        f"expected {layer.weight.shape}"
                    )
                if b.shape != layer.bias.shape:
                    raise ValueError(
                        f"Bias shape mismatch for {layer}: got {b.shape}, "
                        f"expected {layer.bias.shape}"
                    )
                layer.weight.data = torch.from_numpy(w).to(
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                layer.bias.data = torch.from_numpy(b).to(
                    device=layer.bias.device,
                    dtype=layer.bias.dtype,
                )

            def assign_locally_connected(layer, w, b):
                w = w.astype(numpy.float32)
                b = b.astype(numpy.float32)
                if len(w.shape) == 5 and w.shape[1] == 1:
                    out_len, _, k, in_ch, out_ch = w.shape
                    w = w.squeeze(1)
                    w = w.reshape(out_len, k * in_ch, out_ch)
                    w = w.transpose(0, 2, 1)
                elif len(w.shape) == 3 and w.shape[0] == layer.output_length:
                    # Keras (out_len, k*in_ch, out_ch) -> PyTorch (out_len, out_ch, in_ch*k)
                    if w.shape[1] == layer.weight.shape[2] and w.shape[2] == layer.weight.shape[1]:
                        w = w.transpose(0, 2, 1)
                    else:
                        kernel_size = layer.kernel_size
                        out_len = w.shape[0]
                        k_times_in_ch = w.shape[1]
                        out_ch = w.shape[2]
                        in_ch = k_times_in_ch // kernel_size
                        w = w.reshape(out_len, kernel_size, in_ch, out_ch)
                        w = w.transpose(0, 2, 1, 3)
                        w = w.reshape(out_len, in_ch * kernel_size, out_ch)
                        w = w.transpose(0, 2, 1)
                if w.shape != layer.weight.shape:
                    raise ValueError(
                        f"Weight shape mismatch for {layer}: got {w.shape}, "
                        f"expected {layer.weight.shape}"
                    )
                if b.shape != layer.bias.shape:
                    raise ValueError(
                        f"Bias shape mismatch for {layer}: got {b.shape}, "
                        f"expected {layer.bias.shape}"
                    )
                layer.weight.data = torch.from_numpy(w).to(
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                layer.bias.data = torch.from_numpy(b).to(
                    device=layer.bias.device,
                    dtype=layer.bias.dtype,
                )

            def assign_batch_norm(layer, gamma, beta, mean, var):
                layer.weight.data = torch.from_numpy(
                    gamma.astype(numpy.float32)
                ).to(device=layer.weight.device, dtype=layer.weight.dtype)
                layer.bias.data = torch.from_numpy(
                    beta.astype(numpy.float32)
                ).to(device=layer.bias.device, dtype=layer.bias.dtype)
                layer.running_mean.data = torch.from_numpy(
                    mean.astype(numpy.float32)
                ).to(
                    device=layer.running_mean.device,
                    dtype=layer.running_mean.dtype,
                )
                layer.running_var.data = torch.from_numpy(
                    var.astype(numpy.float32)
                ).to(
                    device=layer.running_var.device,
                    dtype=layer.running_var.dtype,
                )

            skip_keras_embedding = False
            keras_metadata = getattr(self, "_keras_metadata", None)
            if keras_metadata and keras_metadata.get("skip_embedding_weights", False):
                skip_keras_embedding = True

            for layer in keras_layers:
                layer_class = layer.get("class_name", "")
                layer_name = layer.get("config", {}).get("name", "")

                if layer_class == "Dense":
                    w = weights[idx]
                    b = weights[idx + 1]
                    idx += 2
                    if layer_name == "output":
                        assign_dense(self.output_layer, w, b)
                    elif layer_name.startswith("dense_"):
                        dense_idx = int(layer_name.split("_")[1])
                        assign_dense(self.dense_layers[dense_idx], w, b)
                    elif layer_name.startswith("peptide_dense_"):
                        dense_idx = int(layer_name.split("_")[2])
                        assign_dense(self.peptide_dense_layers[dense_idx], w, b)
                    elif layer_name.startswith("allele_dense_"):
                        dense_idx = int(layer_name.split("_")[2])
                        assign_dense(self.allele_dense_layers[dense_idx], w, b)
                elif layer_class == "LocallyConnected1D":
                    w = weights[idx]
                    b = weights[idx + 1]
                    idx += 2
                    lc_idx = int(layer_name.split("_")[1])
                    assign_locally_connected(self.lc_layers[lc_idx], w, b)
                elif layer_class == "Embedding":
                    w = weights[idx]
                    idx += 1
                    if skip_keras_embedding:
                        continue
                    if self.allele_embedding is None:
                        continue
                    if w.shape == self.allele_embedding.weight.shape:
                        target = self.allele_embedding.weight
                        self.allele_embedding.weight.data = torch.from_numpy(
                            w.astype(numpy.float32)
                        ).to(device=target.device, dtype=target.dtype)
                elif layer_class == "BatchNormalization":
                    gamma = weights[idx]
                    beta = weights[idx + 1]
                    mean = weights[idx + 2]
                    var = weights[idx + 3]
                    idx += 4
                    if layer_name == "batch_norm_early":
                        if self.batch_norm_early is not None:
                            assign_batch_norm(self.batch_norm_early, gamma, beta, mean, var)
                    elif layer_name.startswith("batch_norm_"):
                        bn_idx = int(layer_name.split("_")[2])
                        if self.batch_norms[bn_idx] is not None:
                            assign_batch_norm(self.batch_norms[bn_idx], gamma, beta, mean, var)
                else:
                    continue

            return
        idx = 0

        # Check for keras metadata to know if we need to skip embedding weights
        keras_metadata = getattr(self, '_keras_metadata', None)
        skip_keras_embedding = False
        if keras_metadata and keras_metadata.get('skip_embedding_weights', False):
            skip_keras_embedding = True

        named_modules = dict(self.named_modules()) if auto_convert_keras else {}

        for name, param in self.named_parameters():
            # Skip allele_embedding when loading Keras weights with placeholder
            if skip_keras_embedding and 'allele_embedding' in name:
                # Also skip the corresponding placeholder weight in the weights list
                # Placeholder embeddings have shape (0, embed_dim)
                while idx < len(weights) and len(weights[idx].shape) == 2 and weights[idx].shape[0] == 0:
                    idx += 1
                continue
            w = weights[idx].astype(numpy.float32)
            extra_keras_skip = 0
            module = None
            if auto_convert_keras and "." in name:
                module_name = name.rsplit(".", 1)[0]
                module = named_modules.get(module_name)

            # Skip allele_embedding if shapes don't match (pan-allele models)
            # The embedding will be set by set_allele_representations later
            if 'allele_embedding' in name and w.shape != param.shape:
                # Advance index past this weight
                idx += 1
                continue

            # Auto-detect and convert Keras weights
            if auto_convert_keras:
                # Dense/Linear layer: Keras (in, out) -> PyTorch (out, in)
                # Note: Must transpose even when shapes match (square matrices)
                # Check for weight (not bias) by looking at param name
                is_linear_weight = ('weight' in name and
                                    'embedding' not in name and
                                    len(w.shape) == 2)
                if is_linear_weight and (w.shape == param.shape[::-1] or
                        (w.shape == param.shape and w.shape[0] == w.shape[1])):
                    w = w.T
                # LocallyConnected1D weight: Keras (out_len, 1, k, in_ch, out_ch)
                # -> PyTorch (out_len, out_ch, in_ch * k)
                elif len(w.shape) == 5 and w.shape[1] == 1:
                    out_len, _, k, in_ch, out_ch = w.shape
                    w = w.squeeze(1)  # (out_len, k, in_ch, out_ch)
                    w = w.reshape(out_len, k * in_ch, out_ch)
                    w = w.transpose(0, 2, 1)  # (out_len, out_ch, k * in_ch)
                # LocallyConnected1D weight (3D): Keras (out_len, k*in_ch, out_ch)
                # -> PyTorch (out_len, out_ch, in_ch*k)
                # Note: Keras stores kernel_positions as outer loop, channels as inner
                # PyTorch unfold produces channels as outer loop, kernel_positions as inner
                elif len(w.shape) == 3 and w.shape[0] == param.shape[0] and \
                        w.shape[1] == param.shape[2] and w.shape[2] == param.shape[1]:
                    # LocallyConnected1D weight (3D): Keras (out_len, k*in_ch, out_ch)
                    # -> PyTorch (out_len, out_ch, k*in_ch)
                    w = w.transpose(0, 2, 1)
                # LocallyConnected1D bias: Keras (out_len * out_ch,) -> PyTorch (out_len, out_ch)
                elif len(w.shape) == 1 and len(param.shape) == 2 and \
                        w.shape[0] == param.shape[0] * param.shape[1]:
                    w = w.reshape(param.shape)
                # BatchNorm: Keras provides gamma, beta, moving_mean, moving_var.
                # PyTorch exposes gamma/beta as params and moving stats as buffers.
                if module is not None and isinstance(module, torch.nn.BatchNorm1d):
                    if name.endswith("bias") and idx + 2 < len(weights):
                        running_mean = weights[idx + 1].astype(numpy.float32)
                        running_var = weights[idx + 2].astype(numpy.float32)
                        if module.running_mean.shape == running_mean.shape:
                            module.running_mean.data = torch.from_numpy(
                                running_mean
                            ).to(
                                device=module.running_mean.device,
                                dtype=module.running_mean.dtype,
                            )
                        if module.running_var.shape == running_var.shape:
                            module.running_var.data = torch.from_numpy(
                                running_var
                            ).to(
                                device=module.running_var.device,
                                dtype=module.running_var.dtype,
                            )
                        extra_keras_skip = 2

            if w.shape != param.shape:
                raise ValueError(
                    f"Weight shape mismatch for {name}: "
                    f"got {weights[idx].shape}, expected {param.shape}"
                )

            param.data = torch.from_numpy(w).to(
                device=param.device,
                dtype=param.dtype,
            )
            idx += 1 + extra_keras_skip
        if not auto_convert_keras:
            named_modules_dict = dict(self.named_modules())
            for name, buffer in self.named_buffers():
                tensor = torch.from_numpy(weights[idx]).to(
                    device=buffer.device,
                    dtype=buffer.dtype,
                )
                # Navigate to the correct submodule for nested buffers
                if "." in name:
                    module_path, buffer_name = name.rsplit(".", 1)
                    named_modules_dict[module_path]._buffers[buffer_name] = tensor
                else:
                    self._buffers[name] = tensor
                idx += 1

    def to_json(self):
        """
        Serialize model configuration to JSON string.

        Returns
        -------
        str
            JSON representation of model configuration
        """
        import json

        # Extract layer configurations
        lc_layers_config = []
        for lc_layer in self.lc_layers:
            lc_layers_config.append({
                'in_channels': lc_layer.in_channels,
                'out_channels': lc_layer.out_channels,
                'kernel_size': lc_layer.kernel_size,
                'input_length': lc_layer.input_length,
                'output_length': lc_layer.output_length,
                'activation': lc_layer.activation_name,
            })

        peptide_dense_sizes = [
            layer.out_features for layer in self.peptide_dense_layers
        ]
        allele_dense_sizes = [
            layer.out_features for layer in self.allele_dense_layers
        ]
        layer_sizes = [
            layer.out_features for layer in self.dense_layers
        ]

        config = {
            'class': 'Class1NeuralNetworkModel',
            'peptide_encoding_shape': list(self.peptide_encoding_shape),
            'has_allele': self.has_allele,
            'peptide_allele_merge_method': self.peptide_allele_merge_method,
            'peptide_allele_merge_activation': self.peptide_allele_merge_activation,
            'dropout_probability': self.dropout_probability,
            'topology': self.topology,
            'num_outputs': self.num_outputs,
            'activation': self.activation_name,
            'output_activation': self.output_activation_name,
            'locally_connected_layers': lc_layers_config,
            'peptide_dense_layer_sizes': peptide_dense_sizes,
            'allele_dense_layer_sizes': allele_dense_sizes,
            'layer_sizes': layer_sizes,
            'batch_normalization': self.batch_norm_early is not None,
        }

        return json.dumps(config, sort_keys=True)


class Class1NeuralNetwork(object):
    """
    Low level class I predictor consisting of a single neural network.

    Both single allele and pan-allele prediction are supported.

    Users will generally use Class1AffinityPredictor, which gives a higher-level
    interface and supports ensembles.
    """

    network_hyperparameter_defaults = HyperparameterDefaults(
        allele_amino_acid_encoding="BLOSUM62",
        allele_dense_layer_sizes=[],
        peptide_encoding={
            "vector_encoding_name": "BLOSUM62",
            "alignment_method": "pad_middle",
            "left_edge": 4,
            "right_edge": 4,
            "max_length": 15,
        },
        peptide_dense_layer_sizes=[],
        peptide_allele_merge_method="multiply",
        peptide_allele_merge_activation="",
        layer_sizes=[32],
        dense_layer_l1_regularization=0.001,
        dense_layer_l2_regularization=0.0,
        activation="tanh",
        init="glorot_uniform",
        output_activation="sigmoid",
        dropout_probability=0.0,
        batch_normalization=False,
        locally_connected_layers=[
            {"filters": 8, "activation": "tanh", "kernel_size": 3}
        ],
        topology="feedforward",
        num_outputs=1,
        # Phase 2 of issue openvax/mhcflurry#268: feed the network (N, L)
        # int amino-acid indices and widen to (N, L, 21) BLOSUM62 fp32 via
        # an on-device embedding lookup, instead of shipping the (N, L, 21)
        # int8-or-fp32 BLOSUM tensor directly. Cuts cache size and H2D
        # bandwidth for the peptide path by a further 21× on top of the
        # Phase 4a int8 storage win. False (default) preserves the
        # pre-Phase-2 behavior byte-for-byte — the model still serializes
        # the same weights, same shapes, same forward output given the
        # same inputs.
        peptide_amino_acid_encoding_gpu=False,
    )
    """
    Hyperparameters (and their default values) that affect the neural network
    architecture.
    """

    compile_hyperparameter_defaults = HyperparameterDefaults(
        loss="custom:mse_with_inequalities",
        optimizer="rmsprop",
        learning_rate=None,
    )
    """
    Loss and optimizer hyperparameters.
    """

    fit_hyperparameter_defaults = HyperparameterDefaults(
        max_epochs=500,
        validation_split=0.1,
        early_stopping=True,
        minibatch_size=512,
        data_dependent_initialization_method=None,
        random_negative_affinity_min=20000.0,
        random_negative_affinity_max=50000.0,
        random_negative_output_indices=None,
        # Number of consecutive epochs that share a pre-generated pool of
        # random-negative peptides. 1 (default) reproduces the pre-Phase-1
        # "fresh peptides every epoch" semantics exactly. >1 amortizes the
        # ~17 s/epoch peptide-generation + encoding pair across that many
        # epochs (Phase 1 of issue #268). Within a pool-cycle, consecutive
        # epochs see distinct slices of the same pool rather than fresh
        # samples — a training-time semantics change the user has
        # explicitly opted into. A new pool is generated at every
        # ``epoch // random_negative_pool_epochs`` boundary. 100 is the
        # recommended production value; smaller values preserve more
        # sample diversity but recover less of the encode cost.
        random_negative_pool_epochs=1,
        # Number of DataLoader worker processes for the fit() inner batch
        # loop. 0 (default) runs everything in the training process —
        # bit-identical to pre-issue-#268 behavior. >0 spawns worker procs
        # that prefetch minibatches into pinned host memory, overlapping
        # CPU data-prep with GPU compute. Per-worker memory + CPU budget:
        # each adds one Python process holding ~100-500 MB RSS. Don't
        # exceed (num_cpu_cores - num_training_workers).
        dataloader_num_workers=0,
        # Batch size used for the validation forward pass. ``None`` uses a
        # device-aware heuristic: ``max(4 * minibatch_size, 4096)`` on CUDA
        # and ``4 * minibatch_size`` elsewhere. Separate from minibatch_size
        # because eval has no backward / optimizer state so VRAM headroom is
        # much higher; the larger CUDA default cuts validation-loop overhead
        # substantially on the small pan-allele networks.
        validation_batch_size=None,
    ).extend(RandomNegativePeptides.hyperparameter_defaults)
    """
    Hyperparameters for neural network training.
    """

    early_stopping_hyperparameter_defaults = HyperparameterDefaults(
        patience=20,
        min_delta=0.0,
        # Run the validation pass every N epochs in fit() / fit_generator().
        # Default 1 preserves pre-existing behavior (validate every epoch).
        # Setting to >1 trades resolution-of-early-stop-decision for
        # epoch-level throughput; the fit() loop forces a final validation
        # pass before reporting min_val_loss / breaking on patience so the
        # saved model still reflects an up-to-date val_loss measurement.
        validation_interval=1,
    )
    """
    Hyperparameters for early stopping.
    """

    miscelaneous_hyperparameter_defaults = HyperparameterDefaults(
        train_data={},
    )
    """
    Miscelaneous hyperaparameters. These parameters are not used by this class
    but may be interpreted by other code.
    """

    hyperparameter_defaults = (
        network_hyperparameter_defaults.extend(compile_hyperparameter_defaults)
        .extend(fit_hyperparameter_defaults)
        .extend(early_stopping_hyperparameter_defaults)
        .extend(miscelaneous_hyperparameter_defaults)
    )
    """
    Combined set of all supported hyperparameters and their default values.
    """

    # Hyperparameter renames. Map old_name → new_name (string to rename)
    # or old_name → None (to silently drop). Loading an old config with
    # any of these keys triggers the mapping in ``__init__`` before
    # ``HyperparameterDefaults.check_valid_keys`` runs — so old keys
    # don't cause a ValueError.
    #
    # Note: ``min_delta`` used to be in this table mapping to None,
    # because when it was added (pre-PyTorch) it was flagged "currently
    # unused". It has since been re-introduced as a live early-stopping
    # hyperparameter, so dropping it silently would have caused old
    # configs with explicit non-default ``min_delta`` to lose that
    # value. Fixed by removing from this table — now it passes through
    # to the valid-keys check and gets preserved.
    hyperparameter_renames = {
        "use_embedding": None,
        "pseudosequence_use_embedding": None,
        "monitor": None,
        "verbose": None,
        "mode": None,
        "take_best_epoch": None,
        "kmer_size": None,
        "peptide_amino_acid_encoding": None,
        "embedding_input_dim": None,
        "embedding_output_dim": None,
        "embedding_init_method": None,
        "left_edge": None,
        "right_edge": None,
    }

    @classmethod
    def apply_hyperparameter_renames(cls, hyperparameters):
        """
        Handle hyperparameter renames.

        Parameters
        ----------
        hyperparameters : dict

        Returns
        -------
        dict : updated hyperparameters

        """
        for from_name, to_name in cls.hyperparameter_renames.items():
            if from_name in hyperparameters:
                value = hyperparameters.pop(from_name)
                if to_name:
                    hyperparameters[to_name] = value
        return hyperparameters

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            self.apply_hyperparameter_renames(hyperparameters)
        )

        self._network = None
        self.network_json = None
        self.network_weights = None
        self.network_weights_loader = None

        self.fit_info = []
        self.prediction_cache = weakref.WeakKeyDictionary()

    MODELS_CACHE = {}
    """
    Process-wide model cache, a map from: architecture JSON string to
    (PyTorch model, existing network weights)
    """

    @classmethod
    def clear_model_cache(klass):
        """
        Clear the model cache.
        """
        klass.MODELS_CACHE.clear()

    @classmethod
    def borrow_cached_network(klass, network_json, network_weights):
        """
        Return a PyTorch model with the specified architecture and weights.
        As an optimization, when possible this will reuse architectures from a
        process-wide cache.

        Parameters
        ----------
        network_json : string of JSON
        network_weights : list of numpy.array

        Returns
        -------
        Class1NeuralNetworkModel
        """
        assert network_weights is not None
        key = klass.model_cache_key(network_json)
        config = json.loads(network_json)
        # Detect if weights are from Keras or PyTorch format
        # Keras JSON has 'class_name': 'Model' or 'Functional'; PyTorch has 'hyperparameters'
        is_keras_format = config.get('class_name') in ('Model', 'Functional')

        if key not in klass.MODELS_CACHE:
            # Cache miss - create new model
            network = klass._create_model_from_config(config)
            existing_weights = None
        else:
            # Cache hit
            (network, existing_weights) = klass.MODELS_CACHE[key]

        if existing_weights is not network_weights:
            network.set_weights_list(network_weights, auto_convert_keras=is_keras_format)
            klass.MODELS_CACHE[key] = (network, network_weights)

        return network

    @classmethod
    def _parse_keras_json_config(cls, config):
        """
        Parse a legacy Keras model JSON config to extract hyperparameters.

        Parameters
        ----------
        config : dict
            Keras model JSON config with 'class_name', 'config', etc.

        Returns
        -------
        tuple of (dict, dict)
            First dict: Hyperparameters dict compatible with Class1NeuralNetwork
            Second dict: Metadata about Keras model structure (e.g., embedding info)
        """
        layers = config.get('config', {}).get('layers', [])

        hyperparameters = {
            'locally_connected_layers': [],
            'layer_sizes': [],
            'activation': 'tanh',
            'output_activation': 'sigmoid',
            'dropout_probability': 0.0,
            'batch_normalization': False,
            'dense_layer_l1_regularization': 0.001,
            'dense_layer_l2_regularization': 0.0,
            'peptide_allele_merge_method': 'multiply',  # Default
        }

        # Metadata about Keras structure
        keras_metadata = {
            'has_embedding': False,
            'embedding_input_dim': 0,
            'embedding_output_dim': 0,
            'skip_embedding_weights': False,
        }

        dense_layers = []
        peptide_dense_sizes = []
        allele_dense_sizes = []
        concatenate_count = 0
        for layer in layers:
            layer_class = layer.get('class_name', '')
            layer_config = layer.get('config', {})

            if layer_class == 'LocallyConnected1D':
                lc_config = {
                    'filters': layer_config.get('filters', 8),
                    'kernel_size': layer_config.get('kernel_size', [3])[0] if isinstance(
                        layer_config.get('kernel_size', [3]), list
                    ) else layer_config.get('kernel_size', 3),
                    'activation': layer_config.get('activation', 'tanh'),
                }
                hyperparameters['locally_connected_layers'].append(lc_config)
                hyperparameters['activation'] = lc_config['activation']

            elif layer_class == 'Dense':
                units = layer_config.get('units', 32)
                activation = layer_config.get('activation', 'tanh')
                layer_name = layer_config.get('name', '')
                if layer_name.startswith('peptide_dense_'):
                    peptide_dense_sizes.append(units)
                elif layer_name.startswith('allele_dense_'):
                    allele_dense_sizes.append(units)
                else:
                    dense_layers.append({'units': units, 'activation': activation})

                # Extract regularization from first dense layer
                kernel_reg = layer_config.get('kernel_regularizer')
                if kernel_reg and isinstance(kernel_reg, dict):
                    reg_config = kernel_reg.get('config', {})
                    if 'l1' in reg_config:
                        hyperparameters['dense_layer_l1_regularization'] = reg_config['l1']
                    if 'l2' in reg_config:
                        hyperparameters['dense_layer_l2_regularization'] = reg_config['l2']

            elif layer_class == 'Dropout':
                rate = layer_config.get('rate', 0.0)
                hyperparameters['dropout_probability'] = 1.0 - rate

            elif layer_class == 'BatchNormalization':
                hyperparameters['batch_normalization'] = True

            elif layer_class == 'Embedding':
                keras_metadata['has_embedding'] = True
                keras_metadata['embedding_input_dim'] = layer_config.get('input_dim', 0)
                keras_metadata['embedding_output_dim'] = layer_config.get('output_dim', 0)
                # If input_dim is 0, it's a placeholder and weights should be skipped
                if layer_config.get('input_dim', 0) == 0:
                    keras_metadata['skip_embedding_weights'] = True

            elif layer_class == 'Concatenate':
                concatenate_count += 1
                # Only set merge_method to concatenate if there's just one Concatenate
                # and it's likely for peptide-allele merging (not DenseNet skip connections)
                if concatenate_count == 1:
                    hyperparameters['peptide_allele_merge_method'] = 'concatenate'

            elif layer_class == 'Multiply':
                hyperparameters['peptide_allele_merge_method'] = 'multiply'

        # Multiple Concatenate layers indicate DenseNet topology with skip connections
        # Note: The first Concatenate is typically for peptide-allele merging,
        # subsequent ones are for DenseNet skip connections
        if concatenate_count > 1:
            hyperparameters['topology'] = 'with-skip-connections'
            # Keep the merge method as detected (concatenate from first Concatenate layer)

        # The last Dense layer is the output layer
        if dense_layers:
            hyperparameters['output_activation'] = dense_layers[-1]['activation']
            hyperparameters['num_outputs'] = dense_layers[-1]['units']
            # All other Dense layers contribute to layer_sizes
            hyperparameters['layer_sizes'] = [d['units'] for d in dense_layers[:-1]]
            if dense_layers[:-1]:
                hyperparameters['activation'] = dense_layers[0]['activation']

        if peptide_dense_sizes:
            hyperparameters['peptide_dense_layer_sizes'] = peptide_dense_sizes
        if allele_dense_sizes:
            hyperparameters['allele_dense_layer_sizes'] = allele_dense_sizes

        return hyperparameters, keras_metadata

    @classmethod
    def _create_model_from_config(cls, config, instance_hyperparameters=None):
        """Create a model from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary (either Keras JSON or hyperparameters dict)
        instance_hyperparameters : dict, optional
            Hyperparameters from the Class1NeuralNetwork instance.
            These take precedence for things like peptide_encoding.
        """
        keras_metadata = None

        # Check if this is a merged network config
        if config.get('merged_networks'):
            return cls._create_merged_model_from_config(config, instance_hyperparameters)

        # Check if this is a legacy Keras JSON config
        if config.get('class_name') in ('Model', 'Functional'):
            hyperparameters, keras_metadata = cls._parse_keras_json_config(config)
        else:
            # Extract hyperparameters from config (new format)
            hyperparameters = config.get('hyperparameters', config)

        # Merge with instance hyperparameters if provided
        # Instance hyperparameters take precedence for things like peptide_encoding
        if instance_hyperparameters:
            # Copy to avoid modifying original
            merged = dict(instance_hyperparameters)
            # Update with parsed hyperparameters (architecture-specific settings)
            for key in ['layer_sizes', 'locally_connected_layers', 'dropout_probability',
                        'batch_normalization', 'activation', 'output_activation',
                        'peptide_allele_merge_method']:
                if key in hyperparameters:
                    merged[key] = hyperparameters[key]
            hyperparameters = merged

        # Create a temporary instance to get encoding shape
        temp = cls(**hyperparameters)
        peptide_encoding_shape = temp.peptides_to_network_input([]).shape[1:]
        # Phase 2 (#268) on-device BLOSUM62: peptides_to_network_input
        # returns (N, L) int indices when the flag is on, but the dense
        # layers are still sized against the post-embedding (L, 21) shape.
        # Expand here so the rest of the network-builder sees the same
        # encoding shape regardless of which path is active.
        if (
            hyperparameters.get("peptide_amino_acid_encoding_gpu", False)
            and len(peptide_encoding_shape) == 1
        ):
            from .amino_acid import AMINO_ACIDS
            peptide_encoding_shape = (
                peptide_encoding_shape[0],
                len(AMINO_ACIDS),
            )

        # Get allele representations if present
        allele_representations = config.get('allele_representations')
        if allele_representations is not None:
            allele_representations = numpy.array(allele_representations)

        # For pan-allele Keras models with placeholder embedding (input_dim=0),
        # create a placeholder allele representation to ensure correct architecture
        if (allele_representations is None and keras_metadata is not None
                and keras_metadata.get('has_embedding', False)
                and keras_metadata.get('embedding_output_dim', 0) > 0):
            # Create placeholder with 1 allele and correct embedding dim
            # This will be replaced by set_allele_representations later
            embedding_dim = keras_metadata['embedding_output_dim']
            allele_representations = numpy.zeros((1, embedding_dim), dtype=numpy.float32)

        # For PyTorch-format configs without allele_representations but with
        # allele_amino_acid_encoding (pan-allele models), create placeholder
        # Check has_allele flag to distinguish pan-allele from allele-specific models
        has_allele = config.get('has_allele', True)  # Default True for backward compat
        if (allele_representations is None and keras_metadata is None
                and has_allele and hyperparameters.get('allele_amino_acid_encoding')):
            # Compute embedding dimension from encoding
            from .amino_acid import ENCODING_DATA_FRAMES
            encoding_name = hyperparameters['allele_amino_acid_encoding']
            encoding_df = ENCODING_DATA_FRAMES.get(encoding_name)
            if encoding_df is not None:
                # Standard allele pseudosequence length is 37 amino acids
                allele_seq_length = 37
                embedding_dim = allele_seq_length * len(encoding_df.columns)
                allele_representations = numpy.zeros((1, embedding_dim), dtype=numpy.float32)

        model = Class1NeuralNetworkModel(
            peptide_encoding_shape=peptide_encoding_shape,
            allele_representations=allele_representations,
            locally_connected_layers=hyperparameters.get('locally_connected_layers', []),
            peptide_dense_layer_sizes=hyperparameters.get('peptide_dense_layer_sizes', []),
            allele_dense_layer_sizes=hyperparameters.get('allele_dense_layer_sizes', []),
            layer_sizes=hyperparameters.get('layer_sizes', [32]),
            peptide_allele_merge_method=hyperparameters.get('peptide_allele_merge_method', 'multiply'),
            peptide_allele_merge_activation=hyperparameters.get('peptide_allele_merge_activation', ''),
            activation=hyperparameters.get('activation', 'tanh'),
            output_activation=hyperparameters.get('output_activation', 'sigmoid'),
            dropout_probability=hyperparameters.get('dropout_probability', 0.0),
            batch_normalization=hyperparameters.get('batch_normalization', False),
            dense_layer_l1_regularization=hyperparameters.get('dense_layer_l1_regularization', 0.001),
            dense_layer_l2_regularization=hyperparameters.get('dense_layer_l2_regularization', 0.0),
            topology=hyperparameters.get('topology', 'feedforward'),
            num_outputs=hyperparameters.get('num_outputs', 1),
            init=hyperparameters.get('init', 'glorot_uniform'),
            peptide_input_is_indices=hyperparameters.get(
                'peptide_amino_acid_encoding_gpu', False),
        )

        # Store keras metadata and config for weight loading
        if keras_metadata is not None:
            model._keras_metadata = keras_metadata
            model._keras_config = config

        return model

    @classmethod
    def _create_merged_model_from_config(cls, config, instance_hyperparameters=None):
        """Create a merged model from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'merged_networks' key
        instance_hyperparameters : dict, optional
            Hyperparameters from the Class1NeuralNetwork instance.
        """
        merged_configs = config['merged_networks']
        merge_method = config.get('merge_method', 'average')

        # Create a temporary instance to get encoding shape
        base_hyperparameters = config.get('hyperparameters', {})
        if instance_hyperparameters:
            base_hyperparameters = dict(instance_hyperparameters)
            base_hyperparameters.update(config.get('hyperparameters', {}))
        temp = cls(**base_hyperparameters)
        peptide_encoding_shape = temp.peptides_to_network_input([]).shape[1:]

        # Create placeholder allele representations for pan-allele models
        allele_representations = None
        if base_hyperparameters.get('allele_amino_acid_encoding'):
            from .amino_acid import ENCODING_DATA_FRAMES
            encoding_name = base_hyperparameters['allele_amino_acid_encoding']
            encoding_df = ENCODING_DATA_FRAMES.get(encoding_name)
            if encoding_df is not None:
                allele_seq_length = 37
                embedding_dim = allele_seq_length * len(encoding_df.columns)
                allele_representations = numpy.zeros((1, embedding_dim), dtype=numpy.float32)

        # Create sub-networks
        sub_networks = []
        for sub_config in merged_configs:
            model = Class1NeuralNetworkModel(
                peptide_encoding_shape=peptide_encoding_shape,
                allele_representations=allele_representations,
                locally_connected_layers=sub_config.get('locally_connected_layers', []),
                peptide_dense_layer_sizes=sub_config.get('peptide_dense_layer_sizes', []),
                allele_dense_layer_sizes=sub_config.get('allele_dense_layer_sizes', []),
                layer_sizes=sub_config.get('layer_sizes', [32]),
                peptide_allele_merge_method=sub_config.get('peptide_allele_merge_method', 'multiply'),
                peptide_allele_merge_activation=sub_config.get('peptide_allele_merge_activation', ''),
                activation=sub_config.get('activation', 'tanh'),
                output_activation=sub_config.get('output_activation', 'sigmoid'),
                dropout_probability=sub_config.get('dropout_probability', 0.0),
                batch_normalization=sub_config.get('batch_normalization', False),
                dense_layer_l1_regularization=base_hyperparameters.get('dense_layer_l1_regularization', 0.001),
                dense_layer_l2_regularization=base_hyperparameters.get('dense_layer_l2_regularization', 0.0),
                topology=sub_config.get('topology', 'feedforward'),
                num_outputs=sub_config.get('num_outputs', 1),
                init=base_hyperparameters.get('init', 'glorot_uniform'),
            )
            sub_networks.append(model)

        return MergedClass1NeuralNetwork(sub_networks, merge_method=merge_method)

    @staticmethod
    def model_cache_key(network_json):
        """
        Given a JSON description of a neural network, return a cache key.

        Parameters
        ----------
        network_json : string

        Returns
        -------
        string
        """
        # Remove regularization settings as they don't affect predictions
        def drop_properties(d):
            if isinstance(d, dict):
                d.pop('dense_layer_l1_regularization', None)
                d.pop('dense_layer_l2_regularization', None)
            return d

        description = json.loads(network_json, object_hook=drop_properties)
        return json.dumps(description)

    @staticmethod
    def keras_network_cache_key(network_json):
        """
        Backward-compatible alias for ``model_cache_key``.
        """
        return Class1NeuralNetwork.model_cache_key(network_json)

    def network(self, borrow=False):
        """
        Return the PyTorch model associated with this predictor.

        Parameters
        ----------
        borrow : bool
            Whether to return a cached model if possible

        Returns
        -------
        Class1NeuralNetworkModel
        """
        if self._network is None and self.network_json is not None:
            self.load_weights()
            if borrow:
                return self.borrow_cached_network(
                    self.network_json, self.network_weights
                )
            else:
                config = json.loads(self.network_json)
                # Detect if weights are from Keras or PyTorch format
                # Keras JSON has 'class_name': 'Model' or 'Functional'; PyTorch has 'hyperparameters'
                is_keras_format = config.get('class_name') in ('Model', 'Functional')
                # Pass this instance's hyperparameters to preserve peptide_encoding etc.
                self._network = self._create_model_from_config(
                    config, instance_hyperparameters=self.hyperparameters)
                if self.network_weights is not None:
                    self._network.set_weights_list(
                        self.network_weights,
                        auto_convert_keras=is_keras_format
                    )
                self.network_json = None
                self.network_weights = None
        return self._network

    def update_network_description(self):
        """
        Update self.network_json and self.network_weights properties based on
        this instances's neural network.
        """
        if self._network is not None:
            config = {
                'hyperparameters': dict(self.hyperparameters),
            }

            # Check if this is a merged network
            if isinstance(self._network, MergedClass1NeuralNetwork):
                # Save sub-network configs for merged networks
                sub_configs = []
                for subnet in self._network.networks:
                    sub_config = {}
                    # Get the architecture info from the network itself
                    sub_config['layer_sizes'] = [
                        layer.out_features for layer in subnet.dense_layers
                    ]
                    sub_config['locally_connected_layers'] = [
                        {'filters': layer.out_channels, 'kernel_size': layer.kernel_size}
                        for layer in subnet.lc_layers
                    ] if hasattr(subnet, 'lc_layers') else []
                    sub_config['peptide_dense_layer_sizes'] = [
                        layer.out_features for layer in subnet.peptide_dense_layers
                    ] if hasattr(subnet, 'peptide_dense_layers') else []
                    sub_config['allele_dense_layer_sizes'] = [
                        layer.out_features for layer in subnet.allele_dense_layers
                    ] if hasattr(subnet, 'allele_dense_layers') else []
                    # MHCflurry hyperparameters use keep probability, not
                    # PyTorch Dropout.p (drop probability).
                    sub_config['dropout_probability'] = getattr(
                        subnet,
                        'dropout_probability',
                        0.0,
                    )
                    sub_config['batch_normalization'] = (
                        hasattr(subnet, 'batch_norms') and bool(subnet.batch_norms) and
                        any(bn is not None for bn in subnet.batch_norms)
                    )
                    sub_config['activation'] = subnet.activation_name
                    sub_config['output_activation'] = subnet.output_activation_name
                    sub_config['peptide_allele_merge_method'] = subnet.peptide_allele_merge_method
                    sub_config['peptide_allele_merge_activation'] = subnet.peptide_allele_merge_activation
                    sub_config['topology'] = subnet.topology
                    sub_config['num_outputs'] = subnet.output_layer.out_features
                    sub_configs.append(sub_config)
                config['merged_networks'] = sub_configs
                config['merge_method'] = self._network.merge_method
            else:
                # Save whether the network has allele features
                config['has_allele'] = getattr(self._network, 'has_allele', False)
                # Save allele representations if present in the network
                if hasattr(self._network, 'allele_embedding') and self._network.allele_embedding is not None:
                    allele_embed = self._network.allele_embedding.weight.detach().cpu().numpy()
                    config['allele_representations'] = allele_embed.tolist()

            self.network_json = json.dumps(config)
            self.network_weights = self._network.get_weights_list()

    def get_config(self):
        """
        serialize to a dict all attributes except model weights

        Returns
        -------
        dict
        """
        self.update_network_description()
        result = dict(self.__dict__)
        result["_network"] = None
        result["network_weights"] = None
        result["network_weights_loader"] = None
        result["prediction_cache"] = None
        return result

    @classmethod
    def from_config(cls, config, weights=None, weights_loader=None):
        """
        deserialize from a dict returned by get_config().

        Supports both:
        - Native Class1NeuralNetwork configs with 'hyperparameters' key
        - Legacy Keras model JSON configs with 'class_name', 'config', etc.

        Parameters
        ----------
        config : dict
        weights : list of array, optional
            Network weights to restore
        weights_loader : callable, optional
            Function to call (no arguments) to load weights when needed

        Returns
        -------
        Class1NeuralNetwork
        """
        config = dict(config)

        # Check if this is a legacy Keras JSON config
        if config.get('class_name') in ('Model', 'Functional'):
            hyperparameters, keras_metadata = cls._parse_keras_json_config(config)
            instance = cls(**hyperparameters)
            # Store metadata for weight loading
            instance._keras_metadata = keras_metadata
            # Store the original config as network_json for lazy network creation
            instance.network_json = json.dumps(config)
        else:
            # Standard Class1NeuralNetwork config format
            instance = cls(**config.pop("hyperparameters"))
            instance.__dict__.update(config)

        instance.network_weights = weights
        instance.network_weights_loader = weights_loader
        instance.prediction_cache = weakref.WeakKeyDictionary()
        return instance

    def load_weights(self):
        """
        Load weights by evaluating self.network_weights_loader, if needed.
        """
        if self.network_weights_loader:
            self.network_weights = self.network_weights_loader()
            self.network_weights_loader = None

    def get_weights(self):
        """
        Get the network weights

        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        self.update_network_description()
        self.load_weights()
        return self.network_weights

    def get_weights_list(self):
        """
        Get the network weights as a list of numpy arrays.

        Returns
        -------
        list of numpy.array giving weights for each layer or None if there is no
        network
        """
        return self.get_weights()

    def set_weights_list(self, weights, auto_convert_keras=True):
        """
        Set the network weights from a list of numpy arrays.

        If a network exists, the weights are set directly on it.
        Otherwise, the weights are stored and will be applied when the
        network is created.

        Parameters
        ----------
        weights : list of numpy.array
            Weights for each layer
        auto_convert_keras : bool
            If True, attempt to auto-detect and convert Keras weight formats
            to PyTorch format. Default True.
        """
        if self._network is not None:
            # Network exists, set weights directly
            self._network.set_weights_list(weights, auto_convert_keras=auto_convert_keras)
        else:
            # Store weights for later application
            self.network_weights = weights
            # Store flag for auto-conversion
            self._auto_convert_keras_weights = auto_convert_keras

    def __getstate__(self):
        """
        serialize to a dict. Model weights are included. For pickle support.

        Returns
        -------
        dict

        """
        self.update_network_description()
        self.load_weights()
        result = dict(self.__dict__)
        result["_network"] = None
        result["prediction_cache"] = None
        return result

    def __setstate__(self, state):
        """
        Deserialize. For pickle support.
        """
        self.__dict__.update(state)
        self.prediction_cache = weakref.WeakKeyDictionary()

    def peptides_to_network_input(self, peptides):
        """
        Encode peptides to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        When ``peptide_amino_acid_encoding_gpu`` is True (Phase 2 of
        openvax/mhcflurry#268), the returned array is (N, L) int8 amino-
        acid indices — the network's forward pass widens it to (N, L, 21)
        BLOSUM62 fp32 on device. Otherwise the returned array is the
        traditional (N, L, 21) vector encoding.

        Parameters
        ----------
        peptides : EncodableSequences or list of string

        Returns
        -------
        numpy.array
        """
        encoder = EncodableSequences.create(peptides)
        if self.hyperparameters.get("peptide_amino_acid_encoding_gpu", False):
            encoded = self._peptides_to_indices_raw(encoder)
        else:
            encoded = encoder.variable_length_to_fixed_length_vector_encoding(
                **self.hyperparameters["peptide_encoding"]
            )
        assert len(encoded) == len(peptides)
        return encoded

    def _peptides_to_indices_raw(self, encoder):
        """Produce (N, L) int8 amino-acid indices for ``encoder``.

        Shares alignment_method/left_edge/right_edge/max_length with the
        configured vector encoding but drops the vector_encoding_name /
        trim / allow_unsupported_amino_acids kwargs that only apply to
        the BLOSUM62 path. Phase 2 of #268.
        """
        categorical_kwargs = {
            key: value
            for key, value in self.hyperparameters["peptide_encoding"].items()
            if key in (
                "alignment_method", "left_edge", "right_edge", "max_length"
            )
        }
        return (
            encoder.variable_length_to_fixed_length_categorical(
                **categorical_kwargs
            )
            .astype("int8", copy=False)
        )

    @property
    def supported_peptide_lengths(self):
        """
        (minimum, maximum) lengths of peptides supported, inclusive.

        Returns
        -------
        (int, int) tuple

        """
        try:
            self.peptides_to_network_input([""])
        except EncodingError as e:
            return e.supported_peptide_lengths
        raise RuntimeError("peptides_to_network_input did not raise")

    def allele_encoding_to_network_input(self, allele_encoding):
        """
        Encode alleles to the fixed-length encoding expected by the neural
        network (which depends on the architecture).

        Parameters
        ----------
        allele_encoding : AlleleEncoding

        Returns
        -------
        (numpy.array, numpy.array)

        Indices and allele representations.

        """
        if allele_encoding is None:
            return (None, None)
        return (
            allele_encoding.indices.values,
            allele_encoding.allele_representations(
                self.hyperparameters["allele_amino_acid_encoding"]
            ),
        )

    @staticmethod
    def data_dependent_weights_initialization(network, x_dict=None, method="lsuv", verbose=1):
        """
        Data dependent weights initialization.

        Parameters
        ----------
        network : Class1NeuralNetworkModel
        x_dict : dict of string -> numpy.ndarray
            Training data
        method : string
            Initialization method. Currently only "lsuv" is supported.
        verbose : int
            Status updates printed to stdout if verbose > 0
        """
        if verbose:
            print("Performing data-dependent init: ", method)
        if method == "lsuv":
            assert x_dict is not None, "Data required for LSUV init"
            lsuv_init(network, x_dict, verbose=verbose > 0)
        else:
            raise RuntimeError("Unsupported init method: ", method)

    @staticmethod
    def _regularized_parameters(network):
        """
        Parameters subject to master-branch dense kernel regularization.
        """
        for name, param in network.named_parameters():
            if not param.requires_grad or not name.endswith("weight"):
                continue
            if any(part in name for part in (
                    "peptide_dense_layers",
                    "allele_dense_layers",
                    "dense_layers")):
                yield param

    @staticmethod
    def _regularization_penalty(parameters, l1=0.0, l2=0.0):
        """
        Match Keras kernel_regularizer semantics used on dense kernels.
        """
        parameters = tuple(parameters)
        if not parameters or (not l1 and not l2):
            return None
        penalty = torch.zeros((), device=parameters[0].device)
        for param in parameters:
            if l1:
                penalty = penalty + (l1 * param.abs().sum())
            if l2:
                penalty = penalty + (l2 * param.square().sum())
        return penalty

    def get_device(self):
        """Get the PyTorch device to use."""
        return get_pytorch_device()

    def fit_generator(
            self,
            generator,
            validation_peptide_encoding,
            validation_affinities,
            validation_allele_encoding=None,
            validation_inequalities=None,
            validation_output_indices=None,
            steps_per_epoch=10,
            epochs=1000,
            min_epochs=0,
            patience=10,
            min_delta=0.0,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0,
            generator_factory=None,
            generator_batches_are_encoded=False):
        """
        Fit using a generator. Does not support many of the features of fit(),
        such as random negative peptides.
        """
        device = self.get_device()
        _configure_matmul_precision(device)

        fit_info = collections.defaultdict(list)
        timing_enabled = _timing_enabled()
        fit_info["timing_enabled"] = timing_enabled

        loss_obj = get_pytorch_loss(self.hyperparameters["loss"])

        (
            validation_allele_input,
            allele_representations,
        ) = self.allele_encoding_to_network_input(validation_allele_encoding)

        if self.network() is None:
            self._network = self.make_network(
                allele_representations=allele_representations,
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters)
            )
            if verbose > 0:
                print(self.network())
        network = self.network()
        network.to(device)

        self.set_allele_representations(allele_representations)

        # Setup optimizer
        optimizer = self._create_optimizer(network)
        if self.hyperparameters["learning_rate"] is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.hyperparameters["learning_rate"]
        fit_info["learning_rate"] = optimizer.param_groups[0]['lr']
        regularization_parameters = tuple(self._regularized_parameters(network))
        l1_reg = self.hyperparameters["dense_layer_l1_regularization"]
        l2_reg = self.hyperparameters["dense_layer_l2_regularization"]

        # Prepare validation data
        validation_x_dict = {
            "peptide": self.peptides_to_network_input(validation_peptide_encoding),
        }
        if validation_allele_input is not None:
            validation_x_dict["allele"] = validation_allele_input
        encode_y_kwargs = {}
        if validation_inequalities is not None:
            encode_y_kwargs["inequalities"] = validation_inequalities
        if validation_output_indices is not None:
            encode_y_kwargs["output_indices"] = validation_output_indices

        output = loss_obj.encode_y(from_ic50(validation_affinities), **encode_y_kwargs)

        # --- Validation tensors cached on device (Phase 1 #268 for fit_generator) ---
        # The validation set is constant across pretrain epochs, but the
        # original code re-ran ``torch.from_numpy(...).float().to(device)``
        # on every epoch — three H2D copies × many epochs. For large
        # validation sets that's tens of ms/epoch of pure wasted bandwidth.
        # Hoist the copy to happen once before the epoch loop.
        #
        # ``.to(device).float()`` (GPU-side widening cast; see Phase 4a
        # of #268): when the encoding cache stores int8 (BLOSUM62 native
        # width), H2D transfer is 4× smaller; widening to fp32 on GPU is
        # essentially free. For fp32-native encodings it's a no-op.
        # Phase 2 (#268): 2D int → index-encoded; keep dtype intact so
        # the embedding lookup inside forward() sees int indices. 3D int
        # is the Phase 4a int8 BLOSUM cache payload; widen as before.
        _val_peptide_np = validation_x_dict["peptide"]
        if (
            _val_peptide_np.ndim == 2
            and numpy.issubdtype(_val_peptide_np.dtype, numpy.integer)
        ):
            val_peptide_device = torch.from_numpy(_val_peptide_np).to(device)
        else:
            val_peptide_device = torch.from_numpy(_val_peptide_np).to(device).float()
        val_allele_device = None
        if "allele" in validation_x_dict:
            val_allele_device = torch.from_numpy(
                validation_x_dict["allele"]
            ).to(device).float()
        val_y_device = torch.from_numpy(output.astype(numpy.float32)).to(device)

        # fit_generator batches stay as numpy arrays until the parent
        # training loop moves them to the device. That keeps worker-side
        # prefetch compatible across platforms and still overlaps CSV /
        # encoding work with GPU compute when a picklable encoded-batch
        # generator_factory is supplied (the pretrain path).
        mutable_generator_state = {"yielded_values": 0}
        dataloader_num_workers = self.hyperparameters.get(
            "dataloader_num_workers", 0
        )
        # fit_generator's DataLoader intentionally transports numpy arrays
        # unchanged (``batch_size=None`` + ``_identity_collate``). Those arrays
        # are not pinned, and setting non_blocking=True for pageable
        # numpy-backed tensors lets CUDA retain source pages until later stream
        # synchronization. Keep H2D copies synchronous so each pretrain chunk's
        # CPU buffer can be released/reused immediately.
        non_blocking_h2d = False
        fit_info["dataloader_num_workers"] = dataloader_num_workers
        fit_info["validation_rows"] = len(validation_affinities)
        # Worker-prefetch requires both (a) a picklable ``generator_factory``
        # so each spawned worker can build its own shard, and (b)
        # ``generator_batches_are_encoded=True`` so the dataset never
        # holds bound methods of this ``Class1NeuralNetwork``. Either
        # missing piece forces a single-process path.
        if dataloader_num_workers > 0 and (
            generator_factory is None or not generator_batches_are_encoded
        ):
            logging.warning(
                "fit_generator requested dataloader_num_workers=%s but "
                "worker-prefetch needs generator_factory + "
                "generator_batches_are_encoded=True (got factory=%s, "
                "encoded=%s); downgrading to 0.",
                dataloader_num_workers,
                "present" if generator_factory is not None else "missing",
                generator_batches_are_encoded,
            )
            dataloader_num_workers = 0
        dataset = _FitGeneratorBatchIterableDataset(
            generator=generator,
            generator_factory=generator_factory,
            source_batches_are_encoded=generator_batches_are_encoded,
            allele_encoding_to_input=self.allele_encoding_to_network_input,
            peptides_to_network_input=self.peptides_to_network_input,
        )

        start = time.time()
        iterator_setup_start = time.perf_counter()
        iterator = iter(
            _make_fit_generator_dataloader(
                dataset=dataset,
                num_workers=dataloader_num_workers,
            )
        )
        fit_info["iterator_setup_time"] = (
            time.perf_counter() - iterator_setup_start
        )

        # Data dependent init
        data_dependent_init = self.hyperparameters[
            "data_dependent_initialization_method"
        ]
        if data_dependent_init and not self.fit_info:
            first_chunk = next(iterator)
            init_chunk = _materialize_repeated_peptide_batch(first_chunk)
            first_inputs = {"peptide": init_chunk["peptide"]}
            if "allele" in init_chunk:
                first_inputs["allele"] = init_chunk["allele"]
            self.data_dependent_weights_initialization(
                network,
                first_inputs,
                method=data_dependent_init,
                verbose=verbose,
            )
            iterator = itertools.chain([first_chunk], iterator)

        # Compile AFTER LSUV init — LSUV registers + removes forward hooks
        # during its activation-norm measurement pass, and every hook-count
        # change invalidates dynamo's frame guard (``len(L['self']
        # .dense_layers[0]._forward_hooks) != 1``). Compiling before LSUV
        # burns through the 8-entry cache_size_limit in seconds and falls
        # back to eager. Compiling after gives dynamo a single stable
        # hook state to specialize on.
        network = _maybe_compile_network(network, device)
        eager_network = _uncompiled_network(network)
        # Compile the loss alongside the network — see _maybe_compile_loss
        # docstring. MSEWithInequalities eager-dispatches ~10 kernels
        # per step; fusing them matters more as batch size grows (less
        # compute to amortize launch overhead against).
        loss_obj = _maybe_compile_loss(loss_obj, device)

        min_val_loss_iteration = None
        min_val_loss = None
        last_progress_print = 0
        epoch = 1
        first_batch_time = None

        # The first batch establishes the compiled fast-path shape.
        # Non-conforming later batches (most commonly a short final
        # pretrain chunk) run eagerly to preserve exact semantics without
        # triggering a second compile.
        expected_chunk_shape = None

        while True:
            epoch_start_time = time.time()
            epoch_wall_start = time.perf_counter()
            network.train()

            epoch_losses = []
            epoch_fetch_time = 0.0
            epoch_h2d_time = 0.0
            epoch_train_time = 0.0
            epoch_num_train_rows = 0
            for step in range(steps_per_epoch):
                fetch_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                epoch_fetch_time += time.perf_counter() - fetch_start

                if expected_chunk_shape is None:
                    expected_chunk_shape = batch["peptide"].shape
                # Phase 0 timing (#268): measure H2D separately from the
                # training compute. Both paths pay a cuda.synchronize per
                # batch when MHCFLURRY_ENABLE_TIMING=1; when disabled the
                # _timing_start/stop calls are just time.perf_counter and
                # the two sub-timers collapse into one.
                h2d_start = _timing_start(device, timing_enabled)
                inputs, y_tensor, weights_batch = _move_fit_batch_to_device(
                    batch,
                    device,
                    non_blocking=non_blocking_h2d,
                )
                epoch_h2d_time += _timing_stop(
                    h2d_start, device, timing_enabled
                )
                batch_start = _timing_start(device, timing_enabled)
                loss = _run_training_batch(
                    network=(
                        network
                        if batch["peptide"].shape == expected_chunk_shape
                        else eager_network
                    ),
                    optimizer=optimizer,
                    loss_obj=loss_obj,
                    regularization_parameters=regularization_parameters,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg,
                    inputs=inputs,
                    y_batch=y_tensor,
                    weights_batch=weights_batch,
                )
                batch_time = _timing_stop(batch_start, device, timing_enabled)
                epoch_train_time += batch_time
                if first_batch_time is None:
                    first_batch_time = batch_time
                mutable_generator_state["yielded_values"] += len(batch["y"])
                epoch_num_train_rows += len(batch["y"])
                epoch_losses.append(loss)

            # Compute validation loss in fixed-size batches — reuse the
            # device tensors hoisted before the epoch loop (val data is
            # static). Single-shot forward pass over the entire pretrain
            # validation set (typically 50K+ rows) was dominating VRAM
            # and defeating shape-stable optimization. See Phase 4b of
            # openvax/mhcflurry#268.
            network.eval()
            validation_time = 0.0
            with torch.inference_mode():
                validation_start = _timing_start(device, timing_enabled)
                val_batch_size = _effective_validation_batch_size(
                    device,
                    self.hyperparameters["validation_batch_size"],
                    self.hyperparameters["minibatch_size"],
                    model=eager_network,
                    num_workers_per_gpu=dataloader_num_workers + 1
                    if dataloader_num_workers else 1,
                )
                fit_info["effective_validation_batch_size"] = val_batch_size
                val_loss = _batched_validation_loss(
                    network=network,
                    eager_network=eager_network,
                    val_peptide=val_peptide_device,
                    val_allele=val_allele_device,
                    val_y=val_y_device,
                    val_weights=None,
                    loss_obj=loss_obj,
                    batch_size=val_batch_size,
                )
                regularization_penalty = self._regularization_penalty(
                    regularization_parameters,
                    l1=l1_reg,
                    l2=l2_reg,
                )
                if regularization_penalty is not None:
                    val_loss = val_loss + regularization_penalty.item()
                validation_time = _timing_stop(
                    validation_start, device, timing_enabled
                )

            epoch_time = time.time() - epoch_start_time
            # Single GPU→CPU sync per epoch for the accumulated per-step
            # loss tensors. See the ``epoch_losses.append(loss.detach())``
            # comment inside the step loop above. When timing is disabled
            # this .item() is the first CUDA sync of the epoch and blocks
            # on the entire queued training pass; when enabled we already
            # synced per-step so the drain is near-zero. Phase 0 of #268
            # measures it either way so the fit_info breakdown sums to
            # the wall clock.
            loss_sync_start = _timing_start(device, timing_enabled)
            train_loss = (
                torch.stack(epoch_losses).mean().item()
                if epoch_losses else float('nan')
            )
            epoch_loss_sync_time = _timing_stop(
                loss_sync_start, device, timing_enabled
            )
            fit_info["loss"].append(train_loss)
            fit_info["val_loss"].append(val_loss)
            if timing_enabled:
                fit_info["epoch_fetch_time"].append(epoch_fetch_time)
                fit_info["epoch_h2d_time"].append(epoch_h2d_time)
                fit_info["epoch_train_time"].append(epoch_train_time)
                fit_info["epoch_loss_sync_time"].append(epoch_loss_sync_time)
                fit_info["epoch_validation_time"].append(validation_time)
                fit_info["epoch_num_train_batches"].append(len(epoch_losses))
                fit_info["epoch_num_train_rows"].append(epoch_num_train_rows)
                fit_info["epoch_num_validation_batches"].append(
                    int(numpy.ceil(len(validation_affinities) / val_batch_size))
                )
                fit_info["epoch_total_time"].append(
                    time.perf_counter() - epoch_wall_start
                )

            if min_val_loss is None or val_loss < min_val_loss - min_delta:
                min_val_loss = val_loss
                min_val_loss_iteration = epoch

            patience_epoch_threshold = min(
                epochs, max(min_val_loss_iteration + patience, min_epochs)
            )

            progress_message = (
                "epoch %3d/%3d [%0.2f sec.]: loss=%g val_loss=%g. Min val "
                "loss %g at epoch %s. Cum. points: %d. Stop at epoch %d."
                % (
                    epoch,
                    epochs,
                    epoch_time,
                    train_loss,
                    val_loss,
                    min_val_loss,
                    min_val_loss_iteration,
                    mutable_generator_state["yielded_values"],
                    patience_epoch_threshold,
                )
            ).strip()

            if progress_print_interval is not None and (
                time.time() - last_progress_print > progress_print_interval
            ):
                print(progress_preamble, progress_message)
                last_progress_print = time.time()

            if progress_callback:
                progress_callback()

            if epoch >= patience_epoch_threshold:
                if progress_print_interval is not None:
                    print(progress_preamble, "STOPPING", progress_message)
                break
            epoch += 1

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = mutable_generator_state["yielded_values"]
        if first_batch_time is not None:
            fit_info["first_batch_time"] = first_batch_time
        self.fit_info.append(dict(fit_info))

    def _create_optimizer(self, network):
        """Create an optimizer for the network."""
        optimizer_name = self.hyperparameters["optimizer"].lower()
        lr = (
            self.hyperparameters["learning_rate"]
            if self.hyperparameters["learning_rate"] is not None
            else 0.001
        )

        if optimizer_name == "rmsprop":
            # Match Keras defaults: rho=0.9, epsilon=1e-07
            return torch.optim.RMSprop(
                network.parameters(), lr=lr, alpha=0.9, eps=1e-07)
        elif optimizer_name == "adam":
            # Match Keras default epsilon=1e-07.
            return torch.optim.Adam(network.parameters(), lr=lr, eps=1e-07)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(network.parameters(), lr=lr)
        else:
            return torch.optim.Adam(network.parameters(), lr=lr, eps=1e-07)

    def fit(
            self,
            peptides,
            affinities,
            allele_encoding=None,
            inequalities=None,
            output_indices=None,
            sample_weights=None,
            shuffle_permutation=None,
            verbose=1,
            progress_callback=None,
            progress_preamble="",
            progress_print_interval=5.0,
            random_negative_seed=None):
        """
        Fit the neural network.

        Parameters
        ----------
        peptides : EncodableSequences or list of string
        affinities : list of float
            nM affinities. Must be same length of as peptides.
        allele_encoding : AlleleEncoding
            If not specified, the model will be a single-allele predictor.
        inequalities : list of string, each element one of ">", "<", or "=".
        output_indices : list of int
            For multi-output models only.
        sample_weights : list of float
        shuffle_permutation : list of int
        verbose : int
        progress_callback : function
        progress_preamble : string
        progress_print_interval : float
        """
        device = self.get_device()
        _configure_matmul_precision(device)

        encodable_peptides = EncodableSequences.create(peptides)
        peptide_encoding = self.peptides_to_network_input(encodable_peptides)
        fit_info = collections.defaultdict(list)
        timing_enabled = _timing_enabled()
        fit_info["timing_enabled"] = timing_enabled
        fit_info["dataloader_num_workers"] = self.hyperparameters.get(
            "dataloader_num_workers", 0
        )

        random_negatives_planner = RandomNegativePeptides(
            **RandomNegativePeptides.hyperparameter_defaults.subselect(
                self.hyperparameters
            )
        )
        random_negatives_planner.plan(
            peptides=encodable_peptides.sequences,
            affinities=affinities,
            alleles=allele_encoding.alleles if allele_encoding else None,
            inequalities=inequalities,
        )

        random_negatives_allele_encoding = None
        if allele_encoding is not None:
            random_negatives_allele_encoding = AlleleEncoding(
                random_negatives_planner.get_alleles(), borrow_from=allele_encoding
            )
        num_random_negatives = random_negatives_planner.get_total_count()

        # Phase 1 of issue openvax/mhcflurry#268 — pre-generate the random-
        # negative peptides + encoding once per pool-cycle rather than once
        # per epoch. At pool_epochs=1 the pool regenerates every epoch and
        # the behavior is semantically identical to the pre-Phase-1 path.
        random_negative_pool_epochs = int(
            self.hyperparameters.get("random_negative_pool_epochs", 1) or 1
        )
        if random_negative_pool_epochs < 1:
            random_negative_pool_epochs = 1
        # Semantics: ``random_negative_seed`` is the pool's cross-cycle
        # determinism knob — ignore it when pool_epochs == 1 so the
        # default path stays on numpy's global RNG stream (the pre-
        # Phase-1 behavior). Only actually seed the pool when pooling
        # is enabled; otherwise a seed passed by the training driver
        # would silently change default-path training semantics to
        # deterministic-per-work-item, which is a prediction-affecting
        # change. See openvax/mhcflurry#270 code review.
        pool_seed = (
            random_negative_seed
            if random_negative_pool_epochs > 1
            else None
        )
        random_negatives_pool = RandomNegativesPool(
            planner=random_negatives_planner,
            peptide_encoder=self.peptides_to_network_input,
            pool_epochs=random_negative_pool_epochs,
            seed=pool_seed,
        )
        fit_info["random_negative_pool_epochs"] = random_negative_pool_epochs

        # Allele encoding for random negatives is planned once (the allele
        # list is a deterministic function of the planner's plan_df). Hoist
        # it out of the epoch loop — prior to Phase 1 this was recomputed
        # every epoch on a constant input, which was harmless but wasteful.
        random_negative_x_allele_base = None
        if (
            num_random_negatives > 0
            and random_negatives_allele_encoding is not None
        ):
            (
                random_negative_x_allele_base,
                _,
            ) = self.allele_encoding_to_network_input(
                random_negatives_allele_encoding
            )

        y_values = from_ic50(numpy.asarray(affinities))
        assert numpy.isnan(y_values).sum() == 0, y_values

        if inequalities is not None:
            adjusted_inequalities = (
                pandas.Series(inequalities)
                .map({
                    "=": "=",
                    ">": "<",
                    "<": ">",
                })
                .values
            )
        else:
            adjusted_inequalities = numpy.tile("=", len(y_values))

        if len(adjusted_inequalities) != len(y_values):
            raise ValueError("Inequalities and y_values must have same length")

        x_dict_without_random_negatives = {
            "peptide": peptide_encoding,
        }
        allele_representations = None
        if allele_encoding is not None:
            (
                allele_encoding_input,
                allele_representations,
            ) = self.allele_encoding_to_network_input(allele_encoding)
            x_dict_without_random_negatives["allele"] = allele_encoding_input

        # Shuffle
        if shuffle_permutation is None:
            shuffle_permutation = numpy.random.permutation(len(y_values))
        y_values = y_values[shuffle_permutation]
        adjusted_inequalities = adjusted_inequalities[shuffle_permutation]
        for key in x_dict_without_random_negatives:
            x_dict_without_random_negatives[key] = x_dict_without_random_negatives[key][
                shuffle_permutation
            ]
        if sample_weights is not None:
            sample_weights = numpy.array(sample_weights, copy=False)[shuffle_permutation]
        if output_indices is not None:
            output_indices = numpy.array(output_indices, copy=False)[shuffle_permutation]

        loss_obj = get_pytorch_loss(self.hyperparameters["loss"])

        if not loss_obj.supports_inequalities and (
            any(inequality != "=" for inequality in adjusted_inequalities)
        ):
            raise ValueError("Loss %s does not support inequalities" % loss_obj)

        if (
            not loss_obj.supports_multiple_outputs
            and output_indices is not None
            and (output_indices != 0).any()
        ):
            raise ValueError("Loss %s does not support multiple outputs" % loss_obj)

        if self.hyperparameters["num_outputs"] != 1:
            if output_indices is None:
                raise ValueError("Must supply output_indices for multi-output predictor")

        if self.network() is None:
            self._network = self.make_network(
                allele_representations=allele_representations,
                **self.network_hyperparameter_defaults.subselect(self.hyperparameters)
            )
            if verbose > 0:
                print(self.network())

        network = self.network()
        network.to(device)

        if allele_representations is not None:
            self.set_allele_representations(allele_representations)

        # Guard against silent training-time OOMs. When many workers
        # share one GPU (pan-allele default max_workers_per_gpu=2 on
        # A100-80GB), the naive minibatch_size from the hyperparameters
        # YAML may not fit per-worker. Shrink loudly for the duration
        # of this fit() call only — DO NOT mutate self.hyperparameters,
        # so the saved model config preserves the user's original
        # intent. fit_info carries the actual value used at run time.
        # Issue openvax/mhcflurry#272.
        _requested_minibatch = int(self.hyperparameters["minibatch_size"])
        _effective_minibatch, _shrunk = check_training_batch_fits(
            _requested_minibatch,
            device,
            network,
            num_workers_per_gpu=_env_workers_per_gpu(1),
        )
        if _shrunk:
            fit_info["minibatch_size_shrunk_from"] = _requested_minibatch
            fit_info["minibatch_size_shrunk_to"] = _effective_minibatch
        fit_info["effective_minibatch_size"] = _effective_minibatch

        optimizer = self._create_optimizer(network)
        if self.hyperparameters["learning_rate"] is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.hyperparameters["learning_rate"]
        fit_info["learning_rate"] = optimizer.param_groups[0]['lr']

        # Prepare y values with random negatives
        if loss_obj.supports_inequalities:
            random_negative_ic50 = self.hyperparameters["random_negative_affinity_min"]
            random_negative_target = from_ic50(random_negative_ic50)

            y_with_negatives = numpy.concatenate([
                numpy.tile(random_negative_target, num_random_negatives),
                y_values,
            ])
            adjusted_inequalities_with_random_negatives = (
                ["<"] * num_random_negatives + list(adjusted_inequalities)
            )
        else:
            y_with_negatives = numpy.concatenate([
                from_ic50(
                    numpy.random.uniform(
                        self.hyperparameters["random_negative_affinity_min"],
                        self.hyperparameters["random_negative_affinity_max"],
                        num_random_negatives,
                    )
                ),
                y_values,
            ])
            adjusted_inequalities_with_random_negatives = None

        if sample_weights is not None:
            sample_weights_with_negatives = numpy.concatenate([
                numpy.ones(num_random_negatives),
                sample_weights
            ])
        else:
            sample_weights_with_negatives = None

        if output_indices is not None:
            random_negative_output_indices = (
                self.hyperparameters["random_negative_output_indices"]
                if self.hyperparameters["random_negative_output_indices"]
                else list(range(0, self.hyperparameters["num_outputs"]))
            )
            output_indices_with_negatives = numpy.concatenate([
                pandas.Series(random_negative_output_indices, dtype=int)
                .sample(n=num_random_negatives, replace=True)
                .values,
                output_indices,
            ])
        else:
            output_indices_with_negatives = None

        # Encode y
        encode_y_kwargs = {}
        if adjusted_inequalities_with_random_negatives is not None:
            encode_y_kwargs["inequalities"] = adjusted_inequalities_with_random_negatives
        if output_indices_with_negatives is not None:
            encode_y_kwargs["output_indices"] = output_indices_with_negatives

        y_encoded = loss_obj.encode_y(y_with_negatives, **encode_y_kwargs)

        min_val_loss_iteration = None
        min_val_loss = None

        needs_initialization = (
            self.hyperparameters["data_dependent_initialization_method"] is not None
            and not self.fit_info
        )

        start = time.time()
        last_progress_print = None
        first_batch_time = None

        # Validation split (fixed across epochs; only training data is reshuffled)
        val_split = self.hyperparameters["validation_split"]
        n_total = len(y_encoded)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        indices = numpy.arange(n_total)
        if n_val > 0:
            train_indices_base = indices[:n_train]
            val_indices = indices[n_train:]
        else:
            train_indices_base = indices
            val_indices = None
        fit_info["train_rows"] = int(n_train)
        fit_info["validation_rows"] = int(n_val)
        fit_info["num_random_negatives"] = int(num_random_negatives)

        regularization_parameters = tuple(self._regularized_parameters(network))
        l1_reg = self.hyperparameters["dense_layer_l1_regularization"]
        l2_reg = self.hyperparameters["dense_layer_l2_regularization"]

        # --- Validation tensors cached on device (issue openvax/mhcflurry#268) ---
        # The validation set indexes into the concatenated
        # [random_negs | training] array via val_indices. The training portion
        # is static across epochs; the random-negative portion changes.
        # When val_indices points entirely into the training portion (the
        # common case — with default val_split=0.1 and random_negative_rate=1,
        # val set is the tail 10% of the ~2x-size concatenated array, all in
        # training portion), x_peptide[val_indices] and x_allele[val_indices]
        # are stable across epochs. Materialize them on device once to save
        # ~60 MB+ H2D per epoch.
        #
        # When overlap IS possible (unusual hyperparameter combos), skip the
        # cache and fall back to per-epoch copy (preserved behavior).
        # Check EVERY val index, not just the first — val_indices can be
        # unsorted (scikit-learn's train_test_split sometimes produces
        # shuffled indices). A leading value ≥ num_random_negatives does
        # not guarantee all subsequent values are in the training portion;
        # the weight/peptide slicing below would then index into the
        # random-negative portion, which mutates per epoch → stale tensors.
        _val_cache_safe = (
            val_indices is not None
            and len(val_indices) > 0
            and bool(numpy.all(val_indices >= num_random_negatives))
        )
        fit_info["validation_cache_reused"] = _val_cache_safe
        _val_device_tensors = None
        if _val_cache_safe:
            val_training_indices = val_indices - num_random_negatives
            # Phase 2 (#268): 2D int is index-encoded (keep dtype); 3D int
            # is the Phase 4a BLOSUM int8 cache payload (widen to fp32).
            _val_peptide_np = x_dict_without_random_negatives["peptide"][
                val_training_indices
            ]
            if (
                _val_peptide_np.ndim == 2
                and numpy.issubdtype(_val_peptide_np.dtype, numpy.integer)
            ):
                _val_peptide_device = torch.from_numpy(_val_peptide_np).to(device)
            else:
                _val_peptide_device = torch.from_numpy(_val_peptide_np).float().to(device)
            _val_y_device = torch.from_numpy(
                y_encoded[val_indices].astype(numpy.float32)
            ).to(device)
            _val_allele_device = None
            if "allele" in x_dict_without_random_negatives:
                _val_allele_device = torch.from_numpy(
                    x_dict_without_random_negatives["allele"][val_training_indices]
                ).float().to(device)
            _val_weights_device = None
            if sample_weights_with_negatives is not None:
                _val_weights_device = torch.from_numpy(
                    sample_weights_with_negatives[val_indices]
                ).float().to(device)
            _val_device_tensors = (
                _val_peptide_device,
                _val_y_device,
                _val_allele_device,
                _val_weights_device,
            )

        for epoch in range(self.hyperparameters["max_epochs"]):
            epoch_wall_start = time.perf_counter()
            input_build_time = 0.0
            initialization_time = 0.0
            shuffle_dataset_time = 0.0
            dataloader_setup_time = 0.0
            train_loop_time = 0.0
            epoch_h2d_time = 0.0
            epoch_loss_sync_time = 0.0
            validation_materialize_time = 0.0
            validation_compute_time = 0.0
            callback_time = 0.0
            gc_time = 0.0

            build_start = time.perf_counter()
            # Phase 1 (#268): slice the per-epoch negatives out of a
            # pre-encoded pool. On pool_epochs=1 this regenerates every
            # epoch (legacy behavior). On pool_epochs=N the heavy
            # encoding pass runs once per N epochs; the in-epoch call
            # here is an O(1) array view.
            _, random_negative_peptides_encoding = (
                random_negatives_pool.get_epoch_inputs(epoch)
            )
            dataset = _FitBatchDataset(
                x_peptide=x_dict_without_random_negatives["peptide"],
                x_allele=x_dict_without_random_negatives.get("allele"),
                y_encoded=y_encoded,
                sample_weights_with_negatives=sample_weights_with_negatives,
                train_indices=None,
                random_negative_x_peptide=random_negative_peptides_encoding,
                random_negative_x_allele=random_negative_x_allele_base,
                num_random_negatives=num_random_negatives,
            )
            input_build_time += time.perf_counter() - build_start

            if needs_initialization:
                init_start = _timing_start(device, timing_enabled)
                self.data_dependent_weights_initialization(
                    network,
                    dataset.batch_for_indices(indices),
                    method=self.hyperparameters["data_dependent_initialization_method"],
                    verbose=verbose,
                )
                initialization_time += _timing_stop(
                    init_start, device, timing_enabled
                )
                needs_initialization = False

            # Compile AFTER LSUV hook churn finishes (see fit_generator
            # comment above). Idempotent — _maybe_compile_network returns
            # the OptimizedModule unchanged if ``network`` is already
            # compiled, so it's safe to call every epoch. First epoch's
            # first batch pays the codegen cost; rest runs compiled.
            network = _maybe_compile_network(network, device)
            eager_network = _uncompiled_network(network)
            # Same rationale as fit_generator's loss-compile call.
            loss_obj = _maybe_compile_loss(loss_obj, device)

            # Train/val split (keep validation fixed)
            dataset_start = time.perf_counter()
            train_indices = train_indices_base.copy()
            numpy.random.shuffle(train_indices)
            dataset.train_indices = train_indices
            shuffle_dataset_time += time.perf_counter() - dataset_start

            # Training
            network.train()
            epoch_start = time.time()

            # Create batches via DataLoader — with num_workers=0 this is
            # bit-identical to the old single-process inline-iteration path
            # (DataLoader wraps the same fancy-indexing logic with no
            # reordering). With num_workers>0, prefetcher processes build
            # the next batch while the GPU crunches the current one.
            # See issue openvax/mhcflurry#268.
            batch_size = _effective_minibatch
            dataloader_num_workers = self.hyperparameters.get(
                "dataloader_num_workers", 0
            )
            # _make_fit_dataloader keeps worker batches as numpy arrays to
            # avoid PyTorch CPU tensor/pinned-memory allocator growth. These
            # sources are pageable, so H2D copies must remain blocking.
            use_pinned_memory = False
            non_blocking_h2d = False

            train_losses = []
            epoch_num_train_rows = 0
            full_batch_count = (len(train_indices) // batch_size) * batch_size
            if full_batch_count > 0:
                dataloader_setup_start = time.perf_counter()
                fit_dataset = _FitBatchDataset(
                    x_peptide=x_dict_without_random_negatives["peptide"],
                    x_allele=x_dict_without_random_negatives.get("allele"),
                    y_encoded=dataset.y_encoded,
                    sample_weights_with_negatives=sample_weights_with_negatives,
                    train_indices=train_indices[:full_batch_count],
                    random_negative_x_peptide=random_negative_peptides_encoding,
                    random_negative_x_allele=random_negative_x_allele_base,
                    num_random_negatives=num_random_negatives,
                )
                (
                    effective_fit_dataloader_num_workers,
                    fit_dataloader_downgrade_reason,
                ) = _effective_fit_dataloader_num_workers(
                    dataloader_num_workers,
                    fit_dataset,
                )
                fit_info["fit_dataloader_num_workers"] = (
                    effective_fit_dataloader_num_workers
                )
                if fit_dataloader_downgrade_reason is not None:
                    fit_info["fit_dataloader_downgrade_reason"] = (
                        fit_dataloader_downgrade_reason
                    )
                loader = _make_fit_dataloader(
                    dataset=fit_dataset,
                    batch_size=batch_size,
                    num_workers=effective_fit_dataloader_num_workers,
                    use_pinned_memory=use_pinned_memory,
                    drop_last=False,
                )
                dataloader_setup_time += (
                    time.perf_counter() - dataloader_setup_start
                )

                for batch in loader:
                    # Phase 0 timing (#268): H2D timed separately from
                    # the training compute so the epoch breakdown adds
                    # up to wall-clock.
                    h2d_start = _timing_start(device, timing_enabled)
                    inputs, y_batch, weights_batch = _move_fit_batch_to_device(
                        batch,
                        device,
                        non_blocking=non_blocking_h2d,
                    )
                    epoch_h2d_time += _timing_stop(
                        h2d_start, device, timing_enabled
                    )
                    batch_start = _timing_start(device, timing_enabled)
                    loss = _run_training_batch(
                        network=network,
                        optimizer=optimizer,
                        loss_obj=loss_obj,
                        regularization_parameters=regularization_parameters,
                        l1_reg=l1_reg,
                        l2_reg=l2_reg,
                        inputs=inputs,
                        y_batch=y_batch,
                        weights_batch=weights_batch,
                    )
                    batch_time = _timing_stop(batch_start, device, timing_enabled)
                    train_loop_time += batch_time
                    if first_batch_time is None:
                        first_batch_time = batch_time
                    epoch_num_train_rows += len(batch["y"])
                    train_losses.append(loss)

            tail_indices = train_indices[full_batch_count:]
            if len(tail_indices) > 0:
                tail_batch = dataset.batch_for_indices(tail_indices)
                h2d_start = _timing_start(device, timing_enabled)
                inputs, y_batch, weights_batch = _move_fit_batch_to_device(
                    tail_batch,
                    device,
                    non_blocking=False,
                )
                epoch_h2d_time += _timing_stop(
                    h2d_start, device, timing_enabled
                )
                batch_start = _timing_start(device, timing_enabled)
                loss = _run_training_batch(
                    network=eager_network,
                    optimizer=optimizer,
                    loss_obj=loss_obj,
                    regularization_parameters=regularization_parameters,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg,
                    inputs=inputs,
                    y_batch=y_batch,
                    weights_batch=weights_batch,
                )
                train_loop_time += _timing_stop(batch_start, device, timing_enabled)
                epoch_num_train_rows += len(tail_batch["y"])
                train_losses.append(loss)

            epoch_time = time.time() - epoch_start
            # Single GPU→CPU sync per epoch over accumulated loss tensors.
            # Without timing, this .item() is the first sync of the
            # epoch and blocks on the entire queued training pass; with
            # timing we already synced per-batch so the drain is ~zero.
            # Phase 0 of #268 captures either regime so the fit_info
            # breakdown sums to the wall clock.
            loss_sync_start = _timing_start(device, timing_enabled)
            train_loss = (
                torch.stack(train_losses).mean().item()
                if train_losses else float('nan')
            )
            epoch_loss_sync_time = _timing_stop(
                loss_sync_start, device, timing_enabled
            )
            fit_info["loss"].append(train_loss)

            # Validation — batched so every GPU invocation is
            # fixed-size (torch.compile-friendly) and peak VRAM stays
            # bounded regardless of n_val. See Phase 4b of #268.
            #
            # ``validation_interval`` >1 skips the val pass on
            # off-interval epochs to save the GPU-sync barrier and
            # ~150 ms of forward-pass time. The skipped epochs reuse
            # the most recent measured val_loss so fit_info["val_loss"]
            # stays one-entry-per-epoch for downstream plotting tools.
            # The final epoch (max_epochs - 1) is always measured so
            # the saved model reflects an up-to-date val_loss.
            validation_interval = max(
                1, int(self.hyperparameters.get("validation_interval", 1) or 1)
            )
            is_last_epoch = epoch == self.hyperparameters["max_epochs"] - 1
            should_validate_this_epoch = (
                val_split > 0
                and (epoch % validation_interval == 0 or is_last_epoch)
            )
            if val_split > 0 and not should_validate_this_epoch:
                # Pad val_loss with the previous measurement so fit_info
                # arrays stay aligned with epoch index. ``val_loss`` is
                # also reused for the early-stop check below — since it
                # equals the previous measurement, min_val_loss won't
                # spuriously update on a skipped epoch.
                prev_val_loss = (
                    fit_info["val_loss"][-1]
                    if fit_info["val_loss"] else float("nan")
                )
                fit_info["val_loss"].append(prev_val_loss)
                val_loss = prev_val_loss
            if should_validate_this_epoch:
                network.eval()
                with torch.inference_mode():
                    if _val_device_tensors is not None:
                        # Fast path: reuse tensors materialized once before the
                        # epoch loop. Saves ~60 MB+ H2D copy per epoch. Bit-
                        # identical because val_indices points entirely into
                        # the static training portion of x_peptide/x_allele.
                        (
                            val_peptide,
                            val_y,
                            val_allele,
                            val_weights,
                        ) = _val_device_tensors
                    else:
                        materialize_start = time.perf_counter()
                        val_batch = dataset.batch_for_indices(val_indices)
                        validation_materialize_time += (
                            time.perf_counter() - materialize_start
                        )
                        val_peptide = _batch_value_to_device(
                            val_batch["peptide"],
                            device,
                            non_blocking=False,
                            cast_float=True,
                        )
                        val_y = _batch_value_to_device(
                            val_batch["y"],
                            device,
                            non_blocking=False,
                            cast_float=False,
                        )
                        val_allele = None
                        if "allele" in val_batch:
                            val_allele = _batch_value_to_device(
                                val_batch["allele"],
                                device,
                                non_blocking=False,
                                cast_float=True,
                            )
                        val_weights = None
                        if "weight" in val_batch:
                            val_weights = _batch_value_to_device(
                                val_batch["weight"],
                                device,
                                non_blocking=False,
                                cast_float=True,
                            )
                    val_batch_size = _effective_validation_batch_size(
                        device,
                        self.hyperparameters["validation_batch_size"],
                        batch_size,
                        model=eager_network,
                        num_workers_per_gpu=dataloader_num_workers + 1
                        if dataloader_num_workers else 1,
                    )
                    fit_info["effective_validation_batch_size"] = val_batch_size
                    validation_start = _timing_start(device, timing_enabled)
                    val_loss = _batched_validation_loss(
                        network=network,
                        eager_network=eager_network,
                        val_peptide=val_peptide,
                        val_allele=val_allele,
                        val_y=val_y,
                        val_weights=val_weights,
                        loss_obj=loss_obj,
                        batch_size=val_batch_size,
                    )
                    regularization_penalty = self._regularization_penalty(
                        regularization_parameters,
                        l1=l1_reg,
                        l2=l2_reg,
                    )
                    if regularization_penalty is not None:
                        val_loss = val_loss + regularization_penalty.item()
                    validation_compute_time += _timing_stop(
                        validation_start, device, timing_enabled
                    )
                fit_info["val_loss"].append(val_loss)

            # Progress printing
            if progress_print_interval is not None and (
                not last_progress_print
                or (time.time() - last_progress_print > progress_print_interval)
            ):
                print(
                    (
                        progress_preamble
                        + " "
                        + "Epoch %3d / %3d [%0.2f sec]: loss=%g. "
                        "Min val loss (%s) at epoch %s"
                        % (
                            epoch,
                            self.hyperparameters["max_epochs"],
                            epoch_time,
                            train_loss,
                            str(min_val_loss),
                            min_val_loss_iteration,
                        )
                    ).strip()
                )
                last_progress_print = time.time()

            # Early stopping. ``min_val_loss`` / ``min_val_loss_iteration``
            # only update on epochs where validation actually ran — on
            # skipped epochs ``val_loss`` is the carried-forward previous
            # measurement and would never beat the current min anyway, so
            # restricting the update is for clarity (the patience counter
            # is anchored to the epoch the measurement was taken, not a
            # later epoch that copied the same value).
            if val_split > 0:
                if should_validate_this_epoch and (
                    min_val_loss is None
                    or val_loss < min_val_loss - self.hyperparameters["min_delta"]
                ):
                    min_val_loss = val_loss
                    min_val_loss_iteration = epoch

                if self.hyperparameters["early_stopping"]:
                    threshold = min_val_loss_iteration + self.hyperparameters["patience"]
                    if epoch > threshold:
                        if progress_print_interval is not None:
                            print(
                                (
                                    progress_preamble
                                    + " "
                                    + "Stopping at epoch %3d / %3d: loss=%g. "
                                    "Min val loss (%g) at epoch %s"
                                    % (
                                        epoch,
                                        self.hyperparameters["max_epochs"],
                                        train_loss,
                                        min_val_loss if min_val_loss is not None else numpy.nan,
                                        min_val_loss_iteration,
                                    )
                                ).strip()
                            )
                        break

            if progress_callback:
                callback_start = time.perf_counter()
                progress_callback()
                callback_time += time.perf_counter() - callback_start

            gc_start = time.perf_counter()
            gc.collect()
            gc_time += time.perf_counter() - gc_start
            if timing_enabled:
                fit_info["epoch_input_build_time"].append(input_build_time)
                fit_info["epoch_initialization_time"].append(
                    initialization_time
                )
                fit_info["epoch_shuffle_dataset_time"].append(
                    shuffle_dataset_time
                )
                fit_info["epoch_dataloader_setup_time"].append(
                    dataloader_setup_time
                )
                fit_info["epoch_h2d_time"].append(epoch_h2d_time)
                fit_info["epoch_train_time"].append(train_loop_time)
                fit_info["epoch_loss_sync_time"].append(epoch_loss_sync_time)
                fit_info["epoch_validation_materialize_time"].append(
                    validation_materialize_time
                )
                fit_info["epoch_validation_time"].append(
                    validation_compute_time
                )
                fit_info["epoch_num_train_batches"].append(len(train_losses))
                fit_info["epoch_num_train_rows"].append(epoch_num_train_rows)
                fit_info["epoch_tail_train_rows"].append(len(tail_indices))
                fit_info["epoch_num_validation_batches"].append(
                    int(numpy.ceil(n_val / val_batch_size)) if n_val > 0 else 0
                )
                fit_info["epoch_callback_time"].append(callback_time)
                fit_info["epoch_gc_time"].append(gc_time)
                fit_info["epoch_total_time"].append(
                    time.perf_counter() - epoch_wall_start
                )

        fit_info["time"] = time.time() - start
        fit_info["num_points"] = len(peptides)
        if first_batch_time is not None:
            fit_info["first_batch_time"] = first_batch_time
        self.fit_info.append(dict(fit_info))

    def predict(
            self,
            peptides,
            allele_encoding=None,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE,
            output_index=0,
            num_workers_per_gpu=1):
        """
        Predict affinities.

        Parameters
        ----------
        peptides : EncodableSequences or list of string
        allele_encoding : AlleleEncoding, optional
        batch_size : int or ``"auto"``
            ``"auto"`` (the default) sizes batches to the available GPU
            memory at call time — see ``compute_prediction_batch_size``.
            Pass an explicit int to pin the size.
        output_index : int or None
        num_workers_per_gpu : int
            When multiple training/calibration workers are co-resident on
            the same CUDA device, pass the worker count so the auto-
            sizer partitions the VRAM budget. Ignored for explicit int
            batch_size.

        Returns
        -------
        numpy.array of nM affinity predictions
        """
        assert self.prediction_cache is not None
        use_cache = allele_encoding is None and isinstance(peptides, EncodableSequences)
        if use_cache and peptides in self.prediction_cache:
            return self.prediction_cache[peptides].copy()

        device = self.get_device()
        _configure_matmul_precision(device)

        x_dict = {"peptide": self.peptides_to_network_input(peptides)}

        if allele_encoding is not None:
            (
                allele_encoding_input,
                allele_representations,
            ) = self.allele_encoding_to_network_input(allele_encoding)
            x_dict["allele"] = allele_encoding_input
            self.set_allele_representations(allele_representations)
            network = self.network()
        else:
            network = self.network(borrow=True)

        network.to(device)
        network.eval()

        # Resolve ``"auto"`` once the network is on device so the
        # heuristic has final visibility into VRAM + architecture.
        batch_size = resolve_prediction_batch_size(
            batch_size,
            device,
            model=network,
            num_workers_per_gpu=num_workers_per_gpu,
        )

        # Batch prediction
        n_samples = len(x_dict["peptide"])
        all_predictions = []

        peptide_is_indices = self.hyperparameters.get(
            "peptide_amino_acid_encoding_gpu", False
        )

        def prediction_tensor(batch_array):
            batch_array = numpy.asarray(batch_array)
            if not batch_array.flags.writeable:
                batch_array = batch_array.copy()
            # Phase 2 (#268): keep integer dtype only on the index-encoded
            # peptide path. Allele inputs and Phase-4a 3D int8 BLOSUM
            # arrays still widen to fp32 as before — the flag only
            # governs the peptide tensor, not allele IDs or legacy
            # BLOSUM-int8 payloads.
            keep_int = (
                peptide_is_indices
                and batch_array.ndim == 2
                and numpy.issubdtype(batch_array.dtype, numpy.integer)
            )
            if not keep_int:
                batch_array = numpy.asarray(batch_array, dtype=numpy.float32)
            return torch.from_numpy(batch_array).to(device)

        with torch.no_grad():
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)

                peptide_batch = prediction_tensor(
                    x_dict["peptide"][batch_start:batch_end]
                )

                inputs = {"peptide": peptide_batch}
                if "allele" in x_dict:
                    allele_batch = prediction_tensor(
                        x_dict["allele"][batch_start:batch_end]
                    )
                    inputs["allele"] = allele_batch

                batch_predictions = network(inputs)
                all_predictions.append(batch_predictions.cpu().numpy())

        raw_predictions = numpy.concatenate(all_predictions, axis=0)
        predictions = numpy.array(raw_predictions, dtype="float64")

        if output_index is not None:
            predictions = predictions[:, output_index]

        result = to_ic50(predictions)
        if use_cache:
            self.prediction_cache[peptides] = result
        return result

    @classmethod
    def merge(cls, models, merge_method="average"):
        """
        Merge multiple models at the neural network level.

        Parameters
        ----------
        models : list of Class1NeuralNetwork
        merge_method : string, one of "average", "sum", or "concatenate"

        Returns
        -------
        Class1NeuralNetwork
        """
        if merge_method == "allele-specific":
            raise NotImplementedError("Allele-specific merge is not implemented")
        if len(models) == 1:
            return models[0]
        assert len(models) > 1
        if any(not model.network().has_allele for model in models):
            raise NotImplementedError("Merging allele-specific models is not implemented")

        # For now, we create a simple ensemble wrapper
        # that averages predictions
        result = Class1NeuralNetwork(**dict(models[0].hyperparameters))

        # Remove hyperparameters not shared by all models
        for model in models:
            for key, value in model.hyperparameters.items():
                if result.hyperparameters.get(key, value) != value:
                    del result.hyperparameters[key]

        # Create merged network
        result._network = MergedClass1NeuralNetwork(
            [model.network() for model in models],
            merge_method=merge_method
        )
        result.update_network_description()

        return result

    def make_network(
            self,
            peptide_encoding,
            allele_amino_acid_encoding,
            allele_dense_layer_sizes,
            peptide_dense_layer_sizes,
            peptide_allele_merge_method,
            peptide_allele_merge_activation,
            layer_sizes,
            dense_layer_l1_regularization,
            dense_layer_l2_regularization,
            activation,
            init,
            output_activation,
            dropout_probability,
            batch_normalization,
            locally_connected_layers,
            topology,
            num_outputs=1,
            allele_representations=None,
            peptide_amino_acid_encoding_gpu=False):
        """
        Helper function to make a PyTorch network for class 1 affinity prediction.
        """
        peptide_encoding_shape = self.peptides_to_network_input([]).shape[1:]
        # Phase 2 (#268): index-encoded peptides probe as 1D (L,), but the
        # network's dense layers still size against the post-embedding
        # (L, 21) shape — widen the shape here so the constructor sees
        # the same dims regardless of path.
        if peptide_amino_acid_encoding_gpu and len(peptide_encoding_shape) == 1:
            from .amino_acid import AMINO_ACIDS
            peptide_encoding_shape = (
                peptide_encoding_shape[0],
                len(AMINO_ACIDS),
            )

        return Class1NeuralNetworkModel(
            peptide_encoding_shape=peptide_encoding_shape,
            allele_representations=allele_representations,
            locally_connected_layers=locally_connected_layers,
            peptide_dense_layer_sizes=peptide_dense_layer_sizes,
            allele_dense_layer_sizes=allele_dense_layer_sizes,
            layer_sizes=layer_sizes,
            peptide_allele_merge_method=peptide_allele_merge_method,
            peptide_allele_merge_activation=peptide_allele_merge_activation,
            activation=activation,
            output_activation=output_activation,
            dropout_probability=dropout_probability,
            batch_normalization=batch_normalization,
            dense_layer_l1_regularization=dense_layer_l1_regularization,
            dense_layer_l2_regularization=dense_layer_l2_regularization,
            topology=topology,
            num_outputs=num_outputs,
            init=init,
            peptide_input_is_indices=peptide_amino_acid_encoding_gpu,
        )

    def clear_allele_representations(self):
        """
        Set allele representations to an empty array.
        """
        original_model = self.network()
        if original_model is not None and original_model.allele_embedding is not None:
            existing_shape = original_model.allele_embedding.weight.shape
            new_weight = numpy.zeros(
                shape=(1, existing_shape[1]),
                dtype=numpy.float32
            )
            target = original_model.allele_embedding.weight
            original_model.allele_embedding.weight.data = torch.from_numpy(
                new_weight
            ).to(device=target.device, dtype=target.dtype)
            original_model.allele_embedding.weight.requires_grad = False

    def set_allele_representations(self, allele_representations, force_surgery=False):
        """
        Set the allele representations in use by this model.

        Parameters
        ----------
        allele_representations : numpy.ndarray of shape (a, l, m)
        force_surgery : bool
        """
        network = self.network()
        if network is None:
            return

        if allele_representations is None:
            has_allele_embedding = False
            if isinstance(network, MergedClass1NeuralNetwork):
                has_allele_embedding = any(
                    sub_network.allele_embedding is not None
                    for sub_network in network.networks
                )
            else:
                has_allele_embedding = (
                    hasattr(network, 'allele_embedding') and
                    network.allele_embedding is not None
                )
            if has_allele_embedding:
                raise ValueError(
                    "set_allele_representations(None) called on a pan-allele "
                    "network"
                )
            return

        reshaped = allele_representations.reshape(
            (
                allele_representations.shape[0],
                numpy.prod(allele_representations.shape[1:]),
            )
        ).astype(numpy.float32)

        # Handle merged networks (ensembles)
        if isinstance(network, MergedClass1NeuralNetwork):
            for sub_network in network.networks:
                self._update_embedding(sub_network, reshaped, force_surgery)
        elif hasattr(network, 'allele_embedding') and network.allele_embedding is not None:
            self._update_embedding(network, reshaped, force_surgery)

    def _update_embedding(self, network, reshaped, force_surgery):
        """Update the allele embedding for a single network."""
        if network.allele_embedding is None:
            return

        target_weight = network.allele_embedding.weight
        existing_shape = target_weight.shape
        target_device = target_weight.device
        target_dtype = target_weight.dtype

        if existing_shape[0] > reshaped.shape[0] and not force_surgery:
            # Extend with NaNs
            reshaped = numpy.append(
                reshaped,
                numpy.ones([existing_shape[0] - reshaped.shape[0], reshaped.shape[1]])
                * numpy.nan,
                axis=0,
            )

        if existing_shape != reshaped.shape:
            # Need to resize embedding
            new_embedding = nn.Embedding(
                num_embeddings=reshaped.shape[0],
                embedding_dim=reshaped.shape[1]
            ).to(device=target_device)
            new_embedding.weight.data = torch.from_numpy(reshaped).to(
                device=target_device,
                dtype=target_dtype,
            )
            new_embedding.weight.requires_grad = False
            network.allele_embedding = new_embedding
        else:
            network.allele_embedding.weight.data = torch.from_numpy(
                reshaped
            ).to(device=target_device, dtype=target_dtype)
            network.allele_embedding.weight.requires_grad = False


class MergedClass1NeuralNetwork(nn.Module):
    """
    A merged ensemble of Class1NeuralNetworkModel instances.
    """

    def __init__(self, networks, merge_method="average"):
        super(MergedClass1NeuralNetwork, self).__init__()
        self.networks = nn.ModuleList(networks)
        self.merge_method = merge_method

    def forward(self, inputs):
        outputs = [network(inputs) for network in self.networks]
        stacked = torch.stack(outputs, dim=-1)

        if self.merge_method == "average":
            return stacked.mean(dim=-1)
        elif self.merge_method == "sum":
            return stacked.sum(dim=-1)
        elif self.merge_method == "concatenate":
            return torch.cat(outputs, dim=-1)
        else:
            raise ValueError(f"Unknown merge method: {self.merge_method}")

    def get_weights_list(self):
        """Get all weights as a flat list."""
        weights = []
        for network in self.networks:
            weights.extend(network.get_weights_list())
        return weights

    def set_weights_list(self, weights, auto_convert_keras=False):
        """Set weights from a flat list."""
        idx = 0
        for network in self.networks:
            n_weights = len(list(network.parameters())) + len(list(network.buffers()))
            network.set_weights_list(weights[idx:idx + n_weights], auto_convert_keras=auto_convert_keras)
            idx += n_weights

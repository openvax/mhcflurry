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

"""Training and validation helpers for class I neural networks."""

import logging
import multiprocessing
import os
import time

import numpy
import torch

from .regression_target import from_ic50


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


class _StreamingBatchIterableDataset(torch.utils.data.IterableDataset):
    """IterableDataset backing ``fit_streaming_batches``.

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
    ``fit_streaming_batches`` — so zeroing these out doesn't remove any
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
                "fit_streaming_batches with DataLoader workers requires a "
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
                    "fit_streaming_batches batches."
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


def _batch_value_to_device(value, device, *, non_blocking, cast_float):
    """Move a batch value (numpy array or tensor) to ``device``.

    When ``cast_float=True`` and the source is float64, cast to float32 on
    the CPU *before* transfer. MPS refuses to host float64 tensors, and
    transferring fp64 just to narrow it on-device wastes bandwidth anyway.
    int8 / float32 inputs skip the CPU-side cast so the pattern of shipping
    compact int encodings and widening on the torch device stays intact.

    When ``cast_float=False`` the value is shipped through in its source
    dtype — used by the device-side fixed peptide encoding path where
    peptide tensors are (N, L) int indices that get widened to (N, L, V)
    fp32 via embedding lookup inside the network's forward pass.
    """
    if isinstance(value, numpy.ndarray):
        value = _torch_from_numpy(value)
    if cast_float and value.dtype == torch.float64:
        value = value.float()
    value = value.to(device, non_blocking=non_blocking)
    if cast_float:
        value = value.float()
    return value


def _torch_from_numpy(value):
    if not value.flags.writeable:
        value = value.copy()
    return torch.from_numpy(value)


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


def _move_fit_batch_to_device(batch, device, *, non_blocking):
    """Move a training batch dict to ``device``.

    When ``batch["peptide"]`` is a 2D integer array the torch-side fixed
    peptide encoding path is active — keep it as int so the network's
    embedding lookup can consume it. A 3D integer array is a compact
    vector-encoded cache payload; still widen that to fp32 on device.
    Float arrays flow through as before.
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




def _sync_mean_loss(losses, *, device, timing_enabled):
    """Return mean detached loss with one device sync plus sync timing."""
    loss_sync_start = _timing_start(device, timing_enabled)
    train_loss = (
        torch.stack(losses).mean().item()
        if losses else float("nan")
    )
    loss_sync_time = _timing_stop(loss_sync_start, device, timing_enabled)
    return train_loss, loss_sync_time


def _validation_interval_from_hyperparameters(hyperparameters):
    """Return normalized validation interval for both training paths."""
    return max(1, int(hyperparameters.get("validation_interval", 1) or 1))


def _early_stop_reached(
    *,
    epoch_index,
    min_val_loss_epoch,
    patience,
    min_epochs=0,
    early_stopping=True,
    strict=True,
):
    """Shared early-stopping decision using zero-based epoch indices.

    With ``strict=True`` (the affinity ``fit()`` path) training stops the
    first epoch *strictly after* ``max(min_val_loss_epoch + patience,
    min_epochs - 1)`` — i.e. ``patience`` no-improvement epochs followed by
    one more before stopping. This matches the pre-2.3.0 ``fit()`` condition
    ``epoch > min_val_loss_iteration + patience``.

    With ``strict=False`` (the streaming pretrain path) training stops *at*
    that threshold, reproducing the pre-2.3.0 streaming condition
    ``epoch >= min_val_loss_iteration + patience`` (which used one-based
    epochs). Keeping the streaming path inclusive means pretrained base
    weights reproduce historical runs rather than training one extra epoch.
    """
    if not early_stopping or min_val_loss_epoch is None:
        return False
    threshold = max(
        int(min_val_loss_epoch) + int(patience),
        int(min_epochs) - 1,
    )
    if strict:
        return int(epoch_index) > threshold
    return int(epoch_index) >= threshold


def _should_validate_epoch(
    *,
    validation_enabled,
    epoch_index,
    max_epochs,
    validation_interval,
    early_stopping,
    min_val_loss_epoch,
    patience,
    min_epochs=0,
    strict=True,
):
    """Shared validation cadence for fit() and fit_streaming_batches().

    ``strict`` is forwarded to :func:`_early_stop_reached` so the "validate on
    the epoch we're about to stop" rule uses the same inclusive/exclusive
    threshold as the caller's break condition.
    """
    if not validation_enabled:
        return False
    is_last_epoch = int(epoch_index) == int(max_epochs) - 1
    patience_would_trigger = _early_stop_reached(
        epoch_index=epoch_index,
        min_val_loss_epoch=min_val_loss_epoch,
        patience=patience,
        min_epochs=min_epochs,
        early_stopping=early_stopping,
        strict=strict,
    )
    return (
        int(epoch_index) % int(validation_interval) == 0
        or is_last_epoch
        or patience_would_trigger
    )


def _carry_forward_validation_loss(fit_info):
    """Keep val_loss aligned with epochs when validation is skipped."""
    return fit_info["val_loss"][-1] if fit_info["val_loss"] else float("nan")


def _update_min_validation_loss(
    *,
    val_loss,
    epoch_index,
    min_val_loss,
    min_val_loss_epoch,
    min_delta,
):
    """Update min validation state after a measured validation epoch."""
    if min_val_loss is None or val_loss < min_val_loss - min_delta:
        return val_loss, epoch_index
    return min_val_loss, min_val_loss_epoch


def _effective_num_workers(num_workers):
    """Downgrade ``num_workers`` to 0 when running inside a daemon process.

    Safety net for DataLoader's ``num_workers>0`` path, which requires
    spawning child processes — forbidden from daemon parents. mhcflurry's
    training orchestrator switched to ``NonDaemonPool`` (see
    ``mhcflurry.parallelism``) so in production Pool workers are now
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
            f"to NonDaemonPool (mhcflurry/parallelism); if it's "
            f"your own caller, use a non-daemonic executor or accept the "
            f"downgrade (set dataloader_num_workers=0 explicitly to silence).",
            DeprecationWarning,
            stacklevel=3,
        )
        return 0
    return num_workers


def _make_streaming_batch_dataloader(
    dataset,
    num_workers,
):
    """Construct a DataLoader for ``fit_streaming_batches``.

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


_VALIDATION_INEQUALITY_FALLBACK_WARNED = False


def _warn_validation_inequality_fallback_once(loss_name):
    """Log once per process when val_loss drops to the single-shot inequality path.

    The fallback exists to preserve 2.1.x semantics: ``MSEWithInequalities`` and
    its multi-output variant exclude encoded ``2.0`` targets from the
    denominator, and batchwise averaging would change that count. Users should
    know that validation is then unbatched so they can size ``batch_size``
    knowing the cost.
    """
    global _VALIDATION_INEQUALITY_FALLBACK_WARNED
    if _VALIDATION_INEQUALITY_FALLBACK_WARNED:
        return
    _VALIDATION_INEQUALITY_FALLBACK_WARNED = True
    logging.warning(
        "validation loss running single-shot (no batching) because "
        "%s has inequality targets that would change the legacy denominator "
        "if split across batches. This log fires once per process.",
        loss_name,
    )


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
    """Compute validation loss with a fixed-size fast path and eager tail path.

    Every ``network(inputs)`` call inside this helper sees the same
    shape (``batch_size``), so GPU validation stays memory-bounded and
    optional compiled-validation paths can specialize once and reuse.
    That matters for H100 and A100 training where single-shot forward
    passes over the full validation set (up to ~185K rows for pan-allele
    training) were eating 15+ GB of VRAM per worker.

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

    # Short-circuit cheap predicates first so we only sync the device for the
    # inequality-denominator check when none of the basic conditions apply.
    needs_basic_fallback = (
        n_val < batch_size
        or val_weights is not None
        or not getattr(loss_obj, "supports_independent_samples", True)
    )
    inequality_fallback = (
        not needs_basic_fallback
        and _validation_loss_has_legacy_inequality_denominator(loss_obj, val_y)
    )
    if needs_basic_fallback or inequality_fallback:
        # Fallback for tiny val sets (typical in unit tests) and for
        # weighted / cross-example / inequality losses whose reductions are
        # not decomposable batchwise without changing the legacy denominator.
        if inequality_fallback:
            _warn_validation_inequality_fallback_once(type(loss_obj).__name__)
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


def _validation_loss_has_legacy_inequality_denominator(loss_obj, val_y):
    """Return True when batch-size weighting would change inequality loss.

    ``MSEWithInequalities`` and its multi-output variant keep 2.1.x behavior:
    encoded ``2.0`` targets are excluded from the denominator. If such rows
    are split unevenly across validation batches, averaging batch means by raw
    batch size changes ``val_loss``. Use the old single-shot path only for that
    case; all other independent-sample inequality rows remain batchable.
    """
    if not getattr(loss_obj, "supports_inequalities", False):
        return False
    y_t = val_y.reshape(-1)
    if getattr(loss_obj, "supports_multiple_outputs", False):
        output_indices = (y_t / 10.0).long()
        y_t = y_t - output_indices.to(y_t.dtype) * 10.0
    return bool(torch.any(y_t == 2.0).item())

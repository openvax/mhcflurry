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

from dataclasses import dataclass
import gc
import multiprocessing
import os
import time

import numpy
import pandas
import torch

from .allele_encoding import AlleleEncoding
from .class1_affinity_training_data import AffinityDeviceTrainingData
from .encodable_sequences import EncodableSequences
from .pytorch_training import (
    copy_module_state_dict_to_cpu,
    effective_validation_batch_size,
    maybe_compile_loss,
    maybe_compile_network,
    uncompiled_network,
    validation_forward_network,
)
from .random_negative_peptides import RandomNegativePeptides
from .regression_target import from_ic50


def validation_split_counts(total_rows, validation_split):
    """Return train/validation row counts using Keras split semantics.

    Keras assigns the last rows to validation and computes the boundary as
    ``floor(total_rows * (1 - validation_split))``. Computing validation rows
    first is not equivalent for row counts that are not exact multiples of the
    split denominator.
    """
    total_rows = int(total_rows)
    validation_split = float(validation_split)
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1)")
    if validation_split == 0.0:
        return total_rows, 0

    train_rows = int(numpy.floor(total_rows * (1.0 - validation_split)))
    validation_rows = total_rows - train_rows
    if train_rows == 0 or validation_rows == 0:
        raise ValueError(
            "validation_split=%s leaves %d training rows and %d validation "
            "rows from %d total rows"
            % (validation_split, train_rows, validation_rows, total_rows)
        )
    return train_rows, validation_rows


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
        value = torch_from_numpy(value)
    if cast_float and value.dtype == torch.float64:
        value = value.float()
    value = value.to(device, non_blocking=non_blocking)
    if cast_float:
        value = value.float()
    return value


def torch_from_numpy(value):
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

    Losses with additive numerator/denominator terms accumulate those terms
    across batches, preserving the exact global reduction for inequality and
    weighted losses. Cross-example losses still fall back to single-shot
    computation because samples genuinely couple within the loss.

    Returns a Python float (already detached from autograd).
    """
    n_val = val_peptide.shape[0]
    if n_val == 0:
        return float("nan")

    # A compiled loss keeps its additive reduction helper on the original
    # module; use it directly for validation aggregation.
    reduction_loss = getattr(loss_obj, "_orig_mod", loss_obj)
    reduction_terms = getattr(
        reduction_loss, "loss_numerator_and_denominator", None)
    needs_single_shot = (
        not getattr(reduction_loss, "supports_independent_samples", True)
        or (val_weights is not None and reduction_terms is None)
    )
    if needs_single_shot:
        # Cross-example and unknown weighted reductions are not decomposable.
        inputs = {"peptide": val_peptide}
        if val_allele is not None:
            inputs["allele"] = val_allele
        preds = eager_network(inputs)
        batch_loss = loss_obj(preds, val_y, sample_weights=val_weights)
        return batch_loss.item()

    n_full = (n_val // batch_size) * batch_size
    numerator_accum = None
    denominator_accum = None
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
        if reduction_terms is not None:
            numerator, denominator = reduction_terms(
                preds, val_y[start:end], sample_weights=weights_slice)
            numerator_accum = (
                numerator if numerator_accum is None
                else numerator_accum + numerator
            )
            denominator_accum = (
                denominator if denominator_accum is None
                else denominator_accum + denominator
            )
        else:
            batch_loss = loss_obj(
                preds, val_y[start:end], sample_weights=weights_slice
            )
            loss_accum += batch_loss.item() * batch_size
            weight_accum += batch_size
    if n_full < n_val:
        inputs = {"peptide": val_peptide[n_full:n_val]}
        if val_allele is not None:
            inputs["allele"] = val_allele[n_full:n_val]
        tail_predictions = eager_network(inputs)
        tail_size = n_val - n_full
        weights_slice = (
            val_weights[n_full:n_val] if val_weights is not None else None
        )
        if reduction_terms is not None:
            numerator, denominator = reduction_terms(
                tail_predictions,
                val_y[n_full:n_val],
                sample_weights=weights_slice,
            )
            numerator_accum = (
                numerator if numerator_accum is None
                else numerator_accum + numerator
            )
            denominator_accum = (
                denominator if denominator_accum is None
                else denominator_accum + denominator
            )
        else:
            tail_loss = loss_obj(
                tail_predictions,
                val_y[n_full:n_val],
                sample_weights=weights_slice,
            )
            loss_accum += tail_loss.item() * tail_size
            weight_accum += tail_size
    if numerator_accum is not None:
        return (
            numerator_accum / torch.clamp(denominator_accum, min=1.0)
        ).item()
    return loss_accum / weight_accum


# ---------------------------------------------------------------------------
# Public training-phase helpers shared by fit() and fit_streaming_batches()
# ---------------------------------------------------------------------------


@dataclass
class EarlyStoppingState:
    """Mutable early-stopping and best-checkpoint state across epochs."""

    min_val_loss: float | None = None
    min_val_loss_epoch: int | None = None
    best_state_dict: dict | None = None


@dataclass
class PreparedAffinityInputs:
    """Encoded, shuffled, validated real training rows plus random-negative plan."""

    x_dict_without_random_negatives: dict
    allele_representations: numpy.ndarray | None
    y_values: numpy.ndarray
    adjusted_inequalities: numpy.ndarray
    sample_weights: numpy.ndarray | None
    output_indices: numpy.ndarray | None
    random_negatives_pool: object
    random_negative_pool_residency: str
    random_negative_pool_epochs: int
    random_negative_x_allele_base: numpy.ndarray | None
    num_random_negatives: int


@dataclass
class PreparedAffinityDeviceData:
    """Device-resident training table, split indices, and optional val cache."""

    affinity_training_data: AffinityDeviceTrainingData
    indices: numpy.ndarray
    train_indices_base: numpy.ndarray
    val_indices: numpy.ndarray | None
    val_device_tensors: tuple | None
    val_cache_reused: bool
    random_neg_template: object
    n_train: int
    n_val: int


@dataclass
class StreamingIteratorState:
    """Tracks the streaming DataLoader iterator across restarts and epochs."""

    iterator: object
    make_iterator: object
    generator_factory: object
    expected_chunk_shape: tuple | None = None
    yielded_values: int = 0
    restarts: int = 0


def seed_training_rngs(seed):
    """Seed NumPy and PyTorch global RNGs from a single master seed.

    When ``seed`` is None, RNG states are left untouched.
    """
    if seed is not None:
        numpy.random.seed(int(seed) % (2 ** 32))
        torch.manual_seed(int(seed) & ((1 << 63) - 1))


def compute_regularization_penalty(parameters, l1=0.0, l2=0.0):
    """Match Keras kernel_regularizer semantics used on dense kernels."""
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


def run_training_batch(
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
    regularization_penalty = compute_regularization_penalty(
        regularization_parameters,
        l1=l1_reg,
        l2=l2_reg,
    )
    if regularization_penalty is not None:
        loss = loss + regularization_penalty
    loss.backward()
    optimizer.step()
    return loss.detach()


def run_prepared_training_batches(
    prepared_batches,
    *,
    optimizer,
    loss_obj,
    regularization_parameters,
    l1_reg,
    l2_reg,
    timing_enabled,
    device,
):
    """Run optimizer steps for a prepared batch stream.

    Batch sources are responsible for efficient data plumbing. The in-memory
    affinity path yields device-resident tensors selected by GPU indices; the
    streaming pretrain path yields DataLoader chunks after its H2D copy. This
    helper owns the learning step so ``fit()`` and ``fit_streaming_batches()``
    share identical optimizer, loss, and timing behavior.
    """
    losses = []
    train_time = 0.0
    train_rows = 0
    first_batch_time = None
    for prepared in prepared_batches:
        batch_start = _timing_start(device, timing_enabled)
        loss = run_training_batch(
            network=prepared["network"],
            optimizer=optimizer,
            loss_obj=loss_obj,
            regularization_parameters=regularization_parameters,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            inputs=prepared["inputs"],
            y_batch=prepared["y_batch"],
            weights_batch=prepared.get("weights_batch"),
        )
        batch_time = _timing_stop(batch_start, device, timing_enabled)
        train_time += batch_time
        if first_batch_time is None:
            first_batch_time = batch_time
        train_rows += int(prepared["row_count"])
        losses.append(loss)
    return {
        "losses": losses,
        "train_time": train_time,
        "train_rows": train_rows,
        "first_batch_time": first_batch_time,
    }


def prepare_affinity_inputs(
    *,
    peptides,
    affinities,
    allele_encoding,
    inequalities,
    output_indices,
    sample_weights,
    shuffle_permutation,
    hyperparameters,
    loss_obj,
    device,
    seed,
    peptides_to_network_input,
    allele_encoding_to_network_input,
    random_negatives_pool_factory,
):
    """Encode, plan random negatives, shuffle, and validate affinity fit inputs.

    Preserves the historical RNG order: ``numpy.random.permutation`` runs here
    before network weight initialization in the caller.
    """
    encodable_peptides = EncodableSequences.create(peptides)
    peptide_encoding = peptides_to_network_input(encodable_peptides)

    random_negatives_planner = RandomNegativePeptides(
        **RandomNegativePeptides.hyperparameter_defaults.subselect(
            hyperparameters
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

    random_negative_pool_epochs = int(
        hyperparameters.get("random_negative_pool_epochs", 1) or 1
    )
    if random_negative_pool_epochs < 1:
        random_negative_pool_epochs = 1
    pool_seed = seed if random_negative_pool_epochs > 1 else None
    requested_fit_tensor_residency = hyperparameters.get("fit_tensor_residency")

    (
        random_negatives_pool,
        random_negative_pool_residency,
    ) = random_negatives_pool_factory(
        random_negatives_planner=random_negatives_planner,
        random_negative_pool_epochs=random_negative_pool_epochs,
        pool_seed=pool_seed,
        device=device,
        requested_fit_tensor_residency=requested_fit_tensor_residency,
    )

    random_negative_x_allele_base = None
    if (
        num_random_negatives > 0
        and random_negatives_allele_encoding is not None
    ):
        (
            random_negative_x_allele_base,
            _,
        ) = allele_encoding_to_network_input(random_negatives_allele_encoding)

    y_values = from_ic50(numpy.asarray(affinities))
    assert numpy.isnan(y_values).sum() == 0, y_values

    if inequalities is not None:
        adjusted_inequalities = (
            pandas.Series(inequalities)
            .map({"=": "=", ">": "<", "<": ">"})
            .values
        )
    else:
        adjusted_inequalities = numpy.tile("=", len(y_values))

    if len(adjusted_inequalities) != len(y_values):
        raise ValueError("Inequalities and y_values must have same length")

    x_dict_without_random_negatives = {"peptide": peptide_encoding}
    allele_representations = None
    if allele_encoding is not None:
        (
            allele_encoding_input,
            allele_representations,
        ) = allele_encoding_to_network_input(allele_encoding)
        x_dict_without_random_negatives["allele"] = allele_encoding_input

    if shuffle_permutation is None:
        shuffle_permutation = numpy.random.permutation(len(y_values))
    y_values = y_values[shuffle_permutation]
    adjusted_inequalities = adjusted_inequalities[shuffle_permutation]
    for key in x_dict_without_random_negatives:
        x_dict_without_random_negatives[key] = x_dict_without_random_negatives[
            key
        ][shuffle_permutation]
    if sample_weights is not None:
        sample_weights = numpy.asarray(sample_weights)[shuffle_permutation]
    if output_indices is not None:
        output_indices = numpy.asarray(output_indices)[shuffle_permutation]

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

    if hyperparameters["num_outputs"] != 1 and output_indices is None:
        raise ValueError("Must supply output_indices for multi-output predictor")

    return PreparedAffinityInputs(
        x_dict_without_random_negatives=x_dict_without_random_negatives,
        allele_representations=allele_representations,
        y_values=y_values,
        adjusted_inequalities=adjusted_inequalities,
        sample_weights=sample_weights,
        output_indices=output_indices,
        random_negatives_pool=random_negatives_pool,
        random_negative_pool_residency=random_negative_pool_residency,
        random_negative_pool_epochs=random_negative_pool_epochs,
        random_negative_x_allele_base=random_negative_x_allele_base,
        num_random_negatives=num_random_negatives,
    )


def build_affinity_device_data(
    *,
    prepared: PreparedAffinityInputs,
    hyperparameters,
    loss_obj,
    device,
):
    """Build combined ``[random negatives | real]`` targets, val cache, and GPU buffer."""
    num_random_negatives = prepared.num_random_negatives

    if loss_obj.supports_inequalities:
        random_negative_ic50 = hyperparameters["random_negative_affinity_min"]
        random_negative_target = from_ic50(random_negative_ic50)

        y_with_negatives = numpy.concatenate([
            numpy.tile(random_negative_target, num_random_negatives),
            prepared.y_values,
        ])
        adjusted_inequalities_with_random_negatives = (
            ["<"] * num_random_negatives + list(prepared.adjusted_inequalities)
        )
    else:
        y_with_negatives = numpy.concatenate([
            from_ic50(
                numpy.random.uniform(
                    hyperparameters["random_negative_affinity_min"],
                    hyperparameters["random_negative_affinity_max"],
                    num_random_negatives,
                )
            ),
            prepared.y_values,
        ])
        adjusted_inequalities_with_random_negatives = None

    if prepared.sample_weights is not None:
        sample_weights_with_negatives = numpy.concatenate([
            numpy.ones(num_random_negatives),
            prepared.sample_weights,
        ])
    else:
        sample_weights_with_negatives = None

    if prepared.output_indices is not None:
        random_negative_output_indices = (
            hyperparameters["random_negative_output_indices"]
            if hyperparameters["random_negative_output_indices"]
            else list(range(0, hyperparameters["num_outputs"]))
        )
        output_indices_with_negatives = numpy.concatenate([
            pandas.Series(random_negative_output_indices, dtype=int)
            .sample(n=num_random_negatives, replace=True)
            .values,
            prepared.output_indices,
        ])
    else:
        output_indices_with_negatives = None

    encode_y_kwargs = {}
    if adjusted_inequalities_with_random_negatives is not None:
        encode_y_kwargs["inequalities"] = (
            adjusted_inequalities_with_random_negatives
        )
    if output_indices_with_negatives is not None:
        encode_y_kwargs["output_indices"] = output_indices_with_negatives

    y_encoded = loss_obj.encode_y(y_with_negatives, **encode_y_kwargs)

    val_split = hyperparameters["validation_split"]
    n_total = len(y_encoded)
    n_train, n_val = validation_split_counts(n_total, val_split)
    indices = numpy.arange(n_total)
    if n_val > 0:
        train_indices_base = indices[:n_train]
        val_indices = indices[n_train:]
    else:
        train_indices_base = indices
        val_indices = None

    # Cache static validation tensors on device when val_indices points
    # exclusively into the immutable training portion of [random_negs | real].
    val_cache_safe = (
        val_indices is not None
        and len(val_indices) > 0
        and bool(numpy.all(val_indices >= num_random_negatives))
    )
    val_device_tensors = None
    if val_cache_safe:
        val_training_indices = val_indices - num_random_negatives
        val_peptide_np = prepared.x_dict_without_random_negatives["peptide"][
            val_training_indices
        ]
        if (
            val_peptide_np.ndim == 2
            and numpy.issubdtype(val_peptide_np.dtype, numpy.integer)
        ):
            val_peptide_device = torch_from_numpy(val_peptide_np).to(device)
        else:
            val_peptide_device = (
                torch_from_numpy(val_peptide_np).float().to(device)
            )
        val_y_device = torch_from_numpy(
            y_encoded[val_indices].astype(numpy.float32)
        ).to(device)
        val_allele_device = None
        if "allele" in prepared.x_dict_without_random_negatives:
            val_allele_device = (
                torch_from_numpy(
                    prepared.x_dict_without_random_negatives["allele"][
                        val_training_indices
                    ]
                )
                .float()
                .to(device)
            )
        val_weights_device = None
        if sample_weights_with_negatives is not None:
            val_weights_device = (
                torch_from_numpy(sample_weights_with_negatives[val_indices])
                .float()
                .to(device)
            )
        val_device_tensors = (
            val_peptide_device,
            val_y_device,
            val_allele_device,
            val_weights_device,
        )

    random_neg_template = None
    if num_random_negatives > 0:
        _, random_neg_template = prepared.random_negatives_pool.get_epoch_inputs(
            0
        )
    affinity_training_data = AffinityDeviceTrainingData.from_arrays(
        x_peptide=prepared.x_dict_without_random_negatives["peptide"],
        x_allele=prepared.x_dict_without_random_negatives.get("allele"),
        y_encoded=y_encoded,
        sample_weights=sample_weights_with_negatives,
        random_negative_x_peptide_template=random_neg_template,
        random_negative_x_allele=prepared.random_negative_x_allele_base,
        device=device,
    )

    return PreparedAffinityDeviceData(
        affinity_training_data=affinity_training_data,
        indices=indices,
        train_indices_base=train_indices_base,
        val_indices=val_indices,
        val_device_tensors=val_device_tensors,
        val_cache_reused=val_cache_safe,
        random_neg_template=random_neg_template,
        n_train=int(n_train),
        n_val=int(n_val),
    )


def prepare_streaming_validation_tensors(
    *,
    validation_peptide_encoding,
    validation_affinities,
    validation_allele_input,
    validation_inequalities,
    validation_output_indices,
    loss_obj,
    device,
    peptides_to_network_input,
):
    """Encode and hoist static streaming-pretrain validation tensors to ``device``."""
    validation_x_dict = {
        "peptide": peptides_to_network_input(validation_peptide_encoding),
    }
    if validation_allele_input is not None:
        validation_x_dict["allele"] = validation_allele_input
    encode_y_kwargs = {}
    if validation_inequalities is not None:
        # from_ic50 is decreasing: concentration bounds reverse direction.
        encode_y_kwargs["inequalities"] = (
            pandas.Series(validation_inequalities)
            .map({"=": "=", ">": "<", "<": ">"})
            .values
        )
    if validation_output_indices is not None:
        encode_y_kwargs["output_indices"] = validation_output_indices

    output = loss_obj.encode_y(
        from_ic50(validation_affinities), **encode_y_kwargs
    )

    val_peptide_np = validation_x_dict["peptide"]
    if (
        val_peptide_np.ndim == 2
        and numpy.issubdtype(val_peptide_np.dtype, numpy.integer)
    ):
        val_peptide_device = torch_from_numpy(val_peptide_np).to(device)
    else:
        val_peptide_device = torch_from_numpy(val_peptide_np).to(device).float()
    val_allele_device = None
    if "allele" in validation_x_dict:
        val_allele_device = (
            torch_from_numpy(validation_x_dict["allele"]).to(device).float()
        )
    val_y_device = torch_from_numpy(output.astype(numpy.float32)).to(device)
    return val_peptide_device, val_y_device, val_allele_device


def evaluate_validation_epoch(
    *,
    network,
    eager_network,
    val_peptide,
    val_y,
    val_allele,
    val_weights,
    loss_obj,
    regularization_parameters,
    l1_reg,
    l2_reg,
    fit_info,
    device,
    validation_batch_size_hp,
    minibatch_size,
    num_workers_per_gpu,
    timing_enabled,
):
    """Run batched validation forward + regularization penalty and return ``(val_loss, val_batch_size, val_time)``."""
    network.eval()
    with torch.inference_mode():
        validation_start = _timing_start(device, timing_enabled)
        if "effective_validation_batch_size" not in fit_info:
            fit_info["effective_validation_batch_size"] = (
                effective_validation_batch_size(
                    device,
                    validation_batch_size_hp,
                    minibatch_size,
                    model=eager_network,
                    num_workers_per_gpu=num_workers_per_gpu,
                    total_rows=len(val_y),
                )
            )
        val_batch_size = fit_info["effective_validation_batch_size"]
        val_loss = _batched_validation_loss(
            network=validation_forward_network(network, eager_network),
            eager_network=eager_network,
            val_peptide=val_peptide,
            val_allele=val_allele,
            val_y=val_y,
            val_weights=val_weights,
            loss_obj=loss_obj,
            batch_size=val_batch_size,
        )
        regularization_penalty = compute_regularization_penalty(
            regularization_parameters,
            l1=l1_reg,
            l2=l2_reg,
        )
        if regularization_penalty is not None:
            val_loss = val_loss + regularization_penalty.item()
        validation_time = _timing_stop(validation_start, device, timing_enabled)
    fit_info["val_loss"].append(val_loss)
    return val_loss, val_batch_size, validation_time


def update_early_stopping_and_checkpoint(
    *,
    state: EarlyStoppingState,
    val_loss,
    epoch_index,
    min_delta,
    eager_network,
    retain_best_weights,
):
    """Update ``state`` on a measured validation epoch and snapshot best weights if improved."""
    previous_min_epoch = state.min_val_loss_epoch
    state.min_val_loss, state.min_val_loss_epoch = _update_min_validation_loss(
        val_loss=val_loss,
        epoch_index=epoch_index,
        min_val_loss=state.min_val_loss,
        min_val_loss_epoch=state.min_val_loss_epoch,
        min_delta=min_delta,
    )
    improved = state.min_val_loss_epoch != previous_min_epoch
    if retain_best_weights and improved:
        state.best_state_dict = copy_module_state_dict_to_cpu(eager_network)
    return improved


def iter_device_training_batches(
    *,
    affinity_training_data,
    train_indices_dev_full,
    batch_size,
    network,
    eager_network,
):
    """Yield full compiled batches followed by an optional eager tail batch."""
    n_train_epoch = int(train_indices_dev_full.shape[0])
    full_batch_count = (n_train_epoch // batch_size) * batch_size
    if full_batch_count > 0:
        train_indices_dev = train_indices_dev_full[:full_batch_count]
        num_full_batches = full_batch_count // batch_size
        for step in range(num_full_batches):
            batch_indices = train_indices_dev[
                step * batch_size : (step + 1) * batch_size
            ]
            inputs, y_batch, weights_batch = (
                affinity_training_data.batch_for_indices(batch_indices)
            )
            yield {
                "network": network,
                "inputs": inputs,
                "y_batch": y_batch,
                "weights_batch": weights_batch,
                "row_count": int(y_batch.shape[0]),
            }
    tail_count = n_train_epoch - full_batch_count
    if tail_count > 0:
        tail_indices_dev = train_indices_dev_full[full_batch_count:]
        inputs, y_batch, weights_batch = (
            affinity_training_data.batch_for_indices(tail_indices_dev)
        )
        yield {
            "network": eager_network,
            "inputs": inputs,
            "y_batch": y_batch,
            "weights_batch": weights_batch,
            "row_count": int(y_batch.shape[0]),
        }


def run_affinity_training_epoch(
    *,
    epoch,
    network,
    eager_network,
    optimizer,
    loss_obj,
    regularization_parameters,
    l1_reg,
    l2_reg,
    prepared: PreparedAffinityInputs,
    device_data: PreparedAffinityDeviceData,
    early_stopping_state: EarlyStoppingState,
    hyperparameters,
    fit_info,
    device,
    effective_minibatch,
    num_workers_per_gpu,
    timing_enabled,
    needs_initialization,
    data_dependent_init_fn,
    verbose,
    save_all_checkpoints,
    progress_callback,
    progress_preamble,
    progress_print_interval,
    last_progress_print,
):
    """Execute one complete epoch of device-resident affinity ``fit()``.

    Returns a dict with updated loop state: ``network``, ``eager_network``,
    ``loss_obj``, ``needs_initialization``, ``last_progress_print``,
    ``first_batch_time``, and ``should_stop``.
    """
    epoch_wall_start = time.perf_counter()
    rn_pool_get_time = 0.0
    rn_refill_time = 0.0
    dataset_construction_time = 0.0
    initialization_time = 0.0
    shuffle_dataset_time = 0.0
    dataloader_setup_time = 0.0
    train_loop_time = 0.0
    epoch_h2d_time = 0.0
    validation_materialize_time = 0.0
    validation_compute_time = 0.0
    callback_time = 0.0
    gc_time = 0.0

    rn_get_start = time.perf_counter()
    if epoch == 0 and device_data.random_neg_template is not None:
        random_negative_peptides_encoding = device_data.random_neg_template
        device_data.random_neg_template = None
    else:
        _, random_negative_peptides_encoding = (
            prepared.random_negatives_pool.get_epoch_inputs(epoch)
        )
    rn_pool_get_time += time.perf_counter() - rn_get_start

    rn_refill_start = time.perf_counter()
    device_data.affinity_training_data.refill_random_negative_peptides(
        random_negative_peptides_encoding
    )
    rn_refill_time += time.perf_counter() - rn_refill_start
    input_build_time = rn_pool_get_time + rn_refill_time

    if needs_initialization:
        init_start = _timing_start(device, timing_enabled)
        init_batch = device_data.affinity_training_data.batch_dict_for_indices(
            device_data.indices, device=device
        )
        data_dependent_init_fn(
            network,
            init_batch,
            method=hyperparameters["data_dependent_initialization_method"],
            target=hyperparameters["data_dependent_initialization_target"],
            verbose=verbose,
        )
        initialization_time += _timing_stop(init_start, device, timing_enabled)
        needs_initialization = False

    # Compile AFTER LSUV hook churn finishes (idempotent across epochs).
    network = maybe_compile_network(network, device)
    eager_network = uncompiled_network(network)
    loss_obj = maybe_compile_loss(loss_obj, device)

    dataset_start = time.perf_counter()
    train_indices_dev_full = torch.randperm(
        device_data.train_indices_base.shape[0], device=device
    )
    shuffle_dataset_time += time.perf_counter() - dataset_start

    network.train()
    epoch_start = time.time()
    train_loop_wall_start = time.perf_counter()

    batch_size = effective_minibatch
    n_train_epoch = int(train_indices_dev_full.shape[0])
    full_batch_count = (n_train_epoch // batch_size) * batch_size

    train_epoch_result = run_prepared_training_batches(
        iter_device_training_batches(
            affinity_training_data=device_data.affinity_training_data,
            train_indices_dev_full=train_indices_dev_full,
            batch_size=batch_size,
            network=network,
            eager_network=eager_network,
        ),
        optimizer=optimizer,
        loss_obj=loss_obj,
        regularization_parameters=regularization_parameters,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        timing_enabled=timing_enabled,
        device=device,
    )
    train_loop_time += train_epoch_result["train_time"]
    epoch_time = time.time() - epoch_start
    train_loop_wall_time = time.perf_counter() - train_loop_wall_start
    train_loss, epoch_loss_sync_time = _sync_mean_loss(
        train_epoch_result["losses"],
        device=device,
        timing_enabled=timing_enabled,
    )
    fit_info["loss"].append(train_loss)

    val_split = hyperparameters["validation_split"]
    validation_interval = _validation_interval_from_hyperparameters(
        hyperparameters
    )
    val_batch_size = None
    should_validate_this_epoch = _should_validate_epoch(
        validation_enabled=val_split > 0,
        epoch_index=epoch,
        max_epochs=hyperparameters["max_epochs"],
        validation_interval=validation_interval,
        early_stopping=hyperparameters.get("early_stopping", False),
        min_val_loss_epoch=early_stopping_state.min_val_loss_epoch,
        patience=hyperparameters["patience"],
    )
    val_loss = None
    if val_split > 0 and not should_validate_this_epoch:
        val_loss = _carry_forward_validation_loss(fit_info)
        fit_info["val_loss"].append(val_loss)
    if should_validate_this_epoch:
        if device_data.val_device_tensors is not None:
            (
                val_peptide,
                val_y,
                val_allele,
                val_weights,
            ) = device_data.val_device_tensors
        else:
            materialize_start = time.perf_counter()
            val_indices_dev = torch.as_tensor(
                device_data.val_indices, dtype=torch.long, device=device
            )
            val_inputs, val_y, val_weights = (
                device_data.affinity_training_data.batch_for_indices(
                    val_indices_dev
                )
            )
            val_peptide = val_inputs["peptide"]
            val_allele = val_inputs.get("allele")
            validation_materialize_time += (
                time.perf_counter() - materialize_start
            )
        val_loss, val_batch_size, val_compute_dt = evaluate_validation_epoch(
            network=network,
            eager_network=eager_network,
            val_peptide=val_peptide,
            val_y=val_y,
            val_allele=val_allele,
            val_weights=val_weights,
            loss_obj=loss_obj,
            regularization_parameters=regularization_parameters,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            fit_info=fit_info,
            device=device,
            validation_batch_size_hp=hyperparameters["validation_batch_size"],
            minibatch_size=batch_size,
            num_workers_per_gpu=num_workers_per_gpu,
            timing_enabled=timing_enabled,
        )
        validation_compute_time += val_compute_dt

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
                    hyperparameters["max_epochs"],
                    epoch_time,
                    train_loss,
                    str(early_stopping_state.min_val_loss),
                    early_stopping_state.min_val_loss_epoch,
                )
            ).strip()
        )
        last_progress_print = time.time()

    should_stop = False
    if val_split > 0:
        if should_validate_this_epoch:
            update_early_stopping_and_checkpoint(
                state=early_stopping_state,
                val_loss=val_loss,
                epoch_index=epoch,
                min_delta=hyperparameters["min_delta"],
                eager_network=eager_network,
                retain_best_weights=(
                    hyperparameters["restore_best_weights"]
                    or save_all_checkpoints
                ),
            )
        if hyperparameters["early_stopping"] and _early_stop_reached(
            epoch_index=epoch,
            min_val_loss_epoch=early_stopping_state.min_val_loss_epoch,
            patience=hyperparameters["patience"],
            early_stopping=True,
        ):
            if progress_print_interval is not None:
                print(
                    (
                        progress_preamble
                        + " "
                        + "Stopping at epoch %3d / %3d: loss=%g. "
                        "Min val loss (%g) at epoch %s"
                        % (
                            epoch,
                            hyperparameters["max_epochs"],
                            train_loss,
                            (
                                early_stopping_state.min_val_loss
                                if early_stopping_state.min_val_loss is not None
                                else numpy.nan
                            ),
                            early_stopping_state.min_val_loss_epoch,
                        )
                    ).strip()
                )
            should_stop = True

    if not should_stop:
        if progress_callback:
            callback_start = time.perf_counter()
            progress_callback()
            callback_time += time.perf_counter() - callback_start

        gc_start = time.perf_counter()
        gc.collect()
        gc_time += time.perf_counter() - gc_start
        if timing_enabled:
            fit_info["epoch_input_build_time"].append(input_build_time)
            fit_info["epoch_rn_pool_get_time"].append(rn_pool_get_time)
            fit_info["epoch_random_negative_refill_time"].append(rn_refill_time)
            fit_info["epoch_dataset_construction_time"].append(
                dataset_construction_time
            )
            fit_info["epoch_initialization_time"].append(initialization_time)
            fit_info["epoch_shuffle_dataset_time"].append(shuffle_dataset_time)
            fit_info["epoch_dataloader_setup_time"].append(
                dataloader_setup_time
            )
            fit_info["epoch_h2d_time"].append(epoch_h2d_time)
            fit_info["epoch_train_time"].append(train_loop_time)
            fit_info["epoch_train_loop_wall_time"].append(train_loop_wall_time)
            fit_info["epoch_dataloader_iter_overhead_time"].append(
                max(0.0, train_loop_wall_time - epoch_h2d_time - train_loop_time)
            )
            fit_info["epoch_loss_sync_time"].append(epoch_loss_sync_time)
            fit_info["epoch_validation_materialize_time"].append(
                validation_materialize_time
            )
            fit_info["epoch_validation_time"].append(validation_compute_time)
            fit_info["epoch_num_train_batches"].append(
                len(train_epoch_result["losses"])
            )
            fit_info["epoch_num_train_rows"].append(
                train_epoch_result["train_rows"]
            )
            fit_info["epoch_tail_train_rows"].append(
                n_train_epoch - full_batch_count
            )
            fit_info["epoch_num_validation_batches"].append(
                int(numpy.ceil(device_data.n_val / val_batch_size))
                if val_batch_size
                else 0
            )
            fit_info["epoch_callback_time"].append(callback_time)
            fit_info["epoch_gc_time"].append(gc_time)
            fit_info["epoch_total_time"].append(
                time.perf_counter() - epoch_wall_start
            )

    return {
        "network": network,
        "eager_network": eager_network,
        "loss_obj": loss_obj,
        "needs_initialization": needs_initialization,
        "last_progress_print": last_progress_print,
        "first_batch_time": train_epoch_result["first_batch_time"],
        "should_stop": should_stop,
    }


def run_streaming_training_epoch(
    *,
    epoch_index,
    epochs,
    steps_per_epoch,
    min_epochs,
    patience,
    min_delta,
    network,
    eager_network,
    optimizer,
    loss_obj,
    regularization_parameters,
    l1_reg,
    l2_reg,
    stream_state: StreamingIteratorState,
    val_peptide_device,
    val_y_device,
    val_allele_device,
    validation_count,
    early_stopping_state: EarlyStoppingState,
    hyperparameters,
    fit_info,
    device,
    non_blocking_h2d,
    num_workers_per_gpu,
    timing_enabled,
    validation_interval,
    save_all_checkpoints,
    progress_callback,
    progress_preamble,
    progress_print_interval,
    last_progress_print,
):
    """Execute one complete epoch of ``fit_streaming_batches()``."""
    epoch_num = epoch_index + 1
    epoch_start_time = time.time()
    epoch_wall_start = time.perf_counter()
    network.train()

    epoch_fetch_time = 0.0
    epoch_h2d_time = 0.0

    def prepared_streaming_batches():
        nonlocal epoch_fetch_time, epoch_h2d_time
        for _step in range(steps_per_epoch):
            fetch_start = time.perf_counter()
            try:
                batch = next(stream_state.iterator)
            except StopIteration:
                if stream_state.generator_factory is None:
                    break
                stream_state.restarts += 1
                fit_info["iterator_restarts"] = stream_state.restarts
                stream_state.iterator = stream_state.make_iterator()
                try:
                    batch = next(stream_state.iterator)
                except StopIteration:
                    break
            epoch_fetch_time += time.perf_counter() - fetch_start

            if stream_state.expected_chunk_shape is None:
                stream_state.expected_chunk_shape = batch["peptide"].shape
            h2d_start = _timing_start(device, timing_enabled)
            inputs, y_tensor, weights_batch = _move_fit_batch_to_device(
                batch,
                device,
                non_blocking=non_blocking_h2d,
            )
            epoch_h2d_time += _timing_stop(h2d_start, device, timing_enabled)
            stream_state.yielded_values += len(batch["y"])
            yield {
                "network": (
                    network
                    if batch["peptide"].shape
                    == stream_state.expected_chunk_shape
                    else eager_network
                ),
                "inputs": inputs,
                "y_batch": y_tensor,
                "weights_batch": weights_batch,
                "row_count": len(batch["y"]),
            }

    train_epoch_result = run_prepared_training_batches(
        prepared_streaming_batches(),
        optimizer=optimizer,
        loss_obj=loss_obj,
        regularization_parameters=regularization_parameters,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        timing_enabled=timing_enabled,
        device=device,
    )

    should_validate_this_epoch = _should_validate_epoch(
        validation_enabled=True,
        epoch_index=epoch_index,
        max_epochs=epochs,
        validation_interval=validation_interval,
        early_stopping=True,
        min_val_loss_epoch=early_stopping_state.min_val_loss_epoch,
        patience=patience,
        min_epochs=min_epochs,
        # Streaming pretrain stops inclusively (epoch >= threshold) to
        # match pre-2.3.0 weights — see _early_stop_reached.
        strict=False,
    )
    validation_time = 0.0
    val_batch_size = None
    val_loss = _carry_forward_validation_loss(fit_info)
    if should_validate_this_epoch:
        val_loss, val_batch_size, validation_time = evaluate_validation_epoch(
            network=network,
            eager_network=eager_network,
            val_peptide=val_peptide_device,
            val_y=val_y_device,
            val_allele=val_allele_device,
            val_weights=None,
            loss_obj=loss_obj,
            regularization_parameters=regularization_parameters,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            fit_info=fit_info,
            device=device,
            validation_batch_size_hp=hyperparameters["validation_batch_size"],
            minibatch_size=hyperparameters["minibatch_size"],
            num_workers_per_gpu=num_workers_per_gpu,
            timing_enabled=timing_enabled,
        )
        update_early_stopping_and_checkpoint(
            state=early_stopping_state,
            val_loss=val_loss,
            epoch_index=epoch_index,
            min_delta=min_delta,
            eager_network=eager_network,
            retain_best_weights=(
                hyperparameters["restore_best_weights"] or save_all_checkpoints
            ),
        )
    else:
        fit_info["val_loss"].append(val_loss)

    epoch_time = time.time() - epoch_start_time
    train_loss, epoch_loss_sync_time = _sync_mean_loss(
        train_epoch_result["losses"],
        device=device,
        timing_enabled=timing_enabled,
    )
    fit_info["loss"].append(train_loss)
    if timing_enabled:
        fit_info["epoch_fetch_time"].append(epoch_fetch_time)
        fit_info["epoch_h2d_time"].append(epoch_h2d_time)
        fit_info["epoch_train_time"].append(train_epoch_result["train_time"])
        fit_info["epoch_loss_sync_time"].append(epoch_loss_sync_time)
        fit_info["epoch_validation_time"].append(validation_time)
        fit_info["epoch_num_train_batches"].append(
            len(train_epoch_result["losses"])
        )
        fit_info["epoch_num_train_rows"].append(
            train_epoch_result["train_rows"]
        )
        fit_info["epoch_num_validation_batches"].append(
            int(numpy.ceil(validation_count / val_batch_size))
            if val_batch_size
            else 0
        )
        fit_info["epoch_total_time"].append(
            time.perf_counter() - epoch_wall_start
        )

    progress_message = (
        "epoch %3d/%3d [%0.2f sec.]: loss=%g val_loss=%g. Min val "
        "loss %g at epoch %s. Cum. points: %d."
        % (
            epoch_num,
            epochs,
            epoch_time,
            train_loss,
            val_loss,
            (
                early_stopping_state.min_val_loss
                if early_stopping_state.min_val_loss is not None
                else numpy.nan
            ),
            (
                early_stopping_state.min_val_loss_epoch + 1
                if early_stopping_state.min_val_loss_epoch is not None
                else None
            ),
            stream_state.yielded_values,
        )
    ).strip()

    if progress_print_interval is not None and (
        time.time() - last_progress_print > progress_print_interval
    ):
        print(progress_preamble, progress_message)
        last_progress_print = time.time()

    if progress_callback:
        progress_callback()

    should_stop = _early_stop_reached(
        epoch_index=epoch_index,
        min_val_loss_epoch=early_stopping_state.min_val_loss_epoch,
        patience=patience,
        min_epochs=min_epochs,
        early_stopping=True,
        strict=False,
    )
    if should_stop and progress_print_interval is not None:
        print(progress_preamble, "STOPPING", progress_message)

    return {
        "last_progress_print": last_progress_print,
        "first_batch_time": train_epoch_result["first_batch_time"],
        "should_stop": should_stop,
    }


def record_fit_summary(
    *,
    fit_info,
    early_stopping_state: EarlyStoppingState,
    restored_best_weights,
    saved_checkpoint_policies,
    elapsed_seconds,
    num_points,
    first_batch_time,
):
    """Populate final summary fields on ``fit_info`` and return it as a plain dict."""
    fit_info["best_epoch"] = (
        early_stopping_state.min_val_loss_epoch + 1
        if early_stopping_state.min_val_loss_epoch is not None
        else None
    )
    fit_info["best_val_loss"] = early_stopping_state.min_val_loss
    fit_info["restored_best_weights"] = restored_best_weights
    fit_info["saved_checkpoint_policies"] = saved_checkpoint_policies
    fit_info["time"] = elapsed_seconds
    fit_info["num_points"] = num_points
    if first_batch_time is not None:
        fit_info["first_batch_time"] = first_batch_time
    return dict(fit_info)

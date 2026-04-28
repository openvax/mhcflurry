"""Shared-memory primitives used by mhcflurry training.

mhcflurry has two shared-memory layers, both implementing the same
"build once, share with many readers" pattern but using different OS
mechanisms because the lifecycles differ.

run mmap cache — per-run, file-mmap, orchestrator-built, read-only.
    Random-negative pool, encoding cache. Built ONCE before any
    training worker is forked; workers ``numpy.memmap`` the file and
    the OS page cache holds a single resident copy across N workers.
    Persists to disk so it can be reused across runs.

    Mechanism: ``numpy.memmap`` of files written by the orchestrator.

fit DataLoader SHM — per-fit(), POSIX shm, fit()-built, read+write.
    Dataset backing arrays for a single fit() inner DataLoader.
    Lifetime is one ``fit()`` call. Tensors are allocated in
    ``/dev/shm`` so the DataLoader's spawn workers receive storage
    handles instead of byte copies.

    Mechanism: ``torch.Tensor.share_memory_()``.

Both layers share one resident copy across many readers; both are
controlled by the same idea ("the orchestrator owns the resource;
workers consume it"). The mechanism asymmetry is intentional —
file-mmap fits the run mmap cache's persist-across-runs property,
torch shm fits the fit DataLoader SHM's mutate-per-epoch property —
but the API surface is uniform: a ``setup_*`` factory and a small set
of generic helpers.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy
import torch

from .random_negative_peptides import (
    RandomNegativePeptides,
    RandomNegativesPool,
)


# ---- fit DataLoader SHM: per-fit() torch shared tensors -----------------

def torch_shared_memory_status():
    """Return whether PyTorch CPU tensor sharing works in this process.

    Some local sandboxes and macOS launch paths expose only PyTorch's
    ``file_system`` sharing strategy but block ``torch_shm_manager`` from
    starting. In that state ``share_memory_()`` fails with ``Operation not
    permitted`` even though the strategy is nominally available. Probe the
    actual operation so callers can fall back deliberately instead of
    discovering the failure through skipped tests or mid-fit exceptions.
    """
    try:
        strategy = torch.multiprocessing.get_sharing_strategy()
    except RuntimeError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "strategy": None,
            "strategies": (),
        }
    try:
        strategies = tuple(sorted(torch.multiprocessing.get_all_sharing_strategies()))
    except RuntimeError:
        strategies = (strategy,)

    try:
        torch.empty(1).share_memory_()
    except RuntimeError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "strategy": strategy,
            "strategies": strategies,
        }

    return {
        "available": True,
        "reason": None,
        "strategy": strategy,
        "strategies": strategies,
    }


def share_tensor(value):
    """Return ``value`` as a CPU ``torch.Tensor`` in shared memory.

    Accepts a ``numpy.ndarray`` or ``torch.Tensor``; passes ``None``
    through. The returned tensor's storage lives in POSIX shm
    (``/dev/shm``), so when the dataset is pickled to a DataLoader
    spawn worker the storage handle is forwarded over the socket and
    the worker materializes a tensor pointing at the same bytes — no
    per-worker byte copy.

    Always allocates a fresh storage (clones from the source) before
    sharing so that callers retain their original buffer. The clone is
    a one-time per-fit() memcpy; far cheaper than the per-epoch /
    per-worker copy it eliminates.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().clone()
    else:
        tensor = torch.from_numpy(numpy.ascontiguousarray(value)).clone()
    tensor.share_memory_()
    return tensor


def share_like(template):
    """Allocate a fresh shared-memory tensor with ``template``'s shape/dtype.

    Contents are uninitialized; the caller MUST fill via ``update_shared``
    before any consumer reads. Used for per-epoch buffers that are
    refilled in place each iteration (e.g. fit()'s random-negative
    payload).
    """
    if template is None:
        raise TypeError("share_like: template cannot be None")
    if isinstance(template, torch.Tensor):
        shape, dtype = template.shape, template.dtype
    else:
        # ``torch.from_numpy`` is a zero-copy view, so ``.dtype`` lookup
        # is O(1) and doesn't allocate the full tensor data.
        arr = numpy.ascontiguousarray(template)
        shape, dtype = arr.shape, torch.from_numpy(arr).dtype
    tensor = torch.empty(shape, dtype=dtype)
    tensor.share_memory_()
    return tensor


def update_shared(target, source):
    """Copy ``source`` into ``target`` in place.

    ``target`` is a shared tensor whose storage workers can already
    see; mutating its contents propagates immediately. Safe between
    DataLoader epochs because the parent's iter exhausts before any
    refill, joining all workers.
    """
    if not isinstance(source, torch.Tensor):
        source = torch.from_numpy(numpy.ascontiguousarray(source))
    target.copy_(source)


def array_nbytes(value):
    """Return the byte size of ``value`` (numpy array or torch tensor)."""
    if value is None:
        return 0
    if isinstance(value, numpy.ndarray):
        return int(value.nbytes)
    if isinstance(value, torch.Tensor):
        return int(value.element_size() * value.numel())
    raise TypeError(
        "array_nbytes: unsupported type %s" % type(value).__name__
    )


def numpy_batch_collate(batch):
    """Stack a list of numpy sample dicts into per-key numpy arrays.

    Used by fit()'s DataLoader on the legacy numpy backing path.
    Skipping PyTorch's default-collate ``torch.tensor(numpy_array)``
    call avoids the CPU tensor allocator growth that motivated
    openvax/mhcflurry#270.
    """
    return {
        key: numpy.stack([sample[key] for sample in batch], axis=0)
        for key in batch[0]
    }


def tensor_batch_collate(batch):
    """Stack a list of tensor sample dicts into per-key tensors.

    Used by fit()'s DataLoader on the SHM backing path. Each sample's
    per-key value is a tensor view into shared memory; ``torch.stack``
    materializes a contiguous CPU tensor that the parent then pins
    and copies to device.
    """
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        for key in batch[0]
    }


@dataclass
class FitBacking:
    """Backing arrays for one fit() call.

    Three residency modes:

    * ``residency="host_numpy"``: numpy arrays, batches built by
      ``_FitBatchDataset.__getitem__`` and copied H2D per batch by the
      DataLoader / fit() loop.
    * ``residency="host_shared"``: SHM-shared CPU torch tensors. Same
      batch path but DataLoader workers see zero-copy views via OS-page-
      cache sharing.
    * ``residency="device"``: torch tensors already on the training
      device (CUDA/MPS/CPU). The fit() loop indexes them directly, no
      DataLoader, no per-batch H2D. The default for CUDA when the data
      bundle fits in VRAM (see ``_pick_fit_tensor_residency``).

    ``tensor_backed`` is preserved for back-compat: it is true for both
    ``host_shared`` and ``device``. ``device_backed`` is the new sharper
    flag — true only for ``device``.

    This is an internal transport container, not a model feature. The
    model-side peptide encoding decision is controlled separately by
    ``peptide_amino_acid_encoding_torch``.
    """

    x_peptide: Any
    x_allele: Any = None
    y_encoded: Any = None
    sample_weights: Any = None
    random_negative_x_peptide: Any = None
    random_negative_x_allele: Any = None
    tensor_backed: bool = False
    device_backed: bool = False
    residency: str = "host_numpy"
    # Device-residency only: pre-stitched (RN | real) device tensors that
    # the inner fit() loop indexes into directly. Top slice is RN
    # (refilled per cycle by RandomNegativesPool); bottom slice is the
    # static real-data block. ``random_negative_x_peptide`` and
    # ``x_peptide`` are views into ``combined_peptide`` (and same for
    # allele) so refills propagate without any copy.
    combined_peptide: Any = None
    combined_allele: Any = None

    @classmethod
    def from_numpy(
        cls,
        *,
        x_peptide,
        x_allele,
        y_encoded,
        sample_weights,
        random_negative_x_peptide,
        random_negative_x_allele,
    ):
        return cls(
            x_peptide=x_peptide,
            x_allele=x_allele,
            y_encoded=y_encoded,
            sample_weights=sample_weights,
            random_negative_x_peptide=random_negative_x_peptide,
            random_negative_x_allele=random_negative_x_allele,
            tensor_backed=False,
            device_backed=False,
            residency="host_numpy",
        )

    @classmethod
    def share(
        cls,
        *,
        x_peptide,
        x_allele,
        y_encoded,
        sample_weights,
        random_negative_x_peptide_template,
        random_negative_x_allele,
    ):
        """Materialize a SHM-backed bundle.

        Static arrays (x_peptide, x_allele, y_encoded, sample_weights,
        random_negative_x_allele) are cloned into shared memory once.
        ``random_negative_x_peptide`` is allocated as a fixed-shape
        buffer the size of one cycle's encoding; the caller refills it
        in place each epoch via ``update_shared``.
        """
        return cls(
            x_peptide=share_tensor(x_peptide),
            x_allele=share_tensor(x_allele),
            y_encoded=share_tensor(y_encoded),
            sample_weights=share_tensor(sample_weights),
            random_negative_x_peptide=(
                share_like(random_negative_x_peptide_template)
                if random_negative_x_peptide_template is not None else None
            ),
            random_negative_x_allele=share_tensor(random_negative_x_allele),
            tensor_backed=True,
            device_backed=False,
            residency="host_shared",
        )

    @classmethod
    def from_device(
        cls,
        *,
        x_peptide,
        x_allele,
        y_encoded,
        sample_weights,
        random_negative_x_peptide_template,
        random_negative_x_allele,
        device,
    ):
        """Materialize a device-resident bundle.

        Pre-allocates one combined ``(num_random_negatives + num_real,
        encoded_length)`` peptide tensor on ``device`` (and same for
        allele). The top ``num_random_negatives`` rows are the RN slice
        — refilled per cycle by ``RandomNegativesPool`` via in-place
        copy. The bottom ``num_real`` rows are the static real-data
        block — copied once at construction. ``x_peptide`` and
        ``random_negative_x_peptide`` (and the allele equivalents) are
        returned as views into the combined buffer so refills propagate
        with zero copy.

        Why one combined tensor: the fit() inner loop indexes via a
        single ``index_select`` into ``combined_peptide`` per batch,
        which is what eliminates the per-batch H2D copies that
        dominate the host-DataLoader path. ``train_indices`` already
        addresses the unified ``[0, num_rn + num_real)`` space (see
        ``y_encoded`` / ``sample_weights_with_negatives`` construction
        in ``fit``), so no index translation is needed.

        ``y_encoded`` and ``sample_weights`` are already laid out in
        the unified ``[RN | real]`` order by the caller and just need
        to land on device.

        Inputs may be numpy arrays or CPU torch tensors; the constructor
        handles the .to(device) move uniformly.
        """
        import torch as _torch

        def _to_device(value, dtype=None):
            if value is None:
                return None
            if isinstance(value, _torch.Tensor):
                tensor = value
            else:
                tensor = _torch.as_tensor(value)
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            return tensor.to(device, non_blocking=False)

        x_peptide_dev = _to_device(x_peptide)
        x_allele_dev = _to_device(x_allele)

        combined_peptide = None
        rn_peptide_view = None
        x_peptide_view = x_peptide_dev
        if random_negative_x_peptide_template is not None:
            template = random_negative_x_peptide_template
            if isinstance(template, _torch.Tensor):
                rn_shape = tuple(template.shape)
                rn_dtype = template.dtype
            else:
                rn_shape = tuple(template.shape)
                rn_dtype = _torch.as_tensor(template).dtype
            num_rn = int(rn_shape[0])
            if x_peptide_dev is None:
                raise ValueError(
                    "FitBacking.from_device: x_peptide is required when "
                    "random_negative_x_peptide_template is set."
                )
            if x_peptide_dev.dtype != rn_dtype:
                x_peptide_dev = x_peptide_dev.to(rn_dtype)
            num_real = int(x_peptide_dev.shape[0])
            combined_shape = (num_rn + num_real, *rn_shape[1:])
            if tuple(x_peptide_dev.shape[1:]) != tuple(rn_shape[1:]):
                raise ValueError(
                    "FitBacking.from_device: real and random-negative "
                    "peptide tensors disagree on per-row shape: %r vs %r" % (
                        tuple(x_peptide_dev.shape[1:]),
                        tuple(rn_shape[1:]),
                    )
                )
            combined_peptide = _torch.empty(
                combined_shape, dtype=rn_dtype, device=device
            )
            rn_peptide_view = combined_peptide[:num_rn]
            x_peptide_view = combined_peptide[num_rn:]
            x_peptide_view.copy_(x_peptide_dev)
            rn_peptide_view.zero_()

        rn_allele_dev = _to_device(random_negative_x_allele)
        combined_allele = None
        rn_allele_view = rn_allele_dev
        x_allele_view = x_allele_dev
        if rn_allele_dev is not None and x_allele_dev is not None:
            if rn_allele_dev.dtype != x_allele_dev.dtype:
                # Match dtypes via the wider one (typically float32 for allele).
                target_dtype = (
                    _torch.float32
                    if (rn_allele_dev.dtype.is_floating_point
                        or x_allele_dev.dtype.is_floating_point)
                    else x_allele_dev.dtype
                )
                rn_allele_dev = rn_allele_dev.to(target_dtype)
                x_allele_dev = x_allele_dev.to(target_dtype)
            num_rn_a = int(rn_allele_dev.shape[0])
            num_real_a = int(x_allele_dev.shape[0])
            combined_allele = _torch.empty(
                (num_rn_a + num_real_a, *rn_allele_dev.shape[1:]),
                dtype=rn_allele_dev.dtype,
                device=device,
            )
            combined_allele[:num_rn_a].copy_(rn_allele_dev)
            combined_allele[num_rn_a:].copy_(x_allele_dev)
            rn_allele_view = combined_allele[:num_rn_a]
            x_allele_view = combined_allele[num_rn_a:]

        return cls(
            x_peptide=x_peptide_view,
            x_allele=x_allele_view,
            y_encoded=_to_device(y_encoded, dtype=_torch.float32),
            sample_weights=_to_device(sample_weights, dtype=_torch.float32),
            random_negative_x_peptide=rn_peptide_view,
            random_negative_x_allele=rn_allele_view,
            tensor_backed=True,
            device_backed=True,
            residency="device",
            combined_peptide=combined_peptide,
            combined_allele=combined_allele,
        )


# ---- run mmap cache: per-run mmap random-negative pool ------------------

def _planner_from_hyperparameters(hyperparameters):
    """Build a ``RandomNegativePeptides`` planner from a hyperparameter dict.

    Reads only the random-negative subset; ignores everything else.
    The orchestrator uses this to build planners keyed on
    (fold, random-negative config) before forking workers.
    """
    rn_keys = RandomNegativePeptides.hyperparameter_defaults.defaults.keys()
    sub = {k: hyperparameters[k] for k in rn_keys if k in hyperparameters}
    return RandomNegativePeptides(**sub)


def _random_negative_config_key(hyperparameters):
    """Stable tuple over the random-negative hyperparameter subset.

    Two work items whose plans would be identical (given the same
    training data) share the same key. The orchestrator builds one
    pool per (fold, key); workers look their pool up by their own key.
    """
    rn_keys = sorted(
        RandomNegativePeptides.hyperparameter_defaults.defaults.keys()
    )
    items = []
    for k in rn_keys:
        v = hyperparameters.get(k)
        if isinstance(v, list):
            v = tuple(v)
        items.append((k, v))
    return tuple(items)


def _per_run_seed(out_dir):
    """Deterministic per-run seed derived from the output directory."""
    return int(
        hashlib.sha256(
            ("mhcflurry-shared-pool::" + out_dir).encode("utf-8")
        ).hexdigest()[:8],
        16,
    )


def setup_shared_random_negative_pools(
    *,
    output_root_dir,
    work_items,
    train_data_df,
    folds_df,
    peptide_encoder,
    pool_epochs,
    seed,
    log=None,
):
    """Build per-(fold, random-negative-config) shared mmap pools.

    Run once by the orchestrator BEFORE forking training workers. For
    each unique (fold, random-negative-config) tuple in ``work_items``,
    computes the plan against that fold's training data and writes a
    Phase-3 mmap pool under
    ``output_root_dir/fold_{fold}/cfg_{idx}/``.

    Returns ``{(fold_num, config_key): pool_dir}`` for worker-side
    lookup via ``lookup_pool_dir``.
    """
    if log is None:
        log = logging.info

    by_fold_and_cfg: Dict[Any, Any] = {}
    for item in work_items:
        fold_num = int(item["fold_num"])
        hp = item["hyperparameters"]
        if int(hp.get("random_negative_pool_epochs", 1) or 1) != pool_epochs:
            raise ValueError(
                "setup_shared_random_negative_pools: work item %r has "
                "random_negative_pool_epochs=%r but caller asked for "
                "pool_epochs=%d. All work items must share the same "
                "pool_epochs to use the shared mmap path." % (
                    item.get("work_item_name"),
                    hp.get("random_negative_pool_epochs"),
                    pool_epochs,
                )
            )
        # The shared mmap pool holds exactly ``pool_epochs`` cycles
        # (cycle 0 only — see RandomNegativesPool.from_shared_mmap).
        # If a work item's ``max_epochs`` would advance training past
        # epoch ``pool_epochs - 1``, the worker's
        # ``RandomNegativesPool.get_epoch_inputs`` raises mid-training.
        # Caught here so the orchestrator fails before forking workers
        # rather than at random points hours into the run.
        max_epochs = int(hp.get("max_epochs", 0) or 0)
        if max_epochs > pool_epochs:
            raise ValueError(
                "setup_shared_random_negative_pools: work item %r has "
                "max_epochs=%d > pool_epochs=%d. The shared mmap pool "
                "covers epochs 0..pool_epochs-1; training would crash "
                "at epoch %d. Either raise random_negative_pool_epochs "
                "to >= max_epochs, lower max_epochs, or omit "
                "--random-negative-shared-pool-dir to fall back to "
                "the in-process pool." % (
                    item.get("work_item_name"),
                    max_epochs, pool_epochs, pool_epochs,
                )
            )
        cfg_key = _random_negative_config_key(hp)
        by_fold_and_cfg.setdefault((fold_num, cfg_key), hp)

    fold_pool_dirs: Dict[Any, str] = {}
    for cfg_idx, ((fold_num, cfg_key), hp) in enumerate(
        sorted(by_fold_and_cfg.items())
    ):
        fold_dir = os.path.join(
            output_root_dir, "fold_%d" % fold_num, "cfg_%d" % cfg_idx
        )
        os.makedirs(fold_dir, exist_ok=True)
        fold_col = "fold_%d" % fold_num
        if fold_col not in folds_df.columns:
            raise KeyError(
                "folds_df is missing column %r required for fold %d"
                % (fold_col, fold_num)
            )
        train_subset = train_data_df.loc[folds_df[fold_col]]
        if len(train_subset) == 0:
            raise ValueError(
                "fold_%d has zero training rows; cannot build "
                "shared random-negative pool" % fold_num
            )
        planner = _planner_from_hyperparameters(hp)
        planner.plan(
            peptides=train_subset.peptide.values,
            affinities=train_subset.measurement_value.values,
            alleles=train_subset.allele.values,
            inequalities=(
                train_subset.measurement_inequality.values
                if "measurement_inequality" in train_subset.columns
                else None
            ),
        )
        log(
            "shared_memory: writing fold=%d cfg=%d pool to %s "
            "(pool_epochs=%d, total_count=%d, seed=%d)" % (
                fold_num, cfg_idx, fold_dir, pool_epochs,
                planner.get_total_count(), seed,
            )
        )
        RandomNegativesPool.write_shared_pool(
            output_dir=fold_dir,
            planner=planner,
            peptide_encoder=peptide_encoder,
            pool_epochs=pool_epochs,
            seed=seed + fold_num,
        )
        fold_pool_dirs[(fold_num, cfg_key)] = fold_dir

    return fold_pool_dirs


def lookup_pool_dir(fold_pool_dirs, *, fold_num, hyperparameters):
    """Worker-side O(1) lookup. Returns dir or None if no match.

    ``None`` is the signal for the worker's fit() to fall back to the
    in-process pool — same surface area as "run mmap cache not enabled."
    """
    if not fold_pool_dirs:
        return None
    return fold_pool_dirs.get(
        (int(fold_num), _random_negative_config_key(hyperparameters))
    )


__all__ = [
    "FitBacking",
    "array_nbytes",
    "lookup_pool_dir",
    "numpy_batch_collate",
    "setup_shared_random_negative_pools",
    "share_like",
    "share_tensor",
    "tensor_batch_collate",
    "update_shared",
]

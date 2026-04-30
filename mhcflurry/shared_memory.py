"""Shared-memory primitives used by mhcflurry training.

mhcflurry's training pipeline now uses a single per-fit() data residency
— device-resident — and a per-run shared mmap pool for the random-negative
peptide cycle. The two layers serve different lifecycles:

run mmap cache — per-run, file-mmap, orchestrator-built, read-only.
    Random-negative pool, encoding cache. Built ONCE before any
    training worker is forked; workers ``numpy.memmap`` the file and
    the OS page cache holds a single resident copy across N workers.
    Persists to disk so it can be reused across runs.

    Mechanism: ``numpy.memmap`` of files written by the orchestrator.

per-fit() data — device-resident.
    Backing arrays for a single ``fit()`` call. Live on the training
    device (CUDA/MPS/CPU) for the lifetime of one fit; the inner
    training loop indexes pre-stitched ``[random_negatives | real]``
    device tensors directly, with zero per-batch H2D copies and no
    DataLoader workers. Built by ``FitBacking.from_device``.

The legacy POSIX-shm "fit DataLoader SHM" backing and the host-resident
DataLoader path are gone — the device-resident path is the only one.
``FitBacking`` only carries device-resident state; ``setup_shared_random_negative_pools`` /
``lookup_pool_dir`` continue to back the per-run mmap pool unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

import torch

from .random_negative_peptides import (
    RandomNegativePeptides,
    RandomNegativesPool,
)


@dataclass
class FitBacking:
    """Backing arrays for one fit() call.

    Device-resident only: every static tensor lives on the training
    device. The fit() inner loop indexes the pre-stitched
    ``combined_peptide`` (and ``combined_allele``) buffers directly,
    so there are no per-batch H2D copies and no DataLoader workers.

    ``random_negative_x_peptide`` and ``x_peptide`` are views into
    ``combined_peptide`` (top slice = RN, bottom slice = real); the
    same applies to alleles. Refilling RN bytes propagates to ``x_*``
    automatically.

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
    # Pre-stitched (RN | real) device tensors that the inner fit() loop
    # indexes into directly. Top slice is RN (refilled per cycle by
    # RandomNegativesPool); bottom slice is the static real-data block.
    combined_peptide: Any = None
    combined_allele: Any = None

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
        eliminating per-batch H2D copies. ``train_indices`` already
        addresses the unified ``[0, num_rn + num_real)`` space (see
        ``y_encoded`` / ``sample_weights_with_negatives`` construction
        in ``fit``), so no index translation is needed.

        Inputs may be numpy arrays or CPU torch tensors; the constructor
        handles the .to(device) move uniformly.
        """
        def _to_device(value, dtype=None):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value)
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
            if isinstance(template, torch.Tensor):
                rn_shape = tuple(template.shape)
                rn_dtype = template.dtype
            else:
                rn_shape = tuple(template.shape)
                rn_dtype = torch.as_tensor(template).dtype
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
            combined_peptide = torch.empty(
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
                    torch.float32
                    if (rn_allele_dev.dtype.is_floating_point
                        or x_allele_dev.dtype.is_floating_point)
                    else x_allele_dev.dtype
                )
                rn_allele_dev = rn_allele_dev.to(target_dtype)
                x_allele_dev = x_allele_dev.to(target_dtype)
            num_rn_a = int(rn_allele_dev.shape[0])
            num_real_a = int(x_allele_dev.shape[0])
            combined_allele = torch.empty(
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
            y_encoded=_to_device(y_encoded, dtype=torch.float32),
            sample_weights=_to_device(sample_weights, dtype=torch.float32),
            random_negative_x_peptide=rn_peptide_view,
            random_negative_x_allele=rn_allele_view,
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
    "lookup_pool_dir",
    "setup_shared_random_negative_pools",
]

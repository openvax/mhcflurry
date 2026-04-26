"""Shared-memory primitives used by mhcflurry training.

This module is the consolidated location for shared-memory
infrastructure used across mhcflurry. There are two distinct shared-
memory layers in production training, with different lifecycles and
different OS mechanisms:

Layer 1 — per-run mmap (orchestrator-built, read-only)
    Random-negative pool, encoding cache, and any other large blob
    that is identical across many work items but expensive to
    rebuild. Built ONCE by the training orchestrator before workers
    are forked, then accessed read-only from worker processes via
    ``numpy.memmap`` or equivalent. The OS page cache holds a single
    resident copy; spawning N workers does not multiply RSS.

    Mechanism: filesystem-backed mmap. Persists to disk so it can be
    reused across runs (warm cache).

    Helpers exposed here:
      * ``setup_shared_random_negative_pools(...)`` — orchestrator-side
        write of per-fold pools.
      * ``RandomNegativesPool.write_shared_pool`` /
        ``.from_shared_mmap`` (in ``random_negative_peptides.py``) —
        the underlying primitive.

Layer 2 — per-fit() torch shared tensors (worker-built, read+write)
    Inside a single ``Class1NeuralNetwork.fit()`` call, the dataset's
    backing arrays are materialized as ``torch.Tensor`` instances in
    POSIX shared memory (``/dev/shm``) so the inner DataLoader's
    spawn workers receive storage handles instead of byte copies.
    Lifetime is one ``fit()`` call (one work item).

    Mechanism: ``torch.Tensor.share_memory_()`` (POSIX shm, in-memory).
    Per-fit allocation; no on-disk persistence.

    Helpers exposed here (re-exported from ``class1_neural_network``):
      * ``to_shared_tensor`` — clone an array into a SHM tensor.
      * ``allocate_shared_neg_buffer`` — fixed-size SHM buffer for the
        random-negative payload (refilled in place each epoch).
      * ``refill_shared_neg_buffer`` — in-place copy from new source.
      * ``array_nbytes`` — type-polymorphic size helper.

Why two mechanisms?
    Layer 1 is per-run / cross-worker / read-only — file-backed mmap
    is the natural fit (persists across runs; no need to keep the
    encoded blob alive in any single process). Layer 2 is per-fit /
    intra-worker / mutated each epoch — POSIX shm via torch is the
    natural fit (in-memory, fast refill, auto-cleanup on process
    teardown). Forcing both onto one mechanism would either lose
    cross-run persistence (Layer 1 → torch shm = re-encode every run)
    or add disk-journal overhead per epoch (Layer 2 → mmap = write
    random-neg buffer to disk every epoch). The asymmetry is
    intentional; both layers conceptually do the same thing
    (one-resident-copy, many-readers).

This module is the canonical place to add new shared-memory helpers.
Per-fit tensor helpers live in ``class1_neural_network`` for backward
compatibility but are re-exported here for callers that want a single
import surface.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Optional

# Re-export Layer-2 (per-fit tensor SHM) helpers. Defined in
# class1_neural_network for historical reasons; aliasing them here so
# callers can ``from mhcflurry.shared_memory import to_shared_tensor``.
from .class1_neural_network import (
    _allocate_shared_neg_buffer as allocate_shared_neg_buffer,
    _array_nbytes as array_nbytes,
    _fit_dataloader_shm_enabled as fit_dataloader_shm_enabled,
    _refill_shared_neg_buffer as refill_shared_neg_buffer,
    _to_shared_tensor as to_shared_tensor,
)

from .random_negative_peptides import (
    RandomNegativePeptides,
    RandomNegativesPool,
)


__all__ = [
    "allocate_shared_neg_buffer",
    "array_nbytes",
    "fit_dataloader_shm_enabled",
    "refill_shared_neg_buffer",
    "setup_shared_random_negative_pools",
    "to_shared_tensor",
]


def _planner_from_hyperparameters(hyperparameters):
    """Build a ``RandomNegativePeptides`` planner from a hyperparameter dict.

    Reads only the random-negative subset of hyperparameters; ignores
    everything else. The orchestrator uses this to construct planners
    keyed on (fold, random-negative config tuple) before forking
    workers.
    """
    rn_keys = RandomNegativePeptides.hyperparameter_defaults.defaults.keys()
    sub = {k: hyperparameters[k] for k in rn_keys if k in hyperparameters}
    return RandomNegativePeptides(**sub)


def _random_negative_config_key(hyperparameters):
    """Stable tuple key over the random-negative hyperparameter subset.

    Two work items whose plans would be identical (given the same
    training data) share the same key. The orchestrator builds one
    pool per (fold, key); work items look up their pool by their own
    (fold, key).
    """
    rn_keys = sorted(
        RandomNegativePeptides.hyperparameter_defaults.defaults.keys()
    )
    items = []
    for k in rn_keys:
        v = hyperparameters.get(k)
        # Lists (e.g. random_negative_lengths) need to be hashable.
        if isinstance(v, list):
            v = tuple(v)
        items.append((k, v))
    return tuple(items)


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
    each unique (fold, random-negative-config) tuple present in
    ``work_items``, computes the plan against that fold's training
    data and writes a Phase-3 mmap pool under
    ``output_root_dir/fold_{fold}/cfg_{idx}/``.

    Parameters
    ----------
    output_root_dir : str
        Directory under which per-fold subdirectories are written. The
        caller is responsible for cleanup at run end (typically the
        run output dir, so it gets archived alongside models).
    work_items : list of dict
        Same structure as ``train_pan_allele_models_command``'s
        ``work_items``. Reads ``fold_num`` and ``hyperparameters``.
    train_data_df : pandas.DataFrame
        Full training data. Must have columns: peptide, allele,
        measurement_value, optionally measurement_inequality.
    folds_df : pandas.DataFrame
        Per-row fold-membership boolean columns (``fold_0``,
        ``fold_1``, ...). Same row index as ``train_data_df``.
    peptide_encoder : callable
        Receives an ``EncodableSequences`` and returns an encoded
        numpy array. Pass
        ``Class1NeuralNetwork.peptides_to_network_input`` from a
        sentinel network configured with the run's peptide-encoding
        hyperparameters.
    pool_epochs : int
        Number of epochs of random negatives to encode into each
        pool. Larger = lower per-epoch generation cost but bigger
        on-disk pool. Must equal the work items'
        ``random_negative_pool_epochs`` hyperparameter.
    seed : int
        Cross-cycle determinism seed. The orchestrator typically
        passes a per-run seed; per-worker permutation diversity comes
        from the ``permutation_seed`` passed to
        ``RandomNegativesPool.from_shared_mmap`` at worker time.
    log : callable, optional
        ``log("...")`` is called at each pool boundary. Defaults to
        ``logging.info``.

    Returns
    -------
    dict
        ``{(fold_num, config_key): pool_dir_path}`` — workers look
        their pool up by ``(work_item['fold_num'],
        _random_negative_config_key(work_item['hyperparameters']))``.
    """
    if log is None:
        log = logging.info

    by_fold_and_cfg = {}
    for item in work_items:
        fold_num = int(item["fold_num"])
        hp = item["hyperparameters"]
        if int(hp.get("random_negative_pool_epochs", 1) or 1) != pool_epochs:
            raise ValueError(
                "setup_shared_random_negative_pools: work item %r has "
                "random_negative_pool_epochs=%r but caller asked for "
                "pool_epochs=%d. All work items must share the same "
                "pool_epochs to use the shared mmap path."
                % (
                    item.get("work_item_name"),
                    hp.get("random_negative_pool_epochs"),
                    pool_epochs,
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

        # Build planner against this fold's training data.
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
            "(pool_epochs=%d, total_count=%d, seed=%d)"
            % (
                fold_num,
                cfg_idx,
                fold_dir,
                pool_epochs,
                planner.get_total_count(),
                seed,
            )
        )
        RandomNegativesPool.write_shared_pool(
            output_dir=fold_dir,
            planner=planner,
            peptide_encoder=peptide_encoder,
            pool_epochs=pool_epochs,
            seed=seed + fold_num,  # distinct seed per fold
        )
        fold_pool_dirs[(fold_num, cfg_key)] = fold_dir

    return fold_pool_dirs


def lookup_pool_dir_for_work_item(fold_pool_dirs, work_item):
    """Look up the shared-pool dir for a work item.

    Returns the path string when the orchestrator has registered a
    pool for this work item's (fold, random-negative-config), or
    ``None`` when the work item should fall back to in-process
    ``RandomNegativesPool`` construction.
    """
    if not fold_pool_dirs:
        return None
    key = (
        int(work_item["fold_num"]),
        _random_negative_config_key(work_item["hyperparameters"]),
    )
    return fold_pool_dirs.get(key)

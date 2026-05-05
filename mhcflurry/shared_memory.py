"""Opt-in mmap random-negative pools used by affinity training workers.

This module owns only the random-negative mmap pool. The default affinity
``fit()`` path keeps one fit's row space in device tensors, implemented in
``class1_affinity_training_data.AffinityDeviceTrainingData``.

When ``--random-negative-shared-pool-dir`` is set, the training orchestrator
builds each work item's first encoded random-negative pool before forking local
workers. Workers reopen their own files with ``numpy.memmap`` and populate later
cycles lazily in the same directory. The mmap path avoids a long-lived Python
heap array and keeps a bounded ``pool_epochs`` window on disk without making
different model fits share the same random negatives.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict

from .random_negative_peptides import (
    RandomNegativePeptides,
    RandomNegativesPool,
)


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


def _work_item_identity(item):
    """Stable identity for a model fit work item."""
    return (
        str(item.get("architecture_num", "")),
        str(item.get("fold_num", "")),
        str(item.get("replicate_num", "")),
        str(item.get("work_item_name", "")),
    )


def random_negative_seed_for_work_item(item):
    """Return the deterministic random-negative seed for one work item."""
    return int(
        hashlib.sha1("|".join(_work_item_identity(item)).encode()).hexdigest()[:16],
        16,
    )


def _random_negative_pool_key(item):
    """Pool lookup key preserving per-work-item random-negative identity."""
    return (
        int(item["fold_num"]),
        _random_negative_config_key(item["hyperparameters"]),
        _work_item_identity(item),
    )


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
    each work item in ``work_items``,
    computes the plan against that fold's training data and writes a
    mmap pool under ``output_root_dir/fold_{fold}/work_{idx}/``.

    Returns ``{(fold_num, config_key, work_item_identity): pool_dir}`` for worker-side
    lookup via ``lookup_pool_dir``.
    """
    if log is None:
        log = logging.info

    by_pool_key: Dict[Any, Any] = {}
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
        # The orchestrator builds each work item's cycle 0 before forking.
        # Workers open their own pool with the model's peptide encoder, so
        # later cycles are populated lazily under per-cycle mmap subdirectories.
        # Keeping work items separate preserves the 2.1/2.2 behavior where each
        # model fit samples its own random negatives, rather than making all
        # workers for a fold/config train against the same negative peptides.
        by_pool_key[_random_negative_pool_key(item)] = item

    fold_pool_dirs: Dict[Any, str] = {}
    for cfg_idx, (pool_key, item) in enumerate(
        sorted(by_pool_key.items())
    ):
        fold_num = pool_key[0]
        hp = item["hyperparameters"]
        fold_dir = os.path.join(
            output_root_dir, "fold_%d" % fold_num, "work_%d" % cfg_idx
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
        work_item_seed = random_negative_seed_for_work_item(item)
        log(
            "shared_memory: writing fold=%d cfg=%d pool to %s "
            "(pool_epochs=%d, total_count=%d, work_item_seed=%d)" % (
                fold_num, cfg_idx, fold_dir, pool_epochs,
                planner.get_total_count(), work_item_seed,
            )
        )
        RandomNegativesPool.write_shared_pool(
            output_dir=fold_dir,
            planner=planner,
            peptide_encoder=peptide_encoder,
            pool_epochs=pool_epochs,
            seed=work_item_seed,
        )
        fold_pool_dirs[pool_key] = fold_dir

    return fold_pool_dirs


def lookup_pool_dir(fold_pool_dirs, *, fold_num, hyperparameters, work_item=None):
    """Worker-side O(1) lookup. Returns dir or None if no match.

    ``None`` is the signal for the worker's fit() to fall back to the
    in-process pool — same surface area as "run mmap cache not enabled."
    """
    if not fold_pool_dirs:
        return None
    if work_item is None:
        # Compatibility with older callers/tests: only succeeds against
        # legacy two-part maps. New shared pools are keyed per work item.
        return fold_pool_dirs.get(
            (int(fold_num), _random_negative_config_key(hyperparameters))
        )
    return fold_pool_dirs.get(_random_negative_pool_key(work_item))


__all__ = [
    "lookup_pool_dir",
    "random_negative_seed_for_work_item",
    "setup_shared_random_negative_pools",
]

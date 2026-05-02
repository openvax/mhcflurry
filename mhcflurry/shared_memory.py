"""Opt-in mmap random-negative pools used by affinity training workers.

This module owns only the random-negative mmap pool. The default affinity
``fit()`` path keeps one fit's row space in device tensors, implemented in
``class1_affinity_training_data.AffinityDeviceTrainingData``.

When ``--random-negative-shared-pool-dir`` is set, the training orchestrator
builds encoded random-negative pools before forking local workers. Workers
reopen those files with ``numpy.memmap`` so the operating-system page cache can
hold one resident copy across processes. This is useful for host/vector encoded
training or worker fan-out where regenerating identical host pools dominates.
It is deliberately opt-in because the default torch-index path can generate
random negatives directly for the active torch device.
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
    mmap pool under ``output_root_dir/fold_{fold}/cfg_{idx}/``.

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
    "lookup_pool_dir",
    "setup_shared_random_negative_pools",
]

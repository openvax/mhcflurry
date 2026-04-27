"""Tests for the consolidated shared-memory module.

Covers run-mmap-cache (per-run mmap random-negative pool) wiring. The
fit DataLoader SHM (per-fit tensor SHM) helpers are re-exports and
tested in ``test_pytorch_regressions.py``; this file only checks
they're importable from the new module.
"""

import os
import pytest

import numpy as np
import pandas as pd

from mhcflurry import shared_memory as shm
from mhcflurry.testing_utils import startup, cleanup


@pytest.fixture(autouse=True)
def setup_teardown():
    startup()
    yield
    cleanup()


def test_layer2_public_api():
    """All fit DataLoader SHM helpers are importable by their canonical names."""
    assert callable(shm.share_tensor)
    assert callable(shm.share_like)
    assert callable(shm.update_shared)
    assert callable(shm.array_nbytes)
    assert callable(shm.numpy_batch_collate)
    assert callable(shm.tensor_batch_collate)
    assert hasattr(shm, "FitBacking")


def test_random_negative_config_key_stable_across_dict_order():
    """Equivalent random-neg configs must hash to the same key."""
    a = {
        "random_negative_rate": 1.0,
        "random_negative_constant": 1,
        "random_negative_method": "by_allele",
        "random_negative_lengths": [9, 10, 11],
        "unrelated": "foo",
    }
    b = {
        "unrelated": "bar",
        "random_negative_lengths": [9, 10, 11],
        "random_negative_method": "by_allele",
        "random_negative_constant": 1,
        "random_negative_rate": 1.0,
    }
    assert shm._random_negative_config_key(a) == shm._random_negative_config_key(b)


def test_random_negative_config_key_distinguishes_distinct_configs():
    a = {"random_negative_rate": 1.0, "random_negative_constant": 1}
    b = {"random_negative_rate": 2.0, "random_negative_constant": 1}
    assert shm._random_negative_config_key(a) != shm._random_negative_config_key(b)


def test_planner_from_hyperparameters_extracts_subset():
    """Planner ignores non-random-negative hyperparameters."""
    hp = {
        "random_negative_rate": 1.0,
        "random_negative_constant": 0,
        "random_negative_method": "by_allele",
        "random_negative_lengths": [9],
        "minibatch_size": 4096,  # ignored
        "layer_sizes": [256],    # ignored
    }
    planner = shm._planner_from_hyperparameters(hp)
    assert planner.hyperparameters["random_negative_rate"] == 1.0
    assert planner.hyperparameters["random_negative_method"] == "by_allele"


def _make_minimal_train_data(num_rows=20):
    """Build minimal pandas data structures matching the orchestrator's contract."""
    rng = np.random.default_rng(42)
    peptides = ["A" * 9 for _ in range(num_rows)]
    df = pd.DataFrame({
        "peptide": peptides,
        "allele": ["HLA-A*02:01"] * num_rows,
        "measurement_value": rng.uniform(50.0, 50000.0, size=num_rows),
        "measurement_inequality": ["="] * num_rows,
    })
    folds_df = pd.DataFrame({
        "fold_0": [True] * num_rows,
    })
    return df, folds_df


def _trivial_peptide_encoder(encodable_sequences):
    """Tiny encoder for tests: maps each peptide to a length-fixed int8 row."""
    seqs = list(encodable_sequences.sequences)
    return np.zeros((len(seqs), 9), dtype=np.int8)


def test_setup_shared_random_negative_pools_builds_files(tmp_path):
    """Orchestrator helper writes manifest + encoded mmap per (fold, cfg)."""
    df, folds_df = _make_minimal_train_data(num_rows=20)
    hp = {
        "random_negative_rate": 1.0,
        "random_negative_constant": 0,
        "random_negative_method": "by_allele",
        "random_negative_lengths": [9],
        "random_negative_match_distribution": False,
        "random_negative_pool_epochs": 3,
    }
    work_items = [
        {"fold_num": 0, "hyperparameters": hp, "work_item_name": "wi-0"},
        {"fold_num": 0, "hyperparameters": hp, "work_item_name": "wi-1"},
    ]
    fold_pool_dirs = shm.setup_shared_random_negative_pools(
        output_root_dir=str(tmp_path),
        work_items=work_items,
        train_data_df=df,
        folds_df=folds_df,
        peptide_encoder=_trivial_peptide_encoder,
        pool_epochs=3,
        seed=12345,
    )
    assert len(fold_pool_dirs) == 1  # one (fold, cfg) entry
    pool_dir = list(fold_pool_dirs.values())[0]
    # Phase 3 contract: pool dir contains manifest + encoded + peptides files.
    assert os.path.exists(os.path.join(pool_dir, "random_negatives_pool.json"))
    assert os.path.exists(
        os.path.join(pool_dir, "random_negatives_encoded.int8.mmap")
    )
    assert os.path.exists(os.path.join(pool_dir, "random_negatives_peptides.json"))


def test_setup_shared_random_negative_pools_rejects_max_epochs_exceeds_pool(tmp_path):
    """Reject when ``max_epochs > pool_epochs`` — would crash mid-training.

    The shared mmap pool covers only epochs 0..pool_epochs-1. If
    ``max_epochs`` exceeds that, the worker's ``get_epoch_inputs``
    would raise mid-training. Caught at orchestrator time so users
    don't waste hours of training on a doomed run.
    """
    df, folds_df = _make_minimal_train_data(num_rows=20)
    hp = {
        "random_negative_rate": 1.0,
        "random_negative_constant": 0,
        "random_negative_method": "by_allele",
        "random_negative_lengths": [9],
        "random_negative_match_distribution": False,
        "random_negative_pool_epochs": 3,
        "max_epochs": 100,  # > pool_epochs=3 → must reject
    }
    work_items = [
        {"fold_num": 0, "hyperparameters": hp, "work_item_name": "wi"},
    ]
    with pytest.raises(ValueError, match="max_epochs"):
        shm.setup_shared_random_negative_pools(
            output_root_dir=str(tmp_path),
            work_items=work_items,
            train_data_df=df,
            folds_df=folds_df,
            peptide_encoder=_trivial_peptide_encoder,
            pool_epochs=3,
            seed=1,
        )


def test_setup_shared_random_negative_pools_accepts_max_epochs_equals_pool(tmp_path):
    """Boundary: ``max_epochs == pool_epochs`` is fine (epochs 0..N-1 covered)."""
    df, folds_df = _make_minimal_train_data(num_rows=20)
    hp = {
        "random_negative_rate": 1.0,
        "random_negative_constant": 0,
        "random_negative_method": "by_allele",
        "random_negative_lengths": [9],
        "random_negative_match_distribution": False,
        "random_negative_pool_epochs": 5,
        "max_epochs": 5,  # equal to pool_epochs → epochs 0..4 are covered
    }
    work_items = [
        {"fold_num": 0, "hyperparameters": hp, "work_item_name": "wi"},
    ]
    fold_pool_dirs = shm.setup_shared_random_negative_pools(
        output_root_dir=str(tmp_path),
        work_items=work_items,
        train_data_df=df,
        folds_df=folds_df,
        peptide_encoder=_trivial_peptide_encoder,
        pool_epochs=5,
        seed=1,
    )
    assert len(fold_pool_dirs) == 1


def test_setup_shared_random_negative_pools_rejects_pool_epochs_mismatch(tmp_path):
    """All work items must share the same pool_epochs."""
    df, folds_df = _make_minimal_train_data(num_rows=20)
    hp_3 = {
        "random_negative_rate": 1.0,
        "random_negative_constant": 0,
        "random_negative_method": "by_allele",
        "random_negative_lengths": [9],
        "random_negative_match_distribution": False,
        "random_negative_pool_epochs": 3,
    }
    hp_5 = dict(hp_3, random_negative_pool_epochs=5)
    work_items = [
        {"fold_num": 0, "hyperparameters": hp_3, "work_item_name": "a"},
        {"fold_num": 0, "hyperparameters": hp_5, "work_item_name": "b"},
    ]
    with pytest.raises(ValueError, match="pool_epochs"):
        shm.setup_shared_random_negative_pools(
            output_root_dir=str(tmp_path),
            work_items=work_items,
            train_data_df=df,
            folds_df=folds_df,
            peptide_encoder=_trivial_peptide_encoder,
            pool_epochs=3,
            seed=1,
        )


def test_lookup_pool_dir_round_trip(tmp_path):
    """Workers find their own (fold, cfg) pool via the lookup helper."""
    df, folds_df = _make_minimal_train_data(num_rows=20)
    hp = {
        "random_negative_rate": 1.0,
        "random_negative_constant": 0,
        "random_negative_method": "by_allele",
        "random_negative_lengths": [9],
        "random_negative_match_distribution": False,
        "random_negative_pool_epochs": 2,
    }
    work_items = [
        {"fold_num": 0, "hyperparameters": hp, "work_item_name": "wi"},
    ]
    fold_pool_dirs = shm.setup_shared_random_negative_pools(
        output_root_dir=str(tmp_path),
        work_items=work_items,
        train_data_df=df,
        folds_df=folds_df,
        peptide_encoder=_trivial_peptide_encoder,
        pool_epochs=2,
        seed=1,
    )
    found = shm.lookup_pool_dir(
        fold_pool_dirs,
        fold_num=work_items[0]["fold_num"],
        hyperparameters=work_items[0]["hyperparameters"],
    )
    assert found is not None
    assert os.path.isdir(found)


def test_lookup_pool_dir_returns_none_when_no_pools():
    """Empty/None pool dirs map -> caller falls back to in-process pool."""
    work_item = {
        "fold_num": 0,
        "hyperparameters": {"random_negative_rate": 1.0},
    }
    assert shm.lookup_pool_dir(
        None, fold_num=work_item["fold_num"],
        hyperparameters=work_item["hyperparameters"],
    ) is None
    assert shm.lookup_pool_dir(
        {}, fold_num=work_item["fold_num"],
        hyperparameters=work_item["hyperparameters"],
    ) is None


def test_lookup_pool_dir_returns_none_when_unmatched(tmp_path):
    """A work item whose (fold, cfg) was not pre-built returns None."""
    df, folds_df = _make_minimal_train_data(num_rows=20)
    hp_built = {
        "random_negative_rate": 1.0,
        "random_negative_constant": 0,
        "random_negative_method": "by_allele",
        "random_negative_lengths": [9],
        "random_negative_match_distribution": False,
        "random_negative_pool_epochs": 2,
    }
    work_items_built = [
        {"fold_num": 0, "hyperparameters": hp_built, "work_item_name": "wi"},
    ]
    fold_pool_dirs = shm.setup_shared_random_negative_pools(
        output_root_dir=str(tmp_path),
        work_items=work_items_built,
        train_data_df=df,
        folds_df=folds_df,
        peptide_encoder=_trivial_peptide_encoder,
        pool_epochs=2,
        seed=1,
    )
    # Different config (different rate) → no match.
    hp_other = dict(hp_built, random_negative_rate=2.0)
    other_work_item = {"fold_num": 0, "hyperparameters": hp_other}
    assert shm.lookup_pool_dir(
        fold_pool_dirs,
        fold_num=other_work_item["fold_num"],
        hyperparameters=other_work_item["hyperparameters"],
    ) is None

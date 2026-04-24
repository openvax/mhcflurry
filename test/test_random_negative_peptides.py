"""Tests for random negative peptide generation."""

import numpy
import pandas
import math

from mhcflurry.common import random_peptides
from mhcflurry.random_negative_peptides import (
    RandomNegativePeptides,
    RandomNegativesPool,
)


def test_random_negative_peptides_by_allele_equalize_nonbinders():
    planner = RandomNegativePeptides(
        random_negative_method="by_allele",
        random_negative_binder_threshold=500,
        random_negative_rate=1.0,
        random_negative_constant=2)

    data_rows = [
        ("HLA-A*02:01", "SIINFEKL", 400, "="),
        ("HLA-A*02:01", "SIINFEKLL", 300, "="),
        ("HLA-A*02:01", "SIINFEKLL", 300, "="),
        ("HLA-A*02:01", "SIINFEKLQ", 1000, "="),
        ("HLA-A*02:01", "SIINFEKLZZ", 12000, ">"),
    ]
    for peptide in random_peptides(1000, length=9):
        data_rows.append(("HLA-B*44:02", peptide, 100, "="))
    for peptide in random_peptides(1000, length=9):
        data_rows.append(("HLA-B*44:02", peptide, 1000, "="))
    for peptide in random_peptides(5, length=10):
        data_rows.append(("HLA-B*44:02", peptide, 100, "="))

    data = pandas.DataFrame(
        data_rows,
        columns=["allele", "peptide", "affinity", "inequality"])
    data["length"] = data.peptide.str.len()

    planner.plan(
        peptides=data.peptide.values,
        affinities=data.affinity.values,
        alleles=data.allele.values,
        inequalities=data.inequality.values)
    result_df = pandas.DataFrame({
        "allele": planner.get_alleles(),
        "peptide": planner.get_peptides(),
    })

    result_df["length"] = result_df.peptide.str.len()
    random_negatives = result_df.groupby(["allele", "length"]).peptide.count().unstack()
    data.groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    data.loc[
        data.affinity <= 500
    ].groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    real_nonbinders = data.loc[
        data.affinity > 500
    ].groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    random_negatives + real_nonbinders

    assert (random_negatives.loc["HLA-A*02:01"] == 1.0).all()
    assert (random_negatives.loc["HLA-B*44:02"] == math.ceil(1007 / 8)).all(), (
        random_negatives.loc["HLA-B*44:02"], math.ceil(1007 / 8))



def test_random_negative_peptides_by_allele():
    planner = RandomNegativePeptides(
        random_negative_method="by_allele_equalize_nonbinders",
        random_negative_binder_threshold=500,
        random_negative_rate=1.0,
        random_negative_constant=2)
    data_rows = [
        ("HLA-A*02:01", "SIINFEKL", 400, "="),
        ("HLA-A*02:01", "SIINFEKLL", 300, "="),
        ("HLA-A*02:01", "SIINFEKLL", 300, "="),
        ("HLA-A*02:01", "SIINFEKLQ", 1000, "="),
        ("HLA-A*02:01", "SIINFEKLZZ", 12000, ">"),
        ("HLA-C*01:02", "SIINFEKLQ", 100, "="),  # only binders
        ("HLA-C*07:02", "SIINFEKLL", 1000, "=")   # only non-binders

    ]
    for peptide in random_peptides(1000, length=9):
        data_rows.append(("HLA-B*44:02", peptide, 100, "="))
    for peptide in random_peptides(1000, length=9):
        data_rows.append(("HLA-B*44:02", peptide, 1000, "="))
    for peptide in random_peptides(5, length=10):
        data_rows.append(("HLA-B*44:02", peptide, 100, "="))

    data = pandas.DataFrame(
        data_rows,
        columns=["allele", "peptide", "affinity", "inequality"])
    data["length"] = data.peptide.str.len()

    planner.plan(
        peptides=data.peptide.values,
        affinities=data.affinity.values,
        alleles=data.allele.values,
        inequalities=data.inequality.values)
    result_df = pandas.DataFrame({
        "allele": planner.get_alleles(),
        "peptide": planner.get_peptides(),
    })
    result_df["length"] = result_df.peptide.str.len()
    random_negatives = result_df.groupby(["allele", "length"]).peptide.count().unstack()
    real_data = data.groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    data.loc[
        data.affinity <= 500
    ].groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    real_nonbinders = data.loc[
        data.affinity > 500
    ].groupby(["allele", "length"]).peptide.count().unstack().fillna(0)
    for length in random_negatives.columns:
        if length not in real_nonbinders.columns:
            real_nonbinders[length] = 0
    total_nonbinders = (
            random_negatives.reindex(real_data.index).fillna(0) +
            real_nonbinders.reindex(real_data.index).fillna(0))

    assert (total_nonbinders.loc["HLA-A*02:01"] == 2.0).all(), total_nonbinders
    assert (total_nonbinders.loc["HLA-B*44:02"] == 1126).all(), total_nonbinders

    assert not total_nonbinders.isnull().any().any()


def _planner_for_pool_tests():
    planner = RandomNegativePeptides(
        random_negative_rate=1.0,
        random_negative_constant=0,
        random_negative_method="by_length",
        random_negative_lengths=[9],
    )
    planner.plan(
        peptides=[
            "SIINFEKLA",
            "GILGFVFTL",
            "SLYNTVATL",
            "NLVPMVATV",
        ],
        affinities=[10.0, 100.0, 1000.0, 5000.0],
    )
    return planner


def _tag_encoder(encodable_sequences):
    # Trivial encoder for the pool tests — we care about slicing/determinism,
    # not the actual encoding. Treat each peptide string as one row.
    return numpy.array(list(encodable_sequences.sequences), dtype=object)


def test_random_negatives_pool_pool_epochs_one_is_per_epoch():
    # With pool_epochs=1 every epoch triggers a rebuild, so the slice
    # returned for each consecutive epoch is a fresh draw (different
    # peptides on every call) — matching pre-Phase-1 fit() semantics.
    planner = _planner_for_pool_tests()
    pool = RandomNegativesPool(planner, _tag_encoder, pool_epochs=1, seed=None)
    epoch0 = pool.get_epoch_inputs(0)[1]
    epoch1 = pool.get_epoch_inputs(1)[1]
    assert len(epoch0) == planner.get_total_count()
    assert len(epoch1) == planner.get_total_count()
    # Not a strict guarantee across RNG states, but extremely likely
    # for 4 9-mers drawn independently.
    assert not numpy.array_equal(epoch0, epoch1), (
        "pool_epochs=1 must regenerate every epoch"
    )


def test_random_negatives_pool_within_cycle_slices_are_distinct():
    planner = _planner_for_pool_tests()
    pool = RandomNegativesPool(planner, _tag_encoder, pool_epochs=3, seed=1234)
    total = planner.get_total_count()
    e0 = pool.get_epoch_inputs(0)[1]
    e1 = pool.get_epoch_inputs(1)[1]
    e2 = pool.get_epoch_inputs(2)[1]
    assert len(e0) == total and len(e1) == total and len(e2) == total
    # Three slices inside a single cycle must be disjoint sections of the
    # pool — not duplicates of each other.
    assert not numpy.array_equal(e0, e1)
    assert not numpy.array_equal(e1, e2)
    assert not numpy.array_equal(e0, e2)


def test_random_negatives_pool_cycle_boundary_triggers_rebuild():
    planner = _planner_for_pool_tests()
    pool = RandomNegativesPool(planner, _tag_encoder, pool_epochs=3, seed=1234)
    e0 = pool.get_epoch_inputs(0)[1]
    # Epoch 3 is the first epoch of cycle 1 → new seed → new peptides.
    e3 = pool.get_epoch_inputs(3)[1]
    assert not numpy.array_equal(e0, e3)


def test_random_negatives_pool_deterministic_with_seed():
    planner = _planner_for_pool_tests()
    pool_a = RandomNegativesPool(planner, _tag_encoder, pool_epochs=5, seed=7)
    pool_b = RandomNegativesPool(planner, _tag_encoder, pool_epochs=5, seed=7)
    for epoch in [0, 2, 9, 12]:
        numpy.testing.assert_array_equal(
            pool_a.get_epoch_inputs(epoch)[1],
            pool_b.get_epoch_inputs(epoch)[1],
        )


def test_random_negatives_pool_different_seeds_diverge():
    planner = _planner_for_pool_tests()
    pool_a = RandomNegativesPool(planner, _tag_encoder, pool_epochs=5, seed=7)
    pool_b = RandomNegativesPool(planner, _tag_encoder, pool_epochs=5, seed=8)
    # Two workers with different seeds must see different random-negative
    # content — preserves cross-worker diversity when the pool is shared
    # across epochs within a worker.
    assert not numpy.array_equal(
        pool_a.get_epoch_inputs(0)[1],
        pool_b.get_epoch_inputs(0)[1],
    )


# ---- Phase 3 of openvax/mhcflurry#268: shared-mmap pool ----

def _int8_encoder(encodable_sequences):
    # Deterministic int8 encoder — good enough to verify the shared-mmap
    # API preserves byte-for-byte content.
    seqs = list(encodable_sequences.sequences)
    if not seqs:
        return numpy.zeros((0, 9), dtype="int8")
    length = len(seqs[0])
    arr = numpy.zeros((len(seqs), length), dtype="int8")
    for i, pep in enumerate(seqs):
        for j, aa in enumerate(pep[:length]):
            arr[i, j] = ord(aa) % 127
    return arr


def _planner_with_ten_peptides():
    planner = RandomNegativePeptides(
        random_negative_rate=1.0,
        random_negative_constant=0,
        random_negative_method="by_length",
        random_negative_lengths=[9],
    )
    peptides = [
        c * 9 for c in "ABCDEFGHIJ"
    ]
    planner.plan(peptides=peptides, affinities=[10.0] * len(peptides))
    return planner


def test_shared_pool_round_trip_preserves_content(tmp_path):
    planner = _planner_with_ten_peptides()
    RandomNegativesPool.write_shared_pool(
        str(tmp_path),
        planner,
        _int8_encoder,
        pool_epochs=3,
        seed=101,
    )
    # Loading a mmap pool without a permutation_seed must yield the
    # same bytes the writer produced.
    loaded = RandomNegativesPool.from_shared_mmap(
        str(tmp_path), planner, permutation_seed=None
    )
    reference = RandomNegativesPool(
        planner, _int8_encoder, pool_epochs=3, seed=101
    )
    for epoch in range(3):
        numpy.testing.assert_array_equal(
            numpy.asarray(loaded.get_epoch_inputs(epoch)[1]),
            numpy.asarray(reference.get_epoch_inputs(epoch)[1]),
        )


def test_shared_pool_permutation_preserves_content_across_workers(tmp_path):
    planner = _planner_with_ten_peptides()
    RandomNegativesPool.write_shared_pool(
        str(tmp_path),
        planner,
        _int8_encoder,
        pool_epochs=3,
        seed=42,
    )
    worker_a = RandomNegativesPool.from_shared_mmap(
        str(tmp_path), planner, permutation_seed=111
    )
    worker_b = RandomNegativesPool.from_shared_mmap(
        str(tmp_path), planner, permutation_seed=222
    )
    for epoch in range(3):
        a_pep, a_enc = worker_a.get_epoch_inputs(epoch)
        b_pep, b_enc = worker_b.get_epoch_inputs(epoch)
        # Permutations must diverge — otherwise we've lost worker diversity.
        assert not numpy.array_equal(numpy.asarray(a_enc), numpy.asarray(b_enc))
        # But the multisets must match — both workers read the same mmap.
        assert sorted(a_pep) == sorted(b_pep)


def test_shared_pool_same_permutation_seed_reproducible(tmp_path):
    planner = _planner_with_ten_peptides()
    RandomNegativesPool.write_shared_pool(
        str(tmp_path), planner, _int8_encoder, pool_epochs=3, seed=42,
    )
    a = RandomNegativesPool.from_shared_mmap(
        str(tmp_path), planner, permutation_seed=7
    )
    b = RandomNegativesPool.from_shared_mmap(
        str(tmp_path), planner, permutation_seed=7
    )
    for epoch in range(3):
        numpy.testing.assert_array_equal(
            numpy.asarray(a.get_epoch_inputs(epoch)[1]),
            numpy.asarray(b.get_epoch_inputs(epoch)[1]),
        )


def test_shared_pool_mismatched_planner_raises(tmp_path):
    writer_planner = _planner_with_ten_peptides()
    RandomNegativesPool.write_shared_pool(
        str(tmp_path),
        writer_planner,
        _int8_encoder,
        pool_epochs=3,
        seed=42,
    )

    # Loader planner has a different total_count — must refuse to load
    # rather than silently slice into garbage.
    loader_planner = RandomNegativePeptides(
        random_negative_rate=1.0,
        random_negative_constant=5,
        random_negative_method="by_length",
        random_negative_lengths=[9],
    )
    loader_planner.plan(
        peptides=["A" * 9, "B" * 9],
        affinities=[10.0, 100.0],
    )

    import pytest
    with pytest.raises(ValueError, match="total_count"):
        RandomNegativesPool.from_shared_mmap(str(tmp_path), loader_planner)


def test_shared_pool_write_requires_seed(tmp_path):
    planner = _planner_with_ten_peptides()
    import pytest
    with pytest.raises(ValueError, match="seed"):
        RandomNegativesPool.write_shared_pool(
            str(tmp_path), planner, _int8_encoder, pool_epochs=3, seed=None,
        )

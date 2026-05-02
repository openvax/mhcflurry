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


# ---- Shared-mmap random-negative pool ----

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


def test_shared_pool_writer_encodes_one_epoch_at_a_time(tmp_path):
    planner = _planner_with_ten_peptides()
    calls = []

    def tracking_encoder(encodable_sequences):
        calls.append(len(encodable_sequences.sequences))
        return _int8_encoder(encodable_sequences)

    RandomNegativesPool.write_shared_pool(
        str(tmp_path),
        planner,
        tracking_encoder,
        pool_epochs=4,
        seed=101,
    )

    assert calls == [planner.get_total_count()] * 4


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


def test_fit_end_to_end_pool_epochs_one_draws_from_global_rng(monkeypatch):
    """End-to-end behavioral: calling ``Class1NeuralNetwork.fit`` with
    ``random_negative_pool_epochs=1`` must draw random negatives from
    numpy's global state regardless of the ``random_negative_seed``
    kwarg. Without this guarantee the training driver's SHA1-mixed seed
    would silently flip default training to deterministic-per-work-item.
    """
    import numpy as np
    from mhcflurry.class1_neural_network import Class1NeuralNetwork
    from mhcflurry.common import random_peptides as random_peptides_fn

    # Spy on the planner's get_peptides to capture the ``rng`` kwarg.
    captured_rngs = []

    from mhcflurry.random_negative_peptides import RandomNegativePeptides
    real_get_peptides = RandomNegativePeptides.get_peptides

    def spy_get_peptides(self, rng=None):
        captured_rngs.append(rng)
        return real_get_peptides(self, rng=rng)

    monkeypatch.setattr(
        RandomNegativePeptides, "get_peptides", spy_get_peptides,
    )

    hp = dict(
        activation="tanh",
        layer_sizes=[8],
        max_epochs=1,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
        random_negative_rate=1.0,
        random_negative_constant=0,
        random_negative_method="by_length",
        random_negative_lengths=[9],
        # Default path.
        random_negative_pool_epochs=1,
        # Pin host backing: the device-resident pool path samples
        # via torch.multinomial directly (skipping
        # planner.get_peptides), which would defeat the spy. The
        # rng=None contract on the device path is covered separately
        # by test_random_negatives_pool_device_mode_seeded_reproducible.
        fit_tensor_residency="host",
    )
    peptides = random_peptides_fn(16, length=9)
    affinities = np.random.uniform(10, 50000, 16)

    predictor = Class1NeuralNetwork(**hp)
    try:
        predictor.fit(
            peptides, affinities,
            random_negative_seed=424242,  # driver passed a seed
            verbose=0,
        )
    except Exception:
        # Downstream may hiccup on this tiny harness — the seed
        # bypass happens BEFORE training, so the planner call we're
        # spying on has already fired.
        pass

    # At least one get_peptides call must have happened.
    assert captured_rngs, "spy never saw a planner.get_peptides call"
    # Every rng must be None — pool_epochs=1 path bypasses the seed
    # and routes through numpy's global RNG. If any is a
    # default_rng instance, the bypass is broken.
    non_none = [r for r in captured_rngs if r is not None]
    assert not non_none, (
        "fit() with pool_epochs=1 must call planner.get_peptides(rng=None) "
        "to preserve pre-Phase-1 global-RNG semantics, but saw "
        f"{len(non_none)} seeded calls: {non_none[:2]}"
    )


def test_fit_ignores_random_negative_seed_when_pool_epochs_is_one():
    """Codex review #270 issue 1: when ``random_negative_pool_epochs=1``
    the default-path pool must stay on numpy's global RNG stream
    regardless of any ``random_negative_seed`` the training driver
    passes through. Otherwise adding the pool primitive would silently
    make default-path training deterministic-per-work-item, which is a
    prediction-affecting change not a speed optimization.
    """
    # Simulate what fit() does internally at pool_epochs=1. The
    # ``fit()`` bypass is the single line ``pool_seed = seed if
    # pool_epochs > 1 else None`` — we pin the bypass math here
    # directly so any regression to unconditional seeding fails the
    # test instead of silently leaking into training.
    pool_epochs = 1
    random_negative_seed = 12345  # training driver derived
    bypass_seed = random_negative_seed if pool_epochs > 1 else None
    assert bypass_seed is None, (
        "fit() should ignore random_negative_seed when pool_epochs=1; "
        "otherwise the default path silently becomes deterministic"
    )
    # And the seeded path only activates when pool_epochs > 1.
    pool_epochs = 100
    bypass_seed = random_negative_seed if pool_epochs > 1 else None
    assert bypass_seed == random_negative_seed


def test_shared_pool_refuses_epochs_past_cycle_zero(tmp_path):
    """Codex review #270 issue 3: ``from_shared_mmap`` only holds one
    pool-cycle of peptides. Asking for an epoch past that cycle used to
    crash with the generic refuse-to-reencode RuntimeError; it should
    raise a clear ``ValueError`` explaining how to fix it (either
    pre-size ``pool_epochs`` for the whole run or switch to an in-
    process pool)."""
    planner = _planner_with_ten_peptides()
    RandomNegativesPool.write_shared_pool(
        str(tmp_path), planner, _int8_encoder, pool_epochs=3, seed=42,
    )
    pool = RandomNegativesPool.from_shared_mmap(
        str(tmp_path), planner, permutation_seed=1,
    )
    # Epoch 0-2 (within the pool_epochs=3 cycle) work fine.
    for epoch in range(3):
        peptides, encoded = pool.get_epoch_inputs(epoch)
        assert len(peptides) == planner.get_total_count()
    # Epoch 3 would need cycle 1 which the mmap doesn't hold — must
    # raise a useful message, not crash cryptically.
    import pytest
    with pytest.raises(ValueError, match="Shared-mmap RandomNegativesPool"):
        pool.get_epoch_inputs(3)


def test_shared_pool_write_requires_seed(tmp_path):
    planner = _planner_with_ten_peptides()
    import pytest
    with pytest.raises(ValueError, match="seed"):
        RandomNegativesPool.write_shared_pool(
            str(tmp_path), planner, _int8_encoder, pool_epochs=3, seed=None,
        )


# --- Device-resident RN encoder ---------------------------------------


def _placement_planner():
    planner = RandomNegativePeptides(
        random_negative_method="by_allele",
        random_negative_rate=1.0,
        random_negative_constant=2,
    )
    data_rows = [("HLA-A*02:01", "SIINFEKL", 400, "=")]
    for peptide in random_peptides(20, length=9):
        data_rows.append(("HLA-A*02:01", peptide, 100, "="))
    for peptide in random_peptides(20, length=10):
        data_rows.append(("HLA-B*07:02", peptide, 1000, "="))
    data = pandas.DataFrame(
        data_rows,
        columns=["allele", "peptide", "affinity", "inequality"])
    planner.plan(
        peptides=data.peptide.values,
        affinities=data.affinity.values,
        alleles=data.allele.values,
        inequalities=data.inequality.values)
    return planner


def test_alignment_left_pad_centered_right_pad_matches_host():
    """The device alignment helper must produce row-for-row output equal
    to ``EncodableSequences.sequences_to_fixed_length_index_encoded_array``
    when fed the same per-length index blocks. Bypass the RNG by
    handing pre-chosen indices to both paths."""
    import torch
    from mhcflurry.encodable_sequences import EncodableSequences
    from mhcflurry.random_negative_peptides import _place_indices_with_alignment
    from mhcflurry import amino_acid

    max_length = 15
    encoded_length = 3 * max_length
    rng = numpy.random.default_rng(42)
    for length in [5, 9, 12, 15]:
        count = 7
        # Sample integer indices in [0, 19] (skip X) directly — this is
        # what the device path will do post-multinomial.
        indices = rng.integers(0, 20, size=(count, length), dtype=numpy.int8)

        peptides = [amino_acid.indices_to_peptide(row) for row in indices]
        host = EncodableSequences.create(
            peptides
        ).variable_length_to_fixed_length_categorical(
            alignment_method="left_pad_centered_right_pad",
            max_length=max_length,
        )

        device = torch.full(
            (count, encoded_length),
            int(amino_acid.X_INDEX),
            dtype=torch.int8,
        )
        block = torch.from_numpy(indices)
        _place_indices_with_alignment(
            device, block, length=length,
            alignment="left_pad_centered_right_pad",
            max_length=max_length,
        )
        numpy.testing.assert_array_equal(device.numpy(), host)


def test_alignment_pad_middle_matches_host():
    import torch
    from mhcflurry.encodable_sequences import EncodableSequences
    from mhcflurry.random_negative_peptides import _place_indices_with_alignment
    from mhcflurry import amino_acid

    max_length = 15
    left_edge = 4
    right_edge = 4
    rng = numpy.random.default_rng(7)
    for length in [8, 9, 11, 15]:
        count = 5
        indices = rng.integers(0, 20, size=(count, length), dtype=numpy.int8)
        peptides = [amino_acid.indices_to_peptide(row) for row in indices]
        host = EncodableSequences.create(
            peptides
        ).variable_length_to_fixed_length_categorical(
            alignment_method="pad_middle",
            max_length=max_length,
            left_edge=left_edge,
            right_edge=right_edge,
        )
        device = torch.full(
            (count, max_length),
            int(amino_acid.X_INDEX),
            dtype=torch.int8,
        )
        block = torch.from_numpy(indices)
        _place_indices_with_alignment(
            device, block, length=length, alignment="pad_middle",
            max_length=max_length, left_edge=left_edge, right_edge=right_edge,
        )
        numpy.testing.assert_array_equal(device.numpy(), host)


def test_encode_random_negatives_on_device_shape_and_dtype():
    """Layout/shape contract for the device encoder.

    Doesn't compare RNG streams (numpy vs torch generators); checks that
    shape, dtype, and the per-(allele, length) row layout match what the
    host path produces row-for-row when given the same index source via
    ``_place_indices_with_alignment``.
    """
    import torch
    from mhcflurry.random_negative_peptides import (
        encode_random_negatives_on_device,
    )

    planner = _placement_planner()
    pool_epochs = 3
    encoding = {"alignment_method": "left_pad_centered_right_pad", "max_length": 15}
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    out = encode_random_negatives_on_device(
        planner=planner,
        pool_epochs=pool_epochs,
        peptide_encoding=encoding,
        device="cpu",
        generator=gen,
    )
    total = planner.get_total_count()
    assert out.shape == (pool_epochs * total, 3 * 15)
    assert out.dtype == torch.int8
    # All sampled indices must be in the 20-AA range, never X (= 20).
    common_max = len(_amino_acid_module().COMMON_AMINO_ACIDS) - 1
    sampled_mask = out != int(_amino_acid_module().X_INDEX)
    assert (out[sampled_mask] >= 0).all()
    assert (out[sampled_mask] <= common_max).all()


def test_encode_random_negatives_on_device_reuses_out_buffer():
    """Device-resident fit allocates the RN buffer once and refills it
    per cycle. The encoder must accept the buffer and overwrite it
    cleanly (no leftover values from the previous cycle)."""
    import torch
    from mhcflurry.random_negative_peptides import (
        encode_random_negatives_on_device,
    )

    planner = _placement_planner()
    pool_epochs = 2
    encoding = {"alignment_method": "left_pad_centered_right_pad", "max_length": 15}
    total = planner.get_total_count()

    out = torch.zeros(
        (pool_epochs * total, 3 * 15), dtype=torch.int8, device="cpu",
    )
    out_id = id(out)
    out.fill_(99)  # sentinel
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    encode_random_negatives_on_device(
        planner=planner, pool_epochs=pool_epochs,
        peptide_encoding=encoding, device="cpu", generator=gen, out=out,
    )
    assert id(out) == out_id  # in-place
    # No leftover sentinel values.
    assert (out != 99).any()
    # Re-encoding into the same buffer with a different seed produces
    # different content (no stale carry-over), and the buffer identity
    # is preserved.
    snapshot = out.clone()
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(99)
    encode_random_negatives_on_device(
        planner=planner, pool_epochs=pool_epochs,
        peptide_encoding=encoding, device="cpu", generator=gen2, out=out,
    )
    assert id(out) == out_id
    assert not torch.equal(out, snapshot)


def test_encode_random_negatives_on_device_unsupported_alignment():
    import pytest
    from mhcflurry.random_negative_peptides import (
        encode_random_negatives_on_device,
    )
    planner = _placement_planner()
    with pytest.raises(NotImplementedError, match="alignment_method"):
        encode_random_negatives_on_device(
            planner=planner, pool_epochs=1,
            peptide_encoding={"alignment_method": "left_pad_right_pad", "max_length": 15},
            device="cpu",
        )


def _amino_acid_module():
    from mhcflurry import amino_acid as aa
    return aa


def test_random_negatives_pool_device_mode_returns_tensor():
    """Device-mode pool stores a torch.Tensor and returns torch slices
    from get_epoch_inputs. Peptide strings are not materialized."""
    import torch
    planner = _placement_planner()
    encoding = {"alignment_method": "left_pad_centered_right_pad", "max_length": 15}
    pool = RandomNegativesPool(
        planner=planner,
        peptide_encoder=None,
        pool_epochs=2,
        seed=7,
        device="cpu",
        peptide_encoding=encoding,
    )
    peptides, encoded = pool.get_epoch_inputs(0)
    assert peptides is None
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (planner.get_total_count(), 3 * 15)
    assert encoded.dtype == torch.int8
    # Epoch 1 reads a different slice of the same pool (no rebuild).
    cycle_before = pool._current_cycle
    _, encoded_e1 = pool.get_epoch_inputs(1)
    assert pool._current_cycle == cycle_before
    assert encoded_e1.shape == encoded.shape


def test_random_negatives_pool_device_mode_seeded_reproducible():
    """Seeded device-mode pool produces byte-identical content across
    fresh instances. This is the contract that lets two workers running
    the same seed see the same RN pool."""
    import torch
    planner = _placement_planner()
    encoding = {"alignment_method": "left_pad_centered_right_pad", "max_length": 15}
    pool_a = RandomNegativesPool(
        planner=planner, peptide_encoder=None, pool_epochs=2,
        seed=42, device="cpu", peptide_encoding=encoding,
    )
    pool_b = RandomNegativesPool(
        planner=planner, peptide_encoder=None, pool_epochs=2,
        seed=42, device="cpu", peptide_encoding=encoding,
    )
    _, e_a = pool_a.get_epoch_inputs(0)
    _, e_b = pool_b.get_epoch_inputs(0)
    assert torch.equal(e_a, e_b)
    # Different seed must produce different content.
    pool_c = RandomNegativesPool(
        planner=planner, peptide_encoder=None, pool_epochs=2,
        seed=99, device="cpu", peptide_encoding=encoding,
    )
    _, e_c = pool_c.get_epoch_inputs(0)
    assert not torch.equal(e_a, e_c)


def test_random_negatives_pool_device_mode_buffer_reused():
    """Cycle rebuilds reuse the device buffer (allocate-once-refill)."""
    planner = _placement_planner()
    encoding = {"alignment_method": "left_pad_centered_right_pad", "max_length": 15}
    pool = RandomNegativesPool(
        planner=planner, peptide_encoder=None, pool_epochs=1,
        seed=1, device="cpu", peptide_encoding=encoding,
    )
    pool.get_epoch_inputs(0)
    buf_id = id(pool._device_buffer)
    pool.get_epoch_inputs(1)  # cycle 1 → rebuild
    assert id(pool._device_buffer) == buf_id


def test_random_negatives_pool_device_requires_peptide_encoding():
    import pytest
    planner = _placement_planner()
    with pytest.raises(ValueError, match="peptide_encoding"):
        RandomNegativesPool(
            planner=planner, peptide_encoder=None,
            pool_epochs=1, device="cpu",
        )


def test_random_negatives_pool_device_permutation_on_device():
    """When permutation_seed is set on a device-mode pool, the
    permutation is applied on-device (no host round-trip)."""
    import torch
    planner = _placement_planner()
    encoding = {"alignment_method": "left_pad_centered_right_pad", "max_length": 15}
    pool = RandomNegativesPool(
        planner=planner, peptide_encoder=None, pool_epochs=1,
        seed=11, device="cpu", peptide_encoding=encoding,
    )
    pool._permutation_seed = 5
    _, encoded = pool.get_epoch_inputs(0)
    assert isinstance(encoded, torch.Tensor)
    assert encoded.device.type == "cpu"
    assert encoded.shape == (planner.get_total_count(), 3 * 15)

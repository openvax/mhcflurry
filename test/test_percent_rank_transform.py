
import numpy
import pytest
import torch

from mhcflurry.percent_rank_transform import PercentRankTransform

from numpy.testing import assert_allclose, assert_equal


def test_percent_rank_transform():
    model = PercentRankTransform()
    model.fit(numpy.arange(1000), bins=100)
    assert_allclose(
        model.transform([-2, 0, 50, 100, 2000]),
        [0.0, 0.0, 5.0, 10.0, 100.0],
        err_msg=str(model.__dict__))

    model2 = PercentRankTransform.from_series(model.to_series())
    assert_allclose(
        model2.transform([-2, 0, 50, 100, 2000]),
        [0.0, 0.0, 5.0, 10.0, 100.0],
        err_msg=str(model.__dict__))

    assert_equal(model.cdf, model2.cdf)
    assert_equal(model.bin_edges, model2.bin_edges)


def test_fit_batch_torch_matches_per_row_numpy():
    rng = numpy.random.default_rng(seed=42)
    n_alleles, n_values = 12, 8000
    bin_edges = numpy.linspace(0, 50000, 1001)
    values = rng.uniform(1.0, 49999.0, size=(n_alleles, n_values))

    transforms_torch = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    assert len(transforms_torch) == n_alleles

    for i in range(n_alleles):
        legacy = PercentRankTransform()
        legacy.fit(values[i], bins=bin_edges)
        torch_t = transforms_torch[i]
        assert_allclose(torch_t.cdf, legacy.cdf, rtol=0, atol=1e-9)
        assert_equal(torch_t.bin_edges, legacy.bin_edges)
        # Spot-check transform parity end-to-end
        probes = numpy.array([-1.0, 1.0, 100.0, 25000.0, 50001.0])
        assert_allclose(torch_t.transform(probes), legacy.transform(probes))


def test_fit_batch_torch_handles_out_of_range_values():
    bin_edges = numpy.array([10.0, 20.0, 30.0, 40.0])
    # Row 0: all in-range. Row 1: half above the last edge, half in first bin.
    values = numpy.array([
        [12.0, 22.0, 32.0, 38.0],
        [11.0, 12.0, 50.0, 60.0],
    ])
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    # Row 0: hist [1, 1, 2] (12->bin0, 22->bin1, 32->bin2, 38->bin2)
    # -> cdf cumulative = 25, 50, 100
    assert_allclose(transforms[0].cdf[2:-1], [25.0, 50.0, 100.0], atol=1e-9)
    # Row 1: 2 in bin 0 (11, 12 -> [10,20)). 50 and 60 are strictly
    # above the last edge (40) and are *dropped* by numpy.histogram —
    # the fast path matches that semantics. hist=[2,0,0], cdf=[100,100,100].
    assert_allclose(transforms[1].cdf[2:-1], [100.0, 100.0, 100.0], atol=1e-9)


def test_fit_batch_torch_rejects_all_out_of_range_rows():
    bin_edges = numpy.array([10.0, 20.0, 30.0, 40.0])
    values = numpy.array([
        [12.0, 22.0, 32.0, 38.0],
        [1.0, 2.0, 3.0, 4.0],
    ])
    with pytest.raises(ValueError, match="no values inside the bin range"):
        PercentRankTransform.fit_batch_torch(
            torch.as_tensor(values, dtype=torch.float64),
            torch.as_tensor(bin_edges, dtype=torch.float64),
        )


def test_fit_batch_torch_runs_on_mps_without_float64():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    values = torch.as_tensor(
        [[12.0, 22.0, 32.0, 38.0]], dtype=torch.float32, device="mps",
    )
    bin_edges = torch.as_tensor(
        [10.0, 20.0, 30.0, 40.0], dtype=torch.float32, device="mps",
    )
    transforms = PercentRankTransform.fit_batch_torch(values, bin_edges)
    assert_allclose(transforms[0].cdf[2:-1], [25.0, 50.0, 100.0], atol=1e-5)


def test_fit_batch_torch_includes_value_at_last_edge():
    # numpy.histogram closes the LAST bin: a value exactly equal to
    # edges[-1] is counted in the last bin (not dropped). The torch
    # port uses ``torch.bucketize(..., right=True)``, which returns
    # n_edges for values == edges[-1], so we recover the closed-bin
    # semantics with an explicit equality mask. This test pins that
    # behavior.
    bin_edges = numpy.array([0.0, 1.0, 2.0, 3.0])
    values = numpy.array([[0.5, 1.5, 2.5, 3.0]])  # last value == edges[-1]
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)
    # hist = [1, 1, 2] (0.5->bin0, 1.5->bin1, 2.5->bin2, 3.0->closed bin2),
    # total=4, cumulative cdf [25, 50, 100].
    assert_allclose(
        transforms[0].cdf[2:-1], [25.0, 50.0, 100.0], atol=1e-9)


def test_fit_batch_torch_handles_values_exactly_at_intermediate_edges():
    # numpy.histogram puts values *equal to an intermediate edge* into
    # the higher bin (right-open: edges[k] is the left edge of bin k).
    # torch.bucketize(right=True) returns indices[value=edges[k]] = k+1,
    # which after the (indices-1) adjustment puts the value into bin k
    # — matching numpy.
    bin_edges = numpy.array([0.0, 1.0, 2.0, 3.0])
    values = numpy.array([[1.0, 2.0]])  # exactly at intermediate edges
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)


def test_fit_batch_torch_strict_above_last_edge_is_dropped():
    # Values strictly greater than the last edge are dropped (matches
    # numpy.histogram). Confirms in_range_interior gating doesn't
    # accidentally count overflow values.
    bin_edges = numpy.array([0.0, 1.0, 2.0])
    # Row: 1 in-range value, 1 strictly above last edge
    values = numpy.array([[0.5, 5.0]])
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)


def test_fit_batch_torch_strict_below_first_edge_is_dropped():
    bin_edges = numpy.array([0.0, 1.0, 2.0])
    values = numpy.array([[-0.5, 0.5]])
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)


def test_fit_batch_torch_includes_value_at_first_edge():
    # value == edges[0] is included in bin 0 (half-open [edges[0], edges[1])).
    bin_edges = numpy.array([0.0, 1.0, 2.0])
    values = numpy.array([[0.0, 0.5, 1.5]])
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)


def test_fit_batch_torch_single_bin():
    # n_bins == 1: ensure the cumsum/cdf bookkeeping handles the
    # smallest possible histogram.
    bin_edges = numpy.array([0.0, 10.0])
    values = numpy.array([[1.0, 2.0, 9.0, 10.0]])  # 10.0 hits closed last bin
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)
    assert transforms[0].cdf[2] == pytest.approx(100.0)


def test_fit_batch_torch_all_values_in_one_bin():
    bin_edges = numpy.array([0.0, 1.0, 2.0, 3.0])
    values = numpy.array([[1.5, 1.5, 1.5, 1.6, 1.6]])  # all in middle bin
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)
    # bin 0 empty, bin 1 has all, bin 2 empty -> cdf cumulative 0,100,100
    assert_allclose(transforms[0].cdf[2:-1], [0.0, 100.0, 100.0], atol=1e-12)


def test_fit_batch_torch_nan_inputs_dropped_like_numpy():
    # numpy.histogram silently drops NaN inputs (they don't go into any
    # bin). torch.bucketize on float64 also returns ``n_edges`` for NaN,
    # so the in_range gating already drops them. This test pins it.
    bin_edges = numpy.array([0.0, 1.0, 2.0])
    values = numpy.array([[0.5, float("nan"), 1.5]])
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)


def test_fit_batch_torch_positive_inf_dropped():
    # +inf falls strictly above the last edge → dropped (matches numpy).
    bin_edges = numpy.array([0.0, 1.0, 2.0])
    values = numpy.array([[0.5, float("inf"), 1.5]])
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    assert_allclose(transforms[0].cdf, legacy.cdf, rtol=0, atol=1e-12)


def test_fit_batch_torch_transform_at_exact_bin_edges_matches_legacy():
    # Beyond cdf parity, exercise transform() at exact edge values to
    # detect off-by-one in searchsorted vs the cdf layout.
    rng = numpy.random.default_rng(seed=1)
    bin_edges = numpy.linspace(0, 100, 11)  # 10 bins
    values = rng.uniform(1.0, 99.0, size=(1, 1000))
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    legacy = PercentRankTransform()
    legacy.fit(values[0], bins=bin_edges)
    probe_values = numpy.concatenate([
        bin_edges,                          # every exact edge
        bin_edges + 1e-12,                  # just above
        bin_edges - 1e-12,                  # just below
        numpy.array([numpy.nan, -1.0, 1000.0]),
    ])
    assert_allclose(
        transforms[0].transform(probe_values),
        legacy.transform(probe_values),
        rtol=0,
        atol=1e-9,
    )

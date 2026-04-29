
import numpy
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
    # Row 0: all in-range. Row 1: all below first edge (out of range).
    # Row 2: half in last bin (>= last edge), half in first bin.
    values = numpy.array([
        [12.0, 22.0, 32.0, 38.0],
        [1.0, 2.0, 3.0, 4.0],
        [11.0, 12.0, 50.0, 60.0],
    ])
    transforms = PercentRankTransform.fit_batch_torch(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(bin_edges, dtype=torch.float64),
    )
    # Row 0: hist [1, 1, 2] (12->bin0, 22->bin1, 32->bin2, 38->bin2)
    # -> cdf cumulative = 25, 50, 100
    assert_allclose(transforms[0].cdf[2:-1], [25.0, 50.0, 100.0], atol=1e-9)
    # Row 2: 2 in bin 0 (11, 12), 0 in bin 1, 2 in bin 2 (50, 60 fold-in to last)
    assert_allclose(transforms[2].cdf[2:-1], [50.0, 50.0, 100.0], atol=1e-9)


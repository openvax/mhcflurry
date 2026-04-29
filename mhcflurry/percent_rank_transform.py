"""
Class for transforming arbitrary values into percent ranks given a distribution.
"""
import numpy
import pandas


class PercentRankTransform(object):
    """
    Transform arbitrary values into percent ranks.
    """

    def __init__(self):
        self.cdf = None
        self.bin_edges = None

    def fit(self, values, bins):
        """
        Fit the transform using the given values (e.g. ic50s).

        Parameters
        ----------
        values : predictions (e.g. ic50 values)
        bins : bins for the cumulative distribution function
            Anything that can be passed to numpy.histogram's "bins" argument
            can be used here.
        """
        assert self.cdf is None
        assert self.bin_edges is None
        assert len(values) > 0
        (hist, self.bin_edges) = numpy.histogram(values, bins=bins)
        self.cdf = numpy.ones(len(hist) + 3) * numpy.nan
        self.cdf[0] = 0.0
        self.cdf[1] = 0.0
        self.cdf[-1] = 100.0
        numpy.cumsum(hist * 100.0 / numpy.sum(hist), out=self.cdf[2:-1])
        assert not numpy.isnan(self.cdf).any()

    @classmethod
    def fit_batch_torch(cls, values_2d, bin_edges_1d):
        """Batched GPU fit across multiple distributions.

        Equivalent to calling ``fit(values_2d[i], bins=bin_edges_1d)`` for
        each row i, but vectorized across the row dim — eliminates the
        per-allele Python loop in calibrate's hot path.

        Parameters
        ----------
        values_2d : torch.Tensor of shape (n_distributions, n_values).
            Stays on whatever device it's on; histogram bucketing happens
            there.
        bin_edges_1d : torch.Tensor of shape (n_edges,), monotonically
            increasing. Same device as values_2d.

        Returns
        -------
        List of ``PercentRankTransform`` of length n_distributions.

        Numerically equivalent to the per-row ``numpy.histogram`` path:
        same bin assignment (right-open intervals, last bin inclusive),
        same CDF normalization (% out of in-range count), bit-identical
        ``cdf`` array layout (length n_bins+3, sentinels at [0]=[1]=0,
        [-1]=100, cumsum in [2:-1]).
        """
        import torch
        assert values_2d.dim() == 2, values_2d.shape
        assert bin_edges_1d.dim() == 1, bin_edges_1d.shape
        assert values_2d.device == bin_edges_1d.device
        n_dist, n_values = values_2d.shape
        n_edges = int(bin_edges_1d.numel())
        n_bins = n_edges - 1
        assert n_bins > 0
        assert n_values > 0

        # bucketize semantics with right=False:
        #   x in [edges[k], edges[k+1]) -> index = k+1
        #   x < edges[0]                -> index = 0  (out of range, drop)
        #   x >= edges[-1]              -> index = n_edges (numpy folds into last bin)
        indices = torch.bucketize(values_2d, bin_edges_1d, right=False)
        # For numpy.histogram parity: items >= last edge land in the last
        # bin (right-inclusive on the last interval only). Items strictly
        # below the first edge are dropped. Clamp to [1, n_bins] so that
        # bin_idx after the -1 lands in [0, n_bins-1].
        bin_idx = indices.clamp(min=1, max=n_bins) - 1  # in [0, n_bins-1]

        # In-place scatter_add fills hist[a, bin_idx[a, p]] += 1 vectorized.
        hist = torch.zeros(n_dist, n_bins, dtype=torch.long, device=values_2d.device)
        hist.scatter_add_(1, bin_idx, torch.ones_like(bin_idx))

        # Subtract back the spurious additions to bin 0 from out-of-range
        # values (bucketize returned 0 -> clamped to 1 -> bin 0). For the
        # mhcflurry calibration set IC50 ∈ [1, 50000] this count is 0 in
        # practice, but keep the correction for completeness.
        below_first = (indices == 0).sum(dim=1)
        hist[:, 0] -= below_first

        totals = hist.sum(dim=1, keepdim=True).to(torch.float64)
        # Guard against an entirely-out-of-range row (totals == 0). In
        # practice unreachable for IC50 inputs vs the mhcflurry bin range.
        totals = totals.clamp(min=1)
        cumsum = torch.cumsum(hist.to(torch.float64) * 100.0 / totals, dim=1)

        cumsum_cpu = cumsum.cpu().numpy()
        bin_edges_cpu = bin_edges_1d.cpu().numpy()

        transforms = []
        for i in range(n_dist):
            t = cls()
            cdf = numpy.empty(n_bins + 3, dtype=numpy.float64)
            cdf[0] = 0.0
            cdf[1] = 0.0
            cdf[-1] = 100.0
            cdf[2:-1] = cumsum_cpu[i]
            t.cdf = cdf
            t.bin_edges = bin_edges_cpu
            transforms.append(t)
        return transforms

    def transform(self, values):
        """
        Return percent ranks (range [0, 100]) for the given values.
        """
        assert self.cdf is not None
        assert self.bin_edges is not None
        indices = numpy.searchsorted(self.bin_edges, values)
        result = self.cdf[indices]
        assert len(result) == len(values)

        # NaNs in input become NaNs in output
        result[numpy.isnan(values)] = numpy.nan

        return numpy.minimum(result, 100.0)

    def to_series(self):
        """
        Serialize the fit to a pandas.Series.

        The index on the series gives the bin edges and the values give the CDF.

        Returns
        -------
        pandas.Series

        """
        return pandas.Series(
            self.cdf, index=[numpy.nan] + list(self.bin_edges) + [numpy.nan])

    @staticmethod
    def from_series(series):
        """
        Deseralize a PercentRankTransform the given pandas.Series, as returned
        by `to_series()`.

        Parameters
        ----------
        series : pandas.Series

        Returns
        -------
        PercentRankTransform

        """
        result = PercentRankTransform()
        result.cdf = series.values
        result.bin_edges = series.index.values[1:-1]
        return result

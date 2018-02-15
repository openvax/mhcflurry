"""
Measures of centrality (e.g. mean) used to combine predictions across an
ensemble. The input to these functions are log affinities, and they are expected
to return a centrality measure also in log-space.
"""

import numpy
from functools import partial


def robust_mean(log_values):
    """
    Mean of values falling within the 25-75 percentiles.

    Parameters
    ----------
    log_values : 2-d numpy.array
        Center is computed along the second axis (i.e. per row).

    Returns
    -------
    center : numpy.array of length log_values.shape[1]

    """
    if log_values.shape[1] <= 3:
        # Too few values to use robust mean.
        return numpy.nanmean(log_values, axis=1)
    without_nans = numpy.nan_to_num(log_values)  # replace nan with 0
    mask = (
        (~numpy.isnan(log_values)) &
        (without_nans <= numpy.nanpercentile(log_values, 75, axis=1).reshape((-1, 1))) &
        (without_nans >= numpy.nanpercentile(log_values, 25, axis=1).reshape((-1, 1))))
    return (without_nans * mask.astype(float)).sum(1) / mask.sum(1)


CENTRALITY_MEASURES = {
    "mean": partial(numpy.nanmean, axis=1),
    "median": partial(numpy.nanmedian, axis=1),
    "robust_mean": robust_mean,
}
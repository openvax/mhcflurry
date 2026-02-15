"""
Measures of centrality (e.g. mean) used to combine predictions across an
ensemble. The input to these functions are log affinities, and they are expected
to return a centrality measure also in log-space.
"""

import numpy


def _nanmean_no_warnings(log_values):
    """
    Row-wise nanmean that returns nan for all-nan rows without warnings.
    """
    valid = ~numpy.isnan(log_values)
    counts = valid.sum(axis=1).astype("float64")
    sums = numpy.where(valid, log_values, 0.0).sum(axis=1)
    result = numpy.full(log_values.shape[0], numpy.nan, dtype="float64")
    numpy.divide(sums, counts, out=result, where=counts > 0)
    return result


def _nanmedian_no_warnings(log_values):
    """
    Row-wise nanmedian that returns nan for all-nan rows without warnings.
    """
    result = numpy.full(log_values.shape[0], numpy.nan, dtype="float64")
    row_has_values = (~numpy.isnan(log_values)).any(axis=1)
    if row_has_values.any():
        result[row_has_values] = numpy.nanmedian(log_values[row_has_values], axis=1)
    return result


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
        return _nanmean_no_warnings(log_values)

    result = numpy.full(log_values.shape[0], numpy.nan, dtype="float64")
    row_has_values = (~numpy.isnan(log_values)).any(axis=1)
    if not row_has_values.any():
        return result

    valid_rows = log_values[row_has_values]
    without_nans = numpy.nan_to_num(valid_rows)  # replace nan with 0
    p75 = numpy.nanpercentile(valid_rows, 75, axis=1).reshape((-1, 1))
    p25 = numpy.nanpercentile(valid_rows, 25, axis=1).reshape((-1, 1))
    mask = (
        (~numpy.isnan(valid_rows)) &
        (without_nans <= p75) &
        (without_nans >= p25))
    mask_f = mask.astype("float64")
    numerator = (without_nans * mask_f).sum(axis=1)
    denominator = mask_f.sum(axis=1)
    robust = numpy.full(valid_rows.shape[0], numpy.nan, dtype="float64")
    numpy.divide(numerator, denominator, out=robust, where=denominator > 0)
    result[row_has_values] = robust
    return result


CENTRALITY_MEASURES = {
    "mean": _nanmean_no_warnings,
    "median": _nanmedian_no_warnings,
    "robust_mean": robust_mean,
}

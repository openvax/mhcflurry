from __future__ import (
    print_function,
    division,
    absolute_import,
)
import logging
import sklearn.metrics
import numpy
import scipy

from .regression_target import from_ic50


def make_scores(
        ic50_y,
        ic50_y_pred,
        sample_weight=None,
        threshold_nm=500,
        max_ic50=50000):
    """
    Calculate AUC, F1, and Kendall Tau scores.

    Parameters
    -----------
    ic50_y : float list
        true IC50s (i.e. affinities)

    ic50_y_pred : float list
        predicted IC50s

    sample_weight : float list [optional]

    threshold_nm : float [optional]

    max_ic50 : float [optional]

    Returns
    -----------
    dict with entries "auc", "f1", "tau"
    """

    y_pred = from_ic50(ic50_y_pred, max_ic50)
    try:
        auc = sklearn.metrics.roc_auc_score(
            ic50_y <= threshold_nm,
            y_pred,
            sample_weight=sample_weight)
    except ValueError as e:
        logging.warning(e)
        auc = numpy.nan
    try:
        f1 = sklearn.metrics.f1_score(
            ic50_y <= threshold_nm,
            ic50_y_pred <= threshold_nm,
            sample_weight=sample_weight)
    except ValueError as e:
        logging.warning(e)
        f1 = numpy.nan
    try:
        tau = scipy.stats.kendalltau(ic50_y_pred, ic50_y)[0]
    except ValueError as e:
        logging.warning(e)
        tau = numpy.nan

    return dict(
        auc=auc,
        f1=f1,
        tau=tau)

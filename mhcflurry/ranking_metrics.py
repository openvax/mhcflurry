# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared ranking-evaluation metrics (AUROC, AUPRC, and tie-invariant PPV@N).

All MHCflurry evaluation commands and analysis scripts share this single
implementation so that PPV@N and its tie-handling rule never diverge across
tables, plots, and figures.
"""

import numpy
from sklearn.metrics import average_precision_score, roc_auc_score


def ppv_at_n(y_true, y_score, n=None):
    """Return expected precision among the top ``n`` scored examples.

    Tie handling
    ------------
    When multiple rows share the exact cutoff score (the ``n``-th highest
    score), this function computes the exact analytic expectation of PPV@N
    under uniform random tie-breaking within the cutoff tie group:

    ``(hits strictly above cutoff + remaining_slots * mean(y_true at cutoff)) / n``

    As a result, PPV@N is deterministic and invariant to input row order.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Binary ground-truth labels.
    y_score : array-like of float
        Prediction scores oriented so that higher values are better.
    n : int, optional
        Number of top-ranked rows to evaluate. Defaults to the number of
        positive labels in ``y_true`` (standard PPV@N).

    Returns
    -------
    float
        Expected PPV@N in ``[0, 1]``, or ``numpy.nan`` when ``n <= 0``.

    Raises
    ------
    ValueError
        If ``n`` exceeds the number of scored rows.
    """
    y_true = numpy.asarray(y_true)
    y_score = numpy.asarray(y_score, dtype=float)
    valid = ~numpy.isnan(y_score)
    if not valid.all():
        y_true = y_true[valid]
        y_score = y_score[valid]
    if n is None:
        n = int(y_true.sum())
    if n <= 0:
        return numpy.nan
    if n > len(y_score):
        raise ValueError("N exceeds the number of scored rows")
    cutoff = numpy.partition(y_score, len(y_score) - n)[len(y_score) - n]
    above = y_score > cutoff
    tied = y_score == cutoff
    remaining = n - int(above.sum())
    hits = y_true[above].sum() + remaining * y_true[tied].mean()
    return float(hits) / float(n)


def ranking_metrics(y_true, y_score):
    """Compute AUROC, AUPRC, and tie-invariant PPV@N on finite-scored rows.

    Non-finite score rows are excluded before counting ``n``, ``n_pos``, and
    computing metrics. When either class is absent after filtering, ``roc_auc``,
    ``pr_auc``, and ``ppv_at_n`` are returned as ``numpy.nan``.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Binary ground-truth labels.
    y_score : array-like of float
        Prediction scores oriented so that higher values are better.

    Returns
    -------
    dict
        Dictionary with keys ``n``, ``n_pos``, ``roc_auc``, ``pr_auc``, and
        ``ppv_at_n``.
    """
    y_true = numpy.asarray(y_true)
    y_score = numpy.asarray(y_score)
    mask = numpy.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return dict(
            n=int(len(y_true)),
            n_pos=n_pos,
            roc_auc=numpy.nan,
            pr_auc=numpy.nan,
            ppv_at_n=numpy.nan,
        )
    return dict(
        n=int(len(y_true)),
        n_pos=n_pos,
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        ppv_at_n=ppv_at_n(y_true, y_score, n_pos),
    )

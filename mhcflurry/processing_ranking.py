"""Sample-balanced ranking metrics for processing stopping-validation data."""

import numpy
from sklearn.metrics import average_precision_score


class SampleRankingMonitor:
    """Validate a fixed binary cohort and compute equal-sample AP and PPV@N.

    Parameters
    ----------
    targets : array-like
        Binary labels in the fixed validation prediction order.
    sample_ids : array-like
        Nonempty string sample identifiers, one per label. Each sample must
        contain both labels. No rows or samples are silently excluded.

    Notes
    -----
    AP uses sklearn's threshold-grouped treatment of ties. PPV uses stable
    descending score order, with original validation-row order breaking ties.
    Neither metric uses training sample weights.
    """

    def __init__(self, targets, sample_ids):
        self.targets = numpy.asarray(targets)
        samples = numpy.asarray(sample_ids, dtype=object)
        if (self.targets.ndim != 1 or samples.shape != self.targets.shape
                or not len(samples) or not numpy.isin(self.targets, [0, 1]).all()):
            raise ValueError("Ranking monitor requires aligned nonempty binary targets and sample IDs")
        if not all(isinstance(value, str) and value for value in samples):
            raise ValueError("Ranking monitor requires nonempty string sample IDs")
        self.sample_ids = sorted(set(samples))
        self.groups = [numpy.flatnonzero(samples == value) for value in self.sample_ids]
        if any(len(numpy.unique(self.targets[rows])) != 2 for rows in self.groups):
            raise ValueError("Every ranking-validation sample must contain both labels")

    def __call__(self, predictions):
        """Return macro AP and PPV for all fixed validation rows, failing closed."""
        scores = numpy.asarray(predictions)
        if (scores.shape != self.targets.shape or not numpy.isfinite(scores).all()
                or ((scores < 0) | (scores > 1)).any()):
            raise ValueError("Ranking predictions must be aligned finite probabilities")
        ap, ppv = [], []
        for rows in self.groups:
            labels, values = self.targets[rows], scores[rows]
            ap.append(average_precision_score(labels, values))
            n = int(labels.sum())
            ppv.append(labels[numpy.argsort(-values, kind="stable")[:n]].sum() / n)
        return {"val_macro_ap": float(numpy.mean(ap)), "val_macro_ppv_at_n": float(numpy.mean(ppv))}

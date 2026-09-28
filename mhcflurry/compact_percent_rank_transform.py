"""Compact percentile transforms for affinity, processing, and presentation.

Approximate the background survival function in logit coordinates. Unlike a
histogram lookup, interpolation retains within-interval order. Extreme-tail
extrapolation is a modeling assumption, not an empirical percentile guarantee.
"""

import heapq

import numpy as np


def probability_logits(values):
    """Convert probabilities to finite logits, preserving NaNs.

    Exact zero/one scores use the nearest interior float64 values. This cannot
    recover distinctions already lost by rounding the underlying raw scores.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.any(np.isinf(values) | (values < 0) | (values > 1)):
        raise ValueError("Scores must be probabilities in [0, 1] or NaN")
    interior = np.clip(values, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    return np.log(interior) - np.log1p(-interior)


def select_monotonic_knots(x, y, budget):
    """Greedily compress a strictly monotonic curve to at most budget knots.

    The largest vertical interpolation error determines each next split.
    Endpoints are always retained; equal errors use the earliest source index.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 2 or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any(np.diff(x) <= 0) or np.any(np.diff(y) >= 0)):
        raise ValueError("Expected finite increasing x and strictly decreasing y")
    if isinstance(budget, bool) or int(budget) != budget or budget < 2:
        raise ValueError("Knot budget must be an integer of at least two")
    selected, pending = {0, len(x) - 1}, []

    def add_interval(left, right):
        if right - left <= 1:
            return
        local_x = x[left + 1:right]
        linear = y[left] + (y[right] - y[left]) * (
            (local_x - x[left]) / (x[right] - x[left]))
        errors = np.abs(y[left + 1:right] - linear)
        offset = int(np.argmax(errors))
        heapq.heappush(pending, (-float(errors[offset]), left + 1 + offset, left, right))

    add_interval(0, len(x) - 1)
    while pending and len(selected) < budget:
        negative_error, middle, left, right = heapq.heappop(pending)
        if -negative_error <= 1e-12:
            break
        selected.add(middle)
        add_interval(left, middle)
        add_interval(middle, right)
    return np.array(sorted(selected), dtype=np.int64)


class CompactPercentRankTransform:
    """Small, versioned monotonic approximation to background percentiles.

    The knots store transformed raw scores and logit midrank survival
    probabilities. Linear interpolation in those coordinates followed by a
    sigmoid gives upper-tail percentiles; the complementary sigmoid gives CDF
    percentiles. Duplicate reference scores are weighted by their mass. No
    evaluation labels are required or accepted.

    Parameters
    ----------
    score_transform : {"logit", "log"}
        Use logit for probability-valued processing/presentation scores and
        log for positive quantities such as affinity IC50. This selects the
        input coordinate, not the output direction: pass ``survival=True`` to
        transform processing/presentation scores into lower-is-stronger ranks.

    Notes
    -----
    Low-level ``fit`` uses a fixed budget (64 by default). Predictor calibration
    uses the shared validation selector to decide whether 128 is warranted.
    Tail extrapolation models, rather than establishes, rare frequencies.
    """

    FORMAT = "compact-log-survival-linear-v1"

    def __init__(self, score_transform="logit"):
        if score_transform not in ("logit", "log"):
            raise ValueError("score_transform must be 'logit' or 'log'")
        self.score_transform = score_transform
        self.x = self.y = self.tail_slopes = self.reference_bounds = None
        self.reference_count = None
        self.selection = None

    def _coordinates(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.score_transform == "logit":
            return probability_logits(values)
        if np.any(np.isinf(values) | (values <= 0)):
            raise ValueError("Scores must be positive finite values or NaN")
        return np.log(values)

    def _set_curve(self, x, y, tail_slopes, reference_bounds, reference_count):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.tail_slopes = np.asarray(tail_slopes, dtype=np.float64)
        self.reference_bounds = np.asarray(reference_bounds, dtype=np.float64)
        if (isinstance(reference_count, bool) or
                int(reference_count) != reference_count or reference_count < 2):
            raise ValueError("Invalid reference observation count")
        self.reference_count = int(reference_count)
        if (self.x.ndim != 1 or self.y.shape != self.x.shape or len(self.x) < 2 or
                not np.isfinite(self.x).all() or not np.isfinite(self.y).all() or
                np.any(np.diff(self.x) <= 0) or np.any(np.diff(self.y) >= 0)):
            raise ValueError("Knots must be finite and strictly monotonic")
        if (self.tail_slopes.shape != (2,) or
                not np.isfinite(self.tail_slopes).all() or np.any(self.tail_slopes >= 0)):
            raise ValueError("Both extrapolation slopes must be finite and negative")
        if (self.reference_bounds.shape != (2,) or
                not np.isfinite(self.reference_bounds).all() or
                not self.reference_bounds[0] < self.reference_bounds[1] or
                self.reference_count < 2):
            raise ValueError("Invalid reference bounds or observation count")
        if not np.allclose(self._coordinates(self.reference_bounds), self.x[[0, -1]],
                           rtol=0, atol=1e-12):
            raise ValueError("Reference bounds must match the endpoint knots")

    def fit(self, scores, num_knots=64):
        """Fit a curve with a fixed knot budget; return this instance.

        Parameters
        ----------
        scores : one-dimensional sequence of float
            Finite, nonconstant background scores. Log coordinates require
            positive values; logit coordinates require values in [0, 1].
        num_knots : int, default 64
            Maximum knot count, including endpoints. This low-level method
            does not choose between budgets or use validation data.

        Returns
        -------
        CompactPercentRankTransform
            This fitted instance. Fewer knots may suffice for small or linear
            reference curves. Any previous automatic-selection metadata is cleared.
        """
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
            raise ValueError("Reference scores must be a finite one-dimensional sample")
        self._coordinates(scores)  # Validate before deduplication.
        unique, counts = np.unique(scores, return_counts=True)
        if len(unique) < 2:
            raise ValueError("Cannot calibrate a constant score distribution")
        x = self._coordinates(unique)
        # Endpoint clipping/log rounding can merge already adjacent floats.
        # Aggregate their reference mass rather than constructing zero-width knots.
        x, inverse = np.unique(x, return_inverse=True)
        counts = np.bincount(inverse, weights=counts)
        if len(x) < 2:
            raise ValueError("Cannot calibrate a constant score distribution")
        # Counts above this value plus half the mass at this exact value.
        survival = (len(scores) - np.cumsum(counts) + 0.5 * counts) / len(scores)
        y = probability_logits(survival)
        indices = select_monotonic_knots(x, y, num_knots)
        tail_size = min(32, len(x))
        slopes = []
        for rows in (slice(0, tail_size), slice(-tail_size, None)):
            local_x, local_y = x[rows], y[rows]
            centered_x, centered_y = local_x - local_x.mean(), local_y - local_y.mean()
            slopes.append(float(centered_x @ centered_y / (centered_x @ centered_x)))
        self._set_curve(x[indices], y[indices], slopes, unique[[0, -1]], len(scores))
        self.selection = None
        return self

    def log_percentiles(self, scores, survival=False):
        """Return natural logs of percentiles on the 0--100 scale.

        Parameters
        ----------
        scores : float or array-like
            Scores in the fitted coordinate's domain; NaNs propagate.
        survival : bool, default False
            False returns log CDF-percentile; True returns log upper-tail
            percentile directly. This is not the log of a 0--1 probability.

        Returns
        -------
        numpy.ndarray or numpy scalar
            Log-percentiles, retaining tiny-tail information before ordinary
            percentile exponentiation underflows. Extrapolation is not an
            empirical guarantee of extreme-tail accuracy.
        """
        if self.x is None:
            raise ValueError("Percentile transform has not been fitted")
        z = self._coordinates(scores)
        eta = np.interp(z, self.x, self.y)
        eta = np.where(z < self.x[0], self.y[0] + self.tail_slopes[0] * (z - self.x[0]), eta)
        eta = np.where(z > self.x[-1], self.y[-1] + self.tail_slopes[1] * (z - self.x[-1]), eta)
        if not survival:
            eta = -eta
        with np.errstate(invalid="ignore"):
            return np.log(100.0) - np.logaddexp(0.0, -eta)

    def transform(self, scores, survival=False):
        """Return CDF or upper-tail percentiles in [0, 100].

        Parameters
        ----------
        scores : float or array-like
            Raw scores in the fitted coordinate's domain. NaNs propagate;
            infinities and out-of-domain values raise ValueError.
        survival : bool, default False
            Use False for affinity IC50 (lower raw values get lower ranks).
            Use True for processing/presentation probabilities (higher raw
            values get lower ranks). The upper tail is evaluated directly,
            without subtracting a rounded CDF percentile from 100.

        Returns
        -------
        numpy.ndarray or numpy scalar
            Percentiles. Float64 rounding and underflow can still create ties
            in extreme tails; use ``log_percentiles`` for tiny-tail analysis.
        """
        return np.minimum(100.0, np.exp(self.log_percentiles(scores, survival=survival)))

    def to_dict(self):
        """Return a JSON-serializable, explicitly versioned representation."""
        if self.x is None:
            raise ValueError("Percentile transform has not been fitted")
        return dict(format=self.FORMAT, score_transform=self.score_transform,
                    x=self.x.tolist(), y=self.y.tolist(),
                    tail_slopes=self.tail_slopes.tolist(),
                    reference_bounds=self.reference_bounds.tolist(),
                    reference_count=self.reference_count, selection=self.selection)

    @classmethod
    def from_dict(cls, values):
        """Load a curve, rejecting unsupported versions and invalid knots."""
        if values.get("format") != cls.FORMAT:
            raise ValueError("Unsupported compact-percentile format")
        result = cls(score_transform=values["score_transform"])
        result._set_curve(**{key: values[key] for key in (
            "x", "y", "tail_slopes", "reference_bounds", "reference_count")})
        result.selection = values.get("selection")
        return result


class CompactPresentationPercentiles(CompactPercentRankTransform):
    """Compatibility adapter for the original presentation-only experiment.

    New callers should use CompactPercentRankTransform().fit(...) and pass
    survival=True explicitly. No separate fitting algorithm lives here.
    """

    @classmethod
    def from_scores(cls, scores, num_knots=64):
        """Fit a new adapter instance.

        ``fit`` stays the base class's instance method, which refits the
        receiver in place; overriding it with a constructor would silently
        discard in-place refits such as the knot-budget selection.
        """
        return cls().fit(scores, num_knots)

    def transform(self, scores, survival=True):
        return super().transform(scores, survival=survival)

    def log_percentiles(self, scores, survival=True):
        return super().log_percentiles(scores, survival=survival)

    @classmethod
    def from_dict(cls, values):
        if values.get("format") == "presentation-logit-survival-linear-v1":
            values = dict(values, format=cls.FORMAT, score_transform="logit")
        return super().from_dict(values)

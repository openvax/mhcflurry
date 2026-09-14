"""Shared method selection and persistence for predictor percentile ranks."""

import json
import hashlib
import os
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .compact_percent_rank_transform import CompactPercentRankTransform
from .histogram_percent_rank_transform import HistogramPercentRankTransform


def calibration_method(method, bins):
    """Resolve the default without changing historical explicit-bin calls."""
    if method is None:
        method = "histogram" if bins is not None else "compact"
    if method not in ("histogram", "compact"):
        raise ValueError("Percentile method must be 'histogram' or 'compact'")
    if method == "compact" and bins is not None:
        raise ValueError("Histogram bins cannot be combined with compact calibration")
    return method


def factorize_calibration_groups(groups):
    """Return ``(inverse, group_count)`` codes reusable across aligned fits.

    Calibrating many alleles against one peptide reference repeats the same
    grouping. Factorize it once and pass the result as ``group_codes``; the
    fitted transforms are identical to passing ``groups`` each time.
    """
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return inverse.reshape(-1), len(unique)


def fit_percent_rank_transform(values, *, method=None, bins=None,
                               score_transform="logit", survival=False,
                               max_knots=128, groups=None, group_codes=None):
    """Fit a histogram or a label-free, validation-selected compact curve.

    Compact fitting starts with 64 knots. A deterministic 80/20 background
    split permits 128 only for a meaningful reduction in cutoff-mass error.
    Groups (normally peptides) cannot cross that split. The selected budget
    is refitted on the full reference; no evaluation labels are accepted.

    Parameters
    ----------
    values : one-dimensional sequence of float
        Independent reference scores, not case/control labels.
    method : {"compact", "histogram"}, optional
        Compact unless explicit bins select histogram. Compact plus bins is
        invalid. This low-level helper requires bins for histogram fitting.
    bins : int or sequence, optional
        Histogram bin definition, forwarded without changing its semantics.
    score_transform : {"logit", "log"}
        Compact input coordinate: log for positive affinity IC50, logit for
        probability-valued processing/presentation scores.
    survival : bool, default False
        Percentile direction for budget validation: True for upper-tail
        processing/presentation ranks, False for affinity CDF ranks. This
        does not change the returned transform's default direction; pass
        survival=True explicitly when evaluating upper-tail percentiles.
    max_knots : {64, 128}, default 128
        Compact ceiling. Fit 64 unless held-out cutoff-mass error improves
        beyond max(0.01, 10% of the better error). Ignored for histograms.
    groups : sequence, optional
        Aligned identities that must not cross the selection split. If absent,
        each row is a separate group. Selection uses fixed seed 403, requires
        at least 1,000 groups and two cutoffs with ten expected observations.
    group_codes : tuple of (array of int, int), optional
        Precomputed ``factorize_calibration_groups(groups)`` for the same
        aligned reference rows. Mutually exclusive with ``groups``.

    Returns
    -------
    HistogramPercentRankTransform or CompactPercentRankTransform
        Fitted transform. Compact ``selection`` metadata records the decision
        and available reference-validation diagnostics.
    """
    method = calibration_method(method, bins)
    if method == "histogram":
        if bins is None:
            raise ValueError("Histogram fitting requires bins")
        return HistogramPercentRankTransform().fit(values, bins)
    if max_knots not in (64, 128):
        raise ValueError("max_knots must be 64 or 128")
    values = np.asarray(values, dtype=float)
    if groups is not None and group_codes is not None:
        raise ValueError("Pass calibration groups or group_codes, not both")
    # Validate even references too small for a split.
    model = CompactPercentRankTransform(score_transform).fit(values, 64)
    if group_codes is not None:
        inverse, group_count = group_codes
        inverse = np.asarray(inverse)
        if inverse.shape != values.shape:
            raise ValueError("Calibration groups must align with reference scores")
    elif groups is None:
        inverse = np.arange(len(values))
        group_count = len(values)
    else:
        groups = np.asarray(groups)
        if groups.shape != values.shape:
            raise ValueError("Calibration groups must align with reference scores")
        unique, inverse = np.unique(groups, return_inverse=True)
        group_count = len(unique)
    selected = 64
    diagnostics = dict(selected_knots=64, max_knots=max_knots, seed=403,
                       reference_count=len(values), group_count=group_count,
                       reference_sha256=hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
                       reason="64 knots retained; no validation evidence for 128")
    if max_knots == 128 and group_count >= 1000:
        shuffled = np.random.default_rng(403).permutation(group_count)
        validation_groups = np.zeros(group_count, dtype=bool)
        validation_groups[shuffled[:group_count // 5]] = True
        validation = validation_groups[inverse]
        fit_values, validation_values = values[~validation], values[validation]
        # Avoid trying to validate cutoffs supported by fewer than ten scores.
        cutoffs = np.array([.03, .1, .3, 1., 3., 10.])
        cutoffs = cutoffs[cutoffs * len(validation_values) / 100 >= 10]
        if (len(cutoffs) >= 2 and len(np.unique(fit_values)) >= 2):
            errors = {}
            for budget in (64, 128):
                trial = CompactPercentRankTransform(score_transform).fit(fit_values, budget)
                ranks = np.sort(trial.transform(validation_values, survival=survival))
                counts = np.searchsorted(ranks, cutoffs, side="right")
                observed = 100 * (counts + .5) / (len(ranks) + 1)
                errors[budget] = float(np.sqrt(np.mean(np.log10(observed / cutoffs) ** 2)))
            tolerance = max(.01, .1 * min(errors.values()))
            if errors[64] > errors[128] + tolerance:
                selected = 128
                model.fit(values, 128)
            diagnostics.update(
                selected_knots=selected, fit_count=len(fit_values),
                validation_count=len(validation_values), cutoffs=cutoffs.tolist(),
                validation_mask_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
                rms_log10_error=errors, tolerance=tolerance,
                reason="128 improves validation beyond tolerance" if selected == 128
                else "64 is within validation tolerance")
    diagnostics["actual_knots"] = len(model.x)
    model.selection = diagnostics
    return model


def save_percent_rank_transforms(models_dir, transforms):
    """Save calibrations with atomic replacement of the target file.

    CSV is retained when all histograms share an edge grid. JSON is authoritative
    for compact, mixed, or differing-grid collections. Remove the old
    alternate calibration only after its replacement is successfully written.
    An empty collection leaves existing calibration files untouched, as
    historical saves did, so an uncalibrated or incremental save cannot erase
    a calibration. Files get ordinary umask permissions. Neural-network files
    are never touched. This does not make the entire model-directory save a
    transaction across multiple files.
    """
    directory = Path(models_dir)
    csv_path, json_path = directory / "percent_ranks.csv", directory / "percent_ranks.json"
    if not transforms:
        return
    histogram_only = bool(transforms) and all(
        isinstance(t, HistogramPercentRankTransform) for t in transforms.values())
    series = {name: t.to_series() for name, t in transforms.items()} if histogram_only else {}
    # Historical CSV requires identical edges for all distributions.
    same_edges = histogram_only and all(
        np.array_equal(s.index, next(iter(series.values())).index, equal_nan=True)
        for s in series.values())
    target, alternate = (csv_path, json_path) if same_edges else (json_path, csv_path)
    # Let the kernel apply this process's umask. mkstemp would force
    # owner-only mode, and reading the umask to undo that would race other
    # threads, since the umask is process-global. os.replace keeps the mode.
    temp = directory / (".percent-ranks-%d-%s" % (os.getpid(), uuid.uuid4().hex))
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "w") as stream:
            if same_edges:
                pd.DataFrame(series).to_csv(stream, index=True, index_label="bin")
            else:
                json.dump(dict(format="mhcflurry-percent-ranks-v1", transforms={
                    name: t.to_dict() for name, t in transforms.items()}), stream,
                    allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        alternate.unlink(missing_ok=True)
    finally:
        temp.unlink(missing_ok=True)


def load_percent_rank_transforms(models_dir):
    """Read new calibrations or historical CSV without refitting either."""
    directory = Path(models_dir)
    json_path, csv_path = directory / "percent_ranks.json", directory / "percent_ranks.csv"
    use_json = json_path.exists()
    if json_path.exists() and csv_path.exists():
        # A save replaces its target and only then removes the alternate format,
        # so both existing means it was interrupted. Load the newer file and
        # warn: refusing would make the whole predictor directory unreadable.
        use_json = json_path.stat().st_mtime >= csv_path.stat().st_mtime
        warnings.warn(
            "Both percent_ranks.json and percent_ranks.csv exist in %s; an interrupted "
            "calibration save left %s behind. Loading the newer file; remove the stale "
            "one or recalibrate." % (directory, (csv_path if use_json else json_path).name))
    if use_json:
        with json_path.open() as stream:
            payload = json.load(stream)
        if payload.get("format") != "mhcflurry-percent-ranks-v1":
            raise ValueError("Unsupported percentile collection format")
        result = {}
        for name, values in payload["transforms"].items():
            implementation = (HistogramPercentRankTransform
                              if values.get("format") == "histogram-cdf-v1"
                              else CompactPercentRankTransform)
            result[name] = implementation.from_dict(values)
        return result
    if (directory / "percent_ranks.csv").exists():
        frame = pd.read_csv(directory / "percent_ranks.csv", index_col=0)
        return {name: HistogramPercentRankTransform.from_series(frame[name]) for name in frame}
    return {}

"""Regression tests for the experimental, label-free percentile approximation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit
from sklearn.metrics import average_precision_score

from mhcflurry.compact_percent_rank_transform import (
    CompactPresentationPercentiles, probability_logits, select_monotonic_knots)


def experiment_module():
    path = Path(__file__).resolve().parents[1] / "scripts/training/compare_presentation_percentiles.py"
    spec = importlib.util.spec_from_file_location("compare_presentation_percentiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("budget", [2, 8, 64, 128, 256])
def test_budget_monotonicity_extrapolation_and_roundtrip(budget):
    rng = np.random.default_rng(15)
    reference = expit(rng.normal(-5, 2, 10000))
    model = CompactPresentationPercentiles.from_scores(reference, budget)
    assert 2 <= len(model.x) <= budget
    assert np.all(model.tail_slopes < 0)
    assert_allclose(model.reference_bounds, [reference.min(), reference.max()])
    # Include points on both sides of the observed reference, not only knots.
    probes = expit(np.linspace(-16, 13, 20000))
    mapped = model.transform(probes)
    assert np.all(np.diff(mapped) < 0)
    assert np.all((mapped > 0) & (mapped < 100))
    restored = CompactPresentationPercentiles.from_dict(json.loads(json.dumps(model.to_dict())))
    assert_array_equal(mapped, restored.transform(probes))
    assert_allclose(np.log(mapped), model.log_percentiles(probes), atol=1e-14)


def test_duplicates_use_weighted_midrank_survival():
    model = CompactPresentationPercentiles.from_scores([.1, .1, .2, .9], 20)
    assert_allclose(model.transform([.1, .2, .9]), [75, 37.5, 12.5])
    assert len(model.x) == 3


def test_two_distinct_scores_and_nan_endpoints():
    model = CompactPresentationPercentiles.from_scores([0, 1], 128)
    result = model.transform([0, .25, .75, 1, np.nan])
    assert_allclose(result[[0, 3]], [75, 25])
    assert np.all(np.diff(result[:4]) < 0)
    assert np.isnan(result[-1])
    assert np.isfinite(probability_logits([0, 1])).all()
    assert model.transform(np.array([])).shape == (0,)
    assert np.isscalar(model.transform(.5))


@pytest.mark.parametrize("scores", [[], [.5], [.5, .5], [.1, np.nan], [.1, np.inf],
                                    [-.1, .1], [.1, 1.1], [[.1, .2]]])
def test_invalid_reference_is_rejected(scores):
    with pytest.raises(ValueError):
        CompactPresentationPercentiles.from_scores(scores)


@pytest.mark.parametrize("budget", [0, 1, -3, 3.5, True])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        CompactPresentationPercentiles.from_scores([.1, .2, .5], budget)


def test_invalid_serialized_models_are_rejected():
    original = CompactPresentationPercentiles.from_scores([.1, .2, .5]).to_dict()
    for key, value in [("format", "v999"), ("tail_slopes", [0, -1]),
                       ("y", [0, 0, 0]), ("x", [1, 1, 2]),
                       ("reference_bounds", [.2, .5])]:
        broken = dict(original, **{key: value})
        with pytest.raises(ValueError):
            CompactPresentationPercentiles.from_dict(broken)


def test_greedy_knots_are_deterministic_and_preserve_endpoints():
    x = np.linspace(-4, 4, 1000)
    y = -np.exp(x)
    for budget in [3, 10, 40]:
        indices = select_monotonic_knots(x, y, budget)
        assert len(indices) == budget
        assert_array_equal(indices[[0, -1]], [0, len(x) - 1])
        assert_array_equal(indices, select_monotonic_knots(x, y, budget))
    assert_array_equal(select_monotonic_knots(x, -x, 128), [0, len(x) - 1])


def test_independent_rare_tail_ranking_is_preserved():
    rng = np.random.default_rng(402)
    reference = expit(rng.normal(-6, 1.3, 100000))
    model = CompactPresentationPercentiles.from_scores(reference, 64)
    # Independent evaluation scores in and above the reference's sparse tail.
    raw = expit(rng.normal(0, 1.5, 6000))
    labels = rng.random(len(raw)) < expit((probability_logits(raw) - .2) * 2)
    percentile_scores = -model.transform(raw)
    assert_allclose(average_precision_score(labels, percentile_scores),
                    average_precision_score(labels, raw), rtol=0, atol=1e-14)
    assert np.unique(percentile_scores).size == np.unique(raw).size


def test_background_split_keeps_repeated_peptides_together():
    module = experiment_module()
    peptides = np.tile(np.array(["P%d" % index for index in range(100)]), 3)
    split = module.grouped_reference_split(peptides, 403)
    assert_array_equal(np.bincount(split), [180, 60, 60])
    assert_array_equal(split[:100], split[100:200])
    assert_array_equal(split, module.grouped_reference_split(peptides, 403))
    # Tiling this split for every allele cannot put a peptide in different folds.
    assert_array_equal(np.tile(split, 8).reshape(8, -1), np.broadcast_to(split, (8, len(split))))


def test_calibration_accuracy_reports_raw_counts_and_smoothing():
    module = experiment_module()
    table = module.calibration_accuracy(np.array([.02, .1, 1, 20]), [.01, .1, 1])
    assert_array_equal(table["count"], [0, 2, 3])
    assert_allclose(table.observed_percent, [0, 50, 75])
    assert_allclose(table.smoothed_percent, [10, 50, 70])
    assert np.isfinite(table.log10_error).all()


def test_selection_ignores_test_split_and_prefers_smaller_near_tie():
    module = experiment_module()
    records = []
    for method, error in [("compact_64", .108), ("compact_128", .10), ("compact_256", .099)]:
        for cutoff in module.SELECTION_CUTOFFS:
            records.extend([dict(method=method, cutoff=cutoff, split="validation", log10_error=error),
                            dict(method=method, cutoff=cutoff, split="test", log10_error=100)])
    result = module.select_compact_method(pd.DataFrame(records))
    assert result["selected"] == "compact_64"
    assert_allclose(result["best"], .099)


def test_ppv_is_invariant_to_input_order_with_cutoff_ties():
    module = experiment_module()
    labels = np.array([1, 1, 0, 0])
    for order in [np.arange(4), np.arange(4)[::-1]]:
        metrics = module.ranking_metrics(labels[order], np.zeros(4))
        assert metrics["ppv_expected_random_ties"] == .5
        assert metrics["auprc"] == .5
        assert metrics["auroc"] == .5


def test_saved_step_and_compact_methods_roundtrip(tmp_path):
    module = experiment_module()
    scores = expit(np.random.default_rng(25).normal(size=1000))
    methods = module.fit_methods(scores, [8, 16])
    module.save_methods(methods, tmp_path / "models")
    loaded = module.load_methods(tmp_path / "models")
    for name in methods:
        assert_array_equal(module.percentiles(methods[name], scores),
                           module.percentiles(loaded[name], scores))


def test_adapter_fit_refits_the_receiver_in_place():
    model = CompactPresentationPercentiles.from_scores([.1, .2, .3, .4, .5], 64)
    same = model.fit([index / 2000 for index in range(1, 2000)], 128)
    assert same is model
    assert len(model.x) > 5

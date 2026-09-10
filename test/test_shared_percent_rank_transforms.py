"""Shared calibration APIs, historical persistence, and numerical direction."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit
from sklearn.metrics import average_precision_score

from mhcflurry import (
    Class1AffinityPredictor, Class1PresentationPredictor, Class1ProcessingPredictor,
    CompactPercentRankTransform, HistogramPercentRankTransform)
from mhcflurry.percent_rank_transform import PercentRankTransform
from mhcflurry.percentile_calibration import (
    fit_percent_rank_transform, load_percent_rank_transforms, save_percent_rank_transforms)


def test_historical_import_pickle_and_cdf_are_unchanged():
    assert PercentRankTransform is HistogramPercentRankTransform
    transform = PercentRankTransform().fit([1, 2, 3], [0, 1, 2, 3, 4])
    probes = np.array([0., 1., 2., 3., 4., np.nan])
    expected = transform.transform(probes)
    assert_array_equal(expected[:-1], [0, 0, 0, 100 / 3, 200 / 3])
    # A pickle referring to the old module/class still resolves the alias.
    old_global = pickle.loads(b"cmhcflurry.percent_rank_transform\nPercentRankTransform\n.")
    assert old_global is HistogramPercentRankTransform
    restored = HistogramPercentRankTransform.from_dict(json.loads(json.dumps(transform.to_dict())))
    assert_array_equal(restored.transform(probes), expected)
    assert_array_equal(transform.transform(probes, survival=True), 100 - expected)


@pytest.mark.parametrize("coordinate", ["log", "logit"])
def test_generic_compact_directions_roundtrip_and_ranking(coordinate):
    rng = np.random.default_rng(27)
    convert = np.exp if coordinate == "log" else expit
    reference = convert(rng.normal(-3, 1.5, 5000))
    scores = convert(rng.normal(-2, 1.5, 10000))
    labels = rng.random(len(scores)) < .2
    transform = CompactPercentRankTransform(coordinate).fit(reference)
    assert len(transform.x) <= 64
    restored = CompactPercentRankTransform.from_dict(json.loads(json.dumps(transform.to_dict())))
    for survival in (False, True):
        ranks = transform.transform(scores, survival=survival)
        assert_array_equal(ranks, restored.transform(scores, survival=survival))
        signed_raw = -scores if survival else scores
        assert average_precision_score(labels, ranks) == average_precision_score(labels, signed_raw)
    assert_allclose(transform.transform(scores) + transform.transform(scores, survival=True), 100)
    assert np.isnan(transform.transform([np.nan])[0])
    assert transform.transform([]).size == 0


def test_survival_does_not_subtract_rounded_cdf():
    transform = CompactPercentRankTransform("log").fit(np.exp(np.linspace(-2, 2, 10000)))
    probes = np.exp(np.linspace(2.2, 2.3, 20))
    survival = transform.transform(probes, survival=True)
    assert np.all(survival > 0)
    assert np.all(np.diff(survival) < 0)
    assert np.all(transform.transform(probes) > 99.999)


@pytest.mark.parametrize("predictor_class", [Class1PresentationPredictor, Class1ProcessingPredictor])
def test_probability_predictors_share_api_and_save_load(tmp_path, monkeypatch, predictor_class):
    if predictor_class is Class1ProcessingPredictor:
        predictor = predictor_class(models=[])
        save_kwargs = {}
    else:
        predictor = predictor_class(weights_dataframe=pd.DataFrame({"intercept": [0.]}, index=["without_flanks"]))
        save_kwargs = dict(write_affinity_predictor=False, write_processing_predictor=False)
        monkeypatch.setattr(Class1AffinityPredictor, "load", lambda *args, **kwargs: None)
    scores = expit(np.linspace(-8, 4, 1000))
    predictor.calibrate_percentile_ranks(scores)
    assert type(predictor.percent_rank_transform) is CompactPercentRankTransform
    assert predictor.percent_rank_transform.selection["selected_knots"] == 64
    expected = predictor.percentile_ranks(scores)
    assert np.all(np.diff(expected) < 0)
    predictor.save(str(tmp_path), **save_kwargs)
    restored = predictor_class.load(str(tmp_path))
    assert_array_equal(restored.percentile_ranks(scores), expected)
    assert (tmp_path / "percent_ranks.json").exists()
    assert not (tmp_path / "percent_ranks.csv").exists()
    # An explicit histogram recalibration replaces the compact file, not its weights.
    predictor.calibrate_percentile_ranks(scores, bins=30)
    predictor.save(str(tmp_path), **save_kwargs)
    restored = predictor_class.load(str(tmp_path))
    assert type(restored.percent_rank_transform) is HistogramPercentRankTransform
    assert_allclose(restored.percentile_ranks(scores), predictor.percentile_ranks(scores), rtol=0, atol=1e-12)
    assert not (tmp_path / "percent_ranks.json").exists()
    assert (tmp_path / "percent_ranks.csv").exists()


def test_affinity_compact_api_and_mixed_legacy_persistence(tmp_path, monkeypatch):
    predictor = Class1AffinityPredictor()
    monkeypatch.setattr(predictor, "predict", lambda *args, **kwargs: np.geomspace(1, 50000, 1000))
    predictor.calibrate_percentile_ranks(peptides=["SIINFEKL"] * 1000, alleles=["HLA-A*02:01"])
    compact = predictor.allele_to_percent_rank_transform["HLA-A*02:01"]
    assert compact.score_transform == "log"
    assert compact.selection["group_count"] == 1
    predictor.allele_to_percent_rank_transform["HLA-B*07:02"] = (
        HistogramPercentRankTransform().fit([1., 100, 50000], [1., 100, 50000]))
    expected = predictor.percentile_ranks([10., 1000.], allele="HLA-A*02:01")
    assert expected[0] < expected[1]
    predictor.save(str(tmp_path))
    restored = Class1AffinityPredictor.load(str(tmp_path), optimization_level=0)
    assert_array_equal(restored.percentile_ranks([10., 1000.], allele="HLA-A*02:01"), expected)
    assert type(restored.allele_to_percent_rank_transform["HLA-B*07:02"]) is HistogramPercentRankTransform


def test_reference_budget_decision_is_deterministic_and_persisted():
    values = expit(np.random.default_rng(42).normal(size=30000))
    groups = np.arange(len(values)) // 3
    first = fit_percent_rank_transform(values, groups=groups, survival=True)
    second = fit_percent_rank_transform(values, groups=groups, survival=True)
    assert first.to_dict() == second.to_dict()
    decision = first.selection
    assert decision["selected_knots"] in (64, 128)
    errors = decision["rms_log10_error"]
    needs_128 = errors[64] > errors[128] + decision["tolerance"]
    assert (decision["selected_knots"] == 128) == needs_128
    assert decision["validation_count"] == 6000
    fixed = fit_percent_rank_transform(values, max_knots=64)
    assert fixed.selection["selected_knots"] == 64
    assert len(fixed.x) <= 64


def test_escalation_to_128_requires_validation_evidence(monkeypatch):
    """Exercise the gate without relying on a lucky random sample."""
    import mhcflurry.percentile_calibration as module
    original_fit = module.CompactPercentRankTransform.fit
    def fit_with_budget(self, values, num_knots=64):
        self._test_budget = num_knots
        return original_fit(self, values, num_knots)
    def validation_ranks(self, values, survival=False):
        ranks = (np.arange(len(values)) + .5) * 100 / len(values)
        return ranks if self._test_budget == 128 else ranks / 2
    monkeypatch.setattr(module.CompactPercentRankTransform, "fit", fit_with_budget)
    monkeypatch.setattr(module.CompactPercentRankTransform, "transform", validation_ranks)
    model = module.fit_percent_rank_transform(expit(np.linspace(-4, 4, 30000)))
    assert model.selection["selected_knots"] == 128
    assert len(model.x) <= 128
    assert model.selection["rms_log10_error"][64] > model.selection["rms_log10_error"][128] + .01


def test_processing_cli_uses_explicit_context_and_retains_scores(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mhcflurry.cli import calibrate_percentile_ranks_command as command
    class FakePredictor(Class1ProcessingPredictor):
        @property
        def sequence_lengths(self):
            return dict(peptide=15, n_flank=5, c_flank=5)
        def predict(self, peptides, n_flanks=None, c_flanks=None, **kwargs):
            assert list(n_flanks) == ["AAAAA", "CCCCC", ""]
            assert list(c_flanks) == ["DDDDD", "EEEEE", ""]
            return np.array([.1, .2, .3])
    predictor = FakePredictor([])
    monkeypatch.setattr(command.Class1ProcessingPredictor, "load", lambda *args, **kwargs: predictor)
    monkeypatch.setattr(command, "write_generate_sh", lambda *args: None)
    reference = tmp_path / "reference.csv"
    reference.write_text("peptide,n_flank,c_flank\nSIINFEKL,AAAAA,DDDDD\nSIINFEKA,CCCCC,EEEEE\nSIINFEKC,,\n")
    args = SimpleNamespace(models_dir=str(tmp_path), processing_reference_data=str(reference),
                           prediction_batch_size=10, percentile_method="compact", max_percentile_knots=128)
    command.run_class1_processing_predictor(args)
    with np.load(tmp_path / "percent_rank_reference.npz", allow_pickle=False) as saved:
        assert_array_equal(saved["scores"], [.1, .2, .3])
        assert_array_equal(saved["n_flank"], ["AAAAA", "CCCCC", ""])
    reference.write_text("peptide\nSIINFEKL\nSIINFEKA\n")
    with pytest.raises(ValueError, match="requires columns"):
        command.run_class1_processing_predictor(args)


def test_generic_tail_underflow_has_log_percentile_escape_hatch():
    transform = CompactPercentRankTransform("log").fit(np.exp(np.linspace(-2, 2, 10000)))
    probes = np.exp(np.linspace(10, 11, 20))
    # Floating-point exponentiation eventually underflows; do not claim that
    # ordinary percentiles preserve every distinguishable score indefinitely.
    assert np.all(transform.transform(probes, survival=True) == 0)
    logarithms = transform.log_percentiles(probes, survival=True)
    assert np.isfinite(logarithms).all()
    assert np.all(np.diff(logarithms) < 0)


def test_bad_methods_and_domains_fail_before_save(tmp_path):
    for kwargs in [dict(method="unknown"), dict(method="compact", bins=10), dict(max_knots=256)]:
        with pytest.raises(ValueError):
            fit_percent_rank_transform([.1, .2], **kwargs)
    for values in ([0, 1], [-1, 2], [1, np.inf]):
        with pytest.raises(ValueError):
            CompactPercentRankTransform("log").fit(values)
    with pytest.raises(ValueError, match="align"):
        fit_percent_rank_transform([.1, .2], groups=["one"])
    with pytest.raises(ValueError, match="not been fitted"):
        CompactPercentRankTransform().transform([.5])
    (tmp_path / "percent_ranks.json").write_text('{"format": "future-v999"}')
    with pytest.raises(ValueError, match="Unsupported"):
        load_percent_rank_transforms(tmp_path)


def test_processing_missing_calibration_and_old_directory(tmp_path):
    predictor = Class1ProcessingPredictor([])
    with pytest.raises(ValueError, match="No processing"):
        predictor.percentile_ranks([.1])
    with pytest.warns(UserWarning):
        assert np.isnan(predictor.percentile_ranks([.1], throw=False)[0])
    predictor.save(str(tmp_path))
    assert not (tmp_path / "percent_ranks.json").exists()
    assert Class1ProcessingPredictor.load(str(tmp_path)).percent_rank_transform is None


def test_atomic_calibration_write_failure_preserves_previous(tmp_path, monkeypatch):
    import mhcflurry.percentile_calibration as module
    old = HistogramPercentRankTransform().fit([.1, .2, .5], 3)
    save_percent_rank_transforms(tmp_path, {"score": old})
    before = (tmp_path / "percent_ranks.csv").read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated failed replacement")
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        save_percent_rank_transforms(tmp_path, {"score": CompactPercentRankTransform().fit([.1, .2])})
    assert (tmp_path / "percent_ranks.csv").read_bytes() == before
    assert not list(tmp_path.glob(".percent-ranks-*"))

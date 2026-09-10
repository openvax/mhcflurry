"""Inner-sample ranking checkpoints are deterministic, isolated and optional."""

import numpy
import pandas
import pytest
import torch

from mhcflurry import Class1ProcessingNeuralNetwork, Class1ProcessingPredictor
from mhcflurry.flanking_encoding import FlankingEncoding
from mhcflurry.processing_ranking import SampleRankingMonitor


def test_ranking_macro_and_stable_ties():
    monitor = SampleRankingMonitor([1, 0, 0, 1, 0, 0], ["a", "a", "b", "b", "b", "b"])
    result = monitor([0.5] * 6)
    assert result["val_macro_ap"] == pytest.approx((0.5 + 0.25) / 2)
    assert result["val_macro_ppv_at_n"] == 0.5
    assert monitor([1, 0, 0, 1, 0, 0]) == {"val_macro_ap": 1.0, "val_macro_ppv_at_n": 1.0}


@pytest.mark.parametrize("labels,samples", [([], []), ([1, 0], ["a"]), ([1, 0], ["a", "b"]),
    ([1, 2], ["a", "a"]), ([1, 0], [None, "a"]), ([1, 0], ["", ""]), ([1, 0], [1, 1])])
def test_ranking_invalid_cohort_fails(labels, samples):
    with pytest.raises(ValueError):
        SampleRankingMonitor(labels, samples)


@pytest.mark.parametrize("values", [[numpy.nan, 0], [1.1, 0], [-0.1, 0], [0.5]])
def test_ranking_invalid_scores_fail(values):
    with pytest.raises(ValueError):
        SampleRankingMonitor([0, 1], ["a", "a"])(values)


def tiny_model(**kwargs):
    hp = dict(max_epochs=3, early_stopping=False, minibatch_size=2,
              convolutional_filters=4, convolutional_kernel_size=3,
              n_flank_length=2, c_flank_length=2, dropout_rate=0,
              convolutional_kernel_l1_l2=[0, 0], save_all_checkpoints=True,
              restore_best_weights=True)
    hp.update(kwargs)
    return Class1ProcessingNeuralNetwork(**hp)


def fit_tiny(model, **kwargs):
    args = dict(sample_ids=numpy.array(["train"] * 4 + ["validation"] * 4),
                validation_mask=numpy.array([False] * 4 + [True] * 4),
                seed=42, verbose=-1, progress_print_interval=None)
    args.update(kwargs)
    model.fit(FlankingEncoding(["SIINFEKL", "AAAAAAAA"] * 4, ["AA"] * 8, ["CC"] * 8),
              [0, 1] * 4, **args)
    return model


def test_monitoring_does_not_change_training_or_legacy_weights():
    plain = fit_tiny(tiny_model())
    monitored = fit_tiny(tiny_model(monitor_validation_ranking=True))
    for policy in ("best", "terminal"):
        for left, right in zip(plain.checkpoint_weights[policy], monitored.checkpoint_weights[policy]):
            numpy.testing.assert_array_equal(left, right)
    assert monitored.fit_info[-1]["ranking_validation_input_rows"] == [4, 5, 6, 7]
    assert monitored.fit_info[-1]["ranking_validation_samples"] == ["validation"]
    assert len(monitored.fit_info[-1]["val_macro_ap"]) == 3
    assert plain.fit_info[-1]["restored_checkpoint_policy"] == "best"


def test_best_ap_earliest_tie_is_independent_and_serialized(tmp_path, monkeypatch):
    observed = []
    def metrics(self, predictions):
        observed.append(self.sample_ids)
        return {"val_macro_ap": [0.2, 0.9, 0.9][len(observed) - 1], "val_macro_ppv_at_n": 0.5}
    monkeypatch.setattr(SampleRankingMonitor, "__call__", metrics)
    model = fit_tiny(tiny_model(checkpoint_metric="val_macro_ap"))
    assert observed == [["validation"]] * 3
    assert set(model.checkpoint_weights) == {"best", "best_ap", "terminal"}
    info = model.fit_info[-1]
    assert info["best_ranking_epoch"] == 2 and info["best_macro_ap"] == 0.9
    assert info["restored_checkpoint_policy"] == "best_ap"
    assert info["stopping_metric"] == "val_loss"
    for left, right in zip(model.get_weights(), model.checkpoint_weights["best_ap"]):
        numpy.testing.assert_array_equal(left, right)
    assert any(not numpy.array_equal(left, right) for left, right in
               zip(model.checkpoint_weights["best_ap"], model.checkpoint_weights["terminal"]))
    snapshot = [value.copy() for value in model.checkpoint_weights["best_ap"]]
    with torch.no_grad():
        next(model.network().parameters()).add_(1)
    for left, right in zip(snapshot, model.checkpoint_weights["best_ap"]):
        numpy.testing.assert_array_equal(left, right)
    predictor = Class1ProcessingPredictor([model])
    predictor.save(str(tmp_path))
    loaded = Class1ProcessingPredictor.load(str(tmp_path))
    for left, right in zip(snapshot, loaded.models[0].checkpoint_weights["best_ap"]):
        numpy.testing.assert_array_equal(left, right)
    loaded.models[0].hyperparameters.update(checkpoint_metric="val_loss", monitor_validation_ranking=False,
                                           save_all_checkpoints=False)
    fit_tiny(loaded.models[0])
    loaded.save(str(tmp_path))
    assert pandas.read_csv(tmp_path / "manifest.csv").checkpoint_best_ap_weights.isna().all()
    assert Class1ProcessingPredictor.load(str(tmp_path)).models[0].checkpoint_weights == {}


@pytest.mark.parametrize("kwargs", [dict(sample_ids=None), dict(validation_mask=None),
    dict(sample_ids=["same"] * 8), dict(validation_mask=numpy.ones(8, dtype=bool)),
    dict(validation_mask=numpy.zeros(8, dtype=bool))])
def test_ranking_requires_disjoint_explicit_samples(kwargs):
    with pytest.raises(ValueError):
        fit_tiny(tiny_model(checkpoint_metric="val_macro_ap"), **kwargs)


def test_unknown_checkpoint_metric_rejected():
    with pytest.raises(ValueError, match="checkpoint_metric"):
        fit_tiny(tiny_model(checkpoint_metric="outer_test_ap"))

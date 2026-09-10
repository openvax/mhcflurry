"""The recipe promotion gate is paired, complete and micro-safe."""

import importlib.util
import json
from pathlib import Path

import numpy
import pandas
import pytest
import yaml


@pytest.fixture
def analysis_module(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts/training"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("confirmation_analysis", directory / "analyze_processing_confirmation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_experiment(tmp_path):
    from generate_processing_recipe import build_processing_confirmation_conditions
    records = [dict(condition=name, hyperparameters=hp, **axes)
               for name, hp, axes in build_processing_confirmation_conditions()]
    (tmp_path / "experiment.json").write_text(json.dumps({"design": "processing-ranking-confirmation-v1", "records": records}))
    for record in records:
        rows, pooled = [], []
        for policy in ("best", "best_ap", "terminal"):
            gain = 0 if policy == "best" else 0.02
            if record["optimizer"] == "rmsprop" and record["kernel_width"] == 13:
                gain += 0.01
            for fold in range(4):
                common = dict(condition=record["condition"], checkpoint_policy=policy, fold_num=fold,
                              roc_auc=0.7 + gain, pr_auc=0.6 + gain, ppv_at_n=0.55 + gain)
                pooled.append(dict(common, n=200, n_pos=100))
                for sample in ("a", "b"):
                    rows.append(dict(common, sample_id=sample, n=100, n_pos=50))
        destination = tmp_path / record["condition"]
        destination.mkdir()
        pandas.DataFrame(rows).to_csv(destination / "checkpoint_per_sample.csv", index=False)
        pandas.DataFrame(pooled).to_csv(destination / "checkpoint_micro_by_fold.csv", index=False)
    return records


def test_complete_paired_gate_writes_reusable_candidate(tmp_path, analysis_module):
    fixture_experiment(tmp_path)
    out = tmp_path / "analysis"
    assert analysis_module.main(["--experiment", str(tmp_path), "--out", str(out), "--replicates", "100"]) == 0
    decision = json.loads((out / "decision.json").read_text())
    assert decision["candidate_condition"] == "legacy_5aa__rmsprop_pytorch__k13"
    assert not decision["release_accepted"]
    from generate_processing_recipe import CONFIRMED_PROCESSING_CANDIDATE, confirmed_processing_candidate_hyperparameters
    assert decision["candidate_condition"] == CONFIRMED_PROCESSING_CANDIDATE["condition"]
    exported = yaml.safe_load((out / "candidate_hyperparameters.yaml").read_text())
    assert exported == [confirmed_processing_candidate_hyperparameters()]
    assert (out / "confirmation.pdf").stat().st_size > 1000
    assert (out / "analysis_source.py").is_file()
    with pytest.raises(ValueError, match="new output"):
        analysis_module.main(["--experiment", str(tmp_path), "--out", str(out)])


def test_optional_epoch_diagnostics_and_manifest_hash(tmp_path, analysis_module):
    records = fixture_experiment(tmp_path)
    name = records[0]["condition"]
    model_dir = tmp_path / name / "processing/models.unselected.short_flanks"
    model_dir.mkdir(parents=True)
    configs = [{"fit_info": [{"training_info": {"fold_num": fold}, "loss": [0.6, 0.5],
                "val_loss": [0.6, 0.62], "val_macro_ap": [0.7, 0.8], "val_macro_ppv_at_n": [0.6, 0.7],
                "best_epoch": 1, "best_ranking_epoch": 2, "ranking_validation_samples": ["inner"]}]}
               for fold in range(4)]
    pandas.DataFrame({"config_json": list(map(json.dumps, configs))}).to_csv(model_dir / "manifest.csv", index=False)
    out = tmp_path / "epoch-analysis"
    assert analysis_module.main(["--experiment", str(tmp_path), "--out", str(out), "--replicates", "100"]) == 0
    assert (out / (name + ".epochs.png")).is_file()
    manifest = json.loads((out / "provenance.json").read_text())
    assert any(item["path"].endswith("manifest.csv") for item in manifest["inputs"])


def test_incomplete_panel_cannot_promote(tmp_path, analysis_module):
    records = fixture_experiment(tmp_path)
    for filename in ("checkpoint_per_sample.csv", "checkpoint_micro_by_fold.csv"):
        (tmp_path / records[-1]["condition"] / filename).unlink()
    design, samples, micro, pending, _ = analysis_module.collect_metrics(tmp_path)
    result = analysis_module.compare_and_choose(design, samples, micro, pending, 100)[-1]
    assert result["status"] == "incomplete" and result["candidate_condition"] is None


def test_micro_regression_disqualifies_ap_leader(tmp_path, analysis_module):
    fixture_experiment(tmp_path)
    design, samples, micro, pending, _ = analysis_module.collect_metrics(tmp_path)
    target = "legacy_5aa__rmsprop_pytorch__k13"
    micro.loc[(micro.condition == target) & (micro.checkpoint_policy == "best_ap") & (micro.fold_num == 2), "ppv_at_n"] = 0.50
    decision = analysis_module.compare_and_choose(design, samples, micro, pending, 100)[-1]
    assert decision["candidate_condition"] != target


@pytest.mark.parametrize("change", ["duplicate", "missing_sample", "nonfinite", "unbalanced", "micro_count", "missing_policy"])
def test_bad_metrics_fail_closed(tmp_path, analysis_module, change):
    records = fixture_experiment(tmp_path)
    name = records[-1]["condition"]
    path = tmp_path / name / ("checkpoint_micro_by_fold.csv" if change == "micro_count" else "checkpoint_per_sample.csv")
    frame = pandas.read_csv(path)
    if change == "duplicate":
        frame = pandas.concat([frame, frame.iloc[:1]])
    elif change == "missing_sample":
        frame = frame.loc[frame.sample_id != "a"]
    elif change == "nonfinite":
        frame.loc[0, "pr_auc"] = numpy.nan
    elif change in ("unbalanced", "micro_count"):
        frame.loc[0, "n"] += 1
    else:
        frame = frame.loc[frame.checkpoint_policy != "terminal"]
    frame.to_csv(path, index=False)
    with pytest.raises((ValueError, AssertionError)):
        analysis_module.collect_metrics(tmp_path)

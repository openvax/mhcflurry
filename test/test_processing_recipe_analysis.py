"""Paired factorial analysis must expose interactions and reject cohort drift."""

from copy import deepcopy
import json
from pathlib import Path

import numpy
import pandas
import pytest


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/training"))
    import analyze_processing_recipe as module
    from generate_processing_recipe import build_processing_recipe_conditions
    records = [dict(condition=name, hyperparameters=grid, **axes)
               for name, grid, axes in build_processing_recipe_conditions()]
    experiment = {"design": "processing-training-recipe-v1", "train_data_sha256": "frozen", "records": records}
    rows = []
    for rec in records[:8]:
        for fold, samples in ((0, ("a", "b")), (1, ("a", "c"))):
            batch_effect = 0 if rec["batch_size"] == 512 else (0.02 if rec["initialization"] == "none" else -0.01)
            value = 0.7 + batch_effect + (0.03 if rec["family"] == "legacy_5aa" else 0)
            rows.append({"condition": rec["condition"], "fold_num": fold, "sample_id": samples[0],
                         "n": 20, "n_pos": 10, **{key: value for key in module.METRICS}})
            rows.append({"condition": rec["condition"], "fold_num": fold, "sample_id": samples[1],
                         "n": 40, "n_pos": 20, **{key: value for key in module.METRICS}})
    return module, experiment, pandas.DataFrame(rows)


def test_one_factor_contrasts_preserve_interaction_and_missing_cells(inputs):
    module, experiment, metrics = inputs
    status, samples, means, contrasts, delta = module.analyze(experiment, metrics, replicates=100)
    assert status.status.value_counts().to_dict() == {"pending": 24, "evaluated": 8}
    assert len(means) == 8 and len(samples) == 24
    batch = contrasts.loc[(contrasts.factor == "batch_size") & (contrasts.metric == "pr_auc")]
    assert len(batch) == 4
    for row in batch.itertuples():
        expected = 0.02 if json.loads(row.fixed_settings)["initialization"] == "none" else -0.01
        numpy.testing.assert_allclose([row.delta, row.ci_low, row.ci_high], expected)
        assert row.samples == 3
        assert row.joint_samples_improved == (3 if expected > 0 else 0)
    assert not contrasts.factor.eq("optimizer_recipe").any()
    assert set(delta.sample_id) == {"a", "b", "c"}


def test_repeated_samples_do_not_count_as_independent_units(inputs):
    module, experiment, metrics = inputs
    names = [record["condition"] for record in experiment["records"] if record["family"] == "legacy_5aa" and record["initialization"] == "none"][:2]
    metrics = metrics.loc[metrics.condition.isin(names)].copy()
    metrics[module.METRICS] = 0.5
    metrics.loc[metrics.condition.eq(names[1]) & metrics.sample_id.eq("a"), module.METRICS] = 0.8
    _, _, means, contrasts, _ = module.analyze(experiment, metrics, replicates=50)
    numpy.testing.assert_allclose(contrasts.delta, 0.1)  # not 0.15 from row averaging
    assert means.samples.tolist() == [3, 3]


@pytest.mark.parametrize("problem", ["missing_sample", "missing_fold", "duplicate", "count", "ratio", "nonfinite", "range", "unknown", "policy"])
def test_invalid_metric_tables_fail_closed(inputs, problem):
    module, experiment, metrics = inputs
    if problem == "missing_sample":
        metrics = metrics.iloc[1:]
    elif problem == "missing_fold":
        metrics = metrics.drop(index=[2, 3])
    elif problem == "duplicate":
        metrics = pandas.concat([metrics, metrics.iloc[:1]])
    elif problem == "count":
        metrics.loc[0, ["n", "n_pos"]] = [22, 11]
    elif problem == "ratio":
        metrics.loc[0, "n"] = 21
    elif problem == "nonfinite":
        metrics.loc[0, "pr_auc"] = numpy.inf
    elif problem == "range":
        metrics.loc[0, "pr_auc"] = 1.1
    elif problem == "unknown":
        metrics.loc[0, "condition"] = "unknown"
    else:
        metrics["checkpoint_policy"] = "terminal"
    with pytest.raises(ValueError):
        module.analyze(experiment, metrics, replicates=10)


@pytest.mark.parametrize("problem", ["axes", "hidden_change", "coordinates", "policy", "folds"])
def test_invalid_designs_fail_closed(inputs, problem):
    module, experiment, metrics = inputs
    experiment = deepcopy(experiment)
    rec = experiment["records"][0]
    if problem == "axes":
        rec["hyperparameters"][0]["minibatch_size"] = 128
    elif problem == "hidden_change":
        rec["hyperparameters"][0]["dropout_rate"] = 0.25
    elif problem == "coordinates":
        duplicate = deepcopy(rec)
        duplicate["condition"] = "duplicate-coordinates"
        experiment["records"].append(duplicate)
    elif problem == "policy":
        rec["checkpoint_policy"] = "terminal"
    else:
        rec["fold_count"] = 4
    with pytest.raises(ValueError):
        module.analyze(experiment, metrics, replicates=10)


def test_report_is_reproducible_and_refuses_snapshot_overwrite(inputs, tmp_path):
    module, experiment, metrics = inputs
    design = tmp_path / "experiment.json"
    table = tmp_path / "metrics.csv"
    design.write_text(json.dumps(experiment))
    metrics.to_csv(table, index=False)
    out = tmp_path / "analysis"
    args = ["--experiment", str(design), "--metrics", str(table), "--out", str(out), "--replicates", "30"]
    assert module.main(args) == 0
    assert (out / "recipe-analysis.pdf").stat().st_size > 1000
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["evaluated_conditions"] == 8 and provenance["planned_conditions"] == 32
    assert len(provenance["inputs"]) == 4
    assert (out / "analysis_source/analyze_processing_recipe.py").is_file()
    assert any(row["path"] == "recipe-analysis.pdf" for row in provenance["outputs"])
    with pytest.raises(ValueError, match="new output directory"):
        module.main(args)
    shuffled = metrics.sample(frac=1, random_state=4)
    expected = module.analyze(experiment, metrics, replicates=30)[3]
    actual = module.analyze(experiment, shuffled, replicates=30)[3]
    pandas.testing.assert_frame_equal(expected, actual)


def test_single_completed_condition_has_maps_but_no_invented_contrasts(inputs, tmp_path):
    module, experiment, metrics = inputs
    metrics = metrics.loc[metrics.condition.eq(metrics.condition.iloc[0])]
    records, _, means, contrasts, _ = module.analyze(experiment, metrics, replicates=10)
    assert contrasts.empty
    module.render_report(records, means, contrasts, tmp_path)
    assert (tmp_path / "recipe-analysis.pdf").stat().st_size > 1000

"""Fail-closed orchestration and metric tests for the paired width experiment."""

import importlib.util
import json
from pathlib import Path

import pandas
import pytest

from .test_processing_matching import training_frame


@pytest.fixture
def sweep_module(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts/training"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("kernel_sweep", directory / "run_processing_kernel_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_sweep_preparation_is_matched_and_immutable(tmp_path, sweep_module):
    table = tmp_path / "train.csv"
    training_frame().assign(matching_affinity_reference_sha256="a" * 64).to_csv(table, index=False)
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pandas.DataFrame({"sample_id": ["held-out"]}).to_csv(holdout / "processing_samples.csv", index=False)
    args = ["--out", str(tmp_path / "experiment"), "--train-data", str(table),
            "--public-root", str(tmp_path), "--release-holdout-dir", str(holdout),
            "--source-commit", "source", "--prepare-only"]
    assert sweep_module.main(args) == 0
    assert sweep_module.main(args) == 0
    config = json.loads((tmp_path / "experiment/experiment.json").read_text())
    assert config["networks"] == 48
    assert len(config["records"]) == 12
    with pytest.raises(ValueError, match="change inputs"):
        sweep_module.main(args + ["--random-seed", "99"])
    training_frame().drop(columns="processing_matching_policy").to_csv(table, index=False)
    with pytest.raises(ValueError, match="requires matched"):
        sweep_module.main(args)


def test_kernel_sweep_validation_metrics_keep_folds_and_samples(tmp_path, sweep_module):
    path = tmp_path / "predictions.csv.bz2"
    pandas.DataFrame({"sample_id": ["a"] * 4, "fold_num": [0, 0, 1, 1],
                      "hit": [1, 0, 1, 0], "processing_score": [0.9, 0.1, 0.1, 0.9]}).to_csv(path, index=False)
    result = sweep_module.validation_metrics(path, "test")
    assert result.fold_num.tolist() == [0, 1]
    assert result.roc_auc.tolist() == [1.0, 0.0]
    assert result.pr_auc.tolist() == [1.0, 0.5]


def test_kernel_sweep_renderer_and_member_evaluation(tmp_path, sweep_module):
    """Preserve member identities and exercise both plot panels on toy data."""
    from mhcflurry import Class1ProcessingNeuralNetwork, Class1ProcessingPredictor
    from .test_processing_affinity_control import _cohort

    from mhcflurry.processing_matching import make_affinity_controlled_risk_sets
    matched, _ = make_affinity_controlled_risk_sets(_cohort(), decoys_per_hit=1)
    matched["peptide"] = matched.peptide.str.replace("B", "R")
    model = Class1ProcessingNeuralNetwork(convolutional_filters=4, convolutional_kernel_size=5,
                                         convolutional_padding_mode="unknown", dropout_rate=0)
    model._network = model.make_network(
        **model.network_hyperparameter_defaults.subselect(model.hyperparameters))
    models = tmp_path / "models"
    Class1ProcessingPredictor([model, model]).save(str(models))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    matched.to_csv(evaluation / "matching_assignments.csv.bz2", index=False)
    records = sweep_module.build_kernel_conditions()[:2]
    for name in ["public_2_1_x"] + [r[0] for r in records]:
        sweep_module.evaluate_predictor(tmp_path, name, models, matched)
        scores = pandas.read_csv(evaluation / (name + ".predictions.csv.bz2"))
        assert {"member_00", "member_01", "source_row", name} <= set(scores)
        assert (scores.member_00 == scores.member_01).all()
        if name != "public_2_1_x":
            directory = tmp_path / name
            directory.mkdir()
            pandas.DataFrame({"condition": [name], "sample_id": ["toy"], "fold_num": [0],
                              "roc_auc": [0.5], "pr_auc": [0.5], "ppv_at_n": [0.5]}).to_csv(
                directory / "validation_metrics.csv", index=False)
    sweep_module.render_summary(tmp_path, records)
    assert (evaluation / "kernel_width.pdf").stat().st_size > 1000
    assert (tmp_path / "validation_kernel_width.pdf").stat().st_size > 1000


def test_kernel_sweep_tiny_end_to_end_and_resume(tmp_path, sweep_module, monkeypatch):
    """Exercise real command initialization, fold reuse, fits, selection and plots."""
    conditions = sweep_module.build_kernel_conditions()[:2]
    for _, grid, _ in conditions:
        grid[0].update(convolutional_filters=4, post_convolutional_dense_layer_sizes=[],
                       max_epochs=1, cleavage_boundary_hidden_size=4)
    monkeypatch.setattr(sweep_module, "build_kernel_conditions", lambda: conditions)
    table = tmp_path / "train.csv"
    frames = []
    for sample in range(20):
        frame = training_frame().assign(
            matching_affinity_reference_sha256="a" * 64, sample_id="sample_%d" % sample)
        # Default pandas parsing loses one ULP on the second CSV round trip.
        # Additional metadata must remain identical too, not only model inputs.
        frame["audit_float"] = 182.06260894694452
        frame["peptide"] = frame.peptide.str.replace("B", "R")
        frames.append(frame)
    pandas.concat(frames, ignore_index=True).to_csv(table, index=False)
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pandas.DataFrame({"sample_id": ["held-out"]}).to_csv(holdout / "processing_samples.csv", index=False)
    out = tmp_path / "experiment"
    args = ["--out", str(out), "--train-data", str(table), "--public-root", str(tmp_path),
            "--release-holdout-dir", str(holdout), "--source-commit", "test",
            "--gpus", "0", "--num-jobs", "0", "--evaluation", "none", "--save-all-checkpoints"]
    assert sweep_module.main(args) == 0
    assert (out / "completed.json").exists()
    assert len(pandas.read_csv(out / "validation_summary.csv")) == 2
    assert (out / "validation_kernel_width.pdf").stat().st_size > 1000
    assert sweep_module.main(args) == 0
    for name, _, _ in conditions:
        manifest = pandas.read_csv(out / name / "processing/models.selected.short_flanks/manifest.csv")
        assert len(manifest) == 4
        predictions = pandas.read_csv(out / "checkpoint_predictions" / (name + ".validation_predictions.csv.bz2"))
        assert set(predictions.checkpoint_policy) == {"best", "terminal"}
        assert predictions.model_name.nunique() == 4
        assert set(predictions.fold_num) == {0, 1, 2, 3}
        assert {"sample_id", "peptide", "n_flank", "c_flank", "validation_row_index"} <= set(predictions)

    # Recovery imports only complete fits without invoking any training command.
    recovery = tmp_path / "recovery"
    recovery_args = list(args)
    recovery_args[recovery_args.index("--out") + 1] = str(recovery)
    recovery_args += ["--resume-from", str(out), "--save-all-checkpoints"]
    def unexpected_command(*args, **kwargs):
        raise AssertionError("Recovery of complete conditions must not retrain")
    monkeypatch.setattr(sweep_module.Driver, "run", unexpected_command)
    assert sweep_module.main(recovery_args) == 0
    assert sweep_module.main(recovery_args) == 0
    record = json.loads((recovery / "recovery.json").read_text())
    assert set(record["completed_conditions"]) == {r[0] for r in conditions}
    # A changed artifact cannot silently be accepted on resume.
    altered = recovery / conditions[0][0] / "validation_metrics.csv"
    altered.write_text(altered.read_text() + "\n")
    with pytest.raises(ValueError, match="Recovered artifact changed"):
        sweep_module.main(recovery_args)

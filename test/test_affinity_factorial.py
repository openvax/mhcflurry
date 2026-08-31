"""Tests for the controlled affinity recipe sweep."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy
import pytest
import yaml

from mhcflurry.class1_neural_network import Class1NeuralNetworkModel


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "training" / "run_affinity_factorial.sh"


def load_script(name):
    """Load a training script as a module."""
    path = REPO / "scripts" / "training" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_affinity_factorial_runner_exposes_explicit_cli():
    result = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in (
            "--out",
            "--train-data",
            "--allele-sequences",
            "--pretrain-data",
            "--data-eval-dir",
            "--release-holdout-dir",
            "--source-commit",
            "--mode",
            "--condition",
            "--random-seed",
            "--gpus",
            "--max-workers-per-gpu",
            "--dataloader-num-workers",
            "--max-tasks-per-worker",
            "--torch-compile",
            "--torch-compile-loss",
            "--matmul-precision"):
        assert flag in result.stdout


def test_affinity_factorial_runner_does_not_accept_env_only_configuration(
        monkeypatch,
):
    for name in (
            "MHCFLURRY_OUT",
            "TRAIN_DATA",
            "ALLELE_SEQUENCES",
            "PRETRAIN_DATA",
            "DATA_EVAL_DIR",
            "RELEASE_HOLDOUT_DIR",
            "SOURCE_COMMIT",
            "GPUS"):
        monkeypatch.setenv(name, "/ignored")
    result = subprocess.run(
        ["bash", str(RUNNER)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Missing required argument for MHCFLURRY_OUT" in result.stderr


def test_affinity_factorial_runner_validates_cli_values():
    result = subprocess.run(
        [
            "bash", str(RUNNER),
            "--out", "/unused",
            "--train-data", "/unused",
            "--allele-sequences", "/unused",
            "--pretrain-data", "/unused",
            "--data-eval-dir", "/unused",
            "--release-holdout-dir", "/unused",
            "--source-commit", "deadbeef",
            "--mode", "invalid",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Invalid --mode: invalid" in result.stderr


def test_orthogonal_affinity_initializer():
    model = Class1NeuralNetworkModel(
        peptide_encoding_shape=(2, 3),
        layer_sizes=[4],
        init="orthogonal",
    )
    for layer in (model.dense_layers[0], model.output_layer):
        weights = layer.weight.detach().numpy()
        numpy.testing.assert_allclose(
            weights @ weights.T,
            numpy.eye(weights.shape[0]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_affinity_factorial_has_only_effective_initialization_axes(tmp_path):
    module = load_script("generate_affinity_factorial")
    records = module.build_conditions()

    assert len(records) == 40
    assert len({condition for condition, _, _ in records}) == 40
    assert {len(grid) for _, grid, _ in records} == {2}
    assert sum(
        axes["data_dependent_initialization_method"] == "lsuv"
        for _, _, axes in records
    ) == 16
    assert sum(
        axes["data_dependent_initialization_method"] is None
        for _, _, axes in records
    ) == 24
    for _, grid, axes in records:
        assert {item["minibatch_size"] for item in grid} == {
            axes["minibatch_size"]
        }
        assert {item["optimizer_implementation"] for item in grid} == {
            axes["optimizer_implementation"]
        }
        assert {item["init"] for item in grid} == {axes["init"]}
        assert {
            item["data_dependent_initialization_method"] for item in grid
        } == {axes["data_dependent_initialization_method"]}
        if axes["data_dependent_initialization_method"] == "lsuv":
            assert axes["init"] == "glorot_uniform"
            assert axes["effective_hidden_initializer"] == (
                "orthogonal_then_lsuv"
            )
        else:
            assert axes["lsuv_target"] == "not_applicable"
            assert axes["effective_hidden_initializer"] == axes["init"]

    manifest = module.write_conditions(tmp_path)
    assert len(manifest["records"]) == 40
    assert manifest["fixed_controls"][
        "lsuv_replaces_eligible_weights_with_orthogonal"
    ] is True
    for record in manifest["records"]:
        values = yaml.safe_load(
            (tmp_path / record["hyperparameters_path"]).read_text()
        )
        assert len(values) == 2


def test_affinity_factorial_full_mode_retains_all_architectures():
    module = load_script("generate_affinity_factorial")
    records = module.build_conditions(
        mode="full",
        minibatch_sizes=(128,),
        optimizer_implementations=("keras",),
        lsuv_targets=("post_activation",),
        initializers=("glorot_uniform",),
    )
    assert len(records) == 2
    assert {len(grid) for _, grid, _ in records} == {35}


def metric_summary(a, b):
    """Return a minimal compare-models affinity summary."""
    macro = {
        metric: {"a": a[index], "b": b[index]}
        for index, metric in enumerate(("roc_auc", "pr_auc", "ppv_at_n"))
    }
    micro_a = {
        metric: a[index]
        for index, metric in enumerate(("roc_auc", "pr_auc", "ppv_at_n"))
    }
    micro_b = {
        metric: b[index]
        for index, metric in enumerate(("roc_auc", "pr_auc", "ppv_at_n"))
    }
    return {
        "macro_mean_over_alleles": macro,
        "micro_pooled": {"a": micro_a, "b": micro_b},
        "allele_count": {},
        "n_rows": 10,
        "n_hits": 2,
        "n_alleles_reported": 1,
    }


def test_affinity_factorial_summary_keeps_comparison_sides_separate(tmp_path):
    module = load_script("summarize_affinity_factorial")
    baseline = "baseline"
    candidate = "candidate"
    manifest = {
        "baseline_condition": baseline,
        "records": [
            {"condition": baseline, "minibatch_size": 128},
            {"condition": candidate, "minibatch_size": 256},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    baseline_dir = tmp_path / "baseline-vs-public" / "affinity"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "summary.json").write_text(
        json.dumps(metric_summary((0.9, 0.6, 0.5), (0.8, 0.4, 0.3)))
    )
    candidate_dir = (
        tmp_path / candidate / "comparison-vs-baseline" / "affinity"
    )
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "summary.json").write_text(
        json.dumps(metric_summary((0.91, 0.63, 0.52), (0.9, 0.6, 0.5)))
    )

    records = module.summarize(tmp_path)
    by_condition = {record["condition"]: record for record in records}
    assert by_condition[candidate]["macro_pr_auc"] == 0.63
    assert by_condition[candidate]["macro_pr_auc_baseline"] == 0.6
    assert by_condition[candidate]["macro_pr_auc_delta"] == pytest.approx(0.03)
    assert by_condition[baseline]["macro_pr_auc_public"] == 0.4
    assert by_condition[candidate]["macro_pr_auc_public"] is None
    assert (tmp_path / "summary.csv").exists()


def test_affinity_factorial_model_verifier_checks_folds_and_batch(tmp_path):
    generator = load_script("generate_affinity_factorial")
    verifier = load_script("verify_affinity_factorial_models")
    _, grid, _ = generator.build_conditions(
        minibatch_sizes=(128,),
        optimizer_implementations=("keras",),
        lsuv_targets=("post_activation",),
        initializers=(),
    )[0]
    hyperparameters_path = tmp_path / "hyperparameters.yaml"
    hyperparameters_path.write_text(yaml.safe_dump(grid))
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    rows = []
    for architecture_num, hyperparameters in enumerate(grid):
        for fold in range(4):
            model_name = "model-%d-%d" % (architecture_num, fold)
            numpy.savez(models_dir / ("weights_%s.npz" % model_name), value=[1])
            rows.append({
                "model_name": model_name,
                "allele": "pan-class1",
                "config_json": json.dumps({
                    "hyperparameters": hyperparameters,
                    "fit_info": [{
                        "effective_minibatch_size": 128,
                        "training_info": {
                            "fold_num": fold,
                            "num_folds": 4,
                            "train_peptide_hash": "fold-%d" % fold,
                        },
                    }],
                }),
            })
    with (models_dir / "manifest.csv").open("w", newline="") as fd:
        writer = csv.DictWriter(
            fd, fieldnames=("model_name", "allele", "config_json")
        )
        writer.writeheader()
        writer.writerows(rows)

    report = verifier.verify(models_dir, hyperparameters_path)
    assert report["model_count"] == 8
    rows[0]["config_json"] = json.dumps({
        "hyperparameters": grid[0],
        "fit_info": [{
            "effective_minibatch_size": 64,
            "training_info": {
                "fold_num": 0,
                "num_folds": 4,
                "train_peptide_hash": "fold-0",
            },
        }],
    })
    with (models_dir / "manifest.csv").open("w", newline="") as fd:
        writer = csv.DictWriter(
            fd, fieldnames=("model_name", "allele", "config_json")
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Training minibatch shrank"):
        verifier.verify(models_dir, hyperparameters_path)

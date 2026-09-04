"""Tests for the controlled affinity recipe sweep."""

import csv
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy
import pandas
import pytest
import yaml

from mhcflurry.class1_neural_network import Class1NeuralNetworkModel


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "training" / "run_affinity_factorial.sh"
ARCHITECTURE_RUNNER = (
    REPO
    / "scripts"
    / "training"
    / "run_affinity_factorial_architecture_evaluation.sh"
)
PUBLIC_EVALUATOR = (
    REPO
    / "scripts"
    / "training"
    / "evaluate_affinity_factorial_public.sh"
)
PROCESSING_REGULARIZATION_RUNNER = (
    REPO
    / "scripts"
    / "training"
    / "run_processing_regularization_activation.sh"
)


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
            "--public-affinity-dir",
            "--source-commit",
            "--mode",
            "--design",
            "--regularization-base-recipe",
            "--condition",
            "--random-seed",
            "--evaluation",
            "--gpus",
            "--max-workers-per-gpu",
            "--dataloader-num-workers",
            "--max-tasks-per-worker",
            "--torch-compile",
            "--torch-compile-loss",
            "--matmul-precision"):
        assert flag in result.stdout


def test_affinity_public_evaluator_exposes_explicit_cli():
    result = subprocess.run(
        ["bash", str(PUBLIC_EVALUATOR), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in (
            "--factorial-dir",
            "--public-affinity-dir",
            "--data-eval-dir",
            "--release-holdout-dir",
            "--analysis-source-commit",
            "--public-label",
            "--gpus",
            "--max-workers-per-gpu"):
        assert flag in result.stdout
    script = PUBLIC_EVALUATOR.read_text()
    assert "--affinity-source no_additional_ms" in script
    assert "--affinity-training-overlap-policy exclude" in script
    assert "--skip-affinity-predictions" not in script
    assert "mhcflurry train plot-loss-curves" in script
    assert "mhcflurry plot-model-comparison" in script
    assert "mhcflurry eval affinity-candidate-figures" in script


def test_processing_regularization_runner_exposes_explicit_cli():
    result = subprocess.run(
        ["bash", str(PROCESSING_REGULARIZATION_RUNNER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in (
            "--out",
            "--train-data",
            "--data-eval-dir",
            "--release-holdout-dir",
            "--source-commit",
            "--random-seed",
            "--evaluation",
            "--experiments-dir",
            "--experiment-name",
            "--source-archive",
            "--gpus",
            "--num-jobs",
            "--max-workers-per-gpu",
            "--dataloader-num-workers",
            "--max-tasks-per-worker"):
        assert flag in result.stdout


def test_affinity_architecture_runner_exposes_explicit_cli():
    result = subprocess.run(
        ["bash", str(ARCHITECTURE_RUNNER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in (
            "--factorial-dir",
            "--data-eval-dir",
            "--release-holdout-dir",
            "--training-source-commit",
            "--analysis-source-commit",
            "--gpus",
            "--max-workers-per-gpu",
            "--max-tasks-per-worker",
            "--torch-compile",
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
            "--public-affinity-dir", "/unused",
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


@pytest.mark.parametrize(
    "base_recipe,optimizer,lsuv_target,lsuv_method",
    [
        ("native-pre-1024", "pytorch", "pre_activation", "lsuv"),
        ("native-post-1024", "pytorch", "post_activation", "lsuv"),
        ("keras-no-lsuv-1024", "keras", "not_applicable", None),
    ],
)
def test_affinity_regularization_activation_design(
        tmp_path, base_recipe, optimizer, lsuv_target, lsuv_method):
    module = load_script("generate_affinity_factorial")
    records = module.build_regularization_activation_conditions(base_recipe)

    assert len(records) == 10
    assert {len(grid) for _, grid, _ in records} == {2}
    assert {axes["activation"] for _, _, axes in records} == {
        "tanh", "relu", "silu", "gelu"}
    assert {axes["normalization"] for _, _, axes in records} == {
        "none", "batch", "layer"}
    assert {axes["dropout_keep_probability"] for _, _, axes in records} == {
        0.5, 0.75, 1.0}
    assert {axes["restore_best_weights"] for _, _, axes in records} == {
        False, True}
    assert {axes["patience"] for _, _, axes in records} == {20, 40}
    for _, grid, axes in records:
        assert {item["optimizer_implementation"] for item in grid} == {
            optimizer}
        assert {item["data_dependent_initialization_method"] for item in grid} == {
            lsuv_method}
        assert axes["lsuv_target"] == lsuv_target

    out = tmp_path / base_recipe
    manifest = module.write_conditions(
        out,
        design="regularization-activation",
        regularization_base_recipe=base_recipe,
    )
    assert manifest["baseline_condition"] == "%s__control" % base_recipe
    assert len(manifest["records"]) == 10


def test_processing_regularization_activation_design(tmp_path):
    module = load_script("generate_processing_regularization_activation")
    records = module.build_conditions()

    assert len(records) == 20
    for architecture in ("small", "large"):
        architecture_records = [
            record for record in records
            if record[2]["architecture"] == architecture
        ]
        assert len(architecture_records) == 10
        assert {axes["activation"] for _, _, axes in architecture_records} == {
            "tanh", "relu", "silu", "gelu"}
        assert {axes["normalization"] for _, _, axes in architecture_records} == {
            "none", "batch", "layer"}
        assert {axes["restore_best_weights"] for _, _, axes in (
            architecture_records
        )} == {False, True}
        assert {axes["patience"] for _, _, axes in architecture_records} == {
            20, 40}
        for _, grid, axes in architecture_records:
            assert len(grid) == 1
            item = grid[0]
            assert item["n_flank_length"] == 5
            assert item["c_flank_length"] == 5
            assert item["minibatch_size"] == 512
            assert item["optimizer_implementation"] == "keras"
            assert axes["baseline_condition"] == "%s__control" % architecture

    manifest = module.write_conditions(tmp_path / "processing")
    assert len(manifest["records"]) == 20
    assert manifest["fixed_controls"]["flank_length_each_side"] == 5


def test_training_stop_summary_tracks_updates_and_validation_tail(tmp_path):
    module = load_script("summarize_training_stops")
    history = pandas.DataFrame({
        "manifest_path": ["condition/models/manifest.csv"] * 4,
        "model_name": ["model"] * 4,
        "fit_index": [0] * 4,
        "phase": ["finetune"] * 4,
        "fold": [0] * 4,
        "architecture": [1] * 4,
        "replicate": [0] * 4,
        "epoch": [1, 2, 3, 4],
        "val_loss": [0.4, 0.2, 0.21, 0.22],
        "epoch_num_train_batches": [10] * 4,
        "epoch_num_train_rows": [100] * 4,
        "epoch_train_time": [1.5] * 4,
        "epoch_total_time": [2.0] * 4,
    })
    path = tmp_path / "training_history.csv"
    history.to_csv(path, index=False)

    details, summary = module.summarize(path)

    assert details.best_epoch.tolist() == [2]
    assert details.final_epoch.tolist() == [4]
    assert details.epochs_after_best.tolist() == [2]
    assert details.optimizer_steps.tolist() == [40.0]
    assert details.training_rows_seen.tolist() == [400.0]
    assert summary.total_epochs.tolist() == [4]
    assert summary.total_optimizer_steps.tolist() == [40.0]


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
    baseline_dir = (
        tmp_path / "baseline-vs-public-no-additional-ms" / "affinity"
    )
    baseline_dir.mkdir(parents=True)
    baseline_public = metric_summary((0.9, 0.6, 0.5), (0.8, 0.4, 0.3))
    baseline_public["benchmark_identity"] = {
        "algorithm": "test",
        "columns": ["source_file", "hla", "peptide", "hit"],
        "ordered_rows": True,
        "row_count": 10,
        "sha256": "a" * 64,
    }
    (baseline_dir / "summary.json").write_text(json.dumps(baseline_public))
    candidate_dir = (
        tmp_path / candidate / "comparison-vs-baseline" / "affinity"
    )
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "summary.json").write_text(
        json.dumps(metric_summary((0.91, 0.63, 0.52), (0.9, 0.6, 0.5)))
    )
    candidate_public_dir = (
        tmp_path / candidate / "comparison-vs-public-no-additional-ms" /
        "affinity"
    )
    candidate_public_dir.mkdir(parents=True)
    candidate_public = metric_summary(
        (0.905, 0.625, 0.515), (0.8, 0.4, 0.3))
    candidate_public["benchmark_identity"] = baseline_public[
        "benchmark_identity"]
    (candidate_public_dir / "summary.json").write_text(
        json.dumps(candidate_public))

    records = module.summarize(tmp_path)
    by_condition = {record["condition"]: record for record in records}
    assert by_condition[candidate]["macro_pr_auc"] == 0.63
    assert by_condition[candidate]["macro_pr_auc_baseline"] == 0.6
    assert by_condition[candidate]["macro_pr_auc_delta"] == pytest.approx(0.03)
    assert by_condition[baseline]["macro_pr_auc_public"] == 0.4
    assert by_condition[candidate]["macro_pr_auc_public"] == 0.4
    assert by_condition[candidate][
        "macro_pr_auc_vs_public_candidate"] == 0.625
    assert by_condition[candidate]["macro_pr_auc_vs_public_delta"] == (
        pytest.approx(0.225))
    assert by_condition[candidate][
        "public_benchmark_identity_sha256"] == "a" * 64
    assert (tmp_path / "summary.csv").exists()

    candidate_public["benchmark_identity"] = {
        **baseline_public["benchmark_identity"], "sha256": "b" * 64}
    (candidate_public_dir / "summary.json").write_text(
        json.dumps(candidate_public))
    with pytest.raises(ValueError, match="different benchmark rows"):
        module.summarize(tmp_path)


def test_affinity_architecture_summary_classifies_strict_dominance(tmp_path):
    module = load_script("summarize_affinity_factorial_architectures")
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
    architecture_dir = tmp_path / "architecture_evaluation"
    provenance = {
        "architecture_num": 0,
        "architecture": {
            "topology": "feedforward",
            "layer_sizes": [512, 512],
            "dense_layer_l1_regularization": 1e-8,
        },
    }
    for condition in (baseline, candidate):
        subset = (
            architecture_dir
            / "subsets"
            / condition
            / "architecture_0"
        )
        subset.mkdir(parents=True)
        (subset / "subset_provenance.json").write_text(json.dumps(provenance))
    baseline_summary = (
        architecture_dir
        / "baseline-vs-public"
        / "architecture_0"
        / "affinity"
    )
    baseline_summary.mkdir(parents=True)
    (baseline_summary / "summary.json").write_text(
        json.dumps(metric_summary((0.9, 0.6, 0.5), (0.8, 0.4, 0.3)))
    )
    candidate_summary = (
        architecture_dir
        / "comparisons"
        / candidate
        / "architecture_0-vs-baseline"
        / "affinity"
    )
    candidate_summary.mkdir(parents=True)
    (candidate_summary / "summary.json").write_text(
        json.dumps(metric_summary((0.91, 0.63, 0.52), (0.9, 0.6, 0.5)))
    )

    records = module.summarize(tmp_path)
    by_condition = {record["condition"]: record for record in records}
    assert by_condition[baseline]["metric_dominance"] == "reference"
    assert by_condition[candidate]["metric_dominance"] == (
        "strictly_dominates_baseline"
    )
    assert by_condition[candidate]["macro_pr_auc_delta"] == pytest.approx(0.03)
    assert (architecture_dir / "summary.csv").exists()


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
            fitted_hyperparameters = deepcopy(hyperparameters)
            fitted_hyperparameters["learning_rate"] /= 10
            rows.append({
                "model_name": model_name,
                "allele": "pan-class1",
                "config_json": json.dumps({
                    "hyperparameters": fitted_hyperparameters,
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
    wrong_learning_rate = json.loads(rows[0]["config_json"])
    wrong_learning_rate["hyperparameters"]["learning_rate"] = 0.001
    rows[0]["config_json"] = json.dumps(wrong_learning_rate)
    with (models_dir / "manifest.csv").open("w", newline="") as fd:
        writer = csv.DictWriter(
            fd, fieldnames=("model_name", "allele", "config_json")
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="learning_rate mismatch"):
        verifier.verify(models_dir, hyperparameters_path)

    fitted_hyperparameters = deepcopy(grid[0])
    fitted_hyperparameters["learning_rate"] /= 10
    rows[0]["config_json"] = json.dumps({
        "hyperparameters": fitted_hyperparameters,
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


def test_split_affinity_architectures_writes_fold_complete_subsets(tmp_path):
    module = load_script("split_affinity_architectures")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    rows = []
    for architecture_num, topology in enumerate((
            "feedforward", "with-skip-connections")):
        for fold_num in range(4):
            name = "model-%d-%d" % (architecture_num, fold_num)
            numpy.savez(models_dir / ("weights_%s.npz" % name), value=[1])
            rows.append({
                "model_name": name,
                "allele": "pan-class1",
                "config_json": json.dumps({
                    "hyperparameters": {
                        "topology": topology,
                        "layer_sizes": [512, 512],
                        "dense_layer_l1_regularization": 1e-8,
                    },
                    "fit_info": [{
                        "training_info": {
                            "architecture_num": architecture_num,
                            "fold_num": fold_num,
                        },
                    }],
                }),
            })
    pandas = pytest.importorskip("pandas")
    pandas.DataFrame(rows).to_csv(models_dir / "manifest.csv", index=False)
    (models_dir / "train_data.csv.bz2").write_bytes(b"training")
    (models_dir / "allele_sequences.csv").write_text(
        "allele,sequence\nHLA-A*02:01," + "A" * 39 + "\n"
    )

    result = module.split_models(models_dir, tmp_path / "subsets")
    assert len(result) == 2
    for architecture_num in range(2):
        target = tmp_path / "subsets" / ("architecture_%d" % architecture_num)
        subset = pandas.read_csv(target / "manifest.csv")
        assert len(subset) == 4
        assert set(subset.model_name) == {
            "model-%d-%d" % (architecture_num, fold) for fold in range(4)
        }
        assert (target / "train_data.csv.bz2").read_bytes() == b"training"
        provenance = json.loads((target / "subset_provenance.json").read_text())
        assert provenance["folds"] == [0, 1, 2, 3]
        assert provenance["architecture_num"] == architecture_num

    repeated = module.split_models(models_dir, tmp_path / "subsets")
    assert [item["models_dir"] for item in repeated] == [
        str(tmp_path / "subsets" / "architecture_0"),
        str(tmp_path / "subsets" / "architecture_1"),
    ]

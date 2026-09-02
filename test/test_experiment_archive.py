# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime, timezone
import json

import pandas

from mhcflurry.experiment_archive import sha256_file, snapshot_experiment


def test_snapshot_experiment_exports_reconstructable_tables(tmp_path):
    source = tmp_path / "run"
    models = source / "condition-a" / "models.unselected.combined"
    comparison = source / "condition-a" / "comparison-vs-baseline" / "affinity"
    conditions = source / "conditions"
    models.mkdir(parents=True)
    comparison.mkdir(parents=True)
    conditions.mkdir()

    training_info = {
        "phase": "finetune",
        "fold_num": 2,
        "architecture_num": 7,
        "replicate_num": 0,
        "work_item_name": "work-123",
    }
    fit_info = {
        "training_info": training_info,
        "loss": [0.4, 0.2],
        "val_loss": [0.5, 0.3],
        "epoch_train_time": [4.0, 3.5],
        "effective_minibatch_size": 256,
        "time": 12.5,
    }
    config = {
        "hyperparameters": {
            "topology": "feedforward",
            "layer_sizes": [512, 512],
            "minibatch_size": 256,
            "optimizer": "rmsprop",
            "optimizer_implementation": "keras",
            "init": "glorot_uniform",
            "data_dependent_initialization_method": "lsuv",
            "data_dependent_initialization_target": "post_activation",
        },
        "fit_info": [fit_info],
        "network_json": "intentionally omitted from normalized export",
    }
    pandas.DataFrame([{
        "model_name": "PAN-CLASS1-0-test",
        "allele": "pan-class1",
        "config_json": json.dumps(config),
    }]).to_csv(models / "manifest.csv", index=False)
    weight = models / "weights_PAN-CLASS1-0-test.npz"
    weight.write_bytes(b"weight payload")
    (models / "train_data.csv.bz2").write_bytes(b"training payload")
    pandas.DataFrame([{
        "allele": "HLA-A*02:01",
        "a_roc_auc": 0.9,
        "b_roc_auc": 0.8,
    }]).to_csv(comparison / "per_allele.csv", index=False)
    predictions = comparison / "predictions.csv.bz2"
    predictions.write_bytes(
        b"held-out predictions larger than test limit" * 10)
    figure = comparison / "plots" / "model_comparison_figures.pdf"
    figure.parent.mkdir()
    figure.write_bytes(b"pdf")
    (conditions / "condition-a.yaml").write_text("minibatch_size: 256\n")
    (source / "gpu_occupancy.csv").write_text("timestamp,gpu_util\n1,99\n")
    command = source / "command.sh"
    command.write_text("run-affinity --condition condition-a\n")
    input_data = tmp_path / "input.csv"
    input_data.write_text("peptide,allele\nSIINFEKL,HLA-A*02:01\n")
    (source / "provenance.txt").write_text(
        "source_commit=abcdef1234567890\n%s  %s\n" % (
            sha256_file(input_data), input_data))
    source_archive = tmp_path / "source.tar.gz"
    source_archive.write_bytes(b"source archive")

    destination = snapshot_experiment(
        source,
        tmp_path / "experiments",
        "affinity frontier",
        collector_commit="fedcba0987654321",
        source_archive=source_archive,
        command_files=[command],
        input_files=[input_data],
        max_copy_bytes=64,
        captured_at=datetime(2026, 9, 2, 12, 34, 56, tzinfo=timezone.utc),
    )

    assert destination.name == (
        "20260902T123456Z-affinity-frontier-abcdef123456")
    experiment = json.loads((destination / "experiment.json").read_text())
    assert experiment["source_commit"] == "abcdef1234567890"
    assert experiment["collector_commit"] == "fedcba0987654321"
    assert experiment["training_tables"] == {
        "model_manifest_count": 1,
        "model_count": 1,
        "fit_count": 1,
        "epoch_count": 2,
    }

    history = pandas.read_csv(destination / "data" / "training_history.csv")
    assert history.epoch.tolist() == [1, 2]
    assert history.loss.tolist() == [0.4, 0.2]
    assert history.val_loss.tolist() == [0.5, 0.3]
    assert history.epoch_train_time.tolist() == [4.0, 3.5]
    assert history.fold.tolist() == [2, 2]

    models_table = pandas.read_csv(destination / "data" / "models.csv")
    assert models_table.minibatch_size.tolist() == [256]
    assert models_table.optimizer_implementation.tolist() == ["keras"]
    configs = [
        json.loads(line)
        for line in (destination / "data" / "model_configs.jsonl")
        .read_text().splitlines()
    ]
    assert "network_json" not in configs[0]["config"]
    assert configs[0]["config"]["fit_info"][0]["loss"] == [0.4, 0.2]

    inventory = pandas.read_csv(destination / "source_files.csv")
    weight_row = inventory.loc[
        inventory.relative_path.str.endswith(weight.name)].iloc[0]
    assert weight_row.sha256 == sha256_file(weight)
    assert not bool(weight_row.copied)
    metric_row = inventory.loc[
        inventory.relative_path.str.endswith("per_allele.csv")].iloc[0]
    assert bool(metric_row.copied)
    assert (
        destination / "artifacts" / "condition-a" /
        "comparison-vs-baseline" / "affinity" / "per_allele.csv"
    ).exists()
    assert (
        destination / "artifacts" / "condition-a" /
        "comparison-vs-baseline" / "affinity" / "predictions.csv.bz2"
    ).read_bytes() == predictions.read_bytes()
    assert (
        destination / "artifacts" / "condition-a" /
        "comparison-vs-baseline" / "affinity" / "plots" /
        "model_comparison_figures.pdf"
    ).exists()
    assert (destination / "source" / source_archive.name).exists()
    assert (destination / "supplemental" / "command" / command.name).exists()

    external = pandas.read_csv(destination / "external_inputs.csv")
    assert external.recorded_sha256.tolist() == [sha256_file(input_data)]
    assert "training_history.csv" in (destination / "README.md").read_text()


def test_snapshot_rejects_naive_capture_time(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    (source / "provenance.txt").write_text("source_commit=abc1234\n")

    try:
        snapshot_experiment(
            source,
            tmp_path / "experiments",
            "test",
            captured_at=datetime(2026, 9, 2),
        )
    except ValueError as error:
        assert "timezone-aware" in str(error)
    else:
        raise AssertionError("Expected a timezone-aware timestamp error")

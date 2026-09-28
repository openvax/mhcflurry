# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pandas
import pytest

from mhcflurry.cli import main as cli_main
from mhcflurry.cli import plot_loss_curves


def write_manifest(path, model_name, *, fold, architecture, work_item):
    config = {
        "hyperparameters": {
            "layer_sizes": [32],
            "dense_layer_l1_regularization": 0.0,
        },
        "fit_info": [
            {
                "loss": [0.2, 0.1],
                "val_loss": [0.3, 0.2],
                "training_info": {
                    "phase": "finetune",
                    "fold_num": fold,
                    "architecture_num": architecture,
                    "replicate_num": 0,
                    "work_item_name": work_item,
                },
            }
        ],
    }
    pandas.DataFrame(
        [{
            "model_name": model_name,
            "allele": "pan-class1",
            "config_json": json.dumps(config),
        }]
    ).to_csv(path, index=False)


def test_selection_identity_survives_model_rename(tmp_path):
    selected_path = tmp_path / "selected.csv"
    candidate_path = tmp_path / "candidate.csv"
    write_manifest(
        selected_path,
        "PAN-CLASS1-1-selected-copy",
        fold=2,
        architecture=17,
        work_item="same-work-item",
    )
    write_manifest(
        candidate_path,
        "PAN-CLASS1-91-original-candidate",
        fold=2,
        architecture=17,
        work_item="same-work-item",
    )

    selected = plot_loss_curves._load_manifest_curves(selected_path)
    candidates = plot_loss_curves._load_manifest_curves(candidate_path)

    assert selected[0]["model_name"] != candidates[0]["model_name"]
    assert (
        plot_loss_curves._selection_key(selected[0])
        == plot_loss_curves._selection_key(candidates[0])
        == ("work_item_name", "same-work-item")
    )


def test_selection_identity_falls_back_to_training_coordinates():
    left = {
        "model_name": "selected-copy",
        "work_item_name": None,
        "fold": 3,
        "arch_num": 4,
        "replicate": 1,
    }
    right = {**left, "model_name": "original-candidate"}

    assert (
        plot_loss_curves._selection_key(left)
        == plot_loss_curves._selection_key(right)
        == ("training_coordinates", 3, 4, 1)
    )


def test_train_plot_loss_curves_cli_marks_renamed_selection(
        monkeypatch, tmp_path):
    selected_dir = tmp_path / "models.combined"
    candidates_dir = tmp_path / "models.unselected.combined"
    out_dir = tmp_path / "plots"
    selected_dir.mkdir()
    candidates_dir.mkdir()
    write_manifest(
        selected_dir / "manifest.csv",
        "PAN-CLASS1-1-selected-copy",
        fold=2,
        architecture=17,
        work_item="same-work-item",
    )
    write_manifest(
        candidates_dir / "manifest.csv",
        "PAN-CLASS1-91-original-candidate",
        fold=2,
        architecture=17,
        work_item="same-work-item",
    )
    monkeypatch.setattr(plot_loss_curves, "_matplotlib_available", lambda: False)

    status = cli_main.main([
        "train",
        "plot-loss-curves",
        "--selected-dir", str(selected_dir),
        "--unselected-dir", str(candidates_dir),
        "--out", str(out_dir),
    ])

    assert status == 0
    summary = pandas.read_csv(out_dir / "summary.csv")
    assert summary.selected.tolist() == [True]


def test_processing_architecture_identity_and_checkpoint_summary(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    rows = []
    for width, boundary in ((11, 0), (13, 0), (13, 5)):
        config = {
            "hyperparameters": {"convolutional_kernel_size": width,
                                "convolutional_filters": 512,
                                "n_flank_length": 5, "c_flank_length": 5,
                                "cleavage_boundary_flank_length": boundary,
                                "cleavage_boundary_peptide_length": boundary,
                                "convolutional_kernel_l1_l2": [0.000001, 0.000002]},
            "fit_info": [{"loss": [0.5, 0.4, 0.3], "val_loss": [0.5, 0.4, 0.6],
                          "best_epoch": 2, "best_val_loss": 0.4,
                          "restored_best_weights": True,
                          "training_info": {"fold_num": 0}}],
        }
        rows.append({"model_name": "k%d-boundary%d" % (width, boundary), "config_json": json.dumps(config)})
    pandas.DataFrame(rows).to_csv(models / "manifest.csv", index=False)
    loaded = plot_loss_curves._load_manifest_curves(models / "manifest.csv")
    assert len({row["architecture_key"] for row in loaded}) == 3
    assert all(row["model_type"] == "processing" for row in loaded)
    assert "boundary 5x5" in loaded[-1]["architecture"]
    assert plot_loss_curves._epoch_label(loaded) == "Epoch"
    assert all(row["l1"] == 0.000001 and row["l2"] == 0.000002 for row in loaded)
    monkeypatch.setattr(plot_loss_curves, "_matplotlib_available", lambda: False)
    out = tmp_path / "plots"
    assert plot_loss_curves.run_argv(["--selected-dir", str(models), "--out", str(out)]) == 0
    summary = pandas.read_csv(out / "summary.csv")
    assert summary.final_val_loss.tolist() == [0.6] * 3
    assert summary.best_val_loss.tolist() == [0.4] * 3
    assert summary.checkpoint_policy.tolist() == ["best"] * 3


def test_historical_checkpoint_policy_not_inferred(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(path, "old", fold=0, architecture=0, work_item="old")
    model = plot_loss_curves._load_manifest_curves(path)[0]
    assert model["phase_curves"][0]["checkpoint_policy"] == "unknown"
    assert model["architecture"] == "(32,)"
    model["phase_curves"].append(model["phase_curves"][0])
    assert plot_loss_curves._epoch_label([model]) == "Epoch (fit calls concatenated)"


def test_processing_plot_labels_show_real_architectures(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot
    saved = []
    original_close = pyplot.close
    monkeypatch.setattr(pyplot, "close", lambda fig=None: saved.append(fig) if hasattr(fig, "axes") else None)
    base = {"model_name": "processing", "work_item_name": None, "fold": 0,
            "arch_num": 0, "replicate": 0, "layer_sizes": (),
            **plot_loss_curves._architecture_metadata({"convolutional_kernel_size": 13, "convolutional_filters": 512}),
            "phase_curves": [{"loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}]}
    keys = {plot_loss_curves._selection_key(base)}
    plot_loss_curves._plot_all_curves(keys, [base], tmp_path / "all.png")
    assert all(axis.get_xlabel() == "Epoch" for axis in saved[-1].axes)
    assert "CNN k13/f512" in saved[-1].axes[1].get_legend().get_texts()[0].get_text()
    plot_loss_curves._plot_by_arch([base], tmp_path / "arch.png")
    assert "CNN k13/f512" in saved[-1].axes[0].get_legend().get_texts()[0].get_text()
    for fig in saved:
        original_close(fig)


@pytest.mark.parametrize("value", [0.0001, [0.0001], [0.0, 1e-6, 1e-6], "0.0001", None, []])
def test_architecture_metadata_tolerates_malformed_regularization(value):
    metadata = plot_loss_curves._architecture_metadata(
        {"convolutional_kernel_size": 11, "convolutional_kernel_l1_l2": value})
    assert metadata["l1"] is None and metadata["l2"] is None


def test_architecture_metadata_reports_a_well_formed_regularization_pair():
    metadata = plot_loss_curves._architecture_metadata(
        {"convolutional_kernel_size": 11, "convolutional_kernel_l1_l2": [0.0, 1e-6]})
    assert metadata["l1"] == 0.0 and metadata["l2"] == 1e-6

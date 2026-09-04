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

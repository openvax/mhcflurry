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

"""Tests for model-bound allele capability and training-evidence reporting."""

import json
from types import SimpleNamespace

import pandas
import pytest

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork, Class1PresentationPredictor
from mhcflurry import allele_capabilities
from mhcflurry.allele_capabilities import capability_rows, model_fingerprint, training_evidence


@pytest.fixture
def predictor():
    affinity = Class1AffinityPredictor(
        class1_pan_allele_models=[Class1NeuralNetwork()],
        allele_to_sequence={"DLA-88*001:01": "A" * 39, "DLA-88*003:02": "C" * 39,
                            "HLA-A*02:01": "D" * 39})
    processing = SimpleNamespace(sequence_lengths={"peptide": 15}, percent_rank_transform=None)
    return Class1PresentationPredictor(
        affinity_predictor=affinity, processing_predictor_without_flanks=processing,
        weights_dataframe=pandas.DataFrame(index=["without_flanks"]),
        percent_rank_transform=object())


def test_capabilities_preserve_dla_identity_and_unknown_evidence(predictor):
    rows = capability_rows(predictor, ["DLA-88*01:01", "DLA-88*001:01", "DLA-88*03:02"], [9])
    assert [r["model_allele"] for r in rows] == ["DLA-88*001:01", "DLA-88*001:01", "DLA-88*003:02"]
    assert rows[0]["model_input_sequence_sha256"] == rows[1]["model_input_sequence_sha256"]
    assert rows[0]["model_input_sequence_sha256"] != rows[2]["model_input_sequence_sha256"]
    assert all(all(r["supported"].values()) for r in rows)
    for row in rows:
        assert all(v["status"] == "unknown" for v in row["empirical_validation"].values())
        assert row["percentile_calibration"]["presentation"]["available"]
        assert row["percentile_calibration"]["presentation"]["reference_population"] is None
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("allele,status", [("DLA-88*999:99", "unsupported"),
                                         ("DLA-88", "invalid"), ("invalid", "invalid"),
                                         ("DLA-DRB1*001:01", "invalid")])
def test_no_substitution_for_unknown_alleles(predictor, allele, status):
    row = capability_rows(predictor, [allele], [9])[0]
    assert row["model_allele"] is None
    assert row["allele_status"] == status
    assert row["supported"] == {"binding": False, "presentation": False, "processing": True}
    assert row["model_input_sequence_sha256"] is None


def test_unsupported_lengths_do_not_claim_presentation(predictor):
    rows = capability_rows(predictor, ["DLA-88*001:01"], [7, 8, 15, 16])
    assert [r["supported"]["binding"] for r in rows] == [False, True, True, False]
    assert [r["supported"]["presentation"] for r in rows] == [False, True, True, False]
    predictor.processing_predictor_without_flanks = None
    row = capability_rows(predictor, ["DLA-88*001:01"], [9])[0]
    assert row["supported"] == {"binding": True, "processing": False, "presentation": False}


def test_fingerprint_changes_with_any_weight_or_metadata(tmp_path):
    component = tmp_path / "affinity_predictor"
    component.mkdir()
    (component / "weights.npz").write_bytes(b"weights1")
    first = model_fingerprint(tmp_path)
    assert set(first["files"]) == {"affinity_predictor/weights.npz"}
    (component / "weights.npz").write_bytes(b"weights2")
    second = model_fingerprint(tmp_path)
    assert first["sha256"] != second["sha256"]
    (tmp_path / "info.txt").write_text("metadata")
    assert model_fingerprint(tmp_path)["sha256"] != second["sha256"]


@pytest.mark.parametrize("relative_link", ["affinity_predictor", "nested/affinity_predictor"])
def test_fingerprint_rejects_symlinked_component_directories(tmp_path, relative_link):
    models = tmp_path / "models"
    models.mkdir()
    (models / "info.txt").write_text("bundle metadata")
    component = tmp_path / "external_component"
    component.mkdir()
    (component / "weights.npz").write_bytes(b"weights")
    link = models / relative_link
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(component, target_is_directory=True)

    with pytest.raises(ValueError, match="Model bundle contains a symlink") as error:
        model_fingerprint(models)
    assert str(link) in str(error.value)


@pytest.mark.parametrize("target_exists", [True, False])
def test_fingerprint_rejects_file_and_dangling_symlinks(tmp_path, target_exists):
    models = tmp_path / "models"
    models.mkdir()
    (models / "info.txt").write_text("bundle metadata")
    target = tmp_path / "external_weights.npz"
    if target_exists:
        target.write_bytes(b"weights")
    (models / "weights.npz").symlink_to(target)

    with pytest.raises(ValueError, match="Model bundle contains a symlink"):
        model_fingerprint(models)


def test_training_counts_are_allele_length_specific_not_host_inference(tmp_path):
    pandas.DataFrame([
        dict(allele="DLA-88*01:01", peptide="SIINFEKL", hit=1),
        dict(allele="DLA-88*001:01", peptide="GILGFVFTL", hit=0),
        dict(allele="HLA-A*02:01", peptide="SIINFEKL", hit=1),
    ]).to_csv(tmp_path / "affinity_predictor_train_data.csv.bz2", index=False)
    result = training_evidence(tmp_path, ["DLA-88*001:01", "DLA-88*003:02"], [8, 9])
    affinity = result["affinity"]
    assert affinity["allele_rows"] == {"DLA-88*001:01": 2, "DLA-88*003:02": 0}
    assert [r["rows"] for r in affinity["allele_length_rows"]] == [1, 1, 0, 0]
    assert all(r["host_species"] is None and r["source_species"] is None for r in affinity["species_strata"])
    assert result["processing_without_flanks"]["status"] == "unavailable"
    assert not affinity["source_provenance_present"]


def test_missing_training_columns_are_unknown_and_explicit_species_survive(tmp_path):
    path = tmp_path / "affinity_predictor_train_data.csv.bz2"
    pandas.DataFrame([dict(peptide="SIINFEKL")]).to_csv(path, index=False)
    result = training_evidence(tmp_path, ["DLA-88*001:01"], [8])["affinity"]
    assert result["allele_rows"]["DLA-88*001:01"] is None
    assert result["allele_length_rows"][0]["rows"] is None
    pandas.DataFrame([dict(allele="DLA-88*001:01", peptide="SIINFEKL",
                           host_species="Homo sapiens", source_species="Homo sapiens")]).to_csv(path, index=False)
    result = training_evidence(tmp_path, ["DLA-88*001:01"], [8])["affinity"]
    assert result["species_strata"][0]["host_species"] == "Homo sapiens"


def test_eval_allele_capabilities_dispatch(monkeypatch):
    from mhcflurry.cli import eval_command

    monkeypatch.setattr(allele_capabilities, "run_argv", lambda argv, prog: (argv, prog))
    assert eval_command.run_argv(["allele-capabilities", "--help"]) == (
        ["--help"], "mhcflurry eval allele-capabilities")
    assert "allele-capabilities" in eval_command.format_help()

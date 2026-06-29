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

"""Tests for the predict command."""
import pytest

import tempfile
import os

import pandas

import torch

from mhcflurry import predict_command

from mhcflurry.testing_utils import cleanup, startup

torch.manual_seed(1)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    startup()
    yield
    cleanup()


TEST_CSV = '''
Allele,Peptide,Experiment
HLA-A0201,SYNFEKKL,17
HLA-B4403,AAAAAAAAA,17
HLA-B4403,PPPPPPPP,18
'''.strip()


def test_predict_dataframe_chunk_preserves_global_peptide_num():
    class FakePredictor:
        def predict_affinity(
                self,
                peptides,
                alleles,
                sample_names,
                throw,
                include_affinity_percentile,
                model_kwargs):
            return pandas.DataFrame({
                "peptide": list(peptides),
                "peptide_num": list(range(len(peptides))),
                "sample_name": list(sample_names),
                "affinity": [1.0] * len(peptides),
            })

    df = pandas.DataFrame({
        "allele": ["HLA-A*02:01", "HLA-A*02:01"],
        "peptide": ["SIINFEKL", "GILGFVFTL"],
    }, index=[4, 5])
    predictions = predict_command._predict_dataframe_chunk(
        FakePredictor(),
        df,
        {
            "affinity_only": True,
            "allele_column": "allele",
            "peptide_column": "peptide",
            "throw": True,
            "include_affinity_percentile": False,
            "affinity_model_kwargs": {},
        },
    )

    assert list(predictions.index) == [4, 5]
    assert list(predictions.peptide_num) == [4, 5]


def test_predict_dataframe_chunk_passes_flanks_positionally():
    class FakePredictor:
        def predict(
                self,
                peptides,
                n_flanks,
                c_flanks,
                alleles,
                sample_names,
                throw,
                include_affinity_percentile,
                affinity_model_kwargs,
                processing_batch_size):
            peptide_nums = pandas.Series(range(len(peptides)))
            return pandas.DataFrame({
                "peptide": list(peptides),
                "peptide_num": peptide_nums,
                "sample_name": list(sample_names),
                "n_flank": peptide_nums.map(pandas.Series(n_flanks)),
                "c_flank": peptide_nums.map(pandas.Series(c_flanks)),
                "processing_score": [0.1] * len(peptides),
            })

    df = pandas.DataFrame({
        "allele": ["HLA-A*02:01", "HLA-A*02:01"],
        "peptide": ["SIINFEKL", "GILGFVFTL"],
        "n_flank": ["NNN", "AAA"],
        "c_flank": ["CCC", "TTT"],
    }, index=[4, 5])
    predictions = predict_command._predict_dataframe_chunk(
        FakePredictor(),
        df,
        {
            "affinity_only": False,
            "allele_column": "allele",
            "peptide_column": "peptide",
            "n_flank_column": "n_flank",
            "c_flank_column": "c_flank",
            "use_flanking": True,
            "throw": True,
            "include_affinity_percentile": False,
            "affinity_model_kwargs": {},
            "processing_batch_size": "auto",
        },
    )

    assert list(predictions.index) == [4, 5]
    assert list(predictions.peptide_num) == [4, 5]
    assert list(predictions.n_flank) == ["NNN", "AAA"]
    assert list(predictions.c_flank) == ["CCC", "TTT"]


def test_detect_affinity_only_models_uses_file_presence(tmp_path):
    # Presentation bundle: has weights.csv
    pres = tmp_path / "pres"
    pres.mkdir()
    (pres / "weights.csv").write_text("")
    assert predict_command._detect_affinity_only_models(str(pres)) is False

    # Affinity-only bundle: no weights.csv
    aff = tmp_path / "aff"
    aff.mkdir()
    assert predict_command._detect_affinity_only_models(str(aff)) is True


def test_default_auto_workers_are_resolved_before_prediction_kwargs(
        tmp_path, monkeypatch):
    """Default --max-workers-per-gpu=auto must not reach int(...) directly."""
    captured = {}

    class FakePredictor:
        def predict_affinity(
                self,
                peptides,
                alleles,
                sample_names,
                throw,
                include_affinity_percentile,
                model_kwargs):
            del alleles, throw, include_affinity_percentile
            captured["model_kwargs"] = dict(model_kwargs)
            return pandas.DataFrame({
                "peptide": list(peptides),
                "peptide_num": list(range(len(peptides))),
                "sample_name": list(sample_names),
                "best_allele": ["HLA-A*02:01"] * len(peptides),
                "affinity": [1.0] * len(peptides),
            })

    models_dir = tmp_path / "affinity-models"
    models_dir.mkdir()
    out = tmp_path / "predictions.csv"

    monkeypatch.setattr(
        predict_command,
        "_load_predictor_for_command",
        lambda models_dir: (FakePredictor(), True),
    )

    predict_command.run([
        "--models", str(models_dir),
        "--alleles", "HLA-A0201",
        "--peptides", "SIINFEKL",
        "--backend", "cpu",
        "--out", str(out),
    ])

    assert captured["model_kwargs"]["num_workers_per_gpu"] == 1


def test_predict_columns_schema_parity_for_affinity_only():
    """Empty-input schema must enumerate every prediction column the
    populated path would emit, so downstream readers see a stable schema."""

    class FakePredictor:
        @property
        def supports_processing_prediction(self):
            return False

        @property
        def supports_presentation_prediction(self):
            return False

    cols = type(FakePredictor()).__bases__  # noqa: F841 (sanity probe)

    # Import the live class for the real method under test.
    from mhcflurry.class1_presentation_predictor import (
        Class1PresentationPredictor,
    )

    class NoPresentationPredictor(Class1PresentationPredictor):
        def __init__(self):
            pass

        @property
        def supports_processing_prediction(self):
            return False

        @property
        def supports_presentation_prediction(self):
            return False

    p = NoPresentationPredictor()
    affinity_cols = p.predict_columns(
        affinity_only=True,
        use_flanking=False,
        include_affinity_percentile=False)
    assert "affinity" in affinity_cols
    assert "best_allele" in affinity_cols
    assert "peptide_num" in affinity_cols
    assert "affinity_percentile" not in affinity_cols
    assert "processing_score" not in affinity_cols

    affinity_cols_with_pct = p.predict_columns(
        affinity_only=True, include_affinity_percentile=True)
    assert "affinity_percentile" in affinity_cols_with_pct


def test_predict_columns_schema_parity_for_presentation():
    from mhcflurry.class1_presentation_predictor import (
        Class1PresentationPredictor,
    )

    class FullPredictor(Class1PresentationPredictor):
        def __init__(self):
            pass

        @property
        def supports_processing_prediction(self):
            return True

        @property
        def supports_presentation_prediction(self):
            return True

    cols = FullPredictor().predict_columns(
        affinity_only=False,
        use_flanking=True,
        include_affinity_percentile=True)
    for col in (
            "peptide", "peptide_num", "sample_name", "affinity", "best_allele",
            "affinity_percentile", "n_flank", "c_flank", "processing_score",
            "presentation_score", "presentation_percentile"):
        assert col in cols, "Missing column: %s" % col

    no_flank_cols = FullPredictor().predict_columns(
        affinity_only=False, use_flanking=False)
    assert "n_flank" not in no_flank_cols
    assert "c_flank" not in no_flank_cols
    assert "processing_score" in no_flank_cols


def test_allele_string_to_alleles_accepts_semicolon_separated_csv_cells():
    df = pandas.DataFrame({
        "allele": [
            "HLA-A*02:01;HLA-B*44:03",
            "H-2-Kb, H-2-Db",
            "HLA-C*07:02 HLA-A*01:01",
            "HLA-A*02:01;HLA-B*44:03",
        ],
    })

    assert predict_command._allele_string_to_alleles(df, "allele") == {
        "HLA-A*02:01;HLA-B*44:03": ["HLA-A*02:01", "HLA-B*44:03"],
        "H-2-Kb, H-2-Db": ["H-2-Kb", "H-2-Db"],
        "HLA-C*07:02 HLA-A*01:01": ["HLA-C*07:02", "HLA-A*01:01"],
    }


@pytest.mark.slow
def test_csv():
    args = ["--allele-column", "Allele", "--peptide-column", "Peptide"]
    deletes = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as fd:
            fd.write(TEST_CSV.encode())
            deletes.append(fd.name)
        fd_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_out.name)
        full_args = [fd.name] + args + ["--out", fd_out.name]
        print("Running with args: %s" % full_args)
        predict_command.run(full_args)
        result = pandas.read_csv(fd_out.name)
        print(result)
        assert not result.isnull().any().any()
    finally:
        for delete in deletes:
            os.unlink(delete)

    assert result.shape == (3, 8)


@pytest.mark.slow
def test_no_csv():
    args = [
        "--alleles", "HLA-A0201", "H-2-Kb",
        "--peptides", "SIINFEKL", "DENDREKLLL", "PICKLEEE",
        "--prediction-column-prefix", "mhcflurry1_",
        "--affinity-only",
    ]

    deletes = []
    try:
        fd_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_out.name)
        full_args = args + ["--out", fd_out.name]
        print("Running with args: %s" % full_args)
        predict_command.run(full_args)
        result = pandas.read_csv(fd_out.name)
        print(result)
    finally:
        for delete in deletes:
            os.unlink(delete)

    print(result)
    assert len(result) == 6
    sub_result1 = result.loc[result.peptide == "SIINFEKL"].set_index("allele")
    print(sub_result1)
    assert (
        sub_result1.loc["H-2-Kb"].mhcflurry1_affinity <
        sub_result1.loc["HLA-A0201"].mhcflurry1_affinity)

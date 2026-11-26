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

"""Regression tests for CLI UX issues #424, #425, #426, #427, and #428."""

import logging
from argparse import Namespace

import numpy
import pandas
import pytest

from mhcflurry import pseudosequences
from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor
from mhcflurry.cli import (
    downloads_command,
    main,
    parsers,
    predict_command,
    predict_scan_command,
    reassign_mass_spec_training_data,
)
from mhcflurry.encodable_sequences import EncodingError
from mhcflurry.flanking_encoding import FlankingEncoding
from mhcflurry.parallelism import resolve_local_parallelism_args
from mhcflurry.workload_planning import WORKLOAD_PRESENTATION_INFERENCE


# ---------------------------------------------------------------------------
# Issue #424: float format, no duplicated flank columns, stable scan columns
# ---------------------------------------------------------------------------

def test_424_predict_does_not_duplicate_flanks_and_honors_float_format(
    tmp_path, monkeypatch
):
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
            processing_batch_size,
        ):
            return pandas.DataFrame({
                "peptide": list(peptides),
                "n_flank": list(n_flanks),
                "c_flank": list(c_flanks),
                "peptide_num": list(range(len(peptides))),
                "sample_name": list(sample_names),
                "affinity": [16.570974887280226],
                "best_allele": ["HLA-A*02:01"],
                "processing_score": [0.26470962166786194],
                "presentation_score": [0.9123456789012345],
            })

    models_dir = tmp_path / "pres-models"
    models_dir.mkdir()
    (models_dir / "weights.csv").write_text("")
    in_csv = tmp_path / "in.csv"
    in_csv.write_text(
        "allele,peptide,n_flank,c_flank\nHLA-A*02:01,SIINFEKL,MRVL,GLAS\n"
    )
    out_csv = tmp_path / "out.csv"

    monkeypatch.setattr(
        predict_command,
        "_load_predictor_for_command",
        lambda _dir: (FakePredictor(), False),
    )

    predict_command.run([
        str(in_csv),
        "--models", str(models_dir),
        "--backend", "cpu",
        "--num-jobs", "0",
        "--float-format", "%.3f",
        "--out", str(out_csv),
    ])

    text = out_csv.read_text()
    header = text.splitlines()[0].split(",")
    assert "n_flank" in header and "c_flank" in header
    assert "mhcflurry_n_flank" not in header
    assert "mhcflurry_c_flank" not in header
    assert "16.571" in text
    assert "0.265" in text
    assert "0.912" in text


def test_424_processing_only_predict_and_scan_column_order_and_no_sample1(
    tmp_path, monkeypatch
):
    class FakeProcPredictor:
        supports_processing_prediction = True
        supports_affinity_prediction = False
        supports_presentation_prediction = False
        sequence_lengths = {"n_flank": 5, "c_flank": 5}

        def predict(self, peptides, n_flanks=None, c_flanks=None, **_kwargs):
            return numpy.full(len(peptides), 0.42, dtype=float)

    proc = FakeProcPredictor()
    pres = Class1PresentationPredictor(
        processing_predictor_with_flanks=proc,
        processing_predictor_without_flanks=proc,
    )
    df = pres.predict(
        peptides=["SIINFEKL"],
        alleles=[],
        n_flanks=["AAAAA"],
        c_flanks=["CCCCC"],
        verbose=0,
    )
    # peptide must come BEFORE n_flank / c_flank (matching affinity mode).
    assert list(df.columns)[:3] == ["peptide", "n_flank", "c_flank"]

    # In predict-scan processing-only mode, sample_name ("sample1") is omitted
    # and --float-format is respected.
    models_dir = tmp_path / "pres-models"
    models_dir.mkdir()
    out_csv = tmp_path / "scan.csv"
    monkeypatch.setattr(
        predict_scan_command, "_load_predictor_for_command", lambda _dir: pres
    )
    predict_scan_command.run([
        "--sequences", "AAAAASIINFEKLCCCCC",
        "--peptide-lengths", "8",
        "--models", str(models_dir),
        "--backend", "cpu",
        "--num-jobs", "0",
        "--float-format", "%.2f",
        "--out", str(out_csv),
    ])
    header = out_csv.read_text().splitlines()[0].split(",")
    assert header[:5] == ["sequence_name", "pos", "peptide", "n_flank", "c_flank"]
    assert "sample_name" not in header
    assert "0.42" in out_csv.read_text()


# ---------------------------------------------------------------------------
# Issue #425: quiet prediction output, no DataFrame reprs on stderr
# ---------------------------------------------------------------------------

def test_425_inference_workload_plan_does_not_print_to_stderr(monkeypatch, capsys):
    monkeypatch.setenv("MHCFLURRY_SYSTEM_RAM_GB", "64")
    monkeypatch.setenv("MHCFLURRY_SYSTEM_AVAILABLE_RAM_GB", "32")
    args = Namespace(
        max_workers_per_gpu="auto",
        num_jobs="auto",
        gpus=None,
        backend="cpu",
    )
    resolve_local_parallelism_args(
        args, workload_name=WORKLOAD_PRESENTATION_INFERENCE
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Local workload plan:" not in captured.err


def test_425_predict_scan_fasta_does_not_dump_dataframe_repr(
    tmp_path, monkeypatch, capsys
):
    class FakeProcPredictor:
        supports_processing_prediction = True
        supports_affinity_prediction = False
        supports_presentation_prediction = False
        sequence_lengths = {"n_flank": 5, "c_flank": 5}

        def predict(self, peptides, n_flanks=None, c_flanks=None, **_kwargs):
            return numpy.full(len(peptides), 0.5, dtype=float)

    proc = FakeProcPredictor()
    pres = Class1PresentationPredictor(
        processing_predictor_with_flanks=proc,
        processing_predictor_without_flanks=proc,
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(">prot1\nAAAAASIINFEKLCCCCC\n")
    out_csv = tmp_path / "out.csv"

    monkeypatch.setattr(
        predict_scan_command, "_load_predictor_for_command", lambda _dir: pres
    )
    predict_scan_command.run([
        str(fasta),
        "--peptide-lengths", "8",
        "--models", str(models_dir),
        "--backend", "cpu",
        "--num-jobs", "0",
        "--out", str(out_csv),
    ])
    err = capsys.readouterr().err
    assert f"Read 1 sequences from {fasta}" in err
    assert "sequence_id" not in err  # no pandas DataFrame repr dump


def test_425_reassign_verbose_preview_shows_measurement_value(tmp_path, capsys):
    csv_path = tmp_path / "ms.csv"
    pandas.DataFrame({
        "allele": ["HLA-A*02:01"],
        "peptide": ["SIINFEKL"],
        "measurement_kind": ["mass_spec"],
        "measurement_inequality": ["<"],
        "measurement_value": [100.0],
        "extra_col_1": ["x"],
        "extra_col_2": ["y"],
        "original_allele": ["HLA-A*02:01"],
    }).to_csv(csv_path, index=False)

    reassign_mass_spec_training_data.reassign_mass_spec_training_data(
        str(csv_path), set_measurement_value=50.0, verbose=True
    )
    err = capsys.readouterr().err
    assert "measurement_value" in err
    assert "50.0" in err


# ---------------------------------------------------------------------------
# Issue #426: crisp one-line CLI errors instead of raw tracebacks
# ---------------------------------------------------------------------------

def test_426_predict_and_scan_report_clean_cli_errors(tmp_path, capsys):
    missing_models = tmp_path / "does_not_exist"
    for cmd in (predict_command, predict_scan_command):
        with pytest.raises(SystemExit) as exc:
            cmd.run([
                "--models", str(missing_models),
                "--alleles", "HLA-A*02:01",
                "--peptides" if cmd is predict_command else "--sequences",
                "SIINFEKLAA",
            ])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "--models: no such directory:" in err
        assert "implies --affinity-only" not in err

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "weights.csv").write_text("")

    # Bad output delimiter (>1 char)
    with pytest.raises(SystemExit) as exc:
        predict_command.run([
            "--models", str(models_dir),
            "--alleles", "HLA-A*02:01",
            "--peptides", "SIINFEKL",
            "--output-delimiter", ";;",
        ])
    assert exc.value.code == 2
    assert "single character" in capsys.readouterr().err

    # Nonexistent --out parent dir
    with pytest.raises(SystemExit) as exc:
        predict_command.run([
            "--models", str(models_dir),
            "--alleles", "HLA-A*02:01",
            "--peptides", "SIINFEKL",
            "--out", str(tmp_path / "nodir" / "out.csv"),
        ])
    assert exc.value.code == 2
    assert "parent directory does not exist" in capsys.readouterr().err

    # Missing input CSV
    with pytest.raises(SystemExit) as exc:
        predict_command.run([
            str(tmp_path / "missing.csv"),
            "--models", str(models_dir),
        ])
    assert exc.value.code == 2
    assert "missing.csv" in capsys.readouterr().err

    # Bad peptide-lengths on predict-scan
    with pytest.raises(SystemExit) as exc:
        predict_scan_command.run([
            "--models", str(models_dir),
            "--sequences", "SIINFEKLAA",
            "--peptide-lengths", "banana",
        ])
    assert exc.value.code == 2
    assert "Couldn't parse peptide lengths: 'banana'" in capsys.readouterr().err

    # Reversed range on predict-scan
    with pytest.raises(SystemExit) as exc:
        predict_scan_command.run([
            "--models", str(models_dir),
            "--sequences", "SIINFEKLAA",
            "--peptide-lengths", "11-8",
        ])
    assert exc.value.code == 2
    assert "Couldn't parse peptide lengths: '11-8'" in capsys.readouterr().err

    # Unsupported length in pseudosequences path
    with pytest.raises(SystemExit) as exc:
        pseudosequences.main(["path", "--directory", str(tmp_path), "--length", "38"])
    assert exc.value.code == 2
    assert "No canonical pseudosequence filename for 38aa" in capsys.readouterr().err


def test_426_flanking_encoding_error_message_wording():
    enc = FlankingEncoding(
        peptides=["A" * 16, "A" * 17],
        n_flanks=["", ""],
        c_flanks=["", ""],
    )
    with pytest.raises(EncodingError) as exc:
        enc.categorical_encode(
            peptide_max_length=15,
            n_flank_length=5,
            c_flank_length=5,
            throw=True,
        )
    msg = str(exc.value)
    assert "Supported lengths: 1 - 15" in msg
    assert "2 total peptide(s) have unsupported lengths" in msg


# ---------------------------------------------------------------------------
# Issue #427: train/select/calibrate & pseudosequences --help quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subcommand", list(parsers.SUBCOMMAND_PARSER_BUILDERS))
def test_427_train_select_calibrate_help_has_usage_and_no_sigusr1(subcommand, capsys):
    with pytest.raises(SystemExit) as exc:
        main.main([subcommand, "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "To show stack trace" not in captured.out
    assert "kill -s USR1" not in captured.out
    first_line = captured.out.splitlines()[0]
    assert first_line.startswith("usage: ")
    assert "[--" in first_line or "[-h]" in first_line
    assert "``" not in captured.out
    assert "genotypesto" not in captured.out
    assert "col.Default:" not in captured.out


def test_427_pseudosequences_help_is_concise_without_rst(capsys):
    with pytest.raises(SystemExit) as exc:
        pseudosequences.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "``" not in out
    assert "downloads-generation" not in out
    assert len(out.splitlines()) < 30


# ---------------------------------------------------------------------------
# Issue #428: downloads --verbose / --quiet / subcommand help
# ---------------------------------------------------------------------------

def test_428_downloads_help_and_verbose_quiet_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exc:
        downloads_command.run(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for sub in ("fetch", "info", "path", "url"):
        assert sub in out
    assert "Download and install released models" in out
    assert "Print the archive download URL(s)" in out

    # --quiet and --verbose are mutually exclusive
    with pytest.raises(SystemExit) as exc:
        downloads_command.run(["--quiet", "--verbose", "info"])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_428_downloads_verbose_updates_existing_root_logger_level(monkeypatch):
    root_logger = logging.getLogger()
    orig_level = root_logger.level
    monkeypatch.setattr(downloads_command, "info_subcommand", lambda _args: None)
    try:
        root_logger.setLevel(logging.WARNING)
        downloads_command.run(["--verbose", "info"])
        assert root_logger.level == logging.DEBUG
        downloads_command.run(["--quiet", "info"])
        assert root_logger.level == logging.WARNING
    finally:
        root_logger.setLevel(orig_level)

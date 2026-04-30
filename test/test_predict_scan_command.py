
import tempfile
import os

import pandas
import pytest
from numpy.testing import assert_array_less

from mhcflurry import predict_scan_command

from . import data_path

from mhcflurry.testing_utils import cleanup, startup

pytest.fixture(autouse=True, scope="module")
def setup_module():
    startup()
    yield
    cleanup()



def read_output_csv(filename):
    return pandas.read_csv(
        filename,
        converters={"n_flank": str, "c_flank": str})


def test_fasta():
    args = [
        data_path("example.fasta"),
        "--alleles",
        "HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:03,HLA-C*07:02",
    ]
    deletes = []
    try:
        fd_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_out.name)
        full_args = args + ["--out", fd_out.name]
        print("Running with args: %s" % full_args)
        predict_scan_command.run(full_args)
        result = read_output_csv(fd_out.name)
        print(result)
        assert not result.isnull().any().any()
    finally:
        for delete in deletes:
            os.unlink(delete)

    assert (
        result.best_allele.nunique() ==
        6), str(list(result.best_allele.unique()))
    assert result.sequence_name.nunique() == 3
    assert_array_less(result.affinity_percentile, 2.0)


def test_fasta_50nm():
    args = [
        data_path("example.fasta"),
        "--alleles",
        "HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:02,HLA-C*07:02",
        "--threshold-affinity", "50",
    ]
    deletes = []
    try:
        fd_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_out.name)
        full_args = args + ["--out", fd_out.name]
        print("Running with args: %s" % full_args)
        predict_scan_command.run(full_args)
        result = read_output_csv(fd_out.name)
        print(result)
        assert not result.isnull().any().any()
    finally:
        for delete in deletes:
            os.unlink(delete)

    assert len(result) > 0
    assert_array_less(result.affinity, 50.0001)


def test_fasta_percentile():
    args = [
        data_path("example.fasta"),
        "--alleles",
        "HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:02,HLA-C*07:02",
        "--threshold-affinity-percentile", "5.0",
    ]
    deletes = []
    try:
        fd_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_out.name)
        full_args = args + ["--out", fd_out.name]
        print("Running with args: %s" % full_args)
        predict_scan_command.run(full_args)
        result = read_output_csv(fd_out.name)
        print(result)
        assert not result.isnull().any().any()
    finally:
        for delete in deletes:
            os.unlink(delete)

    assert len(result) > 0
    assert_array_less(result.affinity_percentile, 5.0001)


def test_commandline_sequences():
    args = [
        "--sequences", "ASDFGHKL", "QWERTYIPCVNM",
        "--alleles", "HLA-A0201,HLA-A0301", "H-2-Kb",
        "--peptide-lengths", "8",
        "--results-all",
    ]

    deletes = []
    try:
        fd_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_out.name)
        full_args = args + ["--out", fd_out.name]
        print("Running with args: %s" % full_args)
        predict_scan_command.run(full_args)
        result = read_output_csv(fd_out.name)
        print(result)
    finally:
        for delete in deletes:
            os.unlink(delete)

    print(result)

    assert result.sequence_name.nunique() == 2
    assert result.best_allele.nunique() == 3
    assert result.sample_name.nunique() == 2
    assert (result.peptide == "ASDFGHKL").sum() == 2
    assert (result.peptide != "ASDFGHKL").sum() == 10


def test_no_affinity_percentile_does_not_crash_with_default_threshold():
    """Regression: default-threshold path filters by affinity_percentile,
    but --no-affinity-percentile leaves that column out, so the filter
    used to KeyError. We now skip filters whose column is missing."""
    args = [
        "--sequences", "ASDFGHKL", "QWERTYIPCVNM",
        "--alleles", "HLA-A0201,HLA-A0301",
        "--peptide-lengths", "8",
        "--no-affinity-percentile",
    ]
    deletes = []
    try:
        fd_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_out.name)
        full_args = args + ["--out", fd_out.name]
        predict_scan_command.run(full_args)
        result = read_output_csv(fd_out.name)
    finally:
        for delete in deletes:
            os.unlink(delete)
    assert "affinity_percentile" not in result.columns
    assert len(result) > 0


def test_empty_input_csv_does_not_crash():
    """Regression: empty input used to produce a schema-less DataFrame,
    then any threshold filter on a missing column raised AttributeError.
    We now seed the empty result with the expected schema."""
    deletes = []
    try:
        fd_in = tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv", mode="w")
        fd_in.write("sequence\n")
        fd_in.close()
        deletes.append(fd_in.name)
        fd_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_out.name)
        args = [
            fd_in.name,
            "--alleles", "HLA-A0201",
            "--peptide-lengths", "8",
            "--out", fd_out.name,
        ]
        predict_scan_command.run(args)
        result = read_output_csv(fd_out.name)
    finally:
        for delete in deletes:
            os.unlink(delete)
    assert len(result) == 0


def test_parallel_output_globally_sorted():
    """Regression: parallel scan used to concat per-chunk sorted outputs
    without re-sorting globally, so output ranking diverged from serial.
    We now re-sort by presentation_score after concat to match serial."""
    seqs = [
        "ASDFGHKLPLPLPLPL",
        "QWERTYIPCVNMLLLM",
        "MKVAVLAVALLVCLLI",
        "GVRDDQYRSPVDPAPL",
    ]
    alleles = "HLA-A0201,HLA-A0301"
    common = [
        "--sequences"] + seqs + [
        "--alleles", alleles,
        "--peptide-lengths", "9",
        "--results-all",
    ]
    deletes = []
    try:
        fd_serial = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_serial.name)
        predict_scan_command.run(common + ["--out", fd_serial.name])
        serial_df = read_output_csv(fd_serial.name)

        fd_parallel = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        deletes.append(fd_parallel.name)
        predict_scan_command.run(common + [
            "--num-jobs", "2", "--backend", "cpu",
            "--out", fd_parallel.name])
        parallel_df = read_output_csv(fd_parallel.name)
    finally:
        for delete in deletes:
            os.unlink(delete)

    assert len(serial_df) == len(parallel_df)
    # Both should be sorted descending by presentation_score globally.
    score_col = "presentation_score"
    if score_col in serial_df.columns:
        serial_scores = serial_df[score_col].values
        parallel_scores = parallel_df[score_col].values
        assert (serial_scores[:-1] >= serial_scores[1:]).all(), (
            "serial output not globally sorted")
        assert (parallel_scores[:-1] >= parallel_scores[1:]).all(), (
            "parallel output not globally sorted")

import logging
logging.getLogger('matplotlib').disabled = True
logging.getLogger('tensorflow').disabled = True

import tempfile
import os

import pandas
from numpy.testing import assert_equal, assert_array_less, assert_array_equal

from mhcflurry import predict_scan_command


from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup

from . import data_path


def read_output_csv(filename):
    return pandas.read_csv(
        filename,
        converters={"n_flank": str, "c_flank": str})


def test_fasta():
    args = [
        data_path("example.fasta"),
        "--alleles",
        "HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:02,HLA-C*07:02",
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

    assert_equal(result.best_allele.nunique(), 6)
    assert_equal(result.sequence_name.nunique(), 3)
    assert_array_less(result.affinity_percentile, 2.0)


def test_fasta_50nm():
    args = [
        data_path("example.fasta"),
        "--alleles",
        "HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:02,HLA-C*07:02",
        "--results-filtered", "affinity",
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
    assert_array_less(result.affinity, 50)


def test_fasta_best():
    args = [
        data_path("example.fasta"),
        "--alleles",
        "HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:02,HLA-C*07:02",
        "--results-best", "affinity_percentile",
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
    assert_array_equal(
        result.groupby(["sequence_name"]).peptide.count().values, 1)


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

    assert_equal(result.sequence_name.nunique(), 2)
    assert_equal(result.best_allele.nunique(), 3)
    assert_equal(result.sample_name.nunique(), 2)
    assert_equal((result.peptide == "ASDFGHKL").sum(), 2)
    assert_equal((result.peptide != "ASDFGHKL").sum(), 10)
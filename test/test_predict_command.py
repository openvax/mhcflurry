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

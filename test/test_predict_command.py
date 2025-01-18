from . import initialize
initialize()

import tempfile
import os

import pandas
import pytest
from numpy.testing import assert_equal

import tensorflow as tf

tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(1)

from mhcflurry import predict_command

from mhcflurry.testing_utils import cleanup, startup

pytest.fixture(autouse=True, scope="module")
def setup_module():
    startup()
    yield
    cleanup()

TEST_CSV = '''
Allele,Peptide,Experiment
HLA-A0201,SYNFEKKL,17
HLA-B4403,AAAAAAAAA,17
HLA-B4403,PPPPPPPP,18
'''.strip()


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

from . import initialize

initialize()

import tempfile
import os
import errno

import pandas
import pytest
from numpy.testing import assert_array_almost_equal
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


TEST_CSV = """
Allele,Peptide,Experiment
HLA-A0201,SYNFEKKL,17
HLA-B4403,AAAAAAAAA,17
HLA-B4403,PPPPPPPP,18
""".strip()


def test_csv():
    """Test CSV input/output functionality."""
    args = ["--allele-column", "Allele", "--peptide-column", "Peptide"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as input_file:
        input_file.write(TEST_CSV)
        input_file.flush()
        input_path = input_file.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as output_file:
            output_path = output_file.name

            try:
                full_args = [input_path] + args + ["--out", output_path]
                print("Running with args: %s" % full_args)
                predict_command.run(full_args)

                result = pandas.read_csv(output_path)
                print(result)

                # Verify results
                assert not result.isnull().any().any()
                assert result.shape == (3, 8)

            finally:
                # Clean up files
                for path in [input_path, output_path]:
                    try:
                        os.unlink(path)
                    except OSError as e:
                        if e.errno != errno.ENOENT:  # No such file or directory
                            print(f"Error removing file {path}: {e}")


def test_no_csv():
    args = [
        "--alleles",
        "HLA-A0201",
        "H-2-Kb",
        "--peptides",
        "SIINFEKL",
        "DENDREKLLL",
        "PICKLEEE",
        "--prediction-column-prefix",
        "mhcflurry1_",
        "--affinity-only",
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as output_file:
        output_path = output_file.name
        try:
            full_args = args + ["--out", output_path]
            print("Running with args: %s" % full_args)
            predict_command.run(full_args)

            result = pandas.read_csv(output_path)
            print(result)

            # Verify results
            assert len(result) == 6
            sub_result1 = result.loc[result.peptide == "SIINFEKL"].set_index("allele")
            print(sub_result1)
            assert sub_result1.loc["H-2-Kb"].mhcflurry1_affinity < sub_result1.loc["HLA-A0201"].mhcflurry1_affinity

        finally:
            try:
                os.unlink(output_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    print(f"Error removing file {output_path}: {e}")

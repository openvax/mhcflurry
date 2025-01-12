from . import initialize
initialize()

import tempfile
import os

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


def test_tensorflow_vs_pytorch_backends():
    """Test that tensorflow and pytorch backends produce matching results."""
    args = [
        "--alleles", "HLA-A0201",
        "--peptides", "SIINFEKL", "DENDREKLLL",
        "--prediction-column-prefix", "mhcflurry_",
        "--affinity-only",
    ]

    deletes = []
    result_tf = None
    result_torch = None
    
    try:
        # Run with tensorflow backend
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as fd_out_tf:
            deletes.append(fd_out_tf.name)
            tf_args = args + ["--out", fd_out_tf.name, "--backend", "tensorflow"]
            print("Running tensorflow with args: %s" % tf_args)
            predict_command.run(tf_args)
            fd_out_tf.close()  # Explicitly close file
            result_tf = pandas.read_csv(fd_out_tf.name)
            print("TensorFlow results:")
            print(result_tf)

        # Run with pytorch backend
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as fd_out_torch:
            deletes.append(fd_out_torch.name)
            torch_args = args + ["--out", fd_out_torch.name, "--backend", "pytorch"]
            print("Running pytorch with args: %s" % torch_args)
            predict_command.run(torch_args)
            fd_out_torch.close()  # Explicitly close file
            result_torch = pandas.read_csv(fd_out_torch.name)
            print("PyTorch results:")
            print(result_torch)

    finally:
        # Make sure we've closed the files and pandas has released them
        import gc
        gc.collect()  # Force garbage collection
        
        # Add a small delay to ensure files are released
        import time
        time.sleep(0.1)
        
        for delete in deletes:
            try:
                os.unlink(delete)
            except Exception as e:
                print(f"Warning: Could not delete {delete}: {e}")

    # Check that results match
    assert result_tf.shape == result_torch.shape, "Output shapes differ"
    assert all(result_tf.columns == result_torch.columns), "Output columns differ"
    
    # Compare numeric columns with tolerance
    numeric_columns = [
        col for col in result_tf.columns 
        if col.startswith("mhcflurry_") and result_tf[col].dtype in ['float64', 'float32']
    ]
    
    for col in numeric_columns:
        print(f"Comparing {col}:")
        print(f"TensorFlow: {result_tf[col].values}")
        print(f"PyTorch: {result_torch[col].values}")
        assert_array_almost_equal(
            result_tf[col].values,
            result_torch[col].values,
            decimal=4,
            err_msg=f"Values differ in column {col}"
        )

    # Compare non-numeric columns exactly
    other_columns = [col for col in result_tf.columns if col not in numeric_columns]
    for col in other_columns:
        assert all(result_tf[col] == result_torch[col]), f"Values differ in column {col}"


def test_no_csv():
    args = [
        "--alleles", "HLA-A0201", "H-2-Kb",
        "--peptides", "SIINFEKL", "DENDREKLLL", "PICKLEEE",
        "--prediction-column-prefix", "mhcflurry1_",
        "--affinity-only",
    ]

    deletes = []
    result = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as fd_out:
            deletes.append(fd_out.name)
            full_args = args + ["--out", fd_out.name]
            print("Running with args: %s" % full_args)
            predict_command.run(full_args)
            fd_out.close()  # Explicitly close
            result = pandas.read_csv(fd_out.name)
            print(result)
    finally:
        # Make sure we've closed the files and pandas has released them
        import gc
        gc.collect()  # Force garbage collection
        
        # Add a small delay to ensure files are released
        import time
        time.sleep(0.1)
        
        for delete in deletes:
            try:
                os.unlink(delete)
            except Exception as e:
                print(f"Warning: Could not delete {delete}: {e}")

    print(result)
    assert len(result) == 6
    sub_result1 = result.loc[result.peptide == "SIINFEKL"].set_index("allele")
    print(sub_result1)
    assert (
        sub_result1.loc["H-2-Kb"].mhcflurry1_affinity <
        sub_result1.loc["HLA-A0201"].mhcflurry1_affinity)

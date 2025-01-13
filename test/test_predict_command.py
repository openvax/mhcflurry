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

TEST_CSV = '''
Allele,Peptide,Experiment
HLA-A0201,SYNFEKKL,17
HLA-B4403,AAAAAAAAA,17
HLA-B4403,PPPPPPPP,18
'''.strip()


def test_csv():
    """Test CSV input/output functionality."""
    args = ["--allele-column", "Allele", "--peptide-column", "Peptide"]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as input_file:
        input_file.write(TEST_CSV)
        input_file.flush()
        input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as output_file:
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


def test_tensorflow_vs_pytorch_backends():
    """Test that tensorflow and pytorch backends produce matching results."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed - skipping backend comparison test")

    args = [
        "--alleles", "HLA-A0201",
        "--alleles", "HLA-A0201", "HLA-A0301",
        "--peptides", "SIINFEKL", "SIINFEKD", "SIINFEKQ",
        "--prediction-column-prefix", "mhcflurry_",
        "--affinity-only",
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tf_output:
        tf_path = tf_output.name
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as torch_output:
            torch_path = torch_output.name
            
            try:
                # Run with tensorflow backend
                tf_args = args + ["--out", tf_path, "--backend", "tensorflow"]
                print("Running tensorflow with args: %s" % tf_args)
                predict_command.run(tf_args)
                result_tf = pandas.read_csv(tf_path)
                print("TensorFlow results:")
                print(result_tf)

                # Run with pytorch backend  
                torch_args = args + ["--out", torch_path, "--backend", "pytorch"]
                print("Running pytorch with args: %s" % torch_args)
                predict_command.run(torch_args)
                result_torch = pandas.read_csv(torch_path)
                print("PyTorch results:")
                print(result_torch)

            finally:
                # Clean up files
                for path in [tf_path, torch_path]:
                    try:
                        os.unlink(path)
                    except OSError as e:
                        if e.errno != errno.ENOENT:  # No such file or directory
                            print(f"Error removing file {path}: {e}")

    # Verify both backends produced results
    assert result_tf is not None, "TensorFlow backend failed to produce results"
    assert result_torch is not None, "PyTorch backend failed to produce results"
    
    # Verify both results contain predictions
    prediction_columns = [col for col in result_tf.columns if col.startswith("mhcflurry_")]
    assert len(prediction_columns) > 0, "No prediction columns found in TensorFlow results"
    
    # Check that no prediction columns contain all nulls
    for col in prediction_columns:
        assert not result_tf[col].isnull().all(), f"TensorFlow predictions are all null for column {col}"
        assert not result_torch[col].isnull().all(), f"PyTorch predictions are all null for column {col}"
        
        # Verify predictions are numeric and within expected ranges
        assert result_tf[col].dtype in ['float64', 'float32'], f"TensorFlow column {col} is not numeric"
        assert result_torch[col].dtype in ['float64', 'float32'], f"PyTorch column {col} is not numeric"
        
        if "affinity" in col.lower():
            # Affinity predictions should be positive numbers
            assert (result_tf[col] > 0).all(), f"Invalid affinity values in TensorFlow column {col}"
            assert (result_torch[col] > 0).all(), f"Invalid affinity values in PyTorch column {col}"
        elif "percentile" in col.lower():
            # Percentile predictions should be between 0 and 100
            assert ((result_tf[col] >= 0) & (result_tf[col] <= 100)).all(), \
                f"Invalid percentile values in TensorFlow column {col}"
            assert ((result_torch[col] >= 0) & (result_torch[col] <= 100)).all(), \
                f"Invalid percentile values in PyTorch column {col}"

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

    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as output_file:
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
            assert (
                sub_result1.loc["H-2-Kb"].mhcflurry1_affinity <
                sub_result1.loc["HLA-A0201"].mhcflurry1_affinity)

        finally:
            try:
                os.unlink(output_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    print(f"Error removing file {output_path}: {e}")

"""
Tests for training and predicting using Class1 pan-allele models.
"""

import json
import os
import shutil
import tempfile
import subprocess
import sys

import numpy
import pandas
import pytest

from mhcflurry import Class1AffinityPredictor
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.downloads import get_path
from mhcflurry.train_pan_allele_models_command import (
    _pop_train_param,
    pretrain_data_iterator,
    pretrain_network_input_iterator,
)
from .pytest_helpers import mhcflurry_cli

from mhcflurry.testing_utils import cleanup, startup

pytest.fixture(autouse=True, scope="module")
def setup_module():
    startup()
    yield
    cleanup()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MHCFLURRY_CLUSTER_WORKER_COMMAND"] = (
    f"{sys.executable} -m mhcflurry.cluster_worker_entry_point"
)


HYPERPARAMETERS_LIST = [
    {
        'activation': 'tanh',
        'allele_dense_layer_sizes': [],
        'batch_normalization': False,
        'dense_layer_l1_regularization': 0.0,
        'dense_layer_l2_regularization': 0.0,
        'dropout_probability': 0.0,
        'early_stopping': False,
        'init': 'glorot_uniform',
        'layer_sizes': [4],
        'learning_rate': None,
        'locally_connected_layers': [],
        'loss': 'custom:mse_with_inequalities',
        'max_epochs': 1,
        'minibatch_size': 100000,
        'optimizer': 'rmsprop',
        'output_activation': 'sigmoid',
        'patience': 1,
        'peptide_allele_merge_activation': '',
        'peptide_allele_merge_method': 'concatenate',
        'peptide_amino_acid_encoding': 'BLOSUM62',
        'peptide_dense_layer_sizes': [],
        'peptide_encoding': {
            'alignment_method': 'left_pad_centered_right_pad',
            'max_length': 15,
            'vector_encoding_name': 'BLOSUM62',
        },
        'random_negative_affinity_max': 50000.0,
        'random_negative_affinity_min': 20000.0,
        'random_negative_constant': 0,
        'random_negative_distribution_smoothing': 0.0,
        'random_negative_match_distribution': True,
        'random_negative_rate': 0.0,
        'train_data': {
            "pretrain": True,
            'pretrain_peptides_per_step': 4,
            'pretrain_steps_per_epoch': 1,
            'pretrain_max_epochs': 1,
        },
        'validation_split': 0.0,
    },
]

PRETRAIN_DATA = """
,HLA-A*01:01,HLA-A*02:01
SIINFEKL,100.0,200.0
LLFGYPVYV,150.0,250.0
KLGGALQAK,300.0,350.0
""".strip()


def test_train_data_metadata_drops_preexisting_fold_columns():
    """Regression: when ``--data`` already has ``fold_*`` columns
    (public 2.2.0 train_data ships with fold_0..3), the train command's
    metadata-DataFrame join used to produce ``fold_0_x``/``fold_0_y``
    after merging in the freshly-computed ``folds_df``. That left the
    select command parsing ``int("x")`` and crashing. The fix drops
    the stale fold columns before the merge, so the saved metadata
    has clean ``fold_<int>`` columns regardless of input shape."""
    import inspect
    from mhcflurry import train_pan_allele_models_command as mod

    src = inspect.getsource(mod.initialize_training)
    # The drop-stale-folds helper must run before the merge.
    assert "df_no_folds" in src, (
        "initialize_training must drop pre-existing fold_* cols before merge")
    drop_idx = src.index("df_no_folds")
    merge_idx = src.index("pandas.merge(\n                df_no_folds")
    assert drop_idx < merge_idx, (
        "df_no_folds must be computed before being passed to merge")


def test_pop_train_param_supports_pretrain_peptides_per_epoch_alias():
    train_params = {"pretrain_peptides_per_epoch": 64}

    actual = _pop_train_param(
        train_params,
        names=("pretrain_peptides_per_step", "pretrain_peptides_per_epoch"),
        default=1024,
        verbose=0,
    )

    assert actual == 64
    assert train_params == {}


def test_pretrain_data_iterator_keeps_short_final_chunk(tmp_path):
    df = pandas.DataFrame(
        {
            "HLA-A*01:01": [100.0, 200.0, 300.0],
            "HLA-A*02:01": [150.0, 250.0, 350.0],
        },
        index=["SIINFEKL", "LLFGYPVYV", "KLGGALQAK"],
    )
    filename = tmp_path / "pretrain.csv"
    df.to_csv(filename)

    master_allele_encoding = AlleleEncoding(
        ["HLA-A*01:01", "HLA-A*02:01"],
        allele_to_sequence={
            "HLA-A*01:01": "A" * 34,
            "HLA-A*02:01": "C" * 34,
        },
    )

    iterator = pretrain_data_iterator(
        str(filename),
        master_allele_encoding,
        peptides_per_chunk=2,
    )
    first = next(iterator)
    second = next(iterator)

    assert len(first[1]) == 4
    assert len(first[2]) == 4
    assert len(second[1]) == 2
    assert len(second[2]) == 2


def test_pretrain_network_input_iterator_compact_torch_indices(tmp_path):
    df = pandas.DataFrame(
        {
            "HLA-A*01:01": [100.0, 200.0, 300.0],
            "HLA-A*02:01": [150.0, 250.0, 350.0],
        },
        index=["SIINFEKL", "LLFGYPVYV", "KLGGALQAK"],
    )
    filename = tmp_path / "pretrain.csv"
    df.to_csv(filename)

    master_allele_encoding = AlleleEncoding(
        ["HLA-A*01:01", "HLA-A*02:01"],
        allele_to_sequence={
            "HLA-A*01:01": "A" * 34,
            "HLA-A*02:01": "C" * 34,
        },
    )

    iterator = pretrain_network_input_iterator(
        str(filename),
        master_allele_encoding,
        peptide_encoding={
            "alignment_method": "left_pad_centered_right_pad",
            "max_length": 15,
            "vector_encoding_name": "BLOSUM62",
        },
        peptides_per_chunk=2,
        compact_peptide_repeats=True,
        peptide_amino_acid_encoding_torch=True,
    )
    x_dict, y = next(iterator)

    assert x_dict["peptide"].shape == (2, 45)
    assert x_dict["peptide"].dtype == "int8"
    assert x_dict["allele"].shape == (4,)
    assert x_dict["peptide_repeat_count"] == 2
    assert y.shape == (4,)

def run_and_check(n_jobs=0, delete=True, additional_args=[]):
    models_dir = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    hyperparameters_filename = os.path.join(
        models_dir, "hyperparameters.yaml")
    with open(hyperparameters_filename, "w") as fd:
        json.dump(HYPERPARAMETERS_LIST, fd)

    pretrain_data_filename = os.path.join(
        models_dir, "pretrain_data.csv")
    with open(pretrain_data_filename, "w") as fd:
        fd.write(PRETRAIN_DATA)
        fd.write("\n")

    data_df = pandas.read_csv(
        get_path("data_curated", "curated_training_data.affinity.csv.bz2"))
    selected_data_df = data_df.sample(n=50, random_state=0)
    selected_data_df.to_csv(
        os.path.join(models_dir, "_train_data.csv"), index=False)

    args = mhcflurry_cli("mhcflurry-class1-train-pan-allele-models") + [
        "--data", os.path.join(models_dir, "_train_data.csv"),
        "--allele-sequences", get_path("allele_sequences", "allele_sequences.csv"),
        "--pretrain-data", pretrain_data_filename,
        "--hyperparameters", hyperparameters_filename,
        "--out-models-dir", models_dir,
        "--num-jobs", str(n_jobs),
        "--num-folds", "2",
        "--verbosity", "1",
    ] + additional_args
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    # Run model selection
    models_dir_selected = tempfile.mkdtemp(
        prefix="mhcflurry-test-models-selected")
    args = mhcflurry_cli("mhcflurry-class1-select-pan-allele-models") + [
        "--data", os.path.join(models_dir, "train_data.csv.bz2"),
        "--models-dir", models_dir,
        "--out-models-dir", models_dir_selected,
        "--max-models", "1",
        "--num-jobs", str(n_jobs),
    ] + additional_args
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    result = Class1AffinityPredictor.load(
        models_dir_selected, optimization_level=0)
    assert len(result.neural_networks) == 2
    predictions = result.predict(peptides=["SLYNTVATL"],
        alleles=["HLA-A*02:01"])
    assert predictions.shape == (1,)
    assert numpy.isfinite(predictions).all()

    if delete:
        print("Deleting: %s" % models_dir)
        shutil.rmtree(models_dir)
        shutil.rmtree(models_dir_selected)


@pytest.mark.slow
@pytest.mark.integration
def test_run_parallel():
    run_and_check(n_jobs=2)


@pytest.mark.slow
@pytest.mark.integration
def test_run_serial():
    run_and_check(n_jobs=0)


@pytest.mark.slow
@pytest.mark.integration
def test_run_cluster_parallelism():
    run_and_check(n_jobs=0, additional_args=[
        '--cluster-parallelism',
        '--cluster-results-workdir', '/tmp/'
    ])


if __name__ == "__main__":
    # run_and_check(n_jobs=0, delete=False)
    test_run_cluster_parallelism()

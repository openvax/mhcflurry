"""
Tests for training and predicting using Class1 pan-allele models.
"""

import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True


import json
import os
import shutil
import tempfile
import subprocess

import pandas

from numpy.testing import assert_equal, assert_array_less

from mhcflurry import Class1AffinityPredictor,Class1NeuralNetwork
from mhcflurry.downloads import get_path

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup

os.environ["CUDA_VISIBLE_DEVICES"] = ""


HYPERPARAMETERS_LIST = [
{
    'activation': 'tanh',
    'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 0.0,
    'dense_layer_l2_regularization': 0.0,
    'dropout_probability': 0.5,
    'early_stopping': True,
    'init': 'glorot_uniform',
    'layer_sizes': [64],
    'learning_rate': None,
    'locally_connected_layers': [],
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': 0,  # never selected
    'minibatch_size': 256,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': 10,
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
    'random_negative_constant': 25,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 0.2,
    'train_data': {"pretrain": False},
    'validation_split': 0.1,
},
{
    'activation': 'tanh',
    'allele_dense_layer_sizes': [],
    'batch_normalization': False,
    'dense_layer_l1_regularization': 0.0,
    'dense_layer_l2_regularization': 0.0,
    'dropout_probability': 0.5,
    'early_stopping': True,
    'init': 'glorot_uniform',
    'layer_sizes': [32],
    'learning_rate': None,
    'locally_connected_layers': [],
    'loss': 'custom:mse_with_inequalities',
    'max_epochs': 5,
    'minibatch_size': 256,
    'optimizer': 'rmsprop',
    'output_activation': 'sigmoid',
    'patience': 10,
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
    'random_negative_constant': 25,
    'random_negative_distribution_smoothing': 0.0,
    'random_negative_match_distribution': True,
    'random_negative_rate': 0.2,
    'train_data': {
        "pretrain": False,
        'pretrain_peptides_per_epoch': 128,
        'pretrain_max_epochs': 2,
        'pretrain_max_val_loss': 0.2,
    },
    'validation_split': 0.1,
},
]


def run_and_check(n_jobs=0, delete=True, additional_args=[]):
    models_dir = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    hyperparameters_filename = os.path.join(
        models_dir, "hyperparameters.yaml")
    with open(hyperparameters_filename, "w") as fd:
        json.dump(HYPERPARAMETERS_LIST, fd)

    data_df = pandas.read_csv(
        get_path("data_curated", "curated_training_data.affinity.csv.bz2"))
    selected_data_df = data_df.loc[data_df.allele.str.startswith("HLA-A")]
    selected_data_df.to_csv(
        os.path.join(models_dir, "_train_data.csv"), index=False)

    args = [
        "mhcflurry-class1-train-pan-allele-models",
        "--data", os.path.join(models_dir, "_train_data.csv"),
        "--allele-sequences", get_path("allele_sequences", "allele_sequences.csv"),
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
    args = [
        "mhcflurry-class1-select-pan-allele-models",
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
    assert_equal(len(result.neural_networks), 2)
    predictions = result.predict(peptides=["SLYNTVATL"],
        alleles=["HLA-A*02:01"])
    assert_equal(predictions.shape, (1,))
    assert_array_less(predictions, 1000)

    if delete:
        print("Deleting: %s" % models_dir)
        shutil.rmtree(models_dir)
        shutil.rmtree(models_dir_selected)


if os.environ.get("KERAS_BACKEND") != "theano":
    def test_run_parallel():
        run_and_check(n_jobs=1)
        run_and_check(n_jobs=2)


def test_run_serial():
    run_and_check(n_jobs=0)


def test_run_cluster_parallelism():
    run_and_check(n_jobs=0, additional_args=[
        '--cluster-parallelism',
        '--cluster-results-workdir', '/tmp/'
    ])


if __name__ == "__main__":
    # run_and_check(n_jobs=0, delete=False)
    test_run_cluster_parallelism()

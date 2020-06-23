"""
Test train, calibrate percentile ranks, and model selection commands.
"""
import logging
logging.getLogger('matplotlib').disabled = True
logging.getLogger('tensorflow').disabled = True

import json
import os
import shutil
import tempfile
import subprocess
from copy import deepcopy

from numpy.testing import assert_array_less, assert_equal

from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HYPERPARAMETERS = [
    {
        "n_models": 2,
        "max_epochs": 500,
        "patience": 5,
        "minibatch_size": 128,
        "early_stopping": True,
        "validation_split": 0.2,

        "random_negative_rate": 0.0,
        "random_negative_constant": 25,

        "peptide_amino_acid_encoding": "BLOSUM62",
        "use_embedding": False,
        "kmer_size": 15,
        "batch_normalization": False,
        "locally_connected_layers": [
            {
                "filters": 8,
                "activation": "tanh",
                "kernel_size": 3
            }
        ],
        "activation": "tanh",
        "output_activation": "sigmoid",
        "layer_sizes": [
            16
        ],
        "random_negative_affinity_min": 20000.0,
        "random_negative_affinity_max": 50000.0,
        "dense_layer_l1_regularization": 0.001,
        "dropout_probability": 0.0
    }
]


def run_and_check(n_jobs=0):
    models_dir = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    hyperparameters_filename = os.path.join(
        models_dir, "hyperparameters.yaml")
    with open(hyperparameters_filename, "w") as fd:
        json.dump(HYPERPARAMETERS, fd)

    args = [
        "mhcflurry-class1-train-allele-specific-models",
        "--data", get_path("data_curated", "curated_training_data.affinity.csv.bz2"),
        "--hyperparameters", hyperparameters_filename,
        "--allele", "HLA-A*02:01", "HLA-A*03:01",
        "--out-models-dir", models_dir,
        "--num-jobs", str(n_jobs),
    ]
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    # Calibrate percentile ranks
    args = [
        "mhcflurry-calibrate-percentile-ranks",
        "--models-dir", models_dir,
        "--num-peptides-per-length", "10000",
        "--num-jobs", str(n_jobs),
    ]
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    result = Class1AffinityPredictor.load(models_dir)
    predictions = result.predict(
        peptides=["SLYNTVATL"],
        alleles=["HLA-A*02:01"])
    assert_equal(predictions.shape, (1,))
    assert_array_less(predictions, 1000)
    df = result.predict_to_dataframe(
            peptides=["SLYNTVATL"],
            alleles=["HLA-A*02:01"])
    print(df)
    assert "prediction_percentile" in df.columns

    print("Deleting: %s" % models_dir)
    shutil.rmtree(models_dir)


def run_and_check_with_model_selection(n_jobs=1):
    models_dir1 = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    hyperparameters_filename = os.path.join(
        models_dir1, "hyperparameters.yaml")

    # Include one architecture that has max_epochs = 0. We check that it never
    # gets selected in model selection.
    hyperparameters = [
        deepcopy(HYPERPARAMETERS[0]),
        deepcopy(HYPERPARAMETERS[0]),
    ]
    hyperparameters[-1]["max_epochs"] = 0
    with open(hyperparameters_filename, "w") as fd:
        json.dump(hyperparameters, fd)

    args = [
        "mhcflurry-class1-train-allele-specific-models",
        "--data", get_path("data_curated", "curated_training_data.affinity.csv.bz2"),
        "--hyperparameters", hyperparameters_filename,
        "--allele", "HLA-A*02:01", "HLA-A*03:01",
        "--out-models-dir", models_dir1,
        "--num-jobs", str(n_jobs),
        "--held-out-fraction-reciprocal", "10",
        "--n-models", "1",
    ]
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    result = Class1AffinityPredictor.load(models_dir1)
    assert_equal(len(result.neural_networks), 4)

    models_dir2 = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    args = [
        "mhcflurry-class1-select-allele-specific-models",
        "--data",
        get_path("data_curated", "curated_training_data.affinity.csv.bz2"),
        "--exclude-data", models_dir1 + "/train_data.csv.bz2",
        "--out-models-dir", models_dir2,
        "--models-dir", models_dir1,
        "--num-jobs", str(n_jobs),
        "--mse-max-models", "1",
        "--unselected-accuracy-scorer", "combined:mass-spec,mse",
        "--unselected-accuracy-percentile-threshold", "95",
    ]
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    result = Class1AffinityPredictor.load(models_dir2)
    assert_equal(len(result.neural_networks), 2)
    assert_equal(
        len(result.allele_to_allele_specific_models["HLA-A*02:01"]), 1)
    assert_equal(
        len(result.allele_to_allele_specific_models["HLA-A*03:01"]), 1)
    assert_equal(
        result.allele_to_allele_specific_models["HLA-A*02:01"][0].hyperparameters["max_epochs"], 500)
    assert_equal(
        result.allele_to_allele_specific_models["HLA-A*03:01"][
            0].hyperparameters["max_epochs"], 500)

    print("Deleting: %s" % models_dir1)
    print("Deleting: %s" % models_dir2)
    shutil.rmtree(models_dir1)


def test_run_parallel():
    run_and_check(n_jobs=2)
    run_and_check_with_model_selection(n_jobs=2)


def test_run_serial():
    run_and_check(n_jobs=0)
    run_and_check_with_model_selection(n_jobs=0)
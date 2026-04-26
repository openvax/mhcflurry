"""
Test train, calibrate percentile ranks, and model selection commands.
"""

import json
import os
import shutil
import tempfile
from copy import deepcopy
import pytest

import numpy
import pandas

from mhcflurry import Class1AffinityPredictor
from mhcflurry import calibrate_percentile_ranks_command
from mhcflurry import select_allele_specific_models_command
from mhcflurry import train_allele_specific_models_command

from mhcflurry.testing_utils import cleanup, startup


@pytest.fixture(autouse=True, scope="module")
def setup_module():
    startup()
    yield
    cleanup()

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HYPERPARAMETERS = [
    {
        "n_models": 1,
        "max_epochs": 2,
        "patience": 1,
        "minibatch_size": 8,
        "early_stopping": False,
        "validation_split": 0.0,

        "random_negative_rate": 0.0,
        "random_negative_constant": 0,

        "peptide_amino_acid_encoding": "BLOSUM62",
        "use_embedding": False,
        "kmer_size": 15,
        "batch_normalization": False,
        "locally_connected_layers": [],
        "activation": "tanh",
        "output_activation": "sigmoid",
        "layer_sizes": [
            4
        ],
        "random_negative_affinity_min": 20000.0,
        "random_negative_affinity_max": 50000.0,
        "dense_layer_l1_regularization": 0.001,
        "dropout_probability": 0.0
    }
]


TINY_AFFINITY_DATA = [
    ("HLA-A*02:01", "SLYNTVATL", 40.0),
    ("HLA-A*02:01", "GILGFVFTL", 55.0),
    ("HLA-A*02:01", "NLVPMVATV", 80.0),
    ("HLA-A*02:01", "YLQPRTFLL", 120.0),
    ("HLA-A*02:01", "KLVALGINAV", 9000.0),
    ("HLA-A*02:01", "FLRGRAYGL", 12000.0),
    ("HLA-A*02:01", "ELAGIGILTV", 18000.0),
    ("HLA-A*02:01", "RMFPNAPYL", 26000.0),
    ("HLA-A*03:01", "KLGGALQAK", 45.0),
    ("HLA-A*03:01", "ILRGSVAHK", 70.0),
    ("HLA-A*03:01", "RLRPGGKKK", 110.0),
    ("HLA-A*03:01", "ALWGFFPVL", 140.0),
    ("HLA-A*03:01", "SIINFEKL", 10000.0),
    ("HLA-A*03:01", "GILGFVFTL", 13000.0),
    ("HLA-A*03:01", "NLVPMVATV", 19000.0),
    ("HLA-A*03:01", "YLQPRTFLL", 30000.0),
]


def write_tiny_affinity_data(filename):
    pandas.DataFrame(
        [
            {
                "allele": allele,
                "peptide": peptide,
                "measurement_value": measurement_value,
                "measurement_type": "quantitative",
                "measurement_source": "synthetic unit test",
            }
            for allele, peptide, measurement_value in TINY_AFFINITY_DATA
        ]
    ).to_csv(filename, index=False)


def run_command(command_module, args):
    command_module.GLOBAL_DATA.clear()
    try:
        command_module.run(args)
    finally:
        command_module.GLOBAL_DATA.clear()


def test_train_calibrate_and_select_commands():
    models_dir1 = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    models_dir2 = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    try:
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
        train_data_filename = os.path.join(models_dir1, "train_data_input.csv")
        write_tiny_affinity_data(train_data_filename)

        run_command(
            train_allele_specific_models_command,
            [
                "--data", train_data_filename,
                "--hyperparameters", hyperparameters_filename,
                "--allele", "HLA-A*02:01", "HLA-A*03:01",
                "--out-models-dir", models_dir1,
                "--num-jobs", "0",
                "--held-out-fraction-reciprocal", "2",
                "--n-models", "1",
            ],
        )

        result = Class1AffinityPredictor.load(models_dir1)
        assert len(result.neural_networks) == 4

        run_command(
            calibrate_percentile_ranks_command,
            [
                "--models-dir", models_dir1,
                "--num-peptides-per-length", "20",
                "--length-range", "8", "9",
                "--allele", "HLA-A*02:01", "HLA-A*03:01",
                "--num-jobs", "0",
            ],
        )

        result = Class1AffinityPredictor.load(models_dir1)
        predictions = result.predict(
            peptides=["SLYNTVATL"],
            alleles=["HLA-A*02:01"])
        assert predictions.shape == (1,)
        assert numpy.isfinite(predictions).all()
        df = result.predict_to_dataframe(
                peptides=["SLYNTVATL"],
                alleles=["HLA-A*02:01"])
        print(df)
        assert "prediction_percentile" in df.columns

        run_command(
            select_allele_specific_models_command,
            [
                "--data", train_data_filename,
                "--exclude-data", models_dir1 + "/train_data.csv.bz2",
                "--out-models-dir", models_dir2,
                "--models-dir", models_dir1,
                "--num-jobs", "0",
                "--allele", "HLA-A*02:01", "HLA-A*03:01",
                "--scoring", "mse",
                "--mse-max-models", "1",
                "--mse-min-models", "1",
                "--unselected-accuracy-scorer", "",
            ],
        )

        result = Class1AffinityPredictor.load(models_dir2)
        assert len(result.neural_networks) == 2
        assert (
            len(result.allele_to_allele_specific_models["HLA-A*02:01"]) == 1)
        assert (
            len(result.allele_to_allele_specific_models["HLA-A*03:01"]) == 1)
        assert (
            result.allele_to_allele_specific_models[
                "HLA-A*02:01"][0].hyperparameters["max_epochs"] == 2)
        assert (
            result.allele_to_allele_specific_models[
                "HLA-A*03:01"][0].hyperparameters["max_epochs"] == 2)
    finally:
        print("Deleting: %s" % models_dir1)
        print("Deleting: %s" % models_dir2)
        shutil.rmtree(models_dir1, ignore_errors=True)
        shutil.rmtree(models_dir2, ignore_errors=True)

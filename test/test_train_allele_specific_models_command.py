import json
import os
import shutil
import tempfile

from numpy.testing import assert_array_less, assert_equal

from mhcflurry import train_allele_specific_models_command
from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path

HYPERPARAMETERS = [
    {
        "n_models": 2,
        "max_epochs": 2,
        "patience": 10,
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
            },
            {
                "filters": 8,
                "activation": "tanh",
                "kernel_size": 3
            }
        ],
        "activation": "relu",
        "output_activation": "sigmoid",
        "layer_sizes": [
            32
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
        "--data", get_path("data_curated", "curated_training_data.no_mass_spec.csv.bz2"),
        "--hyperparameters", hyperparameters_filename,
        "--allele", "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01",
        "--out-models-dir", models_dir,
        "--percent-rank-calibration-num-peptides-per-length", "10000",
        "--num-jobs", str(n_jobs),
        "--ignore-inequalities",
    ]
    print("Running with args: %s" % args)
    train_allele_specific_models_command.run(args)

    result = Class1AffinityPredictor.load(models_dir)
    predictions = result.predict(
        peptides=["SLYNTVATL"],
        alleles=["HLA-A*02:01"])
    assert_equal(predictions.shape, (1,))
    assert_array_less(predictions, 500)
    df = result.predict_to_dataframe(
            peptides=["SLYNTVATL"],
            alleles=["HLA-A*02:01"])
    print(df)
    assert "prediction_percentile" in df.columns

    print("Deleting: %s" % models_dir)
    shutil.rmtree(models_dir)


def Xtest_run_parallel():
    run_and_check(n_jobs=3)


def test_run_serial():
    run_and_check(n_jobs=1)
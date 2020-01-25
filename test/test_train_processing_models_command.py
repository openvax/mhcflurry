"""
Test processing train and model selection commands.
"""
import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True
import json
import os
import shutil
import tempfile
import subprocess
import re
from copy import deepcopy

from numpy.testing import assert_array_less, assert_equal
from sklearn.metrics import roc_auc_score
import pandas

from mhcflurry.class1_processing_predictor import Class1ProcessingPredictor
from mhcflurry.downloads import get_path
from mhcflurry.common import random_peptides

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HYPERPARAMETERS = [
    {
        "max_epochs": 100,
        "n_flank_length": 5,
        "c_flank_length": 5,
        "convolutional_kernel_size": 3,
    },
    {
        "max_epochs": 1,
        "n_flank_length": 5,
        "c_flank_length": 5,
        "convolutional_kernel_size": 3,
    }
]


def make_dataset(num=10000):
    df = pandas.DataFrame({
        "n_flank": random_peptides(num / 2, 10) + random_peptides(num / 2, 1),
        "c_flank": random_peptides(num, 10),
        "peptide": random_peptides(num / 2, 11) + random_peptides(num / 2, 8),
    }).sample(frac=1.0)
    df["sample_id"] = pandas.Series(
        ["sample_%d" % (i + 1) for i in range(5)]).sample(
        n=len(df), replace=True).values

    n_regex = "[AILQSVWEN].[MNPQYKV]"

    def is_hit(n_flank, c_flank, peptide):
        if re.search(n_regex, peptide):
            return False  # peptide is cleaved
        return bool(re.match(n_regex, n_flank[-1:] + peptide))

    df["hit"] = [
        is_hit(row.n_flank, row.c_flank, row.peptide)
        for (_, row) in df.iterrows()
    ]

    train_df = df.sample(frac=0.9)
    test_df = df.loc[~df.index.isin(train_df.index)].copy()

    print(
        "Generated dataset",
        len(df),
        "hits: ",
        df.hit.sum(),
        "frac:",
        df.hit.mean())

    return (train_df, test_df)


def run_and_check(n_jobs=0, additional_args=[], delete=False):
    (train_df, test_df) = make_dataset()

    models_dir = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    hyperparameters_filename = os.path.join(
        models_dir, "hyperparameters.yaml")
    with open(hyperparameters_filename, "w") as fd:
        json.dump(HYPERPARAMETERS, fd)

    train_filename = os.path.join(models_dir, "training.csv")
    train_df.to_csv(train_filename, index=False)

    args = [
        "mhcflurry-class1-train-processing-models",
        "--data", train_filename,
        "--hyperparameters", hyperparameters_filename,
        "--out-models-dir", models_dir,
        "--held-out-samples", "2",
        "--num-folds", "2",
        "--num-jobs", str(n_jobs),
    ]
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    full_predictor = Class1ProcessingPredictor.load(models_dir)
    print("Loaded models", len(full_predictor.models))
    assert_equal(len(full_predictor.models), 4)

    test_df["full_predictor"] = full_predictor.predict(
        test_df.peptide.values,
        test_df.n_flank.values,
        test_df.c_flank.values)

    test_auc = roc_auc_score(test_df.hit.values, test_df.full_predictor.values)
    print("Full predictor auc", test_auc)

    print("Performing model selection.")

    # Run model selection
    models_dir_selected = tempfile.mkdtemp(
        prefix="mhcflurry-test-models-selected")
    args = [
        "mhcflurry-class1-select-processing-models",
        "--data", os.path.join(models_dir, "train_data.csv.bz2"),
        "--models-dir", models_dir,
        "--out-models-dir", models_dir_selected,
        "--max-models", "1",
        "--num-jobs", str(n_jobs),
    ] + additional_args
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    selected_predictor = Class1ProcessingPredictor.load(models_dir_selected)
    assert_equal(len(selected_predictor.models), 2)

    test_df["selected_predictor"] = selected_predictor.predict(
        test_df.peptide.values,
        test_df.n_flank.values,
        test_df.c_flank.values)

    test_auc = roc_auc_score(test_df.hit.values, test_df.selected_predictor.values)
    print("Selected predictor auc", test_auc)

    if delete:
        print("Deleting: %s" % models_dir)
        shutil.rmtree(models_dir)
        shutil.rmtree(models_dir_selected)

def Xtest_run_parallel():
    run_and_check(n_jobs=2)


def test_run_serial():
    run_and_check(n_jobs=0)

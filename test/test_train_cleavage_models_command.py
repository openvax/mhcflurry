"""
Test cleavage train and model selection commands.
"""

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

from mhcflurry.class1_cleavage_predictor import Class1CleavagePredictor
from mhcflurry.downloads import get_path
from mhcflurry.common import random_peptides

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HYPERPARAMETERS = [
    {
        "max_epochs": 500,
        "n_flank_length": 5,
        "c_flank_length": 5,
        "convolutional_kernel_size": 3,
    }
]


def run_and_check(n_jobs=0, num=50000):
    df = pandas.DataFrame({
        "n_flank": random_peptides(num / 2, 10) + random_peptides(num / 2, 1),
        "c_flank": random_peptides(num, 10),
        "peptide": random_peptides(num / 2, 11) + random_peptides(num / 2, 8),
    }).sample(frac=1.0)
    df["sample_id"] = pandas.Series(
        ["sample_%d" % (i + 1) for i in range(5)]).sample(
        n=len(df), replace=True).values

    n_cleavage_regex = "[AILQSV][SINFEKLH][MNPQYK]"

    def is_hit(n_flank, c_flank, peptide):
        if re.search(n_cleavage_regex, peptide):
            return False  # peptide is cleaved
        return bool(re.match(n_cleavage_regex, n_flank[-1:] + peptide))

    df["hit"] = [
        is_hit(row.n_flank, row.c_flank, row.peptide)
        for (_, row) in df.iterrows()
    ]

    train_df = df.sample(frac=0.1)
    test_df = df.loc[~df.index.isin(train_df.index)]

    print(
        "Generated dataset",
        len(df),
        "hits: ",
        df.hit.sum(),
        "frac:",
        df.hit.mean())

    models_dir = tempfile.mkdtemp(prefix="mhcflurry-test-models")
    hyperparameters_filename = os.path.join(
        models_dir, "hyperparameters.yaml")
    with open(hyperparameters_filename, "w") as fd:
        json.dump(HYPERPARAMETERS, fd)

    train_filename = os.path.join(models_dir, "training.csv")
    train_df.to_csv(train_filename, index=False)

    args = [
        "mhcflurry-class1-train-cleavage-models",
        "--data", train_filename,
        "--hyperparameters", hyperparameters_filename,
        "--out-models-dir", models_dir,
        "--held-out-samples", "2",
        "--num-jobs", str(n_jobs),

    ]
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    predictor = Class1CleavagePredictor.load(models_dir)
    print("Loaded models", len(predictor.models))
    assert len(predictor.models) > 0

    test_df["prediction"] = predictor.predict(
        test_df.peptide.values,
        test_df.n_flank.values,
        test_df.c_flank.values)

    test_auc = roc_auc_score(test_df.hit.values, test_df.prediction.values)
    print("Test auc", test_auc)
    assert test_auc > 0.85

    print("Deleting: %s" % models_dir)
    shutil.rmtree(models_dir)



def Xtest_run_parallel():
    run_and_check(n_jobs=2)
    #run_and_check_with_model_selection(n_jobs=2)


def test_run_serial():
    run_and_check(n_jobs=0)
    #run_and_check_with_model_selection(n_jobs=0)
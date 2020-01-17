"""
Tests for calibrate percentile ranks command
"""

import os
import shutil
import tempfile
import subprocess

from numpy.testing import assert_equal

from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mhcflurry.testing_utils import cleanup, startup
teardown = cleanup
setup = startup


def run_and_check(n_jobs=0, delete=True, additional_args=[]):
    source_models_dir = get_path("models_class1_pan", "models.combined")
    dest_models_dir = tempfile.mkdtemp(prefix="mhcflurry-test-models")

    # Save a new predictor that has no percent rank calibration data.
    original_predictor = Class1AffinityPredictor.load(source_models_dir)
    print("Loaded predictor", source_models_dir)
    new_predictor = Class1AffinityPredictor(
        class1_pan_allele_models=original_predictor.class1_pan_allele_models,
        allele_to_sequence=original_predictor.allele_to_sequence,
    )
    new_predictor.save(dest_models_dir)
    print("Saved predictor to", dest_models_dir)

    new_predictor = Class1AffinityPredictor.load(dest_models_dir)
    assert_equal(len(new_predictor.allele_to_percent_rank_transform), 0)

    args = [
        "mhcflurry-calibrate-percentile-ranks",
        "--models-dir", dest_models_dir,
        "--match-amino-acid-distribution-data", get_path(
            "data_curated", "curated_training_data.affinity.csv.bz2"),
        "--motif-summary",
        "--num-peptides-per-length", "1000",
        "--allele", "HLA-A*02:01", "HLA-B*07:02",
        "--verbosity", "1",
        "--num-jobs", str(n_jobs),
    ] + additional_args
    print("Running with args: %s" % args)
    subprocess.check_call(args)

    new_predictor = Class1AffinityPredictor.load(dest_models_dir)
    assert_equal(len(new_predictor.allele_to_percent_rank_transform), 2)

    if delete:
        print("Deleting: %s" % dest_models_dir)
        shutil.rmtree(dest_models_dir)
    else:
        print("Not deleting: %s" % dest_models_dir)


def test_run_serial():
    run_and_check(n_jobs=0)


def test_run_parallel():
    run_and_check(n_jobs=2)


def test_run_cluster_parallelism(delete=True):
    run_and_check(n_jobs=0, additional_args=[
        '--cluster-parallelism',
        '--cluster-results-workdir', '/tmp/',
        '--cluster-max-retries', '0',
    ], delete=delete)


if __name__ == "__main__":
    # run_and_check(n_jobs=0, delete=False)
    # run_and_check(n_jobs=2, delete=False)
    test_run_cluster_parallelism(delete=False)

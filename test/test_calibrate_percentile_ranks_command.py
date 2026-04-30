"""
Tests for calibrate percentile ranks command
"""

import os
import shutil
import tempfile
import subprocess
import pytest
import sys


from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path
from .pytest_helpers import mhcflurry_cli

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MHCFLURRY_CLUSTER_WORKER_COMMAND"] = (
    f"{sys.executable} -m mhcflurry.cluster_worker_entry_point"
)

from mhcflurry.testing_utils import cleanup, startup

pytest.fixture(autouse=True, scope="module")
def setup_module():
    startup()
    yield
    cleanup()


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
    assert len(new_predictor.allele_to_percent_rank_transform) == 0

    args = mhcflurry_cli("mhcflurry-calibrate-percentile-ranks") + [
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
    assert len(new_predictor.allele_to_percent_rank_transform) == 2

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


def test_filter_canonicalizable_alleles_drops_pseudogene():
    """Pseudogene / null / questionable annotations are dropped, valid kept."""
    from mhcflurry.calibrate_percentile_ranks_command import (
        _filter_canonicalizable_alleles,
    )
    raw = [
        "HLA-A*02:01",
        "Caja-B5*01:01ps",  # pseudogene — mhcgnomes annotation_pseudogene
        "HLA-B*07:02",
    ]
    kept = _filter_canonicalizable_alleles(raw)
    assert "Caja-B5*01:01ps" not in kept
    assert "HLA-A*02:01" in kept
    assert "HLA-B*07:02" in kept


def test_filter_canonicalizable_alleles_passthrough_when_all_valid():
    from mhcflurry.calibrate_percentile_ranks_command import (
        _filter_canonicalizable_alleles,
    )
    raw = ["HLA-A*02:01", "HLA-B*07:02", "HLA-C*04:01"]
    kept = _filter_canonicalizable_alleles(raw)
    assert kept == raw


def test_filter_canonicalizable_alleles_call_sites_cover_both_predictor_kinds():
    """Both run_class1_presentation_predictor and run_class1_affinity_predictor
    invoke the filter on supported_alleles.

    Without coverage on both, a presentation-predictor calibration with a
    pseudogene allele in its coverage map would crash partway through the
    same way the affinity path used to. Reading the source is the
    cheapest regression test we can write here — running calibrate
    end-to-end on a synthetic presentation predictor is too heavy for
    a unit test.
    """
    import inspect
    from mhcflurry import calibrate_percentile_ranks_command as mod

    affinity_src = inspect.getsource(mod.run_class1_affinity_predictor)
    presentation_src = inspect.getsource(mod.run_class1_presentation_predictor)
    assert "_filter_canonicalizable_alleles" in affinity_src, (
        "affinity calibration path must call the filter helper"
    )
    assert "_filter_canonicalizable_alleles" in presentation_src, (
        "presentation calibration path must call the filter helper"
    )


def test_filter_canonicalizable_alleles_memoizes_repeats():
    """Duplicate alleles in the input each get evaluated once."""
    from mhcflurry.calibrate_percentile_ranks_command import (
        _filter_canonicalizable_alleles,
    )
    raw = ["HLA-A*02:01"] * 5 + ["Caja-B5*01:01ps"] * 5 + ["HLA-B*07:02"]
    kept = _filter_canonicalizable_alleles(raw)
    assert kept.count("HLA-A*02:01") == 5  # all kept
    assert kept.count("HLA-B*07:02") == 1
    assert "Caja-B5*01:01ps" not in kept


if __name__ == "__main__":
    # run_and_check(n_jobs=0, delete=False)
    # run_and_check(n_jobs=2, delete=False)
    test_run_cluster_parallelism(delete=False)

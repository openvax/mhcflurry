# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for calibrate percentile ranks command
"""

import os
import shutil
import tempfile
import subprocess
import pytest
import sys
from types import SimpleNamespace

import numpy

from mhcflurry import Class1AffinityPredictor
from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor
from mhcflurry.downloads import get_path
from mhcflurry.percent_rank_transform import PercentRankTransform
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


@pytest.mark.slow
@pytest.mark.integration
def test_run_serial():
    run_and_check(n_jobs=0)


@pytest.mark.slow
@pytest.mark.integration
def test_run_parallel():
    run_and_check(n_jobs=2)


@pytest.mark.slow
@pytest.mark.integration
def test_run_cluster_parallelism(delete=True):
    run_and_check(n_jobs=0, additional_args=[
        '--cluster-parallelism',
        '--cluster-results-workdir', '/tmp/',
        '--cluster-max-retries', '0',
    ], delete=delete)


def test_filter_canonicalizable_alleles_drops_pseudogene():
    """Pseudogene / null / questionable annotations are dropped, valid kept."""
    from mhcflurry.common import filter_canonicalizable_alleles
    raw = [
        "HLA-A*02:01",
        "Caja-B5*01:01ps",  # pseudogene — mhcgnomes annotation_pseudogene
        "HLA-B*07:02",
    ]
    kept = filter_canonicalizable_alleles(raw)
    assert "Caja-B5*01:01ps" not in kept
    assert "HLA-A*02:01" in kept
    assert "HLA-B*07:02" in kept


def test_filter_canonicalizable_alleles_passthrough_when_all_valid():
    from mhcflurry.common import filter_canonicalizable_alleles
    raw = ["HLA-A*02:01", "HLA-B*07:02", "HLA-C*04:01"]
    kept = filter_canonicalizable_alleles(raw)
    assert kept == raw


def test_percent_rank_status_helpers_count_sequence_equivalent_calibration():
    from mhcflurry.calibrate_percentile_ranks_command import (
        missing_percent_rank_alleles,
        percent_rank_status_df,
    )

    transform = PercentRankTransform()
    transform.fit(numpy.array([10.0, 20.0, 30.0]), bins=3)
    predictor = Class1AffinityPredictor(
        allele_to_sequence={
            "HLA-A*02:01": "SEQUENCE1",
            "HLA-A*02:02": "SEQUENCE1",
            "HLA-B*07:02": "SEQUENCE2",
        },
        allele_to_percent_rank_transform={"HLA-A*02:01": transform},
    )
    alleles = ["HLA-A*02:01", "HLA-A*02:02", "HLA-B*07:02"]

    assert missing_percent_rank_alleles(predictor, alleles) == ["HLA-B*07:02"]

    status = percent_rank_status_df(predictor, alleles)
    assert status.to_dict("records") == [
        {
            "allele": "HLA-A*02:01",
            "normalized_allele": "HLA-A*02:01",
            "supported": True,
            "has_affinity_percent_rank": True,
            "affinity_percent_rank_source_allele": "HLA-A*02:01",
        },
        {
            "allele": "HLA-A*02:02",
            "normalized_allele": "HLA-A*02:02",
            "supported": True,
            "has_affinity_percent_rank": True,
            "affinity_percent_rank_source_allele": "HLA-A*02:01",
        },
        {
            "allele": "HLA-B*07:02",
            "normalized_allele": "HLA-B*07:02",
            "supported": True,
            "has_affinity_percent_rank": False,
            "affinity_percent_rank_source_allele": "",
        },
    ]


def test_list_percent_rank_status_stdout_starts_with_csv_header(
        monkeypatch, tmp_path, capsys):
    from mhcflurry import calibrate_percentile_ranks_command as mod

    def fake_status(args):
        print("allele,normalized_allele")
        print("HLA-A*02:01,HLA-A*02:01")
        return 0

    monkeypatch.setattr(
        mod, "run_class1_affinity_percent_rank_status", fake_status)

    assert mod.run([
        "--models-dir", str(tmp_path),
        "--list-percent-rank-status",
    ]) == 0

    stdout = capsys.readouterr().out
    assert stdout.startswith("allele,normalized_allele\n")
    assert "random seed" not in stdout


class FakeAffinityPredictorForAlleleSelection(object):
    supported_alleles = ["HLA-A*02:01", "HLA-B*07:02"]

    def __init__(self):
        self.canonicalized = []

    def canonicalize_allele_name(self, allele):
        self.canonicalized.append(allele)
        return "canonical-%s" % allele


def test_requested_calibration_alleles_uses_presentation_affinity_predictor():
    from mhcflurry.calibrate_percentile_ranks_command import (
        requested_calibration_alleles,
    )

    affinity_predictor = FakeAffinityPredictorForAlleleSelection()
    predictor = Class1PresentationPredictor(
        affinity_predictor=affinity_predictor)
    args = SimpleNamespace(
        allele=["HLA-A*02:01", "HLA-B*07:02"],
        alleles_file=None,
    )

    assert not hasattr(predictor, "canonicalize_allele_name")
    assert requested_calibration_alleles(args, predictor) == [
        "canonical-HLA-A*02:01",
        "canonical-HLA-B*07:02",
    ]
    assert affinity_predictor.canonicalized == [
        "HLA-A*02:01",
        "HLA-B*07:02",
    ]


def test_requested_calibration_alleles_file_uses_presentation_affinity_predictor(
        tmp_path):
    from mhcflurry.calibrate_percentile_ranks_command import (
        requested_calibration_alleles,
    )

    alleles_file = tmp_path / "alleles.csv"
    alleles_file.write_text(
        "allele\nHLA-A*02:01\nCaja-B5*01:01ps\nHLA-B*07:02\n")

    affinity_predictor = FakeAffinityPredictorForAlleleSelection()
    predictor = Class1PresentationPredictor(
        affinity_predictor=affinity_predictor)
    args = SimpleNamespace(allele=None, alleles_file=str(alleles_file))

    assert requested_calibration_alleles(args, predictor) == [
        "canonical-HLA-A*02:01",
        "canonical-HLA-B*07:02",
    ]
    assert affinity_predictor.canonicalized == [
        "HLA-A*02:01",
        "HLA-B*07:02",
    ]


def test_filter_canonicalizable_alleles_memoizes_repeats():
    """Duplicate alleles in the input each get evaluated once."""
    from mhcflurry.common import filter_canonicalizable_alleles
    raw = ["HLA-A*02:01"] * 5 + ["Caja-B5*01:01ps"] * 5 + ["HLA-B*07:02"]
    kept = filter_canonicalizable_alleles(raw)
    assert kept.count("HLA-A*02:01") == 5  # all kept
    assert kept.count("HLA-B*07:02") == 1
    assert "Caja-B5*01:01ps" not in kept


if __name__ == "__main__":
    # run_and_check(n_jobs=0, delete=False)
    # run_and_check(n_jobs=2, delete=False)
    test_run_cluster_parallelism(delete=False)

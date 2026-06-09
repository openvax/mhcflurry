"""
Tight torch-baseline regression guard.

Unlike the master (Keras/TF) compatibility fixtures — which compare against a
pre-migration baseline at a loose 1% tolerance that absorbs the TF->torch
framework switch — this test freezes the predictions of the *released* models as
computed by the current torch code and asserts an exact (1e-6) match on CPU. Any
future change (encoding, loading, a refactor) that perturbs a prediction fails
loudly here. It also covers the standalone processing predictor, which has no
other prediction-baseline test.

Regenerate ``test/data/torch_baseline_predictions.json`` only deliberately, when
a numerics change is intended and reviewed.
"""

import json
import os

import numpy as np
import pytest

from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
from mhcflurry.common import configure_pytorch
from mhcflurry.downloads import get_path
from mhcflurry.testing_utils import cleanup, startup

pytestmark = pytest.mark.downloads

BASELINE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "torch_baseline_predictions.json")

# CPU is deterministic; the baseline was frozen on CPU. Match tightly.
RTOL = 1e-6
ATOL = 1e-6


def setup_module():
    startup()
    configure_pytorch(backend="cpu")


def teardown_module():
    cleanup()


def _baseline():
    with open(BASELINE_PATH) as f:
        return json.load(f)


def _assert_close(actual, expected, label):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=RTOL, atol=ATOL,
        err_msg="torch-baseline regression: %s predictions changed. If this is "
                "an intended numerics change, regenerate "
                "test/data/torch_baseline_predictions.json." % label,
    )


def test_affinity_allele_specific_predictions_match_baseline():
    b = _baseline()
    predictor = Class1AffinityPredictor.load(get_path("models_class1", "models"))
    for allele, expected in b["affinity_allele_specific"].items():
        _assert_close(
            predictor.predict(b["peptides"], allele=allele), expected,
            "affinity allele-specific %s" % allele)


def test_affinity_pan_predictions_match_baseline():
    b = _baseline()
    predictor = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"))
    for allele, expected in b["affinity_pan"].items():
        _assert_close(
            predictor.predict(b["peptides"], allele=allele), expected,
            "affinity pan %s" % allele)


def test_processing_predictions_match_baseline():
    b = _baseline()
    pres = Class1PresentationPredictor.load()
    _assert_close(
        pres.processing_predictor_with_flanks.predict(
            b["peptides"], b["n_flanks"], b["c_flanks"]),
        b["processing_with_flanks"], "processing with flanks")
    empty = [""] * len(b["peptides"])
    _assert_close(
        pres.processing_predictor_without_flanks.predict(
            b["peptides"], empty, empty),
        b["processing_no_flanks"], "processing without flanks")


def test_presentation_predictions_match_baseline():
    b = _baseline()
    pres = Class1PresentationPredictor.load()
    df = pres.predict(
        peptides=b["peptides"], alleles={"sample": b["alleles"]},
        n_flanks=b["n_flanks"], c_flanks=b["c_flanks"], verbose=0)
    _assert_close(
        df["presentation_score"].values, b["presentation_score"],
        "presentation score")

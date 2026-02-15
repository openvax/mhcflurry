"""
Compare released-model predictions against master (TF) fixtures.

Fixtures are generated from the current master branch and stored under
test/data. Tests consume these files without importing TensorFlow.
"""
import json
import os

import numpy as np
import pytest

from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import configure, get_current_release, get_path
from mhcflurry.testing_utils import startup, cleanup


def setup_module():
    startup()


def teardown_module():
    cleanup()


def _load_fixture(name):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, name)
    with open(path, "r") as f:
        return json.load(f)


def test_released_affinity_predictions_match_master():
    fixture = _load_fixture("master_released_class1_affinity_predictions.json")
    # Ensure downloads dir reflects current environment before loading models.
    configure()
    if fixture.get("release") != get_current_release():
        pytest.skip(
            "Fixture was generated from master release "
            f"{fixture.get('release')}, but current downloads are "
            f"{get_current_release()}. Update downloads to compare."
        )

    allele_specific = Class1AffinityPredictor.load(
        get_path("models_class1", "models")
    )
    pan = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined")
    )

    spec = fixture["allele_specific"]
    pan_fx = fixture["pan_allele"]

    preds_specific = allele_specific.predict(
        peptides=spec["peptides"],
        alleles=spec["alleles"],
    )
    preds_pan = pan.predict(
        peptides=pan_fx["peptides"],
        alleles=pan_fx["alleles"],
    )

    np.testing.assert_allclose(
        preds_specific,
        np.array(spec["predictions"], dtype=np.float64),
        rtol=0.01,
        atol=0.0,
    )
    np.testing.assert_allclose(
        preds_pan,
        np.array(pan_fx["predictions"], dtype=np.float64),
        rtol=0.01,
        atol=0.0,
    )

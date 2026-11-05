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
Tight torch-baseline regression guard.

Unlike the master (Keras/TF) compatibility fixtures — which compare against a
pre-migration baseline at a loose 1% tolerance that absorbs the TF->torch
framework switch — this test freezes the predictions of the *released* models as
computed by the current torch code and asserts agreement within 1e-6 on CPU. Any
future change (encoding, loading, a refactor) that perturbs a prediction fails
loudly here. It also covers the standalone processing predictor.

Each baseline belongs to one download catalogue release, recorded in the file
(``test/data/torch_baseline_predictions_<release>.json``), and the tests load
that release's bundles explicitly. Changing the default release therefore never
compares a baseline against other weights (issue #489):

* ``2.2.0``: the 2020 public weights, which catalogues 2.0.0 and 2.2.0
  share. Frozen in June 2026.
* ``2.3.0``: the 2.3.0 weights. Frozen in November 2026.

All baselines score the same peptides, alleles and flanks. Regenerate one only
deliberately, when a numerics change is intended and reviewed, from the
repository root::

    python -m test.test_torch_baseline_regression 2.3.0
"""

import json
import os
import sys

import numpy as np
import pytest

from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
from mhcflurry.common import configure_pytorch
from mhcflurry.downloads import get_path
from mhcflurry.testing_utils import cleanup, startup

pytestmark = pytest.mark.downloads

BASELINE_RELEASES = ["2.2.0", "2.3.0"]

# CPU is deterministic; the baselines were frozen on CPU. Match tightly.
RTOL = 1e-6
ATOL = 1e-6


def setup_module():
    startup()
    configure_pytorch(backend="cpu")


def teardown_module():
    cleanup()


def baseline_path(release):
    """
    Return the path of the baseline file for a model release.

    Parameters
    ----------
    release : str
        Download catalogue release, such as ``"2.3.0"``.

    Returns
    -------
    str
    """
    return os.path.join(
        os.path.dirname(__file__), "data",
        "torch_baseline_predictions_%s.json" % release)


def load_baseline(release):
    """
    Load the baseline inputs and predictions recorded for a model release.

    Parameters
    ----------
    release : str
        Download catalogue release, such as ``"2.3.0"``.

    Returns
    -------
    dict
        The baseline file's contents. Its ``release`` entry names the release
        whose bundles produced the predictions.
    """
    with open(baseline_path(release)) as f:
        baseline = json.load(f)
    assert baseline["release"] == release, baseline_path(release)
    return baseline


def predict_affinity(models_dir, inputs):
    """
    Predict the IC50 of every baseline peptide for every baseline allele.

    Parameters
    ----------
    models_dir : str
        Directory of a saved ``Class1AffinityPredictor``.
    inputs : dict
        Baseline inputs with ``peptides`` and ``alleles`` lists.

    Returns
    -------
    dict
        Maps each allele to a list of IC50 predictions (nM), one per peptide.
    """
    predictor = Class1AffinityPredictor.load(models_dir)
    return {
        allele: predictor.predict(inputs["peptides"], allele=allele).tolist()
        for allele in inputs["alleles"]
    }


def predict_processing(models_dir, inputs):
    """
    Predict processing scores with a presentation bundle's two processing
    predictors.

    Parameters
    ----------
    models_dir : str
        Directory of a saved ``Class1PresentationPredictor``.
    inputs : dict
        Baseline inputs with ``peptides``, ``n_flanks`` and ``c_flanks``
        lists.

    Returns
    -------
    dict
        ``processing_with_flanks`` (scored with the given flanks) and
        ``processing_no_flanks`` (scored with empty flanks), each a list with
        one score per peptide.
    """
    predictor = Class1PresentationPredictor.load(models_dir)
    empty = [""] * len(inputs["peptides"])
    return {
        "processing_with_flanks": predictor.processing_predictor_with_flanks.predict(
            inputs["peptides"], inputs["n_flanks"], inputs["c_flanks"]).tolist(),
        "processing_no_flanks": predictor.processing_predictor_without_flanks.predict(
            inputs["peptides"], empty, empty).tolist(),
    }


def predict_presentation(models_dir, inputs):
    """
    Predict presentation scores for one sample carrying every baseline allele.

    Parameters
    ----------
    models_dir : str
        Directory of a saved ``Class1PresentationPredictor``.
    inputs : dict
        Baseline inputs with ``peptides``, ``alleles``, ``n_flanks`` and
        ``c_flanks`` lists.

    Returns
    -------
    list of float
        One presentation score per peptide.
    """
    predictor = Class1PresentationPredictor.load(models_dir)
    predictions = predictor.predict(
        peptides=inputs["peptides"], alleles={"sample": inputs["alleles"]},
        n_flanks=inputs["n_flanks"], c_flanks=inputs["c_flanks"], verbose=0)
    return predictions["presentation_score"].tolist()


def assert_matches_baseline(actual, expected, release, label):
    """
    Assert that predictions equal a baseline's within ``RTOL`` and ``ATOL``.

    Parameters
    ----------
    actual : array-like
        Predictions from the current code.
    expected : array-like
        Predictions recorded in the baseline.
    release : str
        Release of the baseline, named in the failure message.
    label : str
        Prediction kind, named in the failure message.
    """
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=RTOL, atol=ATOL,
        err_msg="torch-baseline regression: %s predictions of release %s "
                "changed. If this is an intended numerics change, regenerate "
                "test/data/%s." % (
                    label, release, os.path.basename(baseline_path(release))),
    )


@pytest.fixture(params=BASELINE_RELEASES)
def baseline(request):
    return load_baseline(request.param)


def test_affinity_allele_specific_predictions_match_baseline(baseline):
    release = baseline["release"]
    predictions = predict_affinity(
        get_path("models_class1", "models", release=release), baseline)
    for allele, expected in baseline["affinity_allele_specific"].items():
        assert_matches_baseline(
            predictions[allele], expected, release,
            "affinity allele-specific %s" % allele)


def test_affinity_pan_predictions_match_baseline(baseline):
    release = baseline["release"]
    predictions = predict_affinity(
        get_path("models_class1_pan", "models.combined", release=release),
        baseline)
    for allele, expected in baseline["affinity_pan"].items():
        assert_matches_baseline(
            predictions[allele], expected, release, "affinity pan %s" % allele)


def test_processing_predictions_match_baseline(baseline):
    release = baseline["release"]
    predictions = predict_processing(
        get_path("models_class1_presentation", "models", release=release),
        baseline)
    assert_matches_baseline(
        predictions["processing_with_flanks"],
        baseline["processing_with_flanks"], release, "processing with flanks")
    assert_matches_baseline(
        predictions["processing_no_flanks"],
        baseline["processing_no_flanks"], release, "processing without flanks")


def test_presentation_predictions_match_baseline(baseline):
    release = baseline["release"]
    predictions = predict_presentation(
        get_path("models_class1_presentation", "models", release=release),
        baseline)
    assert_matches_baseline(
        predictions, baseline["presentation_score"], release,
        "presentation score")


def generate_baseline(release, inputs):
    """
    Predict every baseline quantity with one release's downloaded bundles.

    Parameters
    ----------
    release : str
        Download catalogue release whose ``models_class1``,
        ``models_class1_pan`` and ``models_class1_presentation`` bundles are
        fetched locally.
    inputs : dict
        Baseline inputs with ``peptides``, ``alleles``, ``n_flanks`` and
        ``c_flanks`` lists.

    Returns
    -------
    dict
        A baseline in the layout of the checked-in files, with predictions
        rounded to 8 decimal places.
    """
    presentation_dir = get_path(
        "models_class1_presentation", "models", release=release)
    predictions = {
        "affinity_allele_specific": predict_affinity(
            get_path("models_class1", "models", release=release), inputs),
        "affinity_pan": predict_affinity(
            get_path("models_class1_pan", "models.combined", release=release),
            inputs),
        **predict_processing(presentation_dir, inputs),
        "presentation_score": predict_presentation(presentation_dir, inputs),
    }
    baseline = {"release": release}
    for key in ("peptides", "alleles", "n_flanks", "c_flanks"):
        baseline[key] = inputs[key]
    for (key, values) in predictions.items():
        if isinstance(values, dict):
            baseline[key] = {
                allele: [round(value, 8) for value in allele_values]
                for (allele, allele_values) in values.items()
            }
        else:
            baseline[key] = [round(value, 8) for value in values]
    return baseline


def write_baseline(release):
    """
    Write the baseline file for a release, scoring the inputs of the
    ``BASELINE_RELEASES[0]`` baseline so every release is compared on the
    same peptides.

    Parameters
    ----------
    release : str
        Download catalogue release whose bundles are fetched locally.

    Returns
    -------
    str
        Path of the written file.
    """
    startup()
    configure_pytorch(backend="cpu")
    baseline = generate_baseline(release, load_baseline(BASELINE_RELEASES[0]))
    path = baseline_path(release)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=1)
        f.write("\n")
    return path


if __name__ == "__main__":
    print(write_baseline(sys.argv[1]))

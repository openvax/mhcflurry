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

"""Tests for the shared ranking metrics module (issue #527)."""

import importlib.util
import itertools
from pathlib import Path

import numpy
import pandas
import pytest

from mhcflurry.cli import compare_models, paper_figures, plot_model_comparison
from mhcflurry.cli.model_comparison_constants import (
    METRIC_DISPLAY_NAMES,
    path_is_within,
    score_values,
)
from mhcflurry.ranking_metrics import ppv_at_n, ranking_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scores,expected", [
    ([1, 1, 1, 1], 0.5),
    ([3, 2, 2, 1], 0.75),
    ([4, 3, 2, 1], 0.5),
    ([3, 3, 1, 1], 0.5),
])
def test_ppv_at_n_is_permutation_invariant_with_ties(scores, expected):
    y = numpy.array([1, 0, 1, 0])
    scores = numpy.array(scores, dtype=float)
    for order in itertools.permutations(range(4)):
        idx = list(order)
        assert ppv_at_n(y[idx], scores[idx], 2) == expected
        assert ppv_at_n(y[idx], scores[idx]) == expected


def test_ppv_at_n_edge_cases():
    assert numpy.isnan(ppv_at_n([], [], 0))
    assert numpy.isnan(ppv_at_n([0, 0], [0.5, 0.2]))
    assert ppv_at_n([1, 0, 1], [1.0, 1.0, 1.0], 3) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="exceeds"):
        ppv_at_n([1], [1.0], 2)


def test_ranking_metrics_filters_non_finite_and_handles_single_class():
    only_pos = ranking_metrics([1, 1, 1], [0.1, 0.2, 0.3])
    assert only_pos["n"] == 3 and only_pos["n_pos"] == 3
    assert numpy.isnan(only_pos["roc_auc"])
    assert numpy.isnan(only_pos["pr_auc"])
    assert numpy.isnan(only_pos["ppv_at_n"])

    with_nan = ranking_metrics([1, 0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2, numpy.nan])
    assert with_nan["n"] == 4 and with_nan["n_pos"] == 2
    assert with_nan["roc_auc"] == 1.0
    assert with_nan["pr_auc"] == 1.0
    assert with_nan["ppv_at_n"] == 1.0


def test_all_four_evaluation_call_sites_agree_on_tied_ppv(tmp_path):
    labels = numpy.array([1, 1, 0, 0])
    tied_scores = numpy.array([0.5, 0.5, 0.5, 0.5])

    # 1. Public ranking_metrics + compare_models alias
    public = ranking_metrics(labels, tied_scores)
    assert public["ppv_at_n"] == 0.5
    assert compare_models._metrics(labels, tied_scores)["ppv_at_n"] == 0.5

    # 2. paper_figures
    predictions_csv = tmp_path / "preds.csv"
    pandas.DataFrame({
        "sample_id": ["s1"] * 4,
        "peptide": ["SIINFEKL", "NLVPMVATV", "GILGFVFTL", "SLYNTVATL"],
        "hit": labels,
        "candidate": tied_scores,
    }).to_csv(predictions_csv, index=False)
    predictor_info = pandas.DataFrame([
        {"predictor": "candidate", "higher_is_better": True}
    ]).set_index("predictor", drop=False)
    pf_scores = paper_figures.score_saved_prediction_table(
        predictions_csv, kind="multiallelic", predictor_info=predictor_info)
    pf_ppv = pf_scores.loc[
        (pf_scores["length_label"] == "All") &
        (pf_scores["predictor"] == "candidate")
    ].iloc[0]["ppv"]
    assert pf_ppv == 0.5

    # 3. scripts/training/compare_presentation_percentiles.py
    percentiles_mod = _load_script(
        "scripts/training/compare_presentation_percentiles.py",
        "compare_presentation_percentiles_under_test")
    assert percentiles_mod.ranking_metrics(
        labels, tied_scores)["ppv_expected_random_ties"] == 0.5

    # 4. scripts/training/summarize_presentation_affinity_ensemble.py
    ensemble_mod = _load_script(
        "scripts/training/summarize_presentation_affinity_ensemble.py",
        "summarize_presentation_affinity_ensemble_under_test")
    for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
        assert ensemble_mod.metric_values(
            labels[order], tied_scores[order])["PPV@N"] == 0.5


def test_shared_comparison_helpers_are_single_source():
    assert compare_models._METRIC_DISPLAY_NAMES is METRIC_DISPLAY_NAMES
    assert plot_model_comparison.METRIC_DISPLAY_NAMES is METRIC_DISPLAY_NAMES
    assert compare_models._score_values is score_values
    assert plot_model_comparison._score_values is score_values
    assert paper_figures._path_is_within is path_is_within
    assert plot_model_comparison._path_is_within is path_is_within
    assert not hasattr(paper_figures, "_ppv_at_n")
    assert not hasattr(paper_figures, "_prediction_tie_breaker")

    df = pandas.DataFrame({
        "a_presentation_score": [0.2, 0.8],
        "a_presentation_percentile": [5.0, 0.5],
        "a_processing_score": [0.1, 0.9],
    })
    numpy.testing.assert_array_equal(
        score_values(df, "a", "presentation_score"), [0.2, 0.8])
    numpy.testing.assert_array_equal(
        score_values(df, "a", "presentation_percentile"), [-5.0, -0.5])
    numpy.testing.assert_array_equal(
        score_values(df, "a", "processing_score"), [0.1, 0.9])
    with pytest.raises(ValueError, match="Unknown score kind"):
        score_values(df, "a", "invalid")

    assert path_is_within(Path("/a/b/c"), Path("/a/b"))
    assert not path_is_within(Path("/a/bc"), Path("/a/b"))

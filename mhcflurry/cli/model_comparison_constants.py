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
Shared constants and small helpers for model comparison metrics and plots.
"""

METRIC_NAMES = ("roc_auc", "pr_auc", "ppv_at_n")
METRIC_DISPLAY_NAMES = {
    "roc_auc": "AUROC",
    "pr_auc": "AUPRC",
    "ppv_at_n": "PPV@N",
}
PROCESSING_MODES = ("with_flanks", "no_flank", "short_flanks")
PRESENTATION_MODES = ("with_flanks", "without_flanks")
PRESENTATION_SCORE_KINDS = ("presentation_score", "presentation_percentile")


def score_values(df, prefix, score_kind):
    """Return comparison scores oriented so that higher values are better.

    ``presentation_score`` and ``processing_score`` are already higher-better;
    ``presentation_percentile`` is lower-better so it is negated.
    """
    if score_kind == "presentation_score":
        return df["%s_presentation_score" % prefix].values
    if score_kind == "presentation_percentile":
        return -df["%s_presentation_percentile" % prefix].values
    if score_kind == "processing_score":
        return df["%s_processing_score" % prefix].values
    raise ValueError("Unknown score kind: %s" % score_kind)


def path_is_within(path, directory):
    """Return whether ``path`` is equal to or nested inside ``directory``."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False

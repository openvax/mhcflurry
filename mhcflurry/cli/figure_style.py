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

"""Shared styling helpers for model-comparison and paper figures."""
from __future__ import annotations

import ast


SIDE_A_COLOR = (0.596, 0.557, 0.835)
SIDE_B_COLOR = (0.545, 0.545, 0.545)
POSITIVE_DELTA_COLOR = (0.353, 0.612, 0.518)
NEGATIVE_DELTA_COLOR = (0.886, 0.290, 0.200)
GRID_COLOR = "0.88"
DIAGONAL_COLOR = "0.55"

PREDICTOR_LABELS = {
    "netmhcpan4.ba": "NetMHCpan 4.0 BA",
    "netmhcpan4.el": "NetMHCpan 4.0 EL",
    "netmhcpan4.2.ba": "NetMHCpan 4.2 BA",
    "netmhcpan4.2.el": "NetMHCpan 4.2 EL",
    "mixmhcpred": "MixMHCpred",
    "mhcflurry_production": "MHCflurry BA",
}

PREDICTOR_COLORS = {
    "netmhcpan4.ba": (0.886, 0.290, 0.200),
    "netmhcpan4.el": (1.000, 0.710, 0.722),
    "netmhcpan4.2.ba": (0.650, 0.120, 0.130),
    "netmhcpan4.2.el": (0.980, 0.520, 0.540),
    "mixmhcpred": (0.204, 0.541, 0.741),
    "mhcflurry_production": SIDE_A_COLOR,
}

FALLBACK_PALETTE = (
    (0.345, 0.467, 0.741),
    (0.459, 0.439, 0.702),
    (0.639, 0.400, 0.667),
    (0.871, 0.443, 0.498),
    (0.922, 0.612, 0.357),
    (0.580, 0.690, 0.392),
    POSITIVE_DELTA_COLOR,
    (0.306, 0.573, 0.702),
    SIDE_B_COLOR,
    (0.737, 0.506, 0.741),
)


def apply_paper_style():
    """Apply the shared publication-style matplotlib defaults."""
    import matplotlib.pyplot as plt

    try:
        import seaborn
        seaborn.set_context("paper")
        seaborn.set_style("white")
    except ImportError:
        pass
    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        try:
            plt.style.use("seaborn-white")
        except OSError:
            pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "0.15",
        "axes.linewidth": 0.8,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "text.usetex": False,
    })


def despine(ax):
    """Hide nonessential axes decoration for paper-style panels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def short_label(predictor_info, predictor):
    """Return a concise display label for a predictor identifier."""
    if predictor in predictor_info.index:
        row = predictor_info.loc[predictor]
        for column in ("short", "description"):
            value = row.get(column)
            if isinstance(value, str) and value and value != "-":
                return value
    if predictor in PREDICTOR_LABELS:
        return PREDICTOR_LABELS[predictor]
    return str(predictor).replace("_", " ")


def predictor_color(predictor_info, predictor):
    """Return a stable paper palette color for a predictor identifier."""
    if predictor in predictor_info.index:
        value = predictor_info.loc[predictor].get("color")
        if isinstance(value, str) and value and value.lower() != "nan":
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)) and len(parsed) in (3, 4):
                    return tuple(float(item) for item in parsed)
            except (SyntaxError, ValueError, TypeError):
                pass
    if predictor in PREDICTOR_COLORS:
        return PREDICTOR_COLORS[predictor]
    index = sum(
        (position + 1) * ord(char)
        for position, char in enumerate(str(predictor))
    ) % len(FALLBACK_PALETTE)
    return FALLBACK_PALETTE[index]

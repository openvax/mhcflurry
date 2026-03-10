"""
Generate an HTML parity/error report against cached master-release fixtures.

The report compares the current branch's released affinity and presentation
predictors against fixture data stored under ``test/data``. It writes a
self-contained HTML file with inline SVG plots plus CSV/JSON summaries.

Example:
  ./.venv/bin/python scripts/generate_fixture_error_report.py \
    --out-dir /tmp/mhcflurry-fixture-error-report
"""

from __future__ import annotations

import argparse
import html
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
from mhcflurry.downloads import (
    configure,
    get_current_release,
    get_default_class1_models_dir,
    get_default_class1_presentation_models_dir,
    get_path,
)
from mhcflurry.testing_utils import cleanup, startup


DATA_DIR = Path(__file__).resolve().parents[1] / "test" / "data"
AFFINITY_FIXTURE_PATH = DATA_DIR / "master_released_class1_affinity_predictions.json"
PRESENTATION_FIXTURE_PATH = (
    DATA_DIR / "master_released_class1_presentation_highscore_rows.csv.gz"
)
PRESENTATION_METADATA_PATH = (
    DATA_DIR / "master_released_class1_presentation_highscore_rows_metadata.json"
)
BASE_COLUMNS = ["row_id", "peptide", "allele", "n_flank", "c_flank"]
SVG_NS = "http://www.w3.org/2000/svg"


@dataclass
class MetricReport:
    key: str
    title: str
    section: str
    unit: str
    df: pd.DataFrame
    reference_label: str
    current_label: str
    log_scale: bool = False
    use_relative_histogram: bool = False

    @property
    def summary(self) -> dict:
        error = self.df["error"].to_numpy(dtype=np.float64)
        abs_error = self.df["abs_error"].to_numpy(dtype=np.float64)
        result = {
            "count": int(len(self.df)),
            "mean_error": float(error.mean()) if len(error) else float("nan"),
            "mean_abs_error": float(abs_error.mean()) if len(abs_error) else float("nan"),
            "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else float("nan"),
            "max_abs_error": float(abs_error.max()) if len(abs_error) else float("nan"),
        }
        if "abs_pct_error" in self.df.columns:
            abs_pct = self.df["abs_pct_error"].to_numpy(dtype=np.float64)
            result["mean_abs_pct_error"] = float(abs_pct.mean()) if len(abs_pct) else float("nan")
            result["max_abs_pct_error"] = float(abs_pct.max()) if len(abs_pct) else float("nan")
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="/tmp/mhcflurry-fixture-error-report",
        help="Directory to receive the HTML report, CSVs, and summary JSON.",
    )
    return parser


def _format_number(value: float, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "nan"
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value >= 1e4 or abs_value < 1e-4:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _format_percent(value: float) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "nan"
    return f"{value:.3e}%"


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _clip_positive(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0]
    floor = float(positive.min()) if positive.size else 1e-12
    return np.clip(values, floor, None)


def _make_error_frame(
    base_df: pd.DataFrame,
    reference_column: str,
    current_column: str,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    extra_columns = extra_columns or []
    keep_columns = [c for c in BASE_COLUMNS if c in base_df.columns] + extra_columns
    result = base_df[keep_columns].copy()
    result["reference"] = base_df[reference_column].to_numpy(dtype=np.float64)
    result["current"] = base_df[current_column].to_numpy(dtype=np.float64)
    result["error"] = result["current"] - result["reference"]
    result["abs_error"] = result["error"].abs()
    positive_ref = np.clip(np.abs(result["reference"]), 1e-12, None)
    result["pct_error"] = 100.0 * result["error"] / positive_ref
    result["abs_pct_error"] = result["pct_error"].abs()
    return result.sort_values("abs_error", ascending=False).reset_index(drop=True)


def _load_affinity_fixture() -> dict:
    with AFFINITY_FIXTURE_PATH.open("r") as handle:
        return json.load(handle)


def _load_presentation_fixture() -> tuple[pd.DataFrame, dict]:
    fixture_df = pd.read_csv(PRESENTATION_FIXTURE_PATH, keep_default_na=False)
    with PRESENTATION_METADATA_PATH.open("r") as handle:
        metadata = json.load(handle)
    return fixture_df, metadata


def _predict_current_outputs() -> tuple[dict, pd.DataFrame, dict]:
    configure()
    default_affinity = Class1AffinityPredictor.load(get_default_class1_models_dir())
    allele_specific = Class1AffinityPredictor.load(get_path("models_class1", "models"))
    pan = Class1AffinityPredictor.load(get_path("models_class1_pan", "models.combined"))
    presentation_predictor = Class1PresentationPredictor.load(
        get_default_class1_presentation_models_dir()
    )

    affinity_fixture = _load_affinity_fixture()
    fixture_df, presentation_metadata = _load_presentation_fixture()

    spec_fx = affinity_fixture["allele_specific"]
    pan_fx = affinity_fixture["pan_allele"]
    spec_current = allele_specific.predict(
        peptides=spec_fx["peptides"],
        alleles=spec_fx["alleles"],
    )
    pan_current = pan.predict(
        peptides=pan_fx["peptides"],
        alleles=pan_fx["alleles"],
    )

    spec_df = pd.DataFrame(
        {
            "peptide": spec_fx["peptides"],
            "allele": spec_fx["alleles"],
            "reference": np.asarray(spec_fx["predictions"], dtype=np.float64),
            "current": np.asarray(spec_current, dtype=np.float64),
        }
    )
    spec_df["error"] = spec_df["current"] - spec_df["reference"]
    spec_df["abs_error"] = spec_df["error"].abs()
    spec_df["pct_error"] = 100.0 * spec_df["error"] / np.clip(
        spec_df["reference"].abs(), 1e-12, None
    )
    spec_df["abs_pct_error"] = spec_df["pct_error"].abs()
    spec_df = spec_df.sort_values("abs_error", ascending=False).reset_index(drop=True)

    pan_df = pd.DataFrame(
        {
            "peptide": pan_fx["peptides"],
            "allele": pan_fx["alleles"],
            "reference": np.asarray(pan_fx["predictions"], dtype=np.float64),
            "current": np.asarray(pan_current, dtype=np.float64),
        }
    )
    pan_df["error"] = pan_df["current"] - pan_df["reference"]
    pan_df["abs_error"] = pan_df["error"].abs()
    pan_df["pct_error"] = 100.0 * pan_df["error"] / np.clip(
        pan_df["reference"].abs(), 1e-12, None
    )
    pan_df["abs_pct_error"] = pan_df["pct_error"].abs()
    pan_df = pan_df.sort_values("abs_error", ascending=False).reset_index(drop=True)

    peptides = fixture_df["peptide"].tolist()
    alleles = fixture_df["allele"].tolist()
    n_flanks = fixture_df["n_flank"].tolist()
    c_flanks = fixture_df["c_flank"].tolist()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Downcasting behavior in `replace` is deprecated.*",
            category=FutureWarning,
        )
        affinity_df = default_affinity.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            throw=False,
            include_percentile_ranks=True,
            include_confidence_intervals=True,
            centrality_measure="mean",
            model_kwargs={"batch_size": 4096},
        )

    sample_names = alleles
    allele_map = {allele: [allele] for allele in sorted(set(alleles))}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Downcasting behavior in `replace` is deprecated.*",
            category=FutureWarning,
        )
        pres_with_df = presentation_predictor.predict(
            peptides=peptides,
            alleles=allele_map,
            sample_names=sample_names,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            include_affinity_percentile=True,
            verbose=0,
            throw=True,
        ).sort_values("peptide_num")
        pres_without_df = presentation_predictor.predict(
            peptides=peptides,
            alleles=allele_map,
            sample_names=sample_names,
            n_flanks=None,
            c_flanks=None,
            include_affinity_percentile=True,
            verbose=0,
            throw=True,
        ).sort_values("peptide_num")

    predicted = fixture_df[BASE_COLUMNS].copy()
    predicted["affinity_prediction_current"] = affinity_df["prediction"].to_numpy(
        dtype=np.float64
    )
    predicted["processing_with_score_current"] = pres_with_df["processing_score"].to_numpy(
        dtype=np.float64
    )
    predicted[
        "pres_with_presentation_score_current"
    ] = pres_with_df["presentation_score"].to_numpy(dtype=np.float64)
    predicted["processing_without_score_current"] = pres_without_df[
        "processing_score"
    ].to_numpy(dtype=np.float64)
    predicted[
        "pres_without_presentation_score_current"
    ] = pres_without_df["presentation_score"].to_numpy(dtype=np.float64)

    current_metadata = {
        "release": get_current_release(),
        "presentation_provenance": presentation_predictor.provenance_string,
        "presentation_internal_affinity_provenance": (
            presentation_predictor.affinity_predictor.provenance_string
        ),
    }
    combined_metadata = {
        "affinity_fixture_release": affinity_fixture.get("release"),
        "presentation_fixture": presentation_metadata,
        "current": current_metadata,
    }
    return {
        "released_affinity_allele_specific": spec_df,
        "released_affinity_pan_allele": pan_df,
        "presentation_fixture_predictions": predicted,
    }, fixture_df, combined_metadata


def _compute_metric_reports(
    current_outputs: dict,
    fixture_df: pd.DataFrame,
) -> list[MetricReport]:
    predicted = current_outputs["presentation_fixture_predictions"]
    combined = fixture_df[BASE_COLUMNS].copy()
    for column in [
        "affinity_prediction",
        "processing_with_score",
        "pres_with_presentation_score",
        "processing_without_score",
        "pres_without_presentation_score",
    ]:
        combined[column] = fixture_df[column]
    for column in predicted.columns:
        if column.endswith("_current"):
            combined[column] = predicted[column]

    return [
        MetricReport(
            key="released_affinity_allele_specific",
            title="Released affinity parity: allele-specific models",
            section="Affinity",
            unit="nM",
            df=current_outputs["released_affinity_allele_specific"],
            reference_label="master fixture affinity",
            current_label="current branch affinity",
            log_scale=True,
            use_relative_histogram=True,
        ),
        MetricReport(
            key="released_affinity_pan_allele",
            title="Released affinity parity: pan-allele models",
            section="Affinity",
            unit="nM",
            df=current_outputs["released_affinity_pan_allele"],
            reference_label="master fixture affinity",
            current_label="current branch affinity",
            log_scale=True,
            use_relative_histogram=True,
        ),
        MetricReport(
            key="presentation_fixture_affinity",
            title="Presentation fixture affinity",
            section="Affinity",
            unit="nM",
            df=_make_error_frame(
                combined,
                "affinity_prediction",
                "affinity_prediction_current",
            ),
            reference_label="master fixture affinity",
            current_label="current branch affinity",
            log_scale=True,
            use_relative_histogram=True,
        ),
        MetricReport(
            key="processing_with_score",
            title="Processing score with flanks",
            section="Processing",
            unit="score",
            df=_make_error_frame(
                combined,
                "processing_with_score",
                "processing_with_score_current",
            ),
            reference_label="master fixture processing score",
            current_label="current branch processing score",
        ),
        MetricReport(
            key="processing_without_score",
            title="Processing score without flanks",
            section="Processing",
            unit="score",
            df=_make_error_frame(
                combined,
                "processing_without_score",
                "processing_without_score_current",
            ),
            reference_label="master fixture processing score",
            current_label="current branch processing score",
        ),
        MetricReport(
            key="presentation_with_score",
            title="Presentation score with flanks",
            section="Presentation",
            unit="score",
            df=_make_error_frame(
                combined,
                "pres_with_presentation_score",
                "pres_with_presentation_score_current",
            ),
            reference_label="master fixture presentation score",
            current_label="current branch presentation score",
        ),
        MetricReport(
            key="presentation_without_score",
            title="Presentation score without flanks",
            section="Presentation",
            unit="score",
            df=_make_error_frame(
                combined,
                "pres_without_presentation_score",
                "pres_without_presentation_score_current",
            ),
            reference_label="master fixture presentation score",
            current_label="current branch presentation score",
        ),
    ]


def _axis_range(values: np.ndarray) -> tuple[float, float]:
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    if data_min == data_max:
        pad = 1.0 if data_min == 0.0 else abs(data_min) * 0.05
        return data_min - pad, data_max + pad
    pad = (data_max - data_min) * 0.05
    return data_min - pad, data_max + pad


def _tick_values(start: float, stop: float, count: int = 5) -> list[float]:
    if count < 2:
        return [start]
    return [start + (stop - start) * i / (count - 1) for i in range(count)]


def _format_tick(value: float, log_scale: bool = False) -> str:
    if log_scale:
        return _format_number(10.0 ** value, digits=3)
    return _format_number(value, digits=3)


def _render_scatter_svg(
    df: pd.DataFrame,
    title: str,
    log_scale: bool,
    x_label: str,
    y_label: str,
) -> str:
    width = 520
    height = 360
    margin_left = 70
    margin_right = 24
    margin_top = 36
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x = df["reference"].to_numpy(dtype=np.float64)
    y = df["current"].to_numpy(dtype=np.float64)
    if log_scale:
        x = np.log10(_clip_positive(x))
        y = np.log10(_clip_positive(y))

    combined = np.concatenate([x, y])
    x_min, x_max = _axis_range(combined)
    y_min, y_max = x_min, x_max

    def scale_x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdfa" rx="18" />',
        f'<text x="{margin_left}" y="22" fill="#1f2937" font-size="16" font-weight="700">{html.escape(title)}</text>',
    ]

    for tick in _tick_values(x_min, x_max):
        x_pos = scale_x(tick)
        y_pos = scale_y(tick)
        parts.append(
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" y2="{margin_top + plot_height}" stroke="#ece4d8" stroke-width="1" />'
        )
        parts.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{margin_left + plot_width}" y2="{y_pos:.2f}" stroke="#ece4d8" stroke-width="1" />'
        )
        tick_label = html.escape(_format_tick(tick, log_scale=log_scale))
        parts.append(
            f'<text x="{x_pos:.2f}" y="{height - 18}" fill="#6b7280" font-size="11" text-anchor="middle">{tick_label}</text>'
        )
        parts.append(
            f'<text x="56" y="{y_pos + 4:.2f}" fill="#6b7280" font-size="11" text-anchor="end">{tick_label}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top}" stroke="#9ca3af" stroke-width="2" stroke-dasharray="6 5" />'
    )
    parts.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#9a8f83" stroke-width="1.5" />'
    )

    for x_val, y_val in zip(x, y):
        parts.append(
            f'<circle cx="{scale_x(float(x_val)):.2f}" cy="{scale_y(float(y_val)):.2f}" r="4.2" fill="#0f766e" fill-opacity="0.8" />'
        )

    parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 4}" fill="#374151" font-size="12" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="18" y="{margin_top + plot_height / 2:.2f}" fill="#374151" font-size="12" text-anchor="middle" transform="rotate(-90 18 {margin_top + plot_height / 2:.2f})">{html.escape(y_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _render_histogram_svg(
    values: np.ndarray,
    title: str,
    x_label: str,
    color: str,
) -> str:
    width = 520
    height = 280
    margin_left = 70
    margin_right = 24
    margin_top = 36
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        finite_values = np.array([0.0], dtype=np.float64)

    bound = float(np.max(np.abs(finite_values)))
    if bound == 0.0:
        bound = 1e-12
    bins = min(24, max(8, int(math.sqrt(finite_values.size)) * 2))
    hist, edges = np.histogram(finite_values, bins=bins, range=(-bound, bound))
    y_max = max(int(hist.max()), 1)

    def scale_x(value: float) -> float:
        return margin_left + (value + bound) / (2.0 * bound) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + plot_height - value / y_max * plot_height

    bar_width = plot_width / bins
    parts = [
        f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdfa" rx="18" />',
        f'<text x="{margin_left}" y="22" fill="#1f2937" font-size="16" font-weight="700">{html.escape(title)}</text>',
    ]

    for tick in _tick_values(-bound, bound):
        x_pos = scale_x(tick)
        parts.append(
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" y2="{margin_top + plot_height}" stroke="#ece4d8" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{height - 18}" fill="#6b7280" font-size="11" text-anchor="middle">{html.escape(_format_number(tick, digits=3))}</text>'
        )

    for tick in _tick_values(0.0, float(y_max), count=5):
        y_pos = scale_y(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{margin_left + plot_width}" y2="{y_pos:.2f}" stroke="#ece4d8" stroke-width="1" />'
        )
        parts.append(
            f'<text x="56" y="{y_pos + 4:.2f}" fill="#6b7280" font-size="11" text-anchor="end">{int(round(tick))}</text>'
        )

    parts.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#9a8f83" stroke-width="1.5" />'
    )
    for i, count in enumerate(hist):
        x_pos = margin_left + i * bar_width + 1.5
        y_pos = scale_y(float(count))
        bar_height = margin_top + plot_height - y_pos
        parts.append(
            f'<rect x="{x_pos:.2f}" y="{y_pos:.2f}" width="{max(bar_width - 3.0, 1.0):.2f}" height="{bar_height:.2f}" fill="{color}" fill-opacity="0.85" />'
        )

    zero_x = scale_x(0.0)
    parts.append(
        f'<line x1="{zero_x:.2f}" y1="{margin_top}" x2="{zero_x:.2f}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5" stroke-dasharray="4 4" />'
    )
    parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 4}" fill="#374151" font-size="12" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="18" y="{margin_top + plot_height / 2:.2f}" fill="#374151" font-size="12" text-anchor="middle" transform="rotate(-90 18 {margin_top + plot_height / 2:.2f})">count</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _render_summary_table(reports: list[MetricReport]) -> str:
    rows = []
    for report in reports:
        summary = report.summary
        mean_abs_pct = summary.get("mean_abs_pct_error")
        max_abs_pct = summary.get("max_abs_pct_error")
        rows.append(
            "<tr>"
            f"<td>{html.escape(report.section)}</td>"
            f"<td>{html.escape(report.title)}</td>"
            f"<td>{summary['count']}</td>"
            f"<td>{html.escape(_format_number(summary['mean_abs_error']))}</td>"
            f"<td>{html.escape(_format_number(summary['max_abs_error']))}</td>"
            f"<td>{html.escape(_format_number(summary['rmse']))}</td>"
            f"<td>{html.escape(_format_percent(mean_abs_pct)) if mean_abs_pct is not None else '-'}</td>"
            f"<td>{html.escape(_format_percent(max_abs_pct)) if max_abs_pct is not None else '-'}</td>"
            "</tr>"
        )
    return (
        "<table class='summary-table'>"
        "<thead><tr><th>Section</th><th>Metric</th><th>N</th>"
        "<th>Mean abs error</th><th>Max abs error</th><th>RMSE</th>"
        "<th>Mean abs pct error</th><th>Max abs pct error</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_top_error_table(report: MetricReport, limit: int = 10) -> str:
    columns = [c for c in ["row_id", "peptide", "allele", "n_flank", "c_flank"] if c in report.df]
    columns += ["reference", "current", "error", "abs_error"]
    if "abs_pct_error" in report.df.columns:
        columns.append("abs_pct_error")
    subset = report.df[columns].head(limit)
    header = "".join(f"<th>{html.escape(col)}</th>" for col in subset.columns)
    body_rows = []
    for _, row in subset.iterrows():
        cells = []
        for col in subset.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                if col == "abs_pct_error":
                    formatted = _format_percent(float(value))
                else:
                    formatted = _format_number(float(value))
            else:
                formatted = str(value)
            cells.append(f"<td>{html.escape(formatted)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table class='detail-table'>"
        f"<thead><tr>{header}</tr></thead>"
        "<tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def _render_metric_section(report: MetricReport) -> str:
    hist_values = (
        report.df["pct_error"].to_numpy(dtype=np.float64)
        if report.use_relative_histogram
        else report.df["error"].to_numpy(dtype=np.float64)
    )
    hist_label = "percent error (%)" if report.use_relative_histogram else "signed error"
    scatter_svg = _render_scatter_svg(
        report.df,
        title=report.title,
        log_scale=report.log_scale,
        x_label=report.reference_label + (" (log10)" if report.log_scale else ""),
        y_label=report.current_label + (" (log10)" if report.log_scale else ""),
    )
    hist_svg = _render_histogram_svg(
        hist_values,
        title=report.title + " error distribution",
        x_label=hist_label,
        color="#c2410c" if report.use_relative_histogram else "#2563eb",
    )
    summary = report.summary
    metric_blurb = (
        f"N={summary['count']}, mean abs error={_format_number(summary['mean_abs_error'])} {report.unit}, "
        f"max abs error={_format_number(summary['max_abs_error'])} {report.unit}, "
        f"RMSE={_format_number(summary['rmse'])} {report.unit}."
    )
    if report.use_relative_histogram:
        metric_blurb += (
            " Mean abs pct error="
            + _format_percent(summary["mean_abs_pct_error"])
            + ", max abs pct error="
            + _format_percent(summary["max_abs_pct_error"])
            + "."
        )
    return (
        "<section class='metric-section'>"
        f"<h3>{html.escape(report.title)}</h3>"
        f"<p class='metric-blurb'>{html.escape(metric_blurb)}</p>"
        "<div class='plot-grid'>"
        f"<div class='plot-card'>{scatter_svg}</div>"
        f"<div class='plot-card'>{hist_svg}</div>"
        "</div>"
        "<div class='table-wrap'>"
        "<h4>Largest absolute errors</h4>"
        f"{_render_top_error_table(report)}"
        "</div>"
        "</section>"
    )


def _write_outputs(
    out_dir: Path,
    reports: list[MetricReport],
    metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    summary_json = {
        "generated_at": datetime.now().isoformat(),
        "metadata": metadata,
        "metrics": {report.key: report.summary for report in reports},
    }
    for report in reports:
        report.df.to_csv(csv_dir / f"{report.key}.csv", index=False)

    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary_json, handle, indent=2, sort_keys=True)

    sections = []
    grouped_reports = {
        "Affinity": [r for r in reports if r.section == "Affinity"],
        "Processing": [r for r in reports if r.section == "Processing"],
        "Presentation": [r for r in reports if r.section == "Presentation"],
    }
    for section_name, section_reports in grouped_reports.items():
        sections.append(
            f"<section class='major-section'><h2>{html.escape(section_name)}</h2>"
            + "".join(_render_metric_section(report) for report in section_reports)
            + "</section>"
        )

    notes = []
    affinity_fixture_release = metadata.get("affinity_fixture_release")
    current_release = metadata["current"]["release"]
    if affinity_fixture_release and affinity_fixture_release != current_release:
        notes.append(
            "Affinity fixture release "
            + str(affinity_fixture_release)
            + " does not match current downloads release "
            + str(current_release)
            + "."
        )
    presentation_fixture = metadata.get("presentation_fixture", {})
    if (
        presentation_fixture.get("release")
        and presentation_fixture.get("release") != current_release
    ):
        notes.append(
            "Presentation fixture release "
            + str(presentation_fixture.get("release"))
            + " does not match current downloads release "
            + str(current_release)
            + "."
        )
    if (
        presentation_fixture.get("presentation_provenance")
        and presentation_fixture.get("presentation_provenance")
        != metadata["current"]["presentation_provenance"]
    ):
        notes.append("Presentation predictor provenance differs from fixture metadata.")
    if (
        presentation_fixture.get("presentation_internal_affinity_provenance")
        and presentation_fixture.get("presentation_internal_affinity_provenance")
        != metadata["current"]["presentation_internal_affinity_provenance"]
    ):
        notes.append("Internal affinity provenance differs from fixture metadata.")
    note_html = (
        "<div class='notes'>"
        + "".join(f"<p>{html.escape(note)}</p>" for note in notes)
        + "</div>"
        if notes
        else "<div class='notes ok'><p>Fixture metadata matches the current downloaded release and predictor provenance.</p></div>"
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MHCflurry fixture error report</title>
  <style>
    :root {{
      --bg: #f5efe3;
      --panel: #fffdfa;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d6c7b5;
      --accent: #0f766e;
      --warn: #b45309;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(194, 65, 12, 0.10), transparent 28%),
        var(--bg);
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    h1, h2, h3, h4 {{
      margin: 0 0 12px;
      font-family: "Avenir Next Condensed", "Gill Sans", sans-serif;
      letter-spacing: 0.02em;
    }}
    h1 {{
      font-size: 40px;
      line-height: 1;
      margin-bottom: 10px;
    }}
    p {{
      margin: 0 0 12px;
      line-height: 1.5;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,253,250,0.96), rgba(254,245,231,0.95));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px 28px 22px;
      box-shadow: 0 16px 48px rgba(60, 49, 33, 0.08);
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    .notes {{
      margin-top: 18px;
      background: rgba(180, 83, 9, 0.08);
      border: 1px solid rgba(180, 83, 9, 0.20);
      border-radius: 16px;
      padding: 14px 16px 2px;
    }}
    .notes.ok {{
      background: rgba(15, 118, 110, 0.08);
      border-color: rgba(15, 118, 110, 0.20);
    }}
    .summary-table, .detail-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      font-size: 14px;
    }}
    .summary-table th, .summary-table td,
    .detail-table th, .detail-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid #efe4d6;
      text-align: left;
      vertical-align: top;
    }}
    .summary-table th, .detail-table th {{
      background: #f9f1e5;
      font-family: "Avenir Next Condensed", "Gill Sans", sans-serif;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .major-section {{
      margin-top: 34px;
    }}
    .metric-section {{
      margin-top: 18px;
      background: rgba(255,253,250,0.72);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 12px 32px rgba(60, 49, 33, 0.06);
    }}
    .metric-blurb {{
      color: var(--muted);
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin: 16px 0 18px;
    }}
    .plot-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 10px;
      overflow-x: auto;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    code {{
      background: rgba(15, 23, 42, 0.06);
      border-radius: 6px;
      padding: 1px 6px;
      font-size: 0.95em;
    }}
    @media (max-width: 720px) {{
      main {{
        padding: 18px 12px 32px;
      }}
      .hero {{
        padding: 20px 18px 16px;
      }}
      h1 {{
        font-size: 32px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>MHCflurry Fixture Error Report</h1>
      <p>This report compares cached master-branch fixtures in <code>test/data</code> against predictions from the current branch's downloaded released models. Affinity uses the released-model JSON fixture; processing and presentation use the curated high-score presentation CSV fixture.</p>
      <p class="meta">Generated {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}. Current release: {html.escape(str(current_release))}. Output dir: {html.escape(str(out_dir))}.</p>
      {note_html}
    </section>
    <section class="major-section">
      <h2>Summary</h2>
      {_render_summary_table(reports)}
    </section>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    startup()
    try:
        current_outputs, fixture_df, metadata = _predict_current_outputs()
        reports = _compute_metric_reports(current_outputs, fixture_df)
        _write_outputs(out_dir, reports, metadata)
        print("Wrote HTML report:", out_dir / "index.html")
        print("Wrote summary JSON:", out_dir / "summary.json")
        print("Wrote CSV directory:", out_dir / "csv")
    finally:
        cleanup()


if __name__ == "__main__":
    main()

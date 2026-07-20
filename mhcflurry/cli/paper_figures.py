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

"""Generate paper-style figures from retraining/evaluation outputs.

This command ports the figure families from the 2023 retraining notebooks
into a reproducible CLI. It reads saved evaluation tables: raw saved
prediction tables such as ``benchmark.multiallelic.csv.bz2``, derived score
tables such as ``accuracy_scores.multiallelic.csv``, and the current
``compare-models`` output directory when supplied. Missing inputs are written
to ``missing_inputs.md`` and ``manifest.csv`` so a training run can distinguish
"not generated because the data is absent" from "plotting silently drifted."
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy
import pandas

from .figure_style import (
    SIDE_A_COLOR,
    SIDE_B_COLOR,
    apply_paper_style as _apply_paper_style,
    despine as _despine,
    predictor_color as _predictor_color,
    short_label as _short_label,
)
from ..common import allele_locus_name


CANDIDATE_PREDICTOR = "mhcflurry_production"

DEFAULT_PREDICTOR_HIGHER_IS_BETTER = {
    "affinity": False,
    "affinity_percentile": False,
    "processing_score": True,
    "presentation_score": True,
    "presentation_percentile": False,
    "mhcflurry_production": False,
    "mhcflurry_production_affinity": False,
    "mhcflurry_production_percentile": False,
    "netmhcpan4.ba": False,
    "netmhcpan4.ba_affinity": False,
    "netmhcpan4.el": True,
    "netmhcpan4.el_rank": False,
    "netmhcpan4.el_percentile": False,
    "netmhcpan4.2.ba": False,
    "netmhcpan4.2.ba_affinity": False,
    "netmhcpan4.2.el": True,
    "netmhcpan4.2.el_rank": False,
    "netmhcpan4.2.el_percentile": False,
    "mixmhcpred": True,
    "presentation_with_flanks_presentation_score": True,
    "presentation_without_flanks_presentation_score": True,
    "presentation_with_flanks_processing_score": True,
    "presentation_without_flanks_processing_score": True,
    "presentation_with_flanks_affinity": False,
    "presentation_without_flanks_affinity": False,
    "presentation_with_flanks_presentation_percentile": False,
    "presentation_without_flanks_presentation_percentile": False,
}

EXTERNAL_BASELINES = (
    ("netmhcpan4.ba", "ba"),
    ("netmhcpan4.el", "el"),
    ("mixmhcpred", "mixmhcpred"),
)

PRESENTATION_PANEL_PREDICTORS = (
    "presentation_without_flanks_presentation_score",
    "presentation_with_flanks_presentation_score",
)

PRESENTATION_PANEL_BASELINES = (
    "netmhcpan4.ba",
    "netmhcpan4.el",
    "mixmhcpred",
    "mhcflurry_production",
)

PREFERRED_PREDICTORS = (
    "netmhcpan4.ba",
    "netmhcpan4.el",
    "mixmhcpred",
    "mhcflurry_production",
    "presentation_without_flanks_presentation_score",
    "presentation_with_flanks_presentation_score",
    "presentation_without_flanks_processing_score",
    "presentation_with_flanks_processing_score",
)

LENGTH_LABEL_ORDER = ("All", "8-mer", "9-mer", "10-mer", "11-mer")
DEFAULT_FORMATS = ("svg", "pdf", "png")


@dataclass(frozen=True)
class PredictorConfig:
    candidate: str
    external_baselines: tuple
    preferred_predictors: tuple
    presentation_panel_predictors: tuple
    presentation_panel_baselines: tuple


@dataclass(frozen=True)
class FigureInputs:
    scores_dir: Path
    comparison_dir: Optional[Path]
    run_dir: Optional[Path]
    multiallelic_predictions: Optional[Path]
    monoallelic_predictions: Optional[Path]


def make_parser():
    """Return a standalone parser for documentation tooling."""
    parser = argparse.ArgumentParser(prog="mhcflurry paper-figures")
    register_subparser(parser)
    return parser


def run_argv(argv):
    """Entry point for the lazy ``mhcflurry paper-figures`` dispatcher."""
    return run(make_parser().parse_args(argv))


def register_subparser(parser):
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = """
Figure input contract:
  * comparison-dir: output from ``mhcflurry eval compare-models``.
  * scores-dir: reusable paper-figure inputs. Common files are
    accuracy_scores.multiallelic.csv, accuracy_scores.monoallelic.csv,
    benchmark.multiallelic.csv(.bz2), benchmark.monoallelic.csv(.bz2),
    sample_table.csv, and predictor_info.csv.
  * saved prediction tables: include hit, sample_id or allele/hla, optional
    peptide metadata, and canonical or explicitly declared score columns.
  * score direction is explicit. Built-in predictor names have defaults; custom
    score columns require predictor_info.csv rows with predictor and
    higher_is_better.

Missing optional figure inputs are recorded in manifest.csv and
missing_inputs.md instead of being silently fabricated.
"""
    parser.add_argument(
        "--scores-dir",
        help=(
            "Directory containing saved figure inputs such as "
            "accuracy_scores.multiallelic.csv, benchmark.multiallelic.csv.bz2, "
            "and predictor_info.csv. Custom predictor rows in predictor_info.csv "
            "should include higher_is_better."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        help=(
            "Compatibility alias for --scores-dir. Prefer --scores-dir in "
            "new scripts."
        ),
    )
    parser.add_argument(
        "--comparison-dir",
        help=(
            "Optional directory produced by ``mhcflurry compare-models``. "
            "When provided, paper-figures generates current-run panels from "
            "fresh MHCflurry side-A vs side-B evaluation outputs instead of "
            "requiring pre-derived score tables."
        ),
    )
    parser.add_argument(
        "--multiallelic-predictions",
        help=(
            "Optional saved multiallelic test-set prediction table. If "
            "accuracy_scores.multiallelic.csv is absent, paper-figures derives "
            "per-sample AUC/PPV tables from this file. Default: "
            "<scores-dir>/benchmark.multiallelic.csv.bz2 when present."
        ),
    )
    parser.add_argument(
        "--monoallelic-predictions",
        help=(
            "Optional saved monoallelic test-set prediction table used to "
            "derive monoallelic AUC/PPV plots when "
            "accuracy_scores.monoallelic.csv is absent."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for paper figures and manifest files.",
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help=(
            "Comma-separated output formats. Default: %(default)s. Use SVG/PDF "
            "for publication; PNG is for quick review."
        ),
    )
    parser.add_argument(
        "--combined-pdf",
        default=None,
        help=(
            "Optional multi-page PDF path. Default: <out>/paper_figures.pdf. "
            "Pass 'none' to skip."
        ),
    )
    parser.add_argument(
        "--sample-table",
        help=(
            "Optional sample table with sample_id and sample_group columns. "
            "When present, panels that were recent-sample-only in the 2023 "
            "notebooks use --sample-group."
        ),
    )
    parser.add_argument(
        "--sample-group",
        default="MULTIALLELIC-RECENT",
        help="Sample group for recent-only multiallelic panels.",
    )
    parser.add_argument(
        "--candidate-predictor",
        default=CANDIDATE_PREDICTOR,
        help=(
            "Predictor to treat as the candidate MHCflurry model in "
            "notebook-style comparison panels. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--external-baselines",
        default=",".join(
            "%s:%s" % (predictor, suffix)
            for predictor, suffix in EXTERNAL_BASELINES),
        help=(
            "Comma-separated external predictor comparators. Each item is "
            "PREDICTOR or PREDICTOR:PERCENT_CHANGE_SUFFIX. Default: "
            "%(default)s."
        ),
    )
    parser.add_argument(
        "--preferred-predictors",
        default=",".join(PREFERRED_PREDICTORS),
        help=(
            "Comma-separated predictors for summary bar panels. Default: "
            "%(default)s."
        ),
    )
    parser.add_argument(
        "--presentation-panel-predictors",
        default=",".join(PRESENTATION_PANEL_PREDICTORS),
        help=(
            "Comma-separated candidate predictors for presentation-vs-baseline "
            "scatter grids. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--presentation-panel-baselines",
        default=",".join(PRESENTATION_PANEL_BASELINES),
        help=(
            "Comma-separated baseline predictors for presentation-vs-baseline "
            "scatter grids. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=20_000,
        help="Subsample scatter plots above this many points.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Return a non-zero exit code if any requested non-metadata figure "
            "family skips or fails."
        ),
    )
    return parser


def run(args):
    import matplotlib
    matplotlib.use("Agg")

    out_dir = Path(args.out)
    formats = _parse_formats(args.formats)
    combined_pdf = _combined_pdf_path(args.combined_pdf, out_dir)
    writer = FigureWriter(out_dir, formats, combined_pdf)
    inputs = _resolve_figure_inputs(args, writer)
    if inputs is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(out_dir, writer.rows)
        _write_missing_inputs(out_dir, writer.rows)
        return 2

    cleanup_conflicts = _paper_figure_cleanup_input_conflicts(
        args, inputs, out_dir, combined_pdf)
    if cleanup_conflicts:
        raise SystemExit(
            "Paper-figure output cleanup would delete input path(s): %s. "
            "Choose a separate --out directory." %
            ", ".join(str(path) for path in cleanup_conflicts)
        )

    _clear_paper_figure_outputs(out_dir, combined_pdf)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            predictors = _parse_predictor_config(args)
        except ValueError as e:
            writer.fail("configuration", "predictor_config", str(e))
            return 2

        _apply_paper_style()
        predictor_info = _read_predictor_info(
            inputs.scores_dir / "predictor_info.csv", writer)
        sample_ids = _read_sample_group_ids(args, inputs, writer)
        _run_figure_family(
            writer, "multiallelic", "all", _generate_multiallelic_figures,
            inputs, predictor_info, sample_ids, args.sample_group,
            args.max_scatter_points, writer, predictors)
        _run_figure_family(
            writer, "model-selection", "all",
            _generate_model_selection_figures,
            inputs, writer)
        _run_figure_family(
            writer, "monoallelic", "all", _generate_monoallelic_figures,
            inputs, predictor_info,
            args.max_scatter_points, writer, predictors)
        _run_figure_family(
            writer, "antigen-processing", "all",
            _generate_processing_notebook_figures,
            inputs, predictor_info, writer, predictors)
        _run_figure_family(
            writer, "proteasome", "all",
            _generate_proteasome_figures, inputs, writer)
        _run_figure_family(
            writer, "architecture", "all",
            _copy_architecture_figures, inputs, writer)
    finally:
        writer.close()
        _write_manifest(out_dir, writer.rows)
        _write_missing_inputs(out_dir, writer.rows)

    if args.strict and any(_strict_failure_row(row) for row in writer.rows):
        return 2
    return 0


def _strict_failure_row(row):
    if row["status"] == "failed":
        return True
    if row["status"] != "skipped":
        return False
    return row["family"] != "metadata"


def _run_figure_family(writer, family, figure, func, *args):
    try:
        func(*args)
    except Exception as e:
        writer.fail(
            family,
            figure,
            "%s: %s" % (type(e).__name__, e),
        )


class FigureWriter:
    """Write one plot to SVG/PDF/PNG and track the output manifest."""

    def __init__(self, out_dir, formats, combined_pdf):
        self.out_dir = Path(out_dir)
        self.formats = tuple(formats)
        self.combined_pdf = Path(combined_pdf) if combined_pdf else None
        self.pdf_pages = None
        self.rows = []

    def save(self, fig, name, family, note=""):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        paths = []
        for fmt in self.formats:
            fmt_dir = self.out_dir / fmt
            fmt_dir.mkdir(parents=True, exist_ok=True)
            path = fmt_dir / ("%s.%s" % (name, fmt))
            fig.savefig(path, transparent=True, bbox_inches="tight")
            paths.append(str(path))
        if self.combined_pdf:
            self.combined_pdf.parent.mkdir(parents=True, exist_ok=True)
            if self.pdf_pages is None:
                self.pdf_pages = PdfPages(self.combined_pdf)
            self.pdf_pages.savefig(fig, transparent=True, bbox_inches="tight")
        plt.close(fig)
        self.rows.append({
            "family": family,
            "figure": name,
            "status": "generated",
            "paths": ";".join(paths),
            "note": note,
            "missing": "",
        })

    def skip(self, family, figure, missing, note):
        if isinstance(missing, (list, tuple)):
            missing = ";".join(str(item) for item in missing)
        self.rows.append({
            "family": family,
            "figure": figure,
            "status": "skipped",
            "paths": "",
            "note": note,
            "missing": str(missing),
        })

    def fail(self, family, figure, note):
        self.rows.append({
            "family": family,
            "figure": figure,
            "status": "failed",
            "paths": "",
            "note": note,
            "missing": "",
        })

    def close(self):
        if self.pdf_pages is not None:
            self.pdf_pages.close()


def _parse_formats(value):
    formats = tuple(
        part.strip().lower() for part in value.split(",") if part.strip())
    if not formats:
        raise ValueError("--formats must contain at least one format")
    allowed = {"svg", "pdf", "png"}
    unknown = set(formats) - allowed
    if unknown:
        raise ValueError(
            "Unsupported figure formats: %s. Allowed: %s" % (
                ", ".join(sorted(unknown)), ", ".join(sorted(allowed))))
    return formats


def _parse_predictor_config(args):
    candidate = args.candidate_predictor.strip()
    if not candidate:
        raise ValueError("--candidate-predictor must be non-empty")
    presentation_panel_predictors = _parse_predictor_list(
        args.presentation_panel_predictors)
    if not presentation_panel_predictors:
        raise ValueError(
            "--presentation-panel-predictors must contain at least one "
            "predictor")
    presentation_panel_baselines = _parse_predictor_list(
        args.presentation_panel_baselines)
    if not presentation_panel_baselines:
        raise ValueError(
            "--presentation-panel-baselines must contain at least one "
            "predictor")
    return PredictorConfig(
        candidate=candidate,
        external_baselines=_parse_external_baselines(args.external_baselines),
        preferred_predictors=_parse_predictor_list(args.preferred_predictors),
        presentation_panel_predictors=presentation_panel_predictors,
        presentation_panel_baselines=presentation_panel_baselines,
    )


def _parse_predictor_list(value):
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_external_baselines(value):
    result = []
    for part in _parse_predictor_list(value):
        if ":" in part:
            predictor, suffix = part.split(":", 1)
        else:
            predictor = part
            suffix = part.rsplit(".", 1)[-1]
        predictor = predictor.strip()
        suffix = suffix.strip()
        if predictor:
            result.append((predictor, suffix))
    if not result:
        raise ValueError("--external-baselines must contain at least one predictor")
    return tuple(result)


def _combined_pdf_path(value, out_dir):
    if value == "none":
        return None
    if value:
        return value
    return str(Path(out_dir) / "paper_figures.pdf")


def _clear_paper_figure_outputs(out_dir, combined_pdf):
    """Remove outputs owned by this command before a clean rerender."""
    out_dir = Path(out_dir)
    if out_dir.is_dir():
        for name in (*DEFAULT_FORMATS, "assets"):
            path = out_dir / name
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        for path in out_dir.glob("*.pdf"):
            path.unlink()
        for name in ("manifest.csv", "missing_inputs.md"):
            path = out_dir / name
            if path.exists() or path.is_symlink():
                path.unlink()
    if combined_pdf is not None:
        combined_pdf = Path(combined_pdf)
        if combined_pdf.exists() or combined_pdf.is_symlink():
            combined_pdf.unlink()


def _paper_figure_cleanup_input_conflicts(
        args, inputs, out_dir, combined_pdf):
    """Return inputs that command-owned cleanup would remove."""
    out_dir = Path(out_dir).resolve()
    cleanup_dirs = [
        (out_dir / name).resolve()
        for name in (*DEFAULT_FORMATS, "assets")
    ]
    cleanup_files = {
        (out_dir / name).resolve()
        for name in ("manifest.csv", "missing_inputs.md")
    }
    if combined_pdf is not None:
        cleanup_files.add(Path(combined_pdf).resolve())

    input_dirs = [
        path for path in (
            inputs.scores_dir,
            inputs.comparison_dir,
            inputs.run_dir,
        ) if path is not None
    ]
    input_files = [
        path for path in (
            inputs.multiallelic_predictions,
            inputs.monoallelic_predictions,
            Path(args.sample_table) if args.sample_table else None,
        ) if path is not None
    ]

    conflicts = []
    for path in input_dirs:
        resolved = Path(path).resolve()
        if resolved == out_dir or any(
                _path_is_within(resolved, directory)
                for directory in cleanup_dirs):
            conflicts.append(resolved)
    for path in input_files:
        resolved = Path(path).resolve()
        if (
                resolved in cleanup_files or
                any(_path_is_within(resolved, directory)
                    for directory in cleanup_dirs) or
                (resolved.parent == out_dir and resolved.suffix == ".pdf")):
            conflicts.append(resolved)
    return tuple(dict.fromkeys(conflicts))


def _path_is_within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _resolve_figure_inputs(args, writer):
    scores_value = args.scores_dir or args.artifacts_dir
    comparison_dir = _resolve_comparison_dir(
        args.comparison_dir,
        Path(scores_value) if scores_value else None,
    )
    multiallelic_predictions = (
        Path(args.multiallelic_predictions)
        if args.multiallelic_predictions
        else None
    )
    monoallelic_predictions = (
        Path(args.monoallelic_predictions)
        if args.monoallelic_predictions
        else None
    )

    if scores_value:
        scores_dir = Path(scores_value)
    elif multiallelic_predictions:
        scores_dir = multiallelic_predictions.parent
    elif monoallelic_predictions:
        scores_dir = monoallelic_predictions.parent
    elif comparison_dir is not None:
        scores_dir = comparison_dir
    else:
        writer.fail(
            "configuration",
            "inputs",
            (
                "Specify --scores-dir, --artifacts-dir, --comparison-dir, "
                "--multiallelic-predictions, or --monoallelic-predictions."
            ),
        )
        return None

    if multiallelic_predictions is None:
        default_path = scores_dir / "benchmark.multiallelic.csv.bz2"
        if default_path.is_file():
            multiallelic_predictions = default_path
    if monoallelic_predictions is None:
        for name in (
                "benchmark.monoallelic.csv.bz2",
                "benchmark.monoallelic.train_excluded.csv.bz2"):
            default_path = scores_dir / name
            if default_path.is_file():
                monoallelic_predictions = default_path
                break

    return FigureInputs(
        scores_dir=scores_dir,
        comparison_dir=comparison_dir,
        run_dir=_resolve_run_dir(comparison_dir),
        multiallelic_predictions=multiallelic_predictions,
        monoallelic_predictions=monoallelic_predictions,
    )


def _resolve_comparison_dir(value, scores_dir):
    if value:
        return Path(value)
    if (
            scores_dir is not None and
            (scores_dir / "release_summary.csv").is_file() and
            (scores_dir / "side_a.json").is_file() and
            (scores_dir / "side_b.json").is_file()):
        return scores_dir
    return None


def _resolve_run_dir(comparison_dir):
    if comparison_dir is None:
        return None
    comparison_dir = Path(comparison_dir)
    if comparison_dir.name == "eval_comparison":
        return comparison_dir.parent
    parent = comparison_dir.parent
    if any(
            (parent / name).exists()
            for name in ("affinity", "processing", "presentation")):
        return parent
    return None


def _current_affinity_per_allele(comparison_dir):
    if comparison_dir is None:
        return None
    path = Path(comparison_dir) / "affinity" / "per_allele.csv"
    if not path.is_file():
        return None
    df = pandas.read_csv(path)
    required = {
        "allele", "n", "n_pos", "a_roc_auc", "b_roc_auc",
        "a_ppv_at_n", "b_ppv_at_n",
    }
    if not required.issubset(df.columns):
        return None
    return df


def _comparison_labels(comparison_dir):
    result = {"a": "Side A", "b": "Side B"}
    if comparison_dir is None:
        return result
    for side in ("a", "b"):
        path = Path(comparison_dir) / ("side_%s.json" % side)
        if not path.is_file():
            continue
        try:
            with open(path) as fd:
                loaded = json.load(fd)
            label = loaded.get("label")
            if label:
                result[side] = str(label)
        except (OSError, ValueError, TypeError):
            pass
    return result


def _read_predictor_info(path, writer):
    if not path.is_file():
        writer.skip(
            "metadata", "predictor_info",
            [path],
            "Predictor labels/colors unavailable; using fallback labels.")
        return pandas.DataFrame(columns=[
            "predictor", "description", "primary", "color", "short", "detail"])
    df = pandas.read_csv(path)
    if "predictor" not in df.columns:
        writer.skip(
            "metadata", "predictor_info",
            [path],
            "predictor_info.csv lacks a predictor column; using fallback labels.")
        return pandas.DataFrame(columns=[
            "predictor", "description", "primary", "color", "short", "detail"])
    return df.set_index("predictor", drop=False)


def _coerce_optional_bool(value):
    if isinstance(value, bool):
        return value
    if pandas.isnull(value):
        return None
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return None


def _predictor_orientations(predictor_info):
    result = dict(DEFAULT_PREDICTOR_HIGHER_IS_BETTER)
    if (
            predictor_info is not None and
            not predictor_info.empty and
            "higher_is_better" in predictor_info.columns):
        for predictor, row in predictor_info.iterrows():
            if "predictor" in predictor_info.columns:
                predictor = row.get("predictor", predictor)
            value = _coerce_optional_bool(row.get("higher_is_better"))
            if value is None:
                continue
            result[str(predictor)] = value
            normalized = _normalize_predictor_name(predictor)
            result[normalized] = value
    return result


def _read_sample_group_ids(args, inputs, writer):
    artifacts_dir = inputs.scores_dir
    path = Path(args.sample_table) if args.sample_table else (
        artifacts_dir / "sample_table.csv")
    if not path.is_file():
        benchmark_paths = []
        if inputs.multiallelic_predictions is not None:
            benchmark_paths.append(Path(inputs.multiallelic_predictions))
        benchmark_paths.extend([
            artifacts_dir / "benchmark.multiallelic.csv.bz2",
            artifacts_dir / "benchmark.multiallelic.csv",
        ])
        seen = set()
        for benchmark_path in benchmark_paths:
            if benchmark_path in seen:
                continue
            seen.add(benchmark_path)
            if not benchmark_path.is_file():
                continue
            try:
                df = pandas.read_csv(
                    benchmark_path,
                    usecols=["sample_id", "sample_group"])
                result = set(df.loc[
                    df["sample_group"] == args.sample_group,
                    "sample_id",
                ])
                if result:
                    return result
            except (OSError, ValueError):
                pass
        writer.skip(
            "sample-groups", args.sample_group,
            [path] + benchmark_paths,
            "Sample-group table absent; recent-only panels use all samples.")
        return None
    df = pandas.read_csv(path)
    sample_col = _first_present(df, ("sample_id", "sample", "id"))
    group_col = _first_present(df, ("sample_group", "group", "category"))
    if sample_col is None or group_col is None:
        writer.skip(
            "sample-groups", args.sample_group,
            [path],
            "Sample table must contain sample_id and sample_group columns.")
        return None
    result = set(df.loc[df[group_col] == args.sample_group, sample_col])
    if not result:
        writer.skip(
            "sample-groups", args.sample_group,
            [path],
            "No rows found for sample group %s." % args.sample_group)
        return None
    return result


def _read_multiallelic_scores(inputs, writer, figure, predictors, predictor_info):
    path = inputs.scores_dir / "accuracy_scores.multiallelic.csv"
    if path.is_file():
        return _normalize_score_predictors(pandas.read_csv(path))
    if inputs.multiallelic_predictions is not None:
        return _scores_from_saved_predictions(
            inputs.multiallelic_predictions,
            index_column="sample_id",
            kind="multiallelic",
            family="multiallelic",
            figure=figure,
            writer=writer,
            predictor_orientations=_predictor_orientations(predictor_info),
            declared_predictors=_predictor_info_names(predictor_info),
            external_baselines=predictors.external_baselines,
        )
    writer.skip(
        "multiallelic",
        figure,
        [path, inputs.scores_dir / "benchmark.multiallelic.csv.bz2"],
        "Multiallelic score table or saved prediction table is required.",
    )
    return None


def _read_monoallelic_scores(inputs, writer, figure, predictors, predictor_info):
    path = inputs.scores_dir / "accuracy_scores.monoallelic.csv"
    if path.is_file():
        return _normalize_score_predictors(pandas.read_csv(path))
    if inputs.monoallelic_predictions is not None:
        return _scores_from_saved_predictions(
            inputs.monoallelic_predictions,
            index_column=None,
            kind="monoallelic",
            family="monoallelic",
            figure=figure,
            writer=writer,
            predictor_orientations=_predictor_orientations(predictor_info),
            declared_predictors=_predictor_info_names(predictor_info),
            external_baselines=predictors.external_baselines,
        )
    return None


def _scores_from_saved_predictions(
        path, index_column, family, figure, writer, row_filter=None, kind=None,
        predictor_orientations=None,
        predictor_columns=None,
        declared_predictors=(),
        external_baselines=EXTERNAL_BASELINES):
    path = Path(path)
    if not path.is_file():
        writer.skip(
            family, figure, [path],
            "Saved prediction table is absent.")
        return None
    df = pandas.read_csv(path)
    if row_filter is not None:
        df = row_filter(df)
    if "hit" not in df.columns:
        writer.skip(
            family, figure, [path],
            "Saved prediction table must contain a hit column.")
        return None
    index_column = index_column or _prediction_index_column(df, kind=kind)
    if index_column is None:
        writer.skip(
            family, figure, [path],
            "Saved prediction table needs sample_id, allele, or hla column.")
        return None
    if "length" not in df.columns:
        if "peptide_len" in df.columns:
            df["length"] = df["peptide_len"]
        elif "peptide" in df.columns:
            df["length"] = df["peptide"].astype(str).str.len()
        else:
            df["length"] = numpy.nan

    try:
        predictor_columns = _prediction_score_columns(
            df,
            predictor_columns=predictor_columns,
            declared_predictors=declared_predictors,
        )
    except ValueError as error:
        writer.skip(family, figure, [path], str(error))
        return None
    if not predictor_columns:
        writer.skip(
            family, figure, [path],
            (
                "Saved prediction table has no recognized predictor score "
                "columns. Use canonical score names, declare custom columns "
                "and their direction in predictor_info.csv, or explicitly "
                "select declared columns with --predictor-columns."
            ))
        return None
    predictor_orientations = (
        predictor_orientations
        if predictor_orientations is not None
        else _predictor_orientations(None)
    )
    unknown_orientation = [
        predictor for predictor in predictor_columns
        if (
            predictor not in predictor_orientations and
            _normalize_predictor_name(predictor) not in predictor_orientations
        )
    ]
    if unknown_orientation:
        writer.skip(
            family, figure, unknown_orientation,
            (
                "Saved prediction table has predictor columns without score "
                "orientation. Add higher_is_better true/false rows to "
                "predictor_info.csv."
            ))
        return None

    rows = []
    for group_value, group in df.groupby(index_column):
        rows.extend(_scores_for_prediction_group(
            group, index_column, group_value, None, "All", predictor_columns,
            predictor_orientations=predictor_orientations))
        for length, length_group in group.groupby("length"):
            if pandas.isnull(length):
                continue
            length = int(length)
            rows.extend(_scores_for_prediction_group(
                length_group,
                index_column,
                group_value,
                length,
                "%d-mer" % length,
                predictor_columns,
                predictor_orientations=predictor_orientations,
            ))
    scores = pandas.DataFrame(rows)
    if scores.empty:
        writer.skip(
            family, figure, [path],
            "Saved prediction table produced no evaluable score rows.")
        return None
    scores = _normalize_score_predictors(scores)
    return _add_percent_change_columns(scores, external_baselines)


class _ScoreTableErrorCollector:
    def __init__(self):
        self.message = None

    def skip(self, family, figure, missing, note):
        self.message = note


def score_saved_prediction_table(
        path, index_column=None, kind=None, predictor_info=None,
        external_baselines=EXTERNAL_BASELINES,
        predictor_columns=None):
    """Return notebook-style AUC/PPV rows from a saved prediction table.

    The input table must contain ``hit`` and one grouping column
    (``sample_id``, ``allele``, or ``hla`` unless ``index_column`` is passed).
    Canonical score columns are selected from the built-in predictor registry.
    Custom columns must be declared in a ``predictor_info`` DataFrame with
    ``predictor`` and ``higher_is_better`` columns. ``predictor_columns`` can
    explicitly restrict selection to canonical or declared columns. Numeric
    metadata is never inferred to be a predictor score.
    When ``kind="monoallelic"``, allele identifiers are preferred over
    ``sample_id`` for automatic grouping.
    """
    writer = _ScoreTableErrorCollector()
    scores = _scores_from_saved_predictions(
        path,
        index_column=index_column,
        family="score-predictions",
        figure="score-predictions",
        writer=writer,
        kind=kind,
        predictor_orientations=_predictor_orientations(predictor_info),
        predictor_columns=predictor_columns,
        declared_predictors=_predictor_info_names(predictor_info),
        external_baselines=external_baselines,
    )
    if scores is None:
        raise ValueError(writer.message or "No evaluable score rows.")
    return scores


def _prediction_index_column(df, kind=None):
    if kind == "monoallelic":
        columns = ("allele", "hla", "sample_id")
    else:
        columns = ("sample_id", "allele", "hla")
    for column in columns:
        if column in df.columns:
            return column
    return None


def _predictor_info_names(predictor_info):
    if predictor_info is None or predictor_info.empty:
        return ()
    if "predictor" in predictor_info.columns:
        values = predictor_info["predictor"]
    else:
        values = predictor_info.index
    return tuple(str(value) for value in values if not pandas.isnull(value))


def _prediction_score_columns(
        df, predictor_columns=None, declared_predictors=()):
    """Select score columns from the canonical or explicitly declared schema."""
    if predictor_columns is None:
        expected = set(DEFAULT_PREDICTOR_HIGHER_IS_BETTER)
        expected.update(str(value) for value in declared_predictors)
        selected = [column for column in df.columns if column in expected]
    else:
        selected = list(dict.fromkeys(str(value) for value in predictor_columns))
        if not selected:
            raise ValueError("--predictor-columns must contain at least one column.")
        missing = [column for column in selected if column not in df.columns]
        if missing:
            raise ValueError(
                "Requested predictor score column(s) are absent: %s." %
                ", ".join(missing)
            )

    result = []
    nonnumeric = []
    for column in selected:
        values = pandas.to_numeric(df[column], errors="coerce")
        if values.notnull().any():
            df[column] = values
            result.append(column)
        else:
            nonnumeric.append(column)
    if nonnumeric:
        raise ValueError(
            "Predictor score column(s) contain no numeric values: %s." %
            ", ".join(nonnumeric)
        )
    return result


def _scores_for_prediction_group(
        group, index_column, group_value, length, length_label,
        predictor_columns, predictor_orientations=None):
    from sklearn.metrics import roc_auc_score

    labels = pandas.to_numeric(group["hit"], errors="coerce")
    valid_labels = labels.isin([0, 1])
    group = group.loc[valid_labels].copy()
    y_true = labels.loc[valid_labels].astype(int).values
    tie_breaker = _prediction_tie_breaker(group)
    oriented_scores = {}
    shared_mask = numpy.ones(len(group), dtype=bool)
    for predictor in predictor_columns:
        score = pandas.to_numeric(group[predictor], errors="coerce").values
        score = _orient_prediction_score(
            predictor, score, predictor_orientations)
        oriented_scores[predictor] = score
        shared_mask &= numpy.isfinite(score)

    # Every predictor in a comparison must be evaluated on the same examples.
    # Otherwise an external predictor can improve its apparent AUC/PPV merely
    # by omitting unsupported or difficult rows.
    y = y_true[shared_mask]
    ties = tie_breaker[shared_mask]
    rows = []
    for predictor in predictor_columns:
        s = oriented_scores[predictor][shared_mask]
        if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
            auc = numpy.nan
            ppv = numpy.nan
        else:
            auc = float(roc_auc_score(y, s))
            ppv = _ppv_at_n(y, s, int(y.sum()), tie_breaker=ties)
        rows.append({
            index_column: group_value,
            "sample_id": group_value,
            "length": length,
            "length_label": length_label,
            "predictor": predictor,
            "ppv": ppv,
            "auc": auc,
        })
    return rows


def _prediction_tie_breaker(group):
    identity_columns = [
        column for column in (
            "sample_id", "allele", "hla", "peptide", "n_flank", "c_flank",
            "length", "peptide_len")
        if column in group.columns
    ]
    if identity_columns:
        values = pandas.util.hash_pandas_object(
            group[identity_columns].astype(str), index=False).values
        return values.astype("float64") / float(numpy.iinfo("uint64").max)
    return numpy.random.default_rng(0).random(len(group))


def _orient_prediction_score(predictor, score, predictor_orientations=None):
    score = numpy.asarray(score, dtype=float)
    if predictor_orientations is None:
        predictor_orientations = _predictor_orientations(None)
    candidates = [str(predictor), _normalize_predictor_name(predictor)]
    for candidate in candidates:
        if candidate in predictor_orientations:
            if predictor_orientations[candidate]:
                return score
            return -score
    raise ValueError(
        "No score orientation configured for predictor %s. Add it to "
        "predictor_info.csv with higher_is_better true/false." % predictor)


def _ppv_at_n(y_true, y_score, n, tie_breaker=None):
    if n <= 0:
        return numpy.nan
    y_score = numpy.asarray(y_score, dtype=float)
    if tie_breaker is None:
        tie_breaker = numpy.zeros(len(y_score))
    tie_breaker = numpy.asarray(tie_breaker, dtype=float)
    order = numpy.lexsort((tie_breaker, -y_score))
    top = order[:n]
    return float(numpy.asarray(y_true)[top].sum()) / float(n)


def _add_percent_change_columns(scores, external_baselines=EXTERNAL_BASELINES):
    result = scores.copy()
    for metric in ("auc", "ppv"):
        pivot = result.pivot_table(
            index=["sample_id", "length_label"],
            columns="predictor",
            values=metric,
            aggfunc="mean",
        )
        for baseline, suffix in external_baselines:
            if baseline not in pivot.columns:
                continue
            baseline_values = pivot[baseline].replace(0, numpy.nan)
            percent = pivot.subtract(baseline_values, axis=0).divide(
                baseline_values, axis=0) * 100.0
            column = "percent_change_%s_%s" % (metric, suffix)
            stacked = percent.reset_index().melt(
                id_vars=["sample_id", "length_label"],
                var_name="predictor",
                value_name=column,
            )
            result = result.merge(
                stacked,
                on=["sample_id", "length_label", "predictor"],
                how="left",
            )
    return result


def _first_present(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def _external_baselines_in_predictors(predictors, available_predictors):
    available = set(available_predictors)
    return tuple(
        (predictor, suffix)
        for predictor, suffix in predictors.external_baselines
        if predictor in available
    )


def _external_baselines_with_percent_change(predictors, columns, metric):
    columns = set(columns)
    return tuple(
        (predictor, suffix)
        for predictor, suffix in predictors.external_baselines
        if "percent_change_%s_%s" % (metric, suffix) in columns
    )


def _generate_multiallelic_figures(
        inputs, predictor_info, recent_sample_ids, sample_group,
        max_scatter_points, writer, predictors):
    scores = _read_multiallelic_scores(
        inputs, writer, "all", predictors, predictor_info)
    if scores is None:
        return
    required = {
        "sample_id", "length_label", "predictor", "auc", "ppv",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        writer.skip(
            "multiallelic", "all",
            [inputs.scores_dir / "accuracy_scores.multiallelic.csv"],
            "Missing required columns: %s" % ", ".join(missing))
        return

    recent_note = (
        "Sample table not supplied; using all samples instead of %s." %
        sample_group if recent_sample_ids is None else
        "Restricted to sample group %s." % sample_group
    )

    _plot_external_scatter_triptych(
        scores, predictor_info, "auc", "AUC",
        "fig.3_scores_plots_multiallelic.scatter.auc.ba",
        max_scatter_points, writer, predictors)
    _plot_external_scatter_triptych(
        scores, predictor_info, "ppv", "PPV",
        "fig.3_scores_plots_multiallelic.scatter.ppv.ba",
        max_scatter_points, writer, predictors)
    _plot_percent_change_by_length(
        scores, predictor_info, "auc", "AUC",
        "fig.3_scores_plots_multiallelic.bar_by_peptide_length.auc.ba",
        writer, predictors)
    _plot_percent_change_bars(
        scores, predictor_info, "auc", "AUC",
        "fig.3_scores_plots_multiallelic.bar.auc.presentation",
        writer, predictors)
    _plot_percent_change_bars(
        scores, predictor_info, "ppv", "PPV",
        "fig.3_scores_plots_multiallelic.bar.ppv.presentation",
        writer, predictors)
    _plot_mean_ppv_small(
        scores, predictor_info, recent_sample_ids, recent_note,
        "fig.3_scores_plots_multiallelic.mean_ppv_small_plot",
        writer, predictors)
    _plot_presentation_scatter_grid(
        scores, predictor_info, recent_sample_ids, recent_note,
        max_scatter_points,
        "fig.3_scores_plots_multiallelic.scatter.ppv.presentation",
        writer, predictors)
    _plot_graphical_abstract_logistic_regression(
        predictor_info,
        "fig.3_scores_plots_multiallelic.graphical_abstract_logistic_regression",
        writer, predictors)


def _plot_external_scatter_triptych(
        scores, predictor_info, metric, metric_label, name, max_points, writer,
        predictors):
    import matplotlib.pyplot as plt

    candidate = predictors.candidate
    pivot = _pivot_all_lengths(scores, metric)
    if candidate not in pivot.columns:
        writer.skip(
            "multiallelic", name, [candidate],
            "Candidate predictor absent from multiallelic scores.")
        return
    baselines = _external_baselines_in_predictors(
        predictors, pivot.columns)
    if not baselines:
        writer.skip(
            "multiallelic", name,
            [predictor for predictor, _ in predictors.external_baselines],
            "No external baseline predictors found in multiallelic scores.")
        return

    n_cols = len(baselines)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(max(2.4, 2.35 * n_cols), 2.2),
        squeeze=False)
    y_label = _short_label(predictor_info, candidate)
    for ax, (baseline, _suffix) in zip(axes[0], baselines):
        sub = pivot[[baseline, candidate]].replace(
            [numpy.inf, -numpy.inf], numpy.nan).dropna()
        _scatter_with_winner_colors(
            ax, sub[baseline].values, sub[candidate].values,
            baseline, candidate, predictor_info, max_points)
        _finish_metric_scatter(
            ax,
            _short_label(predictor_info, baseline),
            y_label,
            "%s vs %s" % (
                y_label, _short_label(predictor_info, baseline)),
            metric_label)
    fig.tight_layout(w_pad=1.0)
    writer.save(fig, name, "multiallelic")


def _plot_percent_change_by_length(
        scores, predictor_info, metric, metric_label, name, writer,
        predictors):
    import matplotlib.pyplot as plt

    predictor = predictors.candidate
    sub = scores.loc[scores["predictor"] == predictor].copy()
    if sub.empty:
        writer.skip(
            "multiallelic", name, [predictor],
            "MHCflurry production scores absent.")
        return

    baselines = _external_baselines_with_percent_change(
        predictors, sub.columns, metric)
    if not baselines:
        writer.skip(
            "multiallelic", name,
            [
                "percent_change_%s_%s" % (metric, suffix)
                for _predictor, suffix in predictors.external_baselines
            ],
            "No percent-change columns found for external baselines.")
        return

    n_cols = len(baselines)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(max(2.4, 2.35 * n_cols), 2.1),
        sharey=True, squeeze=False)
    color = _predictor_color(predictor_info, predictor)
    for ax, (baseline, suffix) in zip(axes[0], baselines):
        column = "percent_change_%s_%s" % (metric, suffix)
        if column not in sub.columns:
            ax.set_visible(False)
            continue
        plot_df = (
            sub[["length_label", column]]
            .replace([numpy.inf, -numpy.inf], numpy.nan)
            .dropna()
        )
        values = (
            plot_df.groupby("length_label")[column]
            .mean()
            .reindex(LENGTH_LABEL_ORDER)
            .dropna()
        )
        x = numpy.arange(len(values))
        ax.bar(x, values.values, color=color, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(values.index, rotation=30, ha="right")
        ax.set_title("vs %s" % _short_label(predictor_info, baseline))
        ax.set_ylabel("%% change in %s" % metric_label)
        _despine(ax)
    fig.tight_layout(w_pad=1.0)
    writer.save(fig, name, "multiallelic")


def _plot_percent_change_bars(
        scores, predictor_info, metric, metric_label, name, writer,
        predictors):
    import matplotlib.pyplot as plt

    sub = _all_length_rows(scores)
    selected_predictors = [
        predictor for predictor in predictors.preferred_predictors
        if predictor in set(sub["predictor"])
    ]
    if not selected_predictors:
        writer.skip(
            "multiallelic", name, list(predictors.preferred_predictors),
            "No preferred predictors found in multiallelic scores.")
        return

    baselines = _external_baselines_with_percent_change(
        predictors, sub.columns, metric)
    if not baselines:
        writer.skip(
            "multiallelic", name,
            [
                "percent_change_%s_%s" % (metric, suffix)
                for _predictor, suffix in predictors.external_baselines
            ],
            "No percent-change columns found for external baselines.")
        return

    n_cols = len(baselines)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(max(2.4, 2.35 * n_cols), 3.3),
        sharey=True, squeeze=False)
    for ax, (baseline, suffix) in zip(axes[0], baselines):
        column = "percent_change_%s_%s" % (metric, suffix)
        if column not in sub.columns:
            ax.set_visible(False)
            continue
        means = (
            sub.loc[
                sub["predictor"].isin(selected_predictors),
                ["predictor", column],
            ]
            .replace([numpy.inf, -numpy.inf], numpy.nan)
            .dropna()
            .groupby("predictor")[column]
            .mean()
            .reindex(selected_predictors)
            .dropna()
        )
        labels = [_short_label(predictor_info, item) for item in means.index]
        colors = [_predictor_color(predictor_info, item) for item in means.index]
        y = numpy.arange(len(means))
        ax.barh(y, means.values, color=colors, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="0.35", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title("vs %s" % _short_label(predictor_info, baseline))
        ax.set_xlabel("%% change in %s" % metric_label)
        _despine(ax)
    fig.tight_layout(w_pad=1.0)
    writer.save(fig, name, "multiallelic")


def _plot_mean_ppv_small(
        scores, predictor_info, recent_sample_ids, note, name, writer,
        predictors):
    import matplotlib.pyplot as plt

    sub = _all_length_rows(scores)
    if recent_sample_ids is not None:
        sub = sub.loc[sub["sample_id"].isin(recent_sample_ids)]
    candidates = [
        predictors.candidate,
        "presentation_with_flanks_processing_score",
        "presentation_with_flanks_presentation_score",
    ]
    rows = []
    for predictor in candidates:
        values = sub.loc[sub["predictor"] == predictor, "ppv"].replace(
            [numpy.inf, -numpy.inf], numpy.nan).dropna()
        if len(values):
            rows.append((predictor, values.mean(), _predictor_color(
                predictor_info, predictor)))
    external_baselines = _external_baselines_in_predictors(
        predictors, sub["predictor"])
    external_means = (
        sub.loc[sub["predictor"].isin([p for p, _ in external_baselines])]
        .assign(
            ppv=lambda df: df["ppv"].replace(
                [numpy.inf, -numpy.inf], numpy.nan))
        .groupby("predictor")["ppv"]
        .mean()
        .dropna()
    )
    if len(external_means):
        rows.append(("external_tools", external_means.mean(), (0.45, 0.45, 0.45)))
    if not rows:
        writer.skip(
            "multiallelic", name, candidates,
            "No PPV values available for selected predictors.")
        return

    labels = [
        "External tools" if predictor == "external_tools"
        else _short_label(predictor_info, predictor)
        for predictor, _mean, _color in rows
    ]
    values = [mean for _predictor, mean, _color in rows]
    colors = [color for _predictor, _mean, color in rows]
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    x = numpy.arange(len(values))
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean PPV")
    ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
    _despine(ax)
    fig.tight_layout()
    writer.save(fig, name, "multiallelic", note=note)


def _plot_presentation_scatter_grid(
        scores, predictor_info, recent_sample_ids, note, max_points, name,
        writer, predictors):
    import matplotlib.pyplot as plt

    pivot = _pivot_all_lengths(scores, "ppv")
    if recent_sample_ids is not None:
        pivot = pivot.loc[pivot.index.isin(recent_sample_ids)]
    candidate_predictors = [
        predictor for predictor in predictors.presentation_panel_predictors
        if predictor in pivot.columns
    ]
    baseline_predictors = [
        predictor for predictor in predictors.presentation_panel_baselines
        if predictor in pivot.columns
    ]
    if not candidate_predictors or not baseline_predictors:
        writer.skip(
            "multiallelic", name,
            (
                list(predictors.presentation_panel_predictors) +
                list(predictors.presentation_panel_baselines)
            ),
            "Required presentation-panel predictors absent.")
        return

    n_rows = len(candidate_predictors)
    n_cols = len(baseline_predictors)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(2.0, 1.8 * n_cols), max(2.0, 1.95 * n_rows)),
        squeeze=False,
    )
    for row_index, candidate in enumerate(candidate_predictors):
        for col_index, baseline in enumerate(baseline_predictors):
            ax = axes[row_index, col_index]
            sub = pivot[[baseline, candidate]].replace(
                [numpy.inf, -numpy.inf], numpy.nan).dropna()
            _scatter_with_winner_colors(
                ax, sub[baseline].values, sub[candidate].values,
                baseline, candidate, predictor_info, max_points)
            ax.set_title(_short_label(predictor_info, baseline))
            if col_index == 0:
                ax.set_ylabel(_short_label(predictor_info, candidate))
            else:
                ax.set_ylabel("")
            if row_index == len(candidate_predictors) - 1:
                ax.set_xlabel("Baseline PPV")
            else:
                ax.set_xlabel("")
            _set_unit_limits(ax)
            _despine(ax)
    fig.tight_layout(w_pad=0.9, h_pad=0.9)
    writer.save(fig, name, "multiallelic", note=note)


def _plot_graphical_abstract_logistic_regression(
        predictor_info, name, writer, predictors):
    import matplotlib.pyplot as plt

    x = numpy.linspace(-5.0, 5.0, 200)
    y = 1.0 / (1.0 + numpy.exp(-x))
    ba_color = _predictor_color(predictor_info, predictors.candidate)
    ap_color = _predictor_color(
        predictor_info, "presentation_with_flanks_processing_score")
    ps_color = _predictor_color(
        predictor_info, "presentation_with_flanks_presentation_score")

    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    ax.plot(x, y, color=ps_color, linewidth=2.0)
    ax.fill_between(x, 0, y, color=ps_color, alpha=0.12, linewidth=0)
    ax.scatter([-2.7, -0.7, 1.8], [0.06, 0.34, 0.86],
               color=[ba_color, ap_color, ps_color],
               s=38, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel("Combined affinity + processing evidence")
    ax.set_ylabel("Presentation probability")
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(-0.02, 1.02)
    _despine(ax)
    fig.tight_layout()
    writer.save(fig, name, "multiallelic")


def _generate_model_selection_figures(inputs, writer):
    path = _first_existing(
        inputs.scores_dir,
        ("model_selection_accuracy.csv", "model_selection_accuracy.xlsx"))
    baseline_col = None
    score_label = "AUC"
    baseline_label = None
    count_label = "Training peptides"
    source_path = path
    if path is not None:
        df = (
            pandas.read_excel(path) if path.suffix == ".xlsx"
            else pandas.read_csv(path)
        )
        allele_col = _first_present(df, ("allele", "hla", "mhc_allele"))
        score_col = _first_present(df, ("auc", "AUC", "score", "accuracy"))
        count_col = _first_present(
            df, ("num_peptides", "peptides", "train_peptides", "train_count"))
        binder_col = _first_present(
            df, ("percent_binders", "binder_percent", "binders_percent"))
    else:
        per_allele = _current_affinity_per_allele(inputs.comparison_dir)
        if per_allele is None:
            writer.skip(
                "model-selection",
                "fig.1_model_selection_predictor_accuracy.scores.by_locus",
                [
                    inputs.scores_dir / "model_selection_accuracy.csv",
                    inputs.scores_dir / "model_selection_accuracy.xlsx",
                    (
                        Path(inputs.comparison_dir) / "affinity" /
                        "per_allele.csv"
                        if inputs.comparison_dir else
                        "compare-models affinity/per_allele.csv"
                    ),
                ],
                "No model-selection table or current affinity comparison table.")
            return
        labels = _comparison_labels(inputs.comparison_dir)
        df = per_allele.copy()
        df["percent_binders"] = (
            pandas.to_numeric(df["n_pos"], errors="coerce") /
            pandas.to_numeric(df["n"], errors="coerce").replace(0, numpy.nan) *
            100.0
        )
        allele_col = "allele"
        score_col = "a_roc_auc"
        baseline_col = "b_roc_auc"
        score_label = "%s AUROC" % labels["a"]
        baseline_label = "%s AUROC" % labels["b"]
        count_col = "n"
        count_label = "Evaluation peptides"
        binder_col = "percent_binders"
        source_path = (
            Path(inputs.comparison_dir) / "affinity" / "per_allele.csv"
            if inputs.comparison_dir else None
        )
    if allele_col is None or score_col is None:
        writer.skip(
            "model-selection",
            "fig.1_model_selection_predictor_accuracy.scores.by_locus",
            [source_path],
            "Table must contain allele and AUC/score columns.")
        return

    import matplotlib.pyplot as plt

    df = df.copy()
    df["locus"] = df[allele_col].map(_allele_locus)
    optional_panels = []
    if count_col:
        optional_panels.append(
            (count_col, count_label, (0.55, 0.55, 0.55), True))
    if binder_col:
        optional_panels.append((binder_col, "% binders", (0.65, 0.39, 0.67), False))
    for locus, label in (
            ("HLA-A", "hla_a"),
            ("HLA-B", "hla_b"),
            ("HLA-C", "hla_c"),
            ("H2", "h2"),
            ("other", "other")):
        sub = df.loc[df["locus"] == locus].sort_values(score_col)
        if sub.empty:
            writer.skip(
                "model-selection",
                "fig.1_model_selection_predictor_accuracy.scores.%s" % label,
                [path],
                "No rows for locus %s." % locus)
            continue
        fig, axes = plt.subplots(
            1, 1 + len(optional_panels),
            figsize=(7.1, max(1.7, 0.18 * len(sub) + 0.6)),
            squeeze=False)
        ax = axes[0, 0]
        y = numpy.arange(len(sub))
        if baseline_col and baseline_col in sub.columns:
            height = 0.36
            ax.barh(
                y - height / 2,
                sub[score_col],
                height=height,
                color=_predictor_color(
                    pandas.DataFrame(), "mhcflurry_production"),
                label=score_label,
            )
            ax.barh(
                y + height / 2,
                sub[baseline_col],
                height=height,
                color=(0.55, 0.55, 0.55),
                label=baseline_label,
            )
            ax.legend(frameon=False, loc="lower right")
        else:
            ax.barh(y, sub[score_col], color=(0.34, 0.46, 0.75))
        ax.set_yticks(y)
        ax.set_yticklabels(sub[allele_col])
        ax.set_xlabel(score_label if not baseline_col else "AUROC")
        ax.set_title(locus)
        _despine(ax)
        for panel_index, (column, xlabel, color, log_scale) in enumerate(
                optional_panels, start=1):
            ax = axes[0, panel_index]
            ax.barh(y, sub[column], color=color)
            ax.set_yticks(y)
            ax.set_yticklabels([])
            if log_scale:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            _despine(ax)
        fig.tight_layout(w_pad=1.0)
        writer.save(
            fig,
            "fig.1_model_selection_predictor_accuracy.scores.%s" % label,
            "model-selection")


def _generate_monoallelic_figures(
        inputs, predictor_info, max_scatter_points, writer, predictors):
    scores = _read_monoallelic_scores(
        inputs,
        writer,
        "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.ba",
        predictors,
        predictor_info,
    )
    if scores is not None:
        _plot_monoallelic_scatter(
            scores, predictor_info, "auc", "AUC", max_scatter_points,
            "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.ba",
            writer, predictors)
        _plot_monoallelic_scatter(
            scores, predictor_info, "ppv", "PPV", max_scatter_points,
            "fig.3_scores_plots_monoallelic.scatter.ppv.monoallelic.ba",
            writer, predictors)
    else:
        _plot_current_affinity_scatter(
            inputs.comparison_dir,
            "roc_auc",
            "AUROC",
            "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.ba",
            writer,
        )
        _plot_current_affinity_scatter(
            inputs.comparison_dir,
            "ppv_at_n",
            "PPV@N",
            "fig.3_scores_plots_monoallelic.scatter.ppv.monoallelic.ba",
            writer,
        )

    novel_path = inputs.scores_dir / "accuracy_scores.monoallelic.novel_alleles.csv"
    if novel_path.is_file():
        scores = _normalize_score_predictors(pandas.read_csv(novel_path))
        _plot_monoallelic_scatter(
            scores, predictor_info, "auc", "AUC", max_scatter_points,
            "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.novel_alleles.ba",
            writer,
            predictors,
            preferred_candidate="no_additional_ms_similar")
    else:
        writer.skip(
            "monoallelic",
            "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.novel_alleles.ba",
            [novel_path],
            "Novel-allele monoallelic accuracy scores absent.")


def _plot_current_affinity_scatter(
        comparison_dir, metric, metric_label, name, writer):
    import matplotlib.pyplot as plt

    per_allele = _current_affinity_per_allele(comparison_dir)
    if per_allele is None:
        writer.skip(
            "monoallelic",
            name,
            ["accuracy_scores.monoallelic.csv", "affinity/per_allele.csv"],
            "No monoallelic score table or current affinity comparison table.")
        return
    labels = _comparison_labels(comparison_dir)
    x_col = "b_%s" % metric
    y_col = "a_%s" % metric
    if x_col not in per_allele.columns or y_col not in per_allele.columns:
        writer.skip(
            "monoallelic", name, [comparison_dir],
            "Current affinity comparison lacks %s columns." % metric)
        return
    sub = per_allele[[x_col, y_col]].replace(
        [numpy.inf, -numpy.inf], numpy.nan).dropna()
    if sub.empty:
        writer.skip(
            "monoallelic", name, [comparison_dir],
            "Current affinity comparison has no finite %s values." % metric)
        return
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    ax.scatter(
        sub[x_col], sub[y_col],
        c=[
            SIDE_A_COLOR if y >= x else SIDE_B_COLOR
            for x, y in zip(sub[x_col], sub[y_col])
        ],
        s=16,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.2,
    )
    _add_diagonal(ax, sub[x_col], sub[y_col])
    ax.set_xlabel("%s %s" % (labels["b"], metric_label))
    ax.set_ylabel("%s %s" % (labels["a"], metric_label))
    ax.set_title("Allele-level %s" % metric_label)
    _set_unit_limits(ax)
    _despine(ax)
    fig.tight_layout()
    writer.save(fig, name, "monoallelic")


def _plot_monoallelic_scatter(
        scores, predictor_info, metric, metric_label, max_points, name, writer,
        predictors, preferred_candidate="no_additional_ms"):
    scores = _all_length_rows(scores)
    if scores.empty:
        writer.skip(
            "monoallelic", name, ["All-length monoallelic scores"],
            "Monoallelic scores have no All-length rows.")
        return
    candidate = (
        preferred_candidate if preferred_candidate in set(scores["predictor"])
        else predictors.candidate
    )
    if candidate not in set(scores["predictor"]):
        writer.skip(
            "monoallelic", name, [candidate],
            "MHCflurry candidate predictor absent.")
        return
    pivot = scores.pivot_table(
        index=_monoallelic_index_columns(scores),
        columns="predictor",
        values=metric,
        aggfunc="mean")
    baselines = _external_baselines_in_predictors(
        predictors, pivot.columns)
    if not baselines:
        writer.skip(
            "monoallelic", name,
            [predictor for predictor, _ in predictors.external_baselines],
            "No external baseline predictors found in monoallelic scores.")
        return
    _plot_scatter_triptych_from_pivot(
        pivot, predictor_info, candidate, metric_label, max_points, name,
        "monoallelic", writer, predictors, baselines=baselines)


def _generate_processing_notebook_figures(
        inputs, predictor_info, writer, predictors):
    no_c_path = inputs.scores_dir / "accuracy_scores.multiallelic.no_C.csv"
    motif_path = inputs.scores_dir / "antigen_processing.motifs.xlsx"
    correlation_path = (
        inputs.scores_dir / "correlation.processing_vs_affinity.sampled.csv.bz2"
    )
    training_path = inputs.scores_dir / "train_data.ap.production.csv"

    no_c_scores = _read_cysteine_removed_scores(
        inputs, no_c_path, writer, predictors, predictor_info)
    if no_c_scores is not None:
        _plot_cysteine_removed_panels(
            inputs, no_c_scores, predictor_info, writer, predictors)
    elif _plot_current_ap_vs_summary(inputs, writer):
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
            [no_c_path],
            "Cysteine-removed score table absent; generated current AP summary only.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.bar",
            [no_c_path],
            "Cysteine-removed score table absent; generated current AP summary only.")
    else:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
            [no_c_path],
            "Cysteine-removed multiallelic benchmark scores absent.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.bar",
            [no_c_path],
            "Cysteine-removed multiallelic benchmark scores absent.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.bar.ap_vs_others",
            [no_c_path],
            "AP-vs-other benchmark table absent.")

    if motif_path.is_file():
        _plot_ap_motif_logo(motif_path, writer)
    elif _plot_ap_motif_from_training(inputs, writer):
        pass
    else:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.logo.ap",
            [motif_path],
            "Antigen-processing motif workbook absent.")

    if correlation_path.is_file() and training_path.is_file():
        _plot_ap_correlation_panels(correlation_path, training_path, writer)
    elif _plot_ap_correlation_from_saved_predictions(inputs, writer):
        pass
    else:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.correlation.ap_by_gene",
            [correlation_path, training_path],
            "AP correlation and training-data tables absent.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.extended.ap_correlation",
            [correlation_path, training_path],
            "AP correlation and training-data tables absent.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.correlation.included_vs_excluded",
            [correlation_path, training_path],
            "AP correlation and training-data tables absent.")


def _read_cysteine_removed_scores(
        inputs, no_c_path, writer, predictors, predictor_info):
    if no_c_path.is_file():
        return _normalize_score_predictors(pandas.read_csv(no_c_path))
    if inputs.multiallelic_predictions is None:
        return None
    return _scores_from_saved_predictions(
        inputs.multiallelic_predictions,
        index_column="sample_id",
        kind="multiallelic",
        family="antigen-processing",
        figure="fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
        writer=writer,
        predictor_orientations=_predictor_orientations(predictor_info),
        declared_predictors=_predictor_info_names(predictor_info),
        external_baselines=predictors.external_baselines,
        row_filter=lambda df: df.loc[
            ~df.get("peptide", pandas.Series("", index=df.index))
            .astype(str)
            .str.contains("C", na=False)
        ].copy(),
    )


def _plot_cysteine_removed_panels(
        inputs, no_c, predictor_info, writer, predictors):
    import matplotlib.pyplot as plt

    full = _read_multiallelic_scores(
        inputs,
        writer,
        "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
        predictors,
        predictor_info,
    )
    if full is None:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
            [inputs.scores_dir / "accuracy_scores.multiallelic.csv"],
            "Full multiallelic benchmark scores absent.")
        return
    no_c = _normalize_score_predictors(no_c)
    predictor = "presentation_without_flanks_processing_score"
    full_pivot = _pivot_all_lengths(full, "auc")
    no_c_pivot = _pivot_all_lengths(no_c, "auc")
    if predictor not in full_pivot.columns or predictor not in no_c_pivot.columns:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
            [predictor],
            "AP predictor absent from full or cysteine-removed benchmark.")
        return
    joined = pandas.DataFrame({
        "full": full_pivot[predictor],
        "c_removed": no_c_pivot[predictor],
    }).dropna()

    fig, ax = plt.subplots(figsize=(2.5, 2.4))
    ax.scatter(joined["full"], joined["c_removed"],
               color=_predictor_color(predictor_info, predictor),
               s=16, alpha=0.8, edgecolor="white", linewidth=0.2)
    _add_diagonal(ax, joined["full"], joined["c_removed"])
    ax.set_xlabel("Full benchmark AUC")
    ax.set_ylabel("Cysteine removed AUC")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
        "antigen-processing")

    means = pandas.Series({
        "Full": joined["full"].mean(),
        "C removed": joined["c_removed"].mean(),
    })
    fig, ax = plt.subplots(figsize=(2.1, 2.2))
    ax.bar(numpy.arange(len(means)), means.values,
           color=_predictor_color(predictor_info, predictor),
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(numpy.arange(len(means)))
    ax.set_xticklabels(means.index, rotation=25, ha="right")
    ax.set_ylabel("Mean AUC")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.auc.ap.c_removed.bar",
        "antigen-processing")

    _plot_ap_vs_others(no_c, predictor_info, writer, predictors)


def _plot_ap_vs_others(scores, predictor_info, writer, predictors):
    import matplotlib.pyplot as plt

    sub = _all_length_rows(scores)
    comparison_predictors = dict.fromkeys((
        "presentation_without_flanks_processing_score",
        "presentation_with_flanks_processing_score",
        predictors.candidate,
        *[predictor for predictor, _ in predictors.external_baselines],
    ))
    selected_predictors = [
        predictor for predictor in comparison_predictors
        if predictor in set(sub["predictor"])
    ]
    if not selected_predictors:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.bar.ap_vs_others",
            [],
            "No AP comparison predictors found.")
        return
    means = (
        sub.loc[sub["predictor"].isin(selected_predictors)]
        .groupby("predictor")["auc"]
        .mean()
        .reindex(selected_predictors)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = numpy.arange(len(means))
    ax.bar(
        x, means.values,
        color=[_predictor_color(predictor_info, p) for p in means.index],
        edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_label(predictor_info, p) for p in means.index],
        rotation=30, ha="right")
    ax.set_ylabel("Mean AUC")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig, "fig.4_processing_predictor_plots.bar.ap_vs_others",
        "antigen-processing")


def _plot_current_ap_vs_summary(inputs, writer):
    if inputs.comparison_dir is None:
        return False
    processing_path = (
        Path(inputs.comparison_dir) / "processing" / "summary_table.csv"
    )
    presentation_path = (
        Path(inputs.comparison_dir) / "presentation" / "summary_table.csv"
    )
    if not processing_path.is_file() and not presentation_path.is_file():
        return False
    frames = []
    if processing_path.is_file():
        processing = pandas.read_csv(processing_path)
        for _, row in processing.iterrows():
            frames.append({
                "label": "AP %s" % str(row.get("mode", "")).replace("_", " "),
                "AUROC": row.get("a_macro_roc_auc"),
                "AUPRC": row.get("a_macro_pr_auc"),
                "PPV@N": row.get("a_macro_ppv_at_n"),
            })
    if presentation_path.is_file():
        presentation = pandas.read_csv(presentation_path)
        if "score_kind" in presentation.columns:
            presentation = presentation.loc[
                presentation["score_kind"] == "presentation_score"
            ]
        for _, row in presentation.iterrows():
            frames.append({
                "label": "PS %s" % str(row.get("mode", "")).replace("_", " "),
                "AUROC": row.get("a_macro_roc_auc"),
                "AUPRC": row.get("a_macro_pr_auc"),
                "PPV@N": row.get("a_macro_ppv_at_n"),
            })
    df = pandas.DataFrame(frames)
    if df.empty:
        return False
    import matplotlib.pyplot as plt

    metrics = ["AUROC", "AUPRC", "PPV@N"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0, 2.6), squeeze=False)
    x = numpy.arange(len(df))
    colors = [
        (0.353, 0.612, 0.518) if label.startswith("AP") else (0.596, 0.557, 0.835)
        for label in df["label"]
    ]
    for ax, metric in zip(axes[0], metrics):
        values = pandas.to_numeric(df[metric], errors="coerce")
        ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=35, ha="right")
        finite = values[numpy.isfinite(values)]
        upper = max(1.0, float(finite.max()) * 1.05) if len(finite) else 1.0
        ax.set_ylim(0, upper)
        _despine(ax)
    axes[0, 0].set_ylabel("Macro mean")
    fig.tight_layout(w_pad=1.0)
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.bar.ap_vs_others",
        "antigen-processing",
        note="Generated from current compare-models processing/presentation summaries.",
    )
    return True


def _plot_ap_motif_logo(path, writer):
    import matplotlib.pyplot as plt

    try:
        import logomaker
    except ImportError:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.logo.ap",
            [path],
            "logomaker is not installed.")
        return
    workbook = pandas.read_excel(path, sheet_name=None)
    sheet_name = next(iter(workbook))
    df = workbook[sheet_name]
    if "position" in df.columns:
        df = df.set_index("position")
    numeric = df.select_dtypes(include=[numpy.number])
    if numeric.empty:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.logo.ap",
            [path],
            "Motif workbook contains no numeric amino-acid matrix.")
        return
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    logomaker.Logo(numeric, ax=ax)
    ax.set_xlabel("Position")
    ax.set_ylabel("Weight")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig, "fig.4_processing_predictor_plots.logo.ap",
        "antigen-processing")


def _plot_ap_motif_from_training(inputs, writer):
    path = _processing_training_data_path(inputs)
    if path is None:
        return False
    import matplotlib.pyplot as plt

    try:
        df = pandas.read_csv(
            path,
            usecols=lambda column: column in ("n_flank", "c_flank", "hit"),
            nrows=200_000,
        )
    except (OSError, ValueError):
        return False
    if "hit" in df.columns:
        df = df.loc[pandas.to_numeric(df["hit"], errors="coerce") == 1]
    if df.empty or "n_flank" not in df.columns or "c_flank" not in df.columns:
        return False

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    positions = ["N-%d" % i for i in range(5, 0, -1)] + [
        "C+%d" % i for i in range(1, 6)
    ]
    counts = pandas.DataFrame(0.0, index=positions, columns=amino_acids)
    for flank_col, offset in (("n_flank", 0), ("c_flank", 5)):
        strings = df[flank_col].fillna("").astype(str)
        for row in strings:
            row = row[-5:] if flank_col == "n_flank" else row[:5]
            row = row.rjust(5, "X") if flank_col == "n_flank" else row.ljust(5, "X")
            for i, aa in enumerate(row):
                if aa in counts.columns:
                    counts.iloc[offset + i, counts.columns.get_loc(aa)] += 1.0
    frequencies = counts.divide(counts.sum(axis=1).replace(0, numpy.nan), axis=0)
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    im = ax.imshow(
        frequencies.fillna(0).values,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=float(numpy.nanmax(frequencies.values)) if numpy.isfinite(
            frequencies.values).any() else 1.0,
    )
    ax.set_xticks(numpy.arange(len(amino_acids)))
    ax.set_xticklabels(amino_acids)
    ax.set_yticks(numpy.arange(len(positions)))
    ax.set_yticklabels(positions)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Flank position")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Frequency")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.logo.ap",
        "antigen-processing",
        note="Generated from current processing training flanks.",
    )
    return True


def _processing_training_data_path(inputs):
    candidates = [
        inputs.scores_dir / "train_data.ap.production.csv",
        inputs.scores_dir / "train_data.csv.bz2",
    ]
    if inputs.run_dir is not None:
        candidates.extend([
            Path(inputs.run_dir) / "processing" / "train_data.csv.bz2",
            Path(inputs.run_dir) / "processing" / "models.selected.with_flanks" /
            "train_data.csv.bz2",
            Path(inputs.run_dir) / "processing" / "models.selected.no_flank" /
            "train_data.csv.bz2",
        ])
    for path in candidates:
        if path.is_file():
            return path
    return None


def _plot_ap_correlation_panels(correlation_path, training_path, writer):
    import matplotlib.pyplot as plt

    corr = pandas.read_csv(correlation_path)
    training = pandas.read_csv(training_path)
    x_col = _first_present(corr, ("affinity_score", "affinity", "ba_score"))
    y_col = _first_present(corr, ("processing_score", "ap_score", "processing"))
    gene_col = _first_present(corr, ("gene", "protein", "protein_id"))
    if x_col is None or y_col is None:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.correlation.ap_by_gene",
            [correlation_path],
            "Correlation table must contain affinity and processing columns.")
        return

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    if gene_col is not None:
        top_genes = corr[gene_col].value_counts().head(6).index
        for gene in top_genes:
            sub = corr.loc[corr[gene_col] == gene]
            ax.scatter(sub[x_col], sub[y_col], s=8, alpha=0.6, label=str(gene))
        ax.legend(frameon=False, loc="best", handlelength=1.0)
    else:
        ax.scatter(corr[x_col], corr[y_col], s=8, alpha=0.6)
    ax.set_xlabel("Affinity score")
    ax.set_ylabel("Processing score")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.correlation.ap_by_gene",
        "antigen-processing")

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    numeric = corr.select_dtypes(include=[numpy.number])
    if len(numeric.columns) < 2:
        plt.close(fig)
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.extended.ap_correlation",
            [correlation_path],
            "Correlation heatmap requires at least two numeric columns.")
    else:
        im = ax.imshow(numeric.corr(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(numpy.arange(len(numeric.columns)))
        ax.set_yticks(numpy.arange(len(numeric.columns)))
        ax.set_xticklabels(numeric.columns, rotation=45, ha="right")
        ax.set_yticklabels(numeric.columns)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
        _despine(ax)
        fig.tight_layout()
        writer.save(
            fig,
            "fig.4_processing_predictor_plots.extended.ap_correlation",
            "antigen-processing")

    if "included" in training.columns and y_col in training.columns:
        included = _coerce_bool_series(training["included"])
        if included is None:
            writer.skip(
                "antigen-processing",
                "fig.4_processing_predictor_plots.correlation.included_vs_excluded",
                [training_path],
                "Included flag contains values that are not parseable as booleans.")
            return
        fig, ax = plt.subplots(figsize=(2.5, 2.3))
        groups = [
            training.loc[included, y_col].dropna(),
            training.loc[~included, y_col].dropna(),
        ]
        try:
            ax.boxplot(
                groups, tick_labels=["Included", "Excluded"],
                showfliers=False)
        except TypeError:
            ax.boxplot(
                groups, labels=["Included", "Excluded"], showfliers=False)
        ax.set_ylabel("Processing score")
        _despine(ax)
        fig.tight_layout()
        writer.save(
            fig,
            "fig.4_processing_predictor_plots.correlation.included_vs_excluded",
            "antigen-processing")
    else:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.correlation.included_vs_excluded",
            [training_path],
            "Training table lacks included flag or processing score column.")


def _plot_ap_correlation_from_saved_predictions(inputs, writer):
    path = inputs.multiallelic_predictions
    if path is None or not Path(path).is_file():
        return False
    import matplotlib.pyplot as plt

    try:
        df = pandas.read_csv(path, nrows=250_000)
    except (OSError, ValueError):
        return False
    affinity_col = _first_present(
        df,
        (
            "presentation_with_flanks_affinity",
            "presentation_without_flanks_affinity",
            "mhcflurry_production_affinity",
        ),
    )
    processing_col = _first_present(
        df,
        (
            "presentation_with_flanks_processing_score",
            "presentation_without_flanks_processing_score",
        ),
    )
    if affinity_col is None or processing_col is None:
        return False
    df = df.copy()
    affinity_columns = [
        column for column in df.columns
        if "affinity" in str(column).lower()
    ]
    for column in affinity_columns:
        df[column] = _affinity_to_evidence_score(df[column])
    df[processing_col] = pandas.to_numeric(
        df[processing_col], errors="coerce")
    finite = df[[affinity_col, processing_col]].replace(
        [numpy.inf, -numpy.inf], numpy.nan).dropna()
    if finite.empty:
        return False

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    if "protein_accession" in df.columns:
        top = df["protein_accession"].value_counts().head(6).index
        for protein in top:
            sub = df.loc[df["protein_accession"] == protein]
            ax.scatter(
                sub[affinity_col], sub[processing_col],
                s=7, alpha=0.55, label=str(protein))
        ax.legend(frameon=False, loc="best", handlelength=1.0, fontsize=6)
    else:
        ax.scatter(finite[affinity_col], finite[processing_col], s=7, alpha=0.55)
    ax.set_xlabel("Affinity evidence")
    ax.set_ylabel("Processing score")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.correlation.ap_by_gene",
        "antigen-processing",
        note="Generated from saved multiallelic predictions.",
    )

    numeric_cols = [
        column for column in df.columns
        if (
            "affinity" in str(column).lower() or
            "processing_score" in str(column).lower() or
            "presentation_score" in str(column).lower()
        )
    ]
    numeric = df[numeric_cols].apply(pandas.to_numeric, errors="coerce")
    if len(numeric.columns) >= 2:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        corr = numeric.corr()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(numpy.arange(len(corr.columns)))
        ax.set_yticks(numpy.arange(len(corr.columns)))
        ax.set_xticklabels(
            [_short_score_label(column) for column in corr.columns],
            rotation=45,
            ha="right",
        )
        ax.set_yticklabels([_short_score_label(column) for column in corr.columns])
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
        _despine(ax)
        fig.tight_layout()
        writer.save(
            fig,
            "fig.4_processing_predictor_plots.extended.ap_correlation",
            "antigen-processing",
            note="Generated from saved multiallelic predictions.",
        )

    if "hit" in df.columns:
        hit = pandas.to_numeric(df["hit"], errors="coerce").fillna(0) > 0
        fig, ax = plt.subplots(figsize=(2.7, 2.3))
        groups = [
            df.loc[hit, processing_col].dropna(),
            df.loc[~hit, processing_col].dropna(),
        ]
        try:
            ax.boxplot(groups, tick_labels=["Hit", "Decoy"], showfliers=False)
        except TypeError:
            ax.boxplot(groups, labels=["Hit", "Decoy"], showfliers=False)
        ax.set_ylabel("Processing score")
        _despine(ax)
        fig.tight_layout()
        writer.save(
            fig,
            "fig.4_processing_predictor_plots.processing_score.hit_vs_decoy",
            "antigen-processing",
            note="Generated from saved multiallelic predictions.",
        )
    return True


def _affinity_to_evidence_score(values):
    return -numpy.log10(
        numpy.clip(pandas.to_numeric(values, errors="coerce"), 1e-3, 1e8)
    )


def _short_score_label(column):
    return (
        str(column)
        .replace("presentation_with_flanks_", "with flanks ")
        .replace("presentation_without_flanks_", "without flanks ")
        .replace("mhcflurry_production_", "MHCflurry ")
        .replace("_", " ")
    )


def _coerce_bool_series(series):
    if pandas.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pandas.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y", "included", "include"}
    false_values = {
        "", "0", "false", "f", "no", "n", "excluded", "exclude"
    }
    parsed = normalized.map(
        lambda value: True if value in true_values else (
            False if value in false_values else numpy.nan
        )
    )
    if parsed.isnull().any():
        return None
    return parsed.astype(bool)


def _generate_proteasome_figures(inputs, writer):
    path = _first_existing(
        inputs.scores_dir,
        ("Additional File 8.csv", "proteasome_mass_spec.csv"))
    if path is None and inputs.run_dir is not None:
        run_path = Path(inputs.run_dir) / "processing" / "hits_with_tpm.csv.bz2"
        if run_path.is_file():
            path = run_path
    if path is None:
        writer.skip(
            "proteasome",
            "fig.1_proteasome_mass_spec.proteosome.ms",
            [
                inputs.scores_dir / "Additional File 8.csv",
                inputs.scores_dir / "proteasome_mass_spec.csv",
                (
                    Path(inputs.run_dir) / "processing" / "hits_with_tpm.csv.bz2"
                    if inputs.run_dir is not None else
                    "run/processing/hits_with_tpm.csv.bz2"
                ),
            ],
            "Proteasome mass-spec source table absent.")
        return
    df = pandas.read_csv(path)
    category_col = _first_present(
        df, ("sample_type", "format", "sample", "category", "condition", "gene"))
    value_col = _first_present(df, ("count", "spectra", "intensity", "value"))
    if value_col is None and "hit_id" in df.columns:
        df["count"] = 1
        value_col = "count"
    if category_col is None or value_col is None:
        writer.skip(
            "proteasome",
            "fig.1_proteasome_mass_spec.proteosome.ms",
            [path],
            "Proteasome table must contain category and numeric value columns.")
        return
    import matplotlib.pyplot as plt

    values = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    x = numpy.arange(len(values))
    ax.bar(x, values.values, color=(0.34, 0.46, 0.75),
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(values.index, rotation=35, ha="right")
    ax.set_ylabel(value_col.replace("_", " ").title())
    _despine(ax)
    fig.tight_layout()
    writer.save(fig, "fig.1_proteasome_mass_spec.proteosome.ms", "proteasome")


def _copy_architecture_figures(inputs, writer):
    patterns = (
        "*architecture*.svg", "*architecture*.pdf", "*architecture*.png",
        "*model_information*.svg", "*model_information*.pdf",
        "*model_information*.png", "*model_info*.svg", "*model_info*.pdf",
        "*model_info*.png",
    )
    copied = []
    asset_dir = writer.out_dir / "assets"
    for pattern in patterns:
        for path in inputs.scores_dir.glob(pattern):
            asset_dir.mkdir(parents=True, exist_ok=True)
            target = asset_dir / path.name
            shutil.copy2(path, target)
            copied.append(str(target))
    if copied:
        writer.rows.append({
            "family": "architecture",
            "figure": "architecture_diagrams",
            "status": "generated",
            "paths": ";".join(copied),
            "note": "Copied existing architecture/model-info artwork.",
            "missing": "",
        })
    elif _plot_current_model_information(inputs, writer):
        return
    else:
        writer.skip(
            "architecture",
            "architecture_diagrams",
            [
                inputs.scores_dir / "*architecture*",
                inputs.scores_dir / "*model_info*",
                (
                    Path(inputs.run_dir) / "**" / "manifest.csv"
                    if inputs.run_dir is not None else
                    "run/**/manifest.csv"
                ),
                (
                    Path(inputs.run_dir) / "presentation" / "models" /
                    "weights.csv"
                    if inputs.run_dir is not None else
                    "run/presentation/models/weights.csv"
                ),
            ],
            "Architecture/model-information source artwork absent.")


def _plot_current_model_information(inputs, writer):
    if inputs.run_dir is None:
        return False
    summary = _current_model_counts(inputs.run_dir)
    if summary.empty:
        return False
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, max(2.2, 0.32 * len(summary))))
    y = numpy.arange(len(summary))
    ax.barh(
        y,
        summary["models"],
        color=(0.596, 0.557, 0.835),
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(summary["component"])
    ax.set_xlabel("Final models")
    ax.set_title("Current model ensemble")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.1_predictor_model_information.model_counts",
        "architecture",
        note="Generated from canonical final run model artifacts.",
    )
    return True


def _current_model_counts(run_dir):
    """Return final ensemble sizes from canonical top-level artifacts."""
    run_dir = Path(run_dir)
    manifests = _final_model_manifest_paths(run_dir)
    rows = []
    for path in manifests:
        try:
            df = pandas.read_csv(path, usecols=["model_name"])
        except (OSError, ValueError):
            continue
        if df.empty:
            continue
        label = _manifest_component_label(path, run_dir)
        rows.append((label, int(df["model_name"].nunique())))

    presentation_weights = (
        run_dir / "presentation" / "models" / "weights.csv")
    try:
        weights = pandas.read_csv(presentation_weights, index_col=0)
    except (OSError, ValueError):
        weights = pandas.DataFrame()
    if not weights.empty:
        rows.append(("Presentation", int(weights.index.nunique())))

    if not rows:
        return pandas.DataFrame(columns=["component", "models"])
    return (
        pandas.DataFrame(rows, columns=["component", "models"])
        .groupby("component", as_index=False)["models"]
        .sum()
        .sort_values("models", ascending=True)
    )


def _final_model_manifest_paths(run_dir):
    """Return model manifests that define the release's final ensembles."""
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "affinity" / "models.combined" / "manifest.csv",
    ]
    candidates.extend(sorted(
        (run_dir / "processing").glob(
            "models.selected.*/manifest.csv")))
    return tuple(path for path in candidates if path.is_file())


def _manifest_component_label(path, run_dir):
    try:
        rel = path.relative_to(run_dir)
    except ValueError:
        rel = path
    parts = rel.parts
    if "affinity" in parts:
        return "Affinity"
    if "presentation" in parts:
        return "Presentation"
    if "processing" in parts:
        parent = path.parent.name
        if parent.startswith("models.selected."):
            return "Processing %s" % parent.replace("models.selected.", "").replace("_", " ")
        return "Processing"
    return path.parent.name.replace("_", " ")


def _plot_scatter_triptych_from_pivot(
        pivot, predictor_info, candidate, metric_label, max_points, name, family,
        writer, predictors, baselines=None):
    import matplotlib.pyplot as plt

    if baselines is None:
        baselines = _external_baselines_in_predictors(
            predictors, pivot.columns)
    if not baselines:
        writer.skip(
            family, name,
            [predictor for predictor, _ in predictors.external_baselines],
            "No external baseline predictors found.")
        return

    n_cols = len(baselines)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(max(2.4, 2.35 * n_cols), 2.2),
        squeeze=False)
    for ax, (baseline, _suffix) in zip(axes[0], baselines):
        sub = pivot[[baseline, candidate]].replace(
            [numpy.inf, -numpy.inf], numpy.nan).dropna()
        _scatter_with_winner_colors(
            ax, sub[baseline].values, sub[candidate].values,
            baseline, candidate, predictor_info, max_points)
        _finish_metric_scatter(
            ax,
            _short_label(predictor_info, baseline),
            _short_label(predictor_info, candidate),
            "%s vs %s" % (
                _short_label(predictor_info, candidate),
                _short_label(predictor_info, baseline)),
            metric_label)
    fig.tight_layout(w_pad=1.0)
    writer.save(fig, name, family)


def _pivot_all_lengths(scores, metric):
    rows = _all_length_rows(scores)
    return rows.pivot_table(
        index="sample_id",
        columns="predictor",
        values=metric,
        aggfunc="mean")


def _normalize_score_predictors(scores):
    if "predictor" not in scores.columns:
        return scores
    scores = scores.copy()
    scores["predictor"] = scores["predictor"].map(_normalize_predictor_name)
    return scores


def _normalize_predictor_name(predictor):
    predictor = str(predictor)
    suffix = "_affinity"
    if predictor.endswith(suffix):
        return predictor[:-len(suffix)]
    return predictor


def _all_length_rows(scores):
    if "length_label" in scores.columns:
        return scores.loc[scores["length_label"] == "All"].copy()
    if "length" in scores.columns:
        return scores.loc[scores["length"].isnull()].copy()
    return scores.copy()


def _monoallelic_index_columns(scores):
    for candidates in (
            ("allele", "peptide_length"),
            ("allele", "length"),
            ("allele",),
            ("sample_id",),
    ):
        if all(
                column in scores.columns and (
                    column not in ("peptide_length", "length") or
                    scores[column].notnull().any()
                )
                for column in candidates):
            return list(candidates)
    return scores.index


def _scatter_with_winner_colors(
        ax, x, y, x_predictor, y_predictor, predictor_info, max_points):
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    mask = numpy.isfinite(x) & numpy.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) > max_points:
        rng = numpy.random.default_rng(0)
        indices = rng.choice(len(x), size=max_points, replace=False)
        x = x[indices]
        y = y[indices]
    x_color = _predictor_color(predictor_info, x_predictor)
    y_color = _predictor_color(predictor_info, y_predictor)
    colors = [y_color if y_val >= x_val else x_color
              for x_val, y_val in zip(x, y)]
    ax.scatter(
        x, y, c=colors, s=14, alpha=0.78,
        edgecolor="white", linewidth=0.2, rasterized=False)
    _add_diagonal(ax, x, y)


def _finish_metric_scatter(ax, x_label, y_label, title, metric_label):
    ax.set_title(title)
    ax.set_xlabel("%s %s" % (x_label, metric_label))
    ax.set_ylabel("%s %s" % (y_label, metric_label))
    _set_unit_limits(ax)
    _despine(ax)


def _set_unit_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low = max(0.0, min(xmin, ymin))
    high = min(1.0, max(xmax, ymax))
    if high <= low:
        low, high = 0.0, 1.0
    pad = (high - low) * 0.05
    ax.set_xlim(max(0.0, low - pad), min(1.0, high + pad))
    ax.set_ylim(max(0.0, low - pad), min(1.0, high + pad))
    ax.set_aspect("equal", adjustable="box")


def _add_diagonal(ax, x, y):
    finite = numpy.asarray(list(x) + list(y), dtype=float)
    finite = finite[numpy.isfinite(finite)]
    if len(finite):
        low = max(0.0, float(finite.min()))
        high = min(1.0, float(finite.max()))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        low, high = 0.0, 1.0
    pad = (high - low) * 0.05
    low = max(0.0, low - pad)
    high = min(1.0, high + pad)
    ax.plot([low, high], [low, high], color="0.35", linewidth=0.8, zorder=0)


def _allele_locus(value):
    locus = allele_locus_name(value)
    return locus if locus in ("HLA-A", "HLA-B", "HLA-C", "H2") else "other"


def _first_existing(directory, names):
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def _write_manifest(out_dir, rows):
    path = Path(out_dir) / "manifest.csv"
    pandas.DataFrame(rows).to_csv(path, index=False)


def _write_missing_inputs(out_dir, rows):
    skipped = [
        row for row in rows if row["status"] in ("skipped", "failed")
    ]
    path = Path(out_dir) / "missing_inputs.md"
    with open(path, "w") as fd:
        fd.write("# Missing or failed paper-figure inputs\n\n")
        if not skipped:
            fd.write("All requested figure families were generated.\n")
            return
        for row in skipped:
            fd.write("## %s / %s\n\n" % (row["family"], row["figure"]))
            if row["missing"]:
                fd.write("Missing: `%s`\n\n" % row["missing"].replace(";", "`, `"))
            fd.write("%s\n\n" % row["note"])


# Module-level parser for sphinx autoprogram; behaves like the legacy
# ``mhcflurry-*`` command modules.
parser = make_parser()


if __name__ == "__main__":
    raise SystemExit(run_argv(os.sys.argv[1:]))

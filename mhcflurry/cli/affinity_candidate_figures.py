# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Build common-cohort affinity figure inputs for shortlisted candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy
import pandas

from mhcflurry.experiment_archive import sha256_file


BASELINE_PUBLIC_COMPARISON = "baseline-vs-public-no-additional-ms"
PUBLIC_COMPARISON = "comparison-vs-public-no-additional-ms"
IDENTITY_COLUMNS = ("source_file", "hla", "peptide", "hit")
BENCHMARK_METADATA_COLUMNS = (
    "source_file",
    "protein_accession",
    "sample_id",
    "hla",
    "peptide",
    "peptide_len",
    "n_flank",
    "c_flank",
    "hit",
    "allele",
)
EXTERNAL_JOIN_COLUMNS = (
    "protein_accession",
    "sample_id",
    "hla",
    "peptide",
    "n_flank",
    "c_flank",
    "hit",
)


def make_parser(prog="mhcflurry eval affinity-candidate-figures"):
    """Build the shortlisted-candidate figure parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--factorial-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Candidate condition to include; repeat for each finalist.",
    )
    parser.add_argument(
        "--public-predictor-name",
        default="mhcflurry_public_2_2",
        help="Column/legend name for the pinned public predictor.",
    )
    parser.add_argument(
        "--external-predictions",
        action="append",
        default=[],
        metavar="CSV[.bz2]",
        help=(
            "Benchmark-aligned table containing NetMHCpan/MixMHCpred "
            "columns. Repeat to merge multiple tables. Every candidate row "
            "must match exactly one external row."
        ),
    )
    parser.add_argument(
        "--formats", default="svg,pdf,png",
        help="Paper-figure output formats. Default: %(default)s.",
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="Build reusable prediction/score tables without rendering figures.",
    )
    return parser


def _comparison_dir(factorial_dir, condition, baseline):
    if condition == baseline:
        return factorial_dir / BASELINE_PUBLIC_COMPARISON
    return factorial_dir / condition / PUBLIC_COMPARISON


def _load_comparison(factorial_dir, condition, baseline):
    comparison = _comparison_dir(factorial_dir, condition, baseline)
    summary_path = comparison / "affinity" / "summary.json"
    predictions_path = comparison / "affinity" / "predictions.csv.bz2"
    if not summary_path.is_file():
        raise ValueError(
            "Missing direct public summary for %s: %s" % (
                condition, summary_path))
    if not predictions_path.is_file():
        raise ValueError(
            "Missing held-out predictions for %s: %s" % (
                condition, predictions_path))
    summary = json.loads(summary_path.read_text())
    identity = summary.get("benchmark_identity")
    if not identity or not identity.get("sha256"):
        raise ValueError(
            "Comparison lacks benchmark_identity: %s" % summary_path)
    return comparison, summary_path, predictions_path, identity


def _predictor_info_row(name, description, primary=False, color=None):
    return {
        "predictor": name,
        "description": description,
        "primary": bool(primary),
        "color": color,
        "short": name,
        "detail": description,
        "higher_is_better": True,
    }


def _merge_external_predictions(combined, paths, canonical_columns):
    """Join canonical external scores with strict row-coverage validation."""
    records = []
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Missing external prediction table: %s" % path)
        external = pandas.read_csv(path)
        missing_keys = [
            column for column in EXTERNAL_JOIN_COLUMNS
            if column not in combined or column not in external
        ]
        if missing_keys:
            raise ValueError(
                "External prediction join lacks stable key column(s) in %s: "
                "%s" % (path, ", ".join(missing_keys)))
        score_columns = [
            column for column in external.columns
            if column in canonical_columns
        ]
        if not score_columns:
            raise ValueError(
                "External prediction table has no supported predictor "
                "columns: %s" % path)
        duplicate_columns = [
            column for column in score_columns if column in combined
        ]
        if duplicate_columns:
            raise ValueError(
                "External predictor column already exists before joining %s: "
                "%s" % (path, ", ".join(duplicate_columns)))
        if combined.duplicated(list(EXTERNAL_JOIN_COLUMNS)).any():
            raise ValueError(
                "Candidate benchmark rows are not unique on external join "
                "columns")
        if external.duplicated(list(EXTERNAL_JOIN_COLUMNS)).any():
            raise ValueError(
                "External prediction rows are not unique on join columns: %s" %
                path)

        original_rows = len(combined)
        combined["_candidate_row_order"] = numpy.arange(original_rows)
        combined = combined.merge(
            external[list(EXTERNAL_JOIN_COLUMNS) + score_columns],
            on=list(EXTERNAL_JOIN_COLUMNS),
            how="left",
            sort=False,
            validate="one_to_one",
            indicator="_external_merge",
        )
        unmatched = int((combined["_external_merge"] != "both").sum())
        if unmatched:
            raise ValueError(
                "External prediction table failed to cover %d of %d candidate "
                "rows: %s" % (unmatched, original_rows, path))
        combined = combined.sort_values("_candidate_row_order")
        combined = combined.drop(
            columns=["_candidate_row_order", "_external_merge"])
        combined.index = pandas.RangeIndex(len(combined))
        if len(combined) != original_rows:
            raise ValueError(
                "External prediction join changed candidate row count: %s" %
                path)
        companion = Path("%s.provenance.json" % path)
        records.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "source_rows": len(external),
            "matched_candidate_rows": original_rows,
            "join_columns": list(EXTERNAL_JOIN_COLUMNS),
            "predictor_columns": score_columns,
            "finite_prediction_rows": {
                column: int(numpy.isfinite(
                    pandas.to_numeric(combined[column], errors="coerce")
                ).sum())
                for column in score_columns
            },
            "companion_provenance": (
                {
                    "path": str(companion),
                    "sha256": sha256_file(companion),
                }
                if companion.is_file() else None
            ),
        })
    return combined, records


def build_candidate_figure_inputs(
        factorial_dir, out_dir, conditions, public_predictor_name,
        external_predictions=()):
    """Write and return common-cohort predictions and score metadata."""
    from . import paper_figures

    factorial_dir = Path(factorial_dir).resolve()
    out_dir = Path(out_dir).resolve()
    manifest_path = factorial_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing factorial manifest: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text())
    baseline = manifest["baseline_condition"]
    known = {record["condition"] for record in manifest["records"]}
    conditions = list(dict.fromkeys(conditions))
    unknown = sorted(set(conditions) - known)
    if unknown:
        raise ValueError("Unknown factorial condition(s): %s" % ", ".join(unknown))
    if not conditions:
        raise ValueError("At least one --condition is required")
    if public_predictor_name in conditions:
        raise ValueError(
            "--public-predictor-name collides with a condition: %s" %
            public_predictor_name)

    records = []
    expected_identity = None
    combined = None
    public_scores = None
    public_prediction_max_abs_diff = 0.0
    predictor_info_rows = []
    candidate_columns = []
    canonical_external_columns = {
        name
        for name in paper_figures.DEFAULT_PREDICTOR_HIGHER_IS_BETTER
        if name.startswith("netmhcpan") or name == "mixmhcpred"
    }

    for index, condition in enumerate(conditions):
        comparison, summary_path, predictions_path, identity = _load_comparison(
            factorial_dir, condition, baseline)
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(
                "Candidate comparisons use different held-out rows: %s has "
                "%s; expected %s" % (
                    condition, identity["sha256"],
                    expected_identity["sha256"]))

        predictions = pandas.read_csv(predictions_path)
        missing = [column for column in (*IDENTITY_COLUMNS, "a_score", "b_score")
                   if column not in predictions]
        if missing:
            raise ValueError(
                "Prediction table for %s lacks: %s" % (
                    condition, ", ".join(missing)))
        if combined is None:
            metadata_columns = [
                column for column in BENCHMARK_METADATA_COLUMNS
                if column in predictions
            ]
            external_columns = [
                column for column in predictions.columns
                if column in canonical_external_columns
            ]
            combined = predictions[metadata_columns + external_columns].copy()
            if public_predictor_name in combined.columns:
                raise ValueError(
                    "--public-predictor-name collides with benchmark metadata "
                    "or an external predictor column: %s" %
                    public_predictor_name)
            public_scores = predictions["b_score"].to_numpy(copy=True)
            combined[public_predictor_name] = public_scores
            predictor_info_rows.append(_predictor_info_row(
                public_predictor_name,
                "Pinned public 2.2 models.no_additional_ms affinity ensemble",
                color="#4c78a8",
            ))
        else:
            if len(predictions) != len(combined):
                raise ValueError(
                    "Prediction row count changed for %s: %d versus %d" % (
                        condition, len(predictions), len(combined)))
            comparison_public_scores = predictions["b_score"].to_numpy()
            finite_difference = numpy.abs(
                comparison_public_scores - public_scores)
            max_abs_diff = float(numpy.nanmax(finite_difference))
            public_prediction_max_abs_diff = max(
                public_prediction_max_abs_diff, max_abs_diff)
            if not numpy.allclose(
                    comparison_public_scores, public_scores,
                    rtol=0.0, atol=1e-7, equal_nan=True):
                raise ValueError(
                    "Pinned public predictions changed across comparisons: %s" %
                    condition)

        candidate_column = condition
        if candidate_column in combined.columns:
            raise ValueError(
                "Condition name collides with benchmark metadata or an "
                "external predictor column: %s" % candidate_column)
        combined[candidate_column] = predictions["a_score"].to_numpy()
        candidate_columns.append(candidate_column)
        predictor_info_rows.append(_predictor_info_row(
            candidate_column,
            "MHCflurry affinity-factorial candidate %s" % condition,
            primary=index == 0,
        ))
        records.append({
            "condition": condition,
            "comparison_dir": str(comparison),
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "predictions_path": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "benchmark_identity_sha256": identity["sha256"],
        })

    combined, external_prediction_sources = _merge_external_predictions(
        combined,
        external_predictions,
        canonical_external_columns,
    )

    external_columns = [
        column for column in combined.columns
        if column in canonical_external_columns
    ]
    for column in external_columns:
        predictor_info_rows.append({
            **_predictor_info_row(
                column, "External benchmark predictor %s" % column),
            "higher_is_better": paper_figures.DEFAULT_PREDICTOR_HIGHER_IS_BETTER[
                column],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_out = out_dir / "benchmark.monoallelic.csv.bz2"
    predictor_info_out = out_dir / "predictor_info.csv"
    scores_out = out_dir / "accuracy_scores.monoallelic.csv"
    combined.to_csv(predictions_out, index=False)
    predictor_info = pandas.DataFrame(predictor_info_rows).drop_duplicates(
        "predictor", keep="first")
    predictor_info.to_csv(predictor_info_out, index=False)
    indexed_info = predictor_info.set_index("predictor", drop=False)
    predictor_columns = (
        candidate_columns + [public_predictor_name] + external_columns)
    scores = paper_figures.score_saved_prediction_table(
        predictions_out,
        kind="monoallelic",
        predictor_info=indexed_info,
        predictor_columns=predictor_columns,
        external_baselines=tuple(
            (name, name.replace(".", "_"))
            for name in [public_predictor_name] + external_columns
        ),
    )
    scores.to_csv(scores_out, index=False)
    provenance = {
        "schema_version": 1,
        "factorial_dir": str(factorial_dir),
        "factorial_manifest": str(manifest_path),
        "factorial_manifest_sha256": sha256_file(manifest_path),
        "benchmark_identity": expected_identity,
        "conditions": conditions,
        "public_predictor_name": public_predictor_name,
        "public_prediction_max_abs_diff": public_prediction_max_abs_diff,
        "external_predictors_included": external_columns,
        "external_prediction_sources": external_prediction_sources,
        "source_comparisons": records,
        "outputs": {
            "predictions": str(predictions_out),
            "predictor_info": str(predictor_info_out),
            "scores": str(scores_out),
        },
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def run(args):
    """Build common inputs and optionally render the paper-figure suite."""
    from . import paper_figures

    provenance = build_candidate_figure_inputs(
        args.factorial_dir,
        args.out,
        args.condition,
        args.public_predictor_name,
        args.external_predictions,
    )
    if not args.skip_render:
        external = [args.public_predictor_name]
        external.extend(provenance["external_predictors_included"])
        preferred = list(args.condition) + external
        paper_args = paper_figures.make_parser().parse_args([
            "--scores-dir", str(Path(args.out).resolve()),
            "--monoallelic-predictions",
            provenance["outputs"]["predictions"],
            "--out", str(Path(args.out).resolve() / "paper_figures"),
            "--formats", args.formats,
            "--candidate-predictor", args.condition[0],
            "--external-baselines", ",".join(external),
            "--preferred-predictors", ",".join(preferred),
        ])
        status = paper_figures.run(paper_args)
        if status:
            return status
    print(Path(args.out).resolve())
    return 0


def run_argv(argv=None, prog="mhcflurry eval affinity-candidate-figures"):
    """Parse arguments and generate shortlisted-candidate figures."""
    return run(make_parser(prog).parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(run_argv())

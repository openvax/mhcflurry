"""
Extract high-presentation TF rows for release regression fixtures.

Given a TF predictions table from `cross_allele_parity_analysis.py` or
`compare_tf_pytorch_random_outputs.py`, this script:
1. Finds peptide+flank contexts where any allele has presentation score > threshold.
2. Keeps all allele rows for those contexts (including low-scoring alleles).
3. Writes a compact fixture CSV and metadata JSON for unit tests.

Example:
  python scripts/extract_high_presentation_fixture.py \
    --tf-predictions-csv /tmp/mhcflurry-cross-allele-1000-randflanks/tf_predictions.csv.gz \
    --out-csv test/data/master_released_class1_presentation_highscore_rows.csv.gz \
    --out-metadata-json test/data/master_released_class1_presentation_highscore_rows_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CONTEXT_COLUMNS = ["peptide", "n_flank", "c_flank"]
SCORE_COLUMNS = [
    "pres_with_presentation_score",
    "pres_without_presentation_score",
]


def _collect_model_metadata() -> dict:
    metadata = {}
    try:
        from mhcflurry import Class1PresentationPredictor  # pylint: disable=import-error
        from mhcflurry.downloads import (  # pylint: disable=import-error
            configure,
            get_current_release,
        )

        configure()
        predictor = Class1PresentationPredictor.load()
        metadata.update(
            {
                "release": get_current_release(),
                "presentation_provenance": predictor.provenance_string,
                "presentation_internal_affinity_provenance": (
                    predictor.affinity_predictor.provenance_string
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - metadata capture is best-effort
        metadata["model_metadata_error"] = str(exc)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf-predictions-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-metadata-json", required=True)
    parser.add_argument("--high-score-threshold", type=float, default=0.9)
    parser.add_argument("--low-score-threshold", type=float, default=0.2)
    parser.add_argument(
        "--allow-contexts-without-low-alleles",
        action="store_true",
        help=(
            "Do not enforce that each selected peptide+flank context has at least one "
            "allele below --low-score-threshold."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    tf_predictions_path = Path(args.tf_predictions_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_metadata = Path(args.out_metadata_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tf_predictions_path, keep_default_na=False)
    missing = [c for c in CONTEXT_COLUMNS + SCORE_COLUMNS + ["allele"] if c not in df.columns]
    if missing:
        raise ValueError("TF predictions missing required columns: %s" % missing)

    context_max = df.groupby(CONTEXT_COLUMNS, observed=True)[SCORE_COLUMNS].max()
    selected_context_mask = (
        (context_max["pres_with_presentation_score"] > args.high_score_threshold)
        | (context_max["pres_without_presentation_score"] > args.high_score_threshold)
    )
    selected_contexts = context_max[selected_context_mask].reset_index()
    if selected_contexts.empty:
        raise ValueError(
            "No contexts found above high score threshold %.3f"
            % args.high_score_threshold
        )

    selected = df.merge(selected_contexts[CONTEXT_COLUMNS], on=CONTEXT_COLUMNS, how="inner")
    selected = selected.sort_values(CONTEXT_COLUMNS + ["allele"]).reset_index(drop=True)

    expected_allele_count = int(df["allele"].nunique())
    alleles_per_context = selected.groupby(CONTEXT_COLUMNS, observed=True)["allele"].nunique()
    if not (alleles_per_context == expected_allele_count).all():
        bad = alleles_per_context[alleles_per_context != expected_allele_count]
        raise ValueError(
            "Expected %d alleles per selected context; got mismatches for %d contexts."
            % (expected_allele_count, int(bad.shape[0]))
        )

    low_score_stats = {}
    for score_col in SCORE_COLUMNS:
        context_min = selected.groupby(CONTEXT_COLUMNS, observed=True)[score_col].min()
        contexts_with_low = int((context_min < args.low_score_threshold).sum())
        low_score_stats[score_col] = {
            "contexts_with_low_allele": contexts_with_low,
            "context_count": int(context_min.shape[0]),
            "low_score_threshold": args.low_score_threshold,
        }
        if (
            contexts_with_low < int(context_min.shape[0])
            and not args.allow_contexts_without_low_alleles
        ):
            raise ValueError(
                "Selected contexts do not all include low-scoring alleles for %s "
                "(%d/%d below %.3f)."
                % (
                    score_col,
                    contexts_with_low,
                    int(context_min.shape[0]),
                    args.low_score_threshold,
                )
            )

    selected.to_csv(out_csv, index=False, compression="gzip")

    metadata = {
        "source_tf_predictions_csv": str(tf_predictions_path),
        "row_count": int(selected.shape[0]),
        "context_count": int(selected_contexts.shape[0]),
        "allele_count": expected_allele_count,
        "high_score_threshold": float(args.high_score_threshold),
        "low_score_threshold": float(args.low_score_threshold),
        "score_columns": SCORE_COLUMNS,
        "low_score_stats": low_score_stats,
    }
    metadata.update(_collect_model_metadata())

    with open(out_metadata, "w") as out:
        json.dump(metadata, out, indent=2, sort_keys=True)

    print("Wrote fixture rows:", selected.shape[0])
    print("Selected contexts:", selected_contexts.shape[0])
    print("Alleles per context:", expected_allele_count)
    print("Fixture CSV:", out_csv)
    print("Fixture metadata:", out_metadata)


if __name__ == "__main__":
    main()

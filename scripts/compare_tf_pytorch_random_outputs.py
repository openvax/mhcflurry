"""
Large-scale TF vs PyTorch MHCflurry comparison on random peptide/allele examples.

This script is designed to run locally and keep TF/PyTorch imports isolated by
running each backend in a subprocess.

Primary workflow:
    python scripts/compare_tf_pytorch_random_outputs.py run \
      --tf-repo-root /tmp/mhcflurry-master-check \
      --num-examples 120000 \
      --out-dir /tmp/mhcflurry-random-parity

Outputs:
    - dataset.csv.gz
    - pt_predictions.csv.gz
    - tf_predictions.csv.gz
    - diff_summary.json
    - diff_report.txt
    - top_outliers.csv.gz
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Downcasting behavior in `replace` is deprecated.*",
    category=FutureWarning,
)


AA20 = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_ALLELE_REGEX = r"^HLA-[ABC]\*"

BASE_COLUMNS = ["row_id", "peptide", "allele", "n_flank", "c_flank"]
STRING_OUTPUT_COLUMNS = ["pres_with_best_allele", "pres_without_best_allele"]

# A compact default panel: common human class I alleles often used for broad
# population coverage checks, plus a few representative animal alleles.
IEDB_POPCOV_HUMAN_ALLELES = [
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-A*24:02",
    "HLA-A*26:01",
    "HLA-A*30:01",
    "HLA-A*30:02",
    "HLA-A*31:01",
    "HLA-A*33:01",
    "HLA-A*68:01",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-B*15:01",
    "HLA-B*35:01",
    "HLA-B*40:01",
    "HLA-B*44:02",
    "HLA-B*44:03",
    "HLA-B*51:01",
    "HLA-B*53:01",
    "HLA-B*57:01",
    "HLA-B*58:01",
    "HLA-C*03:04",
    "HLA-C*04:01",
    "HLA-C*05:01",
    "HLA-C*06:02",
    "HLA-C*07:01",
    "HLA-C*07:02",
    "HLA-C*08:02",
    "HLA-C*12:03",
    "HLA-C*15:02",
]

EXTRA_ANIMAL_ALLELES = [
    "H2-K*b",
    "H2-D*b",
    "H2-K*d",
    "H2-L*d",
    "DLA-88*01:01",
    "SLA-1*04:01",
]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)


def _self_cmd(*args: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), *args]


def _run_subprocess_json(cmd: list[str]) -> dict:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("No JSON output from command: %s" % " ".join(cmd))
    line = stdout.splitlines()[-1]
    return json.loads(line)


def _run_subprocess(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _append_if_set(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parents[1]


def _random_sequences(rng: np.random.Generator, lengths: np.ndarray) -> list[str]:
    chars = np.array(list(AA20), dtype="<U1")
    result = []
    for length in lengths.tolist():
        if length <= 0:
            result.append("")
        else:
            result.append("".join(rng.choice(chars, size=length)))
    return result


def _generate_dataset(
    num_examples: int,
    alleles: list[str],
    peptide_min_len: int,
    peptide_max_len: int,
    n_flank_max: int,
    c_flank_max: int,
    seed: int,
) -> pd.DataFrame:
    if not alleles:
        raise ValueError("No alleles provided after filtering/intersection.")

    rng = np.random.default_rng(seed)
    peptide_lengths = rng.integers(
        peptide_min_len, peptide_max_len + 1, size=num_examples
    )
    n_flank_lengths = rng.integers(0, n_flank_max + 1, size=num_examples)
    c_flank_lengths = rng.integers(0, c_flank_max + 1, size=num_examples)
    allele_indices = rng.integers(0, len(alleles), size=num_examples)

    df = pd.DataFrame(
        {
            "row_id": np.arange(num_examples, dtype=np.int64),
            "allele": [alleles[i] for i in allele_indices.tolist()],
            "peptide": _random_sequences(rng, peptide_lengths),
            "n_flank": _random_sequences(rng, n_flank_lengths),
            "c_flank": _random_sequences(rng, c_flank_lengths),
        }
    )
    return df


def _apply_allele_panel(
    allele_pool: list[str],
    allele_panel: str,
    allele_regex: str | None,
) -> list[str]:
    selected = list(allele_pool)
    if allele_panel == "iedb_plus_animals":
        requested = IEDB_POPCOV_HUMAN_ALLELES + EXTRA_ANIMAL_ALLELES
        selected = [a for a in requested if a in selected]
    elif allele_panel == "all_hla":
        pattern = re.compile(DEFAULT_ALLELE_REGEX)
        selected = [a for a in selected if pattern.search(a)]
    elif allele_panel == "all":
        pass
    else:
        raise ValueError("Unknown allele panel: %s" % allele_panel)

    if allele_regex:
        pattern = re.compile(allele_regex)
        selected = [a for a in selected if pattern.search(a)]
    return selected


def cmd_backend_metadata(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(args.repo_root).resolve()))

    from mhcflurry import (  # pylint: disable=import-error
        Class1AffinityPredictor,
        Class1PresentationPredictor,
    )
    from mhcflurry.downloads import (  # pylint: disable=import-error
        get_default_class1_models_dir,
        get_default_class1_presentation_models_dir,
        get_default_class1_processing_models_dir,
        get_path,
    )

    class1_models_dir = args.class1_models_dir or get_default_class1_models_dir()
    presentation_models_dir = (
        args.presentation_models_dir or get_default_class1_presentation_models_dir()
    )
    processing_with_flanks_dir = (
        args.processing_with_flanks_models_dir
        or get_default_class1_processing_models_dir()
    )
    processing_without_flanks_dir = (
        args.processing_without_flanks_models_dir
        or get_path("models_class1_processing", "models.selected.no_flank")
    )

    affinity_predictor = Class1AffinityPredictor.load(class1_models_dir)
    presentation_predictor = Class1PresentationPredictor.load(presentation_models_dir)

    with_flanks_lengths = (
        presentation_predictor.processing_predictor_with_flanks.sequence_lengths
        if presentation_predictor.processing_predictor_with_flanks is not None
        else None
    )
    without_flanks_lengths = (
        presentation_predictor.processing_predictor_without_flanks.sequence_lengths
        if presentation_predictor.processing_predictor_without_flanks is not None
        else None
    )

    out = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "class1_models_dir": class1_models_dir,
        "presentation_models_dir": presentation_models_dir,
        "processing_with_flanks_models_dir": processing_with_flanks_dir,
        "processing_without_flanks_models_dir": processing_without_flanks_dir,
        "supported_alleles": sorted(affinity_predictor.supported_alleles),
        "alleles_with_percentile_ranks": sorted(
            affinity_predictor.allele_to_percent_rank_transform.keys()
        ),
        "affinity_supported_peptide_lengths": list(
            affinity_predictor.supported_peptide_lengths
        ),
        "with_flanks_lengths": with_flanks_lengths,
        "without_flanks_lengths": without_flanks_lengths,
        "provenance": {
            "affinity": affinity_predictor.provenance_string,
            "presentation": presentation_predictor.provenance_string,
            "presentation_internal_affinity": (
                presentation_predictor.affinity_predictor.provenance_string
            ),
        },
    }
    print(json.dumps(out, default=_json_default))


def cmd_predict_backend(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(args.repo_root).resolve()))
    # Keep TF/Keras logs from overwhelming stdout in large runs.
    try:
        import logging

        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        from tf_keras.models import Model as _KerasModel  # type: ignore

        _orig_predict = _KerasModel.predict

        def _quiet_predict(self, *p_args, **p_kwargs):
            p_kwargs.setdefault("verbose", 0)
            return _orig_predict(self, *p_args, **p_kwargs)

        _KerasModel.predict = _quiet_predict
    except Exception:
        pass

    from mhcflurry import (  # pylint: disable=import-error
        Class1AffinityPredictor,
        Class1PresentationPredictor,
    )

    df = pd.read_csv(args.input_csv, keep_default_na=False)
    if not BASE_COLUMNS or not all(col in df.columns for col in BASE_COLUMNS):
        raise ValueError("Input CSV missing required columns: %s" % BASE_COLUMNS)

    peptides = df["peptide"].tolist()
    alleles = df["allele"].tolist()
    n_flanks = df["n_flank"].tolist()
    c_flanks = df["c_flank"].tolist()

    affinity_predictor = Class1AffinityPredictor.load(args.class1_models_dir)
    presentation_predictor = Class1PresentationPredictor.load(
        args.presentation_models_dir
    )

    aff_df = affinity_predictor.predict_to_dataframe(
        peptides=peptides,
        alleles=alleles,
        throw=False,
        include_percentile_ranks=True,
        include_confidence_intervals=True,
        centrality_measure=args.centrality_measure,
        model_kwargs={"batch_size": args.batch_size},
    )

    sample_names = alleles
    alleles_map = {allele: [allele] for allele in sorted(set(alleles))}

    pres_with_df = presentation_predictor.predict(
        peptides=peptides,
        alleles=alleles_map,
        sample_names=sample_names,
        n_flanks=n_flanks,
        c_flanks=c_flanks,
        include_affinity_percentile=True,
        verbose=0,
        throw=True,
    ).sort_values("peptide_num")

    pres_without_df = presentation_predictor.predict(
        peptides=peptides,
        alleles=alleles_map,
        sample_names=sample_names,
        n_flanks=None,
        c_flanks=None,
        include_affinity_percentile=True,
        verbose=0,
        throw=True,
    ).sort_values("peptide_num")

    out = df[BASE_COLUMNS].copy()

    out["affinity_prediction"] = aff_df["prediction"].values
    out["affinity_prediction_low"] = aff_df.get("prediction_low", np.nan)
    out["affinity_prediction_high"] = aff_df.get("prediction_high", np.nan)
    out["affinity_prediction_percentile"] = aff_df.get("prediction_percentile", np.nan)

    out["pres_with_affinity"] = pres_with_df["affinity"].values
    out["pres_with_best_allele"] = pres_with_df["best_allele"].astype(str).values
    out["pres_with_affinity_percentile"] = pres_with_df["affinity_percentile"].values
    out["processing_with_score"] = pres_with_df["processing_score"].values
    out["pres_with_processing_score"] = pres_with_df["processing_score"].values
    out["pres_with_presentation_score"] = pres_with_df["presentation_score"].values
    out["pres_with_presentation_percentile"] = pres_with_df[
        "presentation_percentile"
    ].values

    out["pres_without_affinity"] = pres_without_df["affinity"].values
    out["pres_without_best_allele"] = pres_without_df["best_allele"].astype(str).values
    out["pres_without_affinity_percentile"] = pres_without_df[
        "affinity_percentile"
    ].values
    out["processing_without_score"] = pres_without_df["processing_score"].values
    out["pres_without_processing_score"] = pres_without_df["processing_score"].values
    out["pres_without_presentation_score"] = pres_without_df[
        "presentation_score"
    ].values
    out["pres_without_presentation_percentile"] = pres_without_df[
        "presentation_percentile"
    ].values

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False, compression="gzip")


def _numeric_stats(
    merged: pd.DataFrame,
    column: str,
    relative_epsilon: float,
    top_k: int,
) -> tuple[dict, list[dict]]:
    pt_col = pd.to_numeric(merged[f"{column}_pt"], errors="coerce")
    tf_col = pd.to_numeric(merged[f"{column}_tf"], errors="coerce")

    valid_mask = np.isfinite(pt_col.values) & np.isfinite(tf_col.values)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return {"count": 0}, []

    pt_values = pt_col.values[valid_mask].astype(np.float64)
    tf_values = tf_col.values[valid_mask].astype(np.float64)
    diff = pt_values - tf_values
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.abs(tf_values), relative_epsilon)

    pearson = float(np.corrcoef(pt_values, tf_values)[0, 1]) if valid_count > 1 else np.nan

    stats = {
        "count": valid_count,
        "pt_mean": float(pt_values.mean()),
        "tf_mean": float(tf_values.mean()),
        "mean_diff": float(diff.mean()),
        "mean_abs_diff": float(abs_diff.mean()),
        "median_abs_diff": float(np.median(abs_diff)),
        "p95_abs_diff": float(np.percentile(abs_diff, 95)),
        "p99_abs_diff": float(np.percentile(abs_diff, 99)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_rel_diff": float(rel_diff.mean()),
        "p95_rel_diff": float(np.percentile(rel_diff, 95)),
        "p99_rel_diff": float(np.percentile(rel_diff, 99)),
        "max_rel_diff": float(rel_diff.max()),
        "pearson_r": pearson,
    }

    valid_idx = np.where(valid_mask)[0]
    sorted_local = np.argsort(abs_diff)[::-1][:top_k]
    outliers = []
    for rank, local_idx in enumerate(sorted_local, start=1):
        global_idx = valid_idx[local_idx]
        row = merged.iloc[global_idx]
        outliers.append(
            {
                "column": column,
                "rank": rank,
                "row_id": int(row["row_id"]),
                "peptide": row["peptide"],
                "allele": row["allele"],
                "n_flank": row["n_flank"],
                "c_flank": row["c_flank"],
                "pt_value": float(pt_values[local_idx]),
                "tf_value": float(tf_values[local_idx]),
                "signed_diff": float(diff[local_idx]),
                "abs_diff": float(abs_diff[local_idx]),
                "rel_diff": float(rel_diff[local_idx]),
            }
        )
    return stats, outliers


def cmd_analyze(args: argparse.Namespace) -> None:
    pt = pd.read_csv(args.pt_predictions_csv, keep_default_na=False)
    tf = pd.read_csv(args.tf_predictions_csv, keep_default_na=False)

    merged = pt.merge(
        tf,
        on=BASE_COLUMNS,
        suffixes=("_pt", "_tf"),
        how="inner",
    )

    numeric_columns = [
        col for col in pt.columns if col not in BASE_COLUMNS + STRING_OUTPUT_COLUMNS
    ]
    string_columns = [col for col in STRING_OUTPUT_COLUMNS if col in pt.columns]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_rows_pt": int(len(pt)),
        "num_rows_tf": int(len(tf)),
        "num_rows_merged": int(len(merged)),
        "numeric_columns": {},
        "string_columns": {},
    }
    outlier_rows = []

    for col in numeric_columns:
        stats, outliers = _numeric_stats(
            merged, col, relative_epsilon=args.relative_epsilon, top_k=args.top_k
        )
        summary["numeric_columns"][col] = stats
        outlier_rows.extend(outliers)

    for col in string_columns:
        pt_values = merged[f"{col}_pt"].astype(str)
        tf_values = merged[f"{col}_tf"].astype(str)
        mismatches = pt_values != tf_values
        summary["string_columns"][col] = {
            "count": int(len(merged)),
            "mismatch_count": int(mismatches.sum()),
            "mismatch_rate": float(mismatches.mean()),
        }

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as out:
        json.dump(summary, out, indent=2, sort_keys=True)

    if outlier_rows:
        outlier_df = pd.DataFrame(outlier_rows)
        outlier_df.to_csv(args.top_outliers_csv, index=False, compression="gzip")
    else:
        pd.DataFrame().to_csv(args.top_outliers_csv, index=False, compression="gzip")

    report_lines = []
    report_lines.append("TF vs PyTorch random comparison report")
    report_lines.append("generated_at_utc: %s" % summary["generated_at_utc"])
    report_lines.append("rows_pt: %d" % summary["num_rows_pt"])
    report_lines.append("rows_tf: %d" % summary["num_rows_tf"])
    report_lines.append("rows_merged: %d" % summary["num_rows_merged"])
    report_lines.append("")
    report_lines.append("Numeric columns:")
    for col, stats in summary["numeric_columns"].items():
        if stats.get("count", 0) == 0:
            report_lines.append("  %s: no valid numeric pairs" % col)
            continue
        report_lines.append(
            (
                "  %s: mean_abs=%.6g p95_abs=%.6g p99_abs=%.6g max_abs=%.6g "
                "mean_rel=%.6g p99_rel=%.6g max_rel=%.6g r=%.8f"
            )
            % (
                col,
                stats["mean_abs_diff"],
                stats["p95_abs_diff"],
                stats["p99_abs_diff"],
                stats["max_abs_diff"],
                stats["mean_rel_diff"],
                stats["p99_rel_diff"],
                stats["max_rel_diff"],
                stats["pearson_r"],
            )
        )
    report_lines.append("")
    report_lines.append("String columns:")
    for col, stats in summary["string_columns"].items():
        report_lines.append(
            "  %s: mismatch_rate=%.6g (%d/%d)"
            % (
                col,
                stats["mismatch_rate"],
                stats["mismatch_count"],
                stats["count"],
            )
        )

    Path(args.report_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_txt, "w") as out:
        out.write("\n".join(report_lines) + "\n")


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_repo_root = str(Path(args.pytorch_repo_root).resolve())
    tf_repo_root = str(Path(args.tf_repo_root).resolve())

    pt_meta_cmd = _self_cmd("backend-metadata", "--repo-root", pt_repo_root)
    tf_meta_cmd = _self_cmd("backend-metadata", "--repo-root", tf_repo_root)
    for cmd in (pt_meta_cmd, tf_meta_cmd):
        _append_if_set(cmd, "--class1-models-dir", args.class1_models_dir)
        _append_if_set(cmd, "--presentation-models-dir", args.presentation_models_dir)
        _append_if_set(
            cmd,
            "--processing-with-flanks-models-dir",
            args.processing_with_flanks_models_dir,
        )
        _append_if_set(
            cmd,
            "--processing-without-flanks-models-dir",
            args.processing_without_flanks_models_dir,
        )

    pt_meta = _run_subprocess_json(pt_meta_cmd)
    tf_meta = _run_subprocess_json(tf_meta_cmd)

    class1_models_dir = args.class1_models_dir or pt_meta["class1_models_dir"]
    presentation_models_dir = (
        args.presentation_models_dir or pt_meta["presentation_models_dir"]
    )
    processing_with_flanks_models_dir = (
        args.processing_with_flanks_models_dir
        or pt_meta["processing_with_flanks_models_dir"]
    )
    processing_without_flanks_models_dir = (
        args.processing_without_flanks_models_dir
        or pt_meta["processing_without_flanks_models_dir"]
    )

    allele_pool = sorted(
        set(pt_meta["supported_alleles"]).intersection(tf_meta["supported_alleles"])
    )
    if not args.allow_missing_affinity_percentiles:
        pt_percentile = set(pt_meta["alleles_with_percentile_ranks"])
        tf_percentile = set(tf_meta["alleles_with_percentile_ranks"])
        allele_pool = [a for a in allele_pool if (a in pt_percentile and a in tf_percentile)]

    allele_pool = _apply_allele_panel(
        allele_pool=allele_pool,
        allele_panel=args.allele_panel,
        allele_regex=args.allele_regex,
    )
    if not allele_pool:
        raise ValueError(
            "No alleles remain after applying panel '%s' and regex filter."
            % args.allele_panel
        )

    if args.allele_subset_size and len(allele_pool) > args.allele_subset_size:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(
            len(allele_pool), size=args.allele_subset_size, replace=False
        )
        allele_pool = sorted([allele_pool[i] for i in indices.tolist()])

    print(
        "Using %d alleles (panel=%s)." % (len(allele_pool), args.allele_panel)
    )

    pt_min_len, pt_max_len = pt_meta["affinity_supported_peptide_lengths"]
    tf_min_len, tf_max_len = tf_meta["affinity_supported_peptide_lengths"]
    peptide_min_len = max(int(pt_min_len), int(tf_min_len))
    peptide_max_len = min(int(pt_max_len), int(tf_max_len))

    pt_with = pt_meta["with_flanks_lengths"]
    tf_with = tf_meta["with_flanks_lengths"]
    n_flank_max = min(int(pt_with["n_flank"]), int(tf_with["n_flank"]))
    c_flank_max = min(int(pt_with["c_flank"]), int(tf_with["c_flank"]))

    dataset_path = out_dir / "dataset.csv.gz"
    pt_predictions_path = out_dir / "pt_predictions.csv.gz"
    tf_predictions_path = out_dir / "tf_predictions.csv.gz"
    summary_path = out_dir / "diff_summary.json"
    report_path = out_dir / "diff_report.txt"
    outliers_path = out_dir / "top_outliers.csv.gz"

    dataset = _generate_dataset(
        num_examples=args.num_examples,
        alleles=allele_pool,
        peptide_min_len=peptide_min_len,
        peptide_max_len=peptide_max_len,
        n_flank_max=n_flank_max,
        c_flank_max=c_flank_max,
        seed=args.seed,
    )
    dataset.to_csv(dataset_path, index=False, compression="gzip")

    pt_predict_cmd = _self_cmd(
        "predict-backend",
        "--repo-root",
        pt_repo_root,
        "--input-csv",
        str(dataset_path),
        "--output-csv",
        str(pt_predictions_path),
        "--batch-size",
        str(args.batch_size),
        "--centrality-measure",
        args.centrality_measure,
    )
    tf_predict_cmd = _self_cmd(
        "predict-backend",
        "--repo-root",
        tf_repo_root,
        "--input-csv",
        str(dataset_path),
        "--output-csv",
        str(tf_predictions_path),
        "--batch-size",
        str(args.batch_size),
        "--centrality-measure",
        args.centrality_measure,
    )
    for cmd in (pt_predict_cmd, tf_predict_cmd):
        _append_if_set(cmd, "--class1-models-dir", class1_models_dir)
        _append_if_set(cmd, "--presentation-models-dir", presentation_models_dir)
        _append_if_set(
            cmd,
            "--processing-with-flanks-models-dir",
            processing_with_flanks_models_dir,
        )
        _append_if_set(
            cmd,
            "--processing-without-flanks-models-dir",
            processing_without_flanks_models_dir,
        )

    _run_subprocess(pt_predict_cmd)
    _run_subprocess(tf_predict_cmd)

    _run_subprocess(
        _self_cmd(
            "analyze",
            "--pt-predictions-csv",
            str(pt_predictions_path),
            "--tf-predictions-csv",
            str(tf_predictions_path),
            "--summary-json",
            str(summary_path),
            "--report-txt",
            str(report_path),
            "--top-outliers-csv",
            str(outliers_path),
            "--top-k",
            str(args.top_k),
            "--relative-epsilon",
            str(args.relative_epsilon),
        )
    )

    print("Wrote dataset:", dataset_path)
    print("Wrote PT predictions:", pt_predictions_path)
    print("Wrote TF predictions:", tf_predictions_path)
    print("Wrote summary:", summary_path)
    print("Wrote report:", report_path)
    print("Wrote outliers:", outliers_path)


def _add_common_model_dir_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--class1-models-dir", default=None)
    parser.add_argument("--presentation-models-dir", default=None)
    parser.add_argument("--processing-with-flanks-models-dir", default=None)
    parser.add_argument("--processing-without-flanks-models-dir", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--tf-repo-root", required=True)
    run.add_argument("--pytorch-repo-root", default=str(_repo_root_default()))
    run.add_argument("--num-examples", type=int, default=120000)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument(
        "--allele-panel",
        choices=["iedb_plus_animals", "all_hla", "all"],
        default="iedb_plus_animals",
        help=(
            "Preset allele list. 'iedb_plus_animals' uses ~30 common human alleles "
            "plus a few animal alleles."
        ),
    )
    run.add_argument(
        "--allele-regex",
        default=None,
        help="Optional extra regex filter applied after allele panel selection.",
    )
    run.add_argument(
        "--allele-subset-size",
        type=int,
        default=None,
        help="Optional random subset size after panel+regex filtering.",
    )
    run.add_argument(
        "--allow-missing-affinity-percentiles",
        action="store_true",
        help=(
            "If set, allow random alleles with missing affinity percentile calibrations. "
            "Default behavior restricts to alleles with percentiles in both backends."
        ),
    )
    run.add_argument("--batch-size", type=int, default=4096)
    run.add_argument("--centrality-measure", default="mean")
    run.add_argument("--out-dir", default="/tmp/mhcflurry-random-parity")
    run.add_argument("--top-k", type=int, default=25)
    run.add_argument("--relative-epsilon", type=float, default=1e-12)
    _add_common_model_dir_args(run)
    run.set_defaults(func=cmd_run)

    metadata = subparsers.add_parser("backend-metadata")
    metadata.add_argument("--repo-root", required=True)
    _add_common_model_dir_args(metadata)
    metadata.set_defaults(func=cmd_backend_metadata)

    predict = subparsers.add_parser("predict-backend")
    predict.add_argument("--repo-root", required=True)
    predict.add_argument("--input-csv", required=True)
    predict.add_argument("--output-csv", required=True)
    predict.add_argument("--batch-size", type=int, default=4096)
    predict.add_argument("--centrality-measure", default="mean")
    _add_common_model_dir_args(predict)
    predict.set_defaults(func=cmd_predict_backend)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--pt-predictions-csv", required=True)
    analyze.add_argument("--tf-predictions-csv", required=True)
    analyze.add_argument("--summary-json", required=True)
    analyze.add_argument("--report-txt", required=True)
    analyze.add_argument("--top-outliers-csv", required=True)
    analyze.add_argument("--top-k", type=int, default=25)
    analyze.add_argument("--relative-epsilon", type=float, default=1e-12)
    analyze.set_defaults(func=cmd_analyze)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Normalize empty-string optional directory args from subprocess command args.
    for key in [
        "class1_models_dir",
        "presentation_models_dir",
        "processing_with_flanks_models_dir",
        "processing_without_flanks_models_dir",
    ]:
        if hasattr(args, key) and getattr(args, key) == "":
            setattr(args, key, None)

    args.func(args)


if __name__ == "__main__":
    main()

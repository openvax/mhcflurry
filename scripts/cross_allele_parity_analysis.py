"""
Cross-allele TF vs PyTorch parity analysis for MHCflurry.

This script:
1. Selects a limited curated allele panel (default: common human + a few animal).
2. Generates random peptides uniformly across lengths (default: 7-15 when supported).
3. Generates random flanks for each peptide (unique N and C flanks across peptide entries).
4. Builds a full cross-product dataset: peptides x alleles.
5. Runs predictions for PyTorch branch and TF master via subprocess isolation.
6. Produces summary stats, outlier tables, and diagnostic plots.

Example:
  python scripts/cross_allele_parity_analysis.py \
    --tf-repo-root /tmp/mhcflurry-master-check \
    --num-peptides 1000 \
    --out-dir /tmp/mhcflurry-cross-allele-parity
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AA20 = "ACDEFGHIKLMNPQRSTVWY"
BASE_COLUMNS = ["row_id", "peptide", "allele", "n_flank", "c_flank"]
STRING_OUTPUT_COLUMNS = ["pres_with_best_allele", "pres_without_best_allele"]

# Common human class I alleles often used for broad coverage checks.
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

# A few non-human alleles to ensure cross-species sanity.
EXTRA_ANIMAL_ALLELES = [
    "H2-K*b",
    "H2-D*b",
    "H2-K*d",
    "H2-L*d",
    "DLA-88*01:01",
    "SLA-1*04:01",
]


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parents[1]


def _compare_script_path() -> Path:
    return Path(__file__).resolve().parent / "compare_tf_pytorch_random_outputs.py"


def _self_cmd(*args: str) -> list[str]:
    return [sys.executable, str(_compare_script_path()), *args]


def _run_subprocess_json(cmd: list[str]) -> dict:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("No JSON output from command: %s" % " ".join(cmd))
    return json.loads(stdout.splitlines()[-1])


def _run_subprocess(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _append_if_set(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def _select_alleles(
    pt_meta: dict,
    tf_meta: dict,
    allele_panel: str,
    allele_regex: str | None,
    require_percentiles: bool,
) -> list[str]:
    allele_pool = sorted(
        set(pt_meta["supported_alleles"]).intersection(tf_meta["supported_alleles"])
    )
    if require_percentiles:
        pt_percentile = set(pt_meta["alleles_with_percentile_ranks"])
        tf_percentile = set(tf_meta["alleles_with_percentile_ranks"])
        allele_pool = [a for a in allele_pool if a in pt_percentile and a in tf_percentile]

    if allele_panel == "iedb_plus_animals":
        requested = IEDB_POPCOV_HUMAN_ALLELES + EXTRA_ANIMAL_ALLELES
        allele_pool = [a for a in requested if a in allele_pool]
    elif allele_panel == "all_hla":
        pattern = re.compile(r"^HLA-[ABC]\*")
        allele_pool = [a for a in allele_pool if pattern.search(a)]
    elif allele_panel == "all":
        pass
    else:
        raise ValueError("Unknown allele panel: %s" % allele_panel)

    if allele_regex:
        pattern = re.compile(allele_regex)
        allele_pool = [a for a in allele_pool if pattern.search(a)]

    if not allele_pool:
        raise ValueError("No alleles remain after panel/filter selection.")
    return allele_pool


def _lengths_to_sample(
    min_supported: int, max_supported: int, min_requested: int, max_requested: int
) -> list[int]:
    start = max(min_supported, min_requested)
    end = min(max_supported, max_requested)
    if end < start:
        raise ValueError(
            "No overlap in peptide lengths. Supported=[%d,%d], requested=[%d,%d]"
            % (min_supported, max_supported, min_requested, max_requested)
        )
    lengths = list(range(start, end + 1))
    # If 7 is not supported but 8 is, satisfy the user's fallback request.
    if 7 not in lengths and 8 in lengths and min_requested <= 7 <= max_requested:
        return [length for length in lengths if length >= 8]
    return lengths


def _random_sequences(
    rng: np.random.Generator,
    length: int,
    n: int,
    used: set[str],
) -> list[str]:
    chars = np.array(list(AA20), dtype="<U1")
    out: list[str] = []
    while len(out) < n:
        needed = n - len(out)
        batch = max(needed * 2, 64)
        arr = rng.choice(chars, size=(batch, length))
        for row in arr:
            seq = "".join(row.tolist())
            if seq in used:
                continue
            used.add(seq)
            out.append(seq)
            if len(out) >= n:
                break
    return out


def _generate_uniform_peptides(
    num_peptides: int,
    lengths: list[int],
    n_flank_length: int,
    c_flank_length: int,
    seed: int,
) -> pd.DataFrame:
    if num_peptides <= 0:
        raise ValueError("num_peptides must be positive")
    if not lengths:
        raise ValueError("No lengths provided")
    if n_flank_length <= 0 or c_flank_length <= 0:
        raise ValueError(
            "Flank lengths must be positive (got n=%d c=%d)."
            % (n_flank_length, c_flank_length)
        )

    rng = np.random.default_rng(seed)
    per_length = num_peptides // len(lengths)
    remainder = num_peptides % len(lengths)

    rows = []
    used: set[str] = set()
    used_n_flanks: set[str] = set()
    used_c_flanks: set[str] = set()
    n_flanks = _random_sequences(
        rng=rng, length=n_flank_length, n=num_peptides, used=used_n_flanks
    )
    c_flanks = _random_sequences(
        rng=rng, length=c_flank_length, n=num_peptides, used=used_c_flanks
    )

    peptide_id = 0
    for i, length in enumerate(lengths):
        count = per_length + (1 if i < remainder else 0)
        seqs = _random_sequences(rng=rng, length=length, n=count, used=used)
        for seq in seqs:
            rows.append(
                {
                    "peptide_id": peptide_id,
                    "peptide": seq,
                    "peptide_length": length,
                    "n_flank": n_flanks[peptide_id],
                    "c_flank": c_flanks[peptide_id],
                }
            )
            peptide_id += 1
    return pd.DataFrame(rows)


def _cross_join_dataset(peptides_df: pd.DataFrame, alleles: list[str]) -> pd.DataFrame:
    n_peptides = len(peptides_df)
    n_alleles = len(alleles)
    total = n_peptides * n_alleles

    peptide_vals = peptides_df["peptide"].values
    length_vals = peptides_df["peptide_length"].values
    peptide_id_vals = peptides_df["peptide_id"].values
    n_flank_vals = peptides_df["n_flank"].values
    c_flank_vals = peptides_df["c_flank"].values

    peptide_repeated = np.repeat(peptide_vals, n_alleles)
    length_repeated = np.repeat(length_vals, n_alleles)
    peptide_id_repeated = np.repeat(peptide_id_vals, n_alleles)
    n_flank_repeated = np.repeat(n_flank_vals, n_alleles)
    c_flank_repeated = np.repeat(c_flank_vals, n_alleles)
    allele_tiled = np.tile(np.array(alleles), n_peptides)

    out = pd.DataFrame(
        {
            "row_id": np.arange(total, dtype=np.int64),
            "peptide_id": peptide_id_repeated,
            "peptide_length": length_repeated,
            "peptide": peptide_repeated,
            "allele": allele_tiled,
            "n_flank": n_flank_repeated,
            "c_flank": c_flank_repeated,
        }
    )
    return out


def _pre_run_sanity_checks(peptides_df: pd.DataFrame, dataset: pd.DataFrame) -> None:
    for col in ["peptide", "n_flank", "c_flank"]:
        if not peptides_df[col].is_unique:
            dupes = int(peptides_df[col].duplicated().sum())
            raise ValueError(
                "Pre-run sanity check failed: %s has %d repeated values "
                "across peptide entries."
                % (col, dupes)
            )
    if peptides_df[["peptide", "n_flank", "c_flank"]].duplicated().any():
        raise ValueError(
            "Pre-run sanity check failed: duplicate (peptide, n_flank, c_flank) tuples."
        )
    if dataset[["peptide", "allele", "n_flank", "c_flank"]].duplicated().any():
        raise ValueError(
            "Pre-run sanity check failed: duplicate (peptide, allele, n_flank, c_flank) rows."
        )


def _enforce_presentation_score_requirements(
    predictions: pd.DataFrame,
    label: str,
    min_fraction_above: float = 0.01,
    threshold: float = 0.2,
    min_max_score: float = 0.9,
) -> dict:
    stats = {}
    for col in ["pres_with_presentation_score", "pres_without_presentation_score"]:
        if col not in predictions.columns:
            continue
        scores = pd.to_numeric(predictions[col], errors="coerce")
        frac = float((scores > threshold).mean())
        max_score = float(scores.max())
        stats[col] = {
            "fraction_gt_threshold": frac,
            "threshold": threshold,
            "max_score": max_score,
            "min_required_fraction": min_fraction_above,
            "min_required_max_score": min_max_score,
        }
        if frac < min_fraction_above:
            raise ValueError(
                "%s failed presentation sanity: %s has %.6f fraction > %.3f, need >= %.3f"
                % (label, col, frac, threshold, min_fraction_above)
            )
        if max_score <= min_max_score:
            raise ValueError(
                "%s failed presentation sanity: %s max %.6f, need > %.3f"
                % (label, col, max_score, min_max_score)
            )
    return stats


def _numeric_output_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLUMNS + STRING_OUTPUT_COLUMNS]


def _make_diff_frame(merged: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    out = merged[BASE_COLUMNS].copy()
    for col in numeric_columns:
        pt = pd.to_numeric(merged[f"{col}_pt"], errors="coerce")
        tf = pd.to_numeric(merged[f"{col}_tf"], errors="coerce")
        out[f"{col}_pt"] = pt
        out[f"{col}_tf"] = tf
        out[f"{col}_diff"] = pt - tf
        out[f"{col}_abs_diff"] = (pt - tf).abs()
    return out


def _per_output_summary(diff_df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_columns:
        abs_diff = pd.to_numeric(diff_df[f"{col}_abs_diff"], errors="coerce").dropna()
        if abs_diff.empty:
            continue
        rows.append(
            {
                "output": col,
                "count": int(abs_diff.shape[0]),
                "mean_abs_diff": float(abs_diff.mean()),
                "median_abs_diff": float(abs_diff.median()),
                "p95_abs_diff": float(abs_diff.quantile(0.95)),
                "p99_abs_diff": float(abs_diff.quantile(0.99)),
                "max_abs_diff": float(abs_diff.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)


def _break_thresholds_for_output(output: str) -> float:
    if output.startswith("affinity_prediction") or output.endswith("_affinity"):
        return 0.1  # nM
    if "affinity_percentile" in output or "presentation_percentile" in output:
        return 0.1  # percentile points
    return 1e-4  # score-scale outputs


def _break_analysis(
    merged: pd.DataFrame,
    diff_df: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    break_rows = []
    max_abs_per_row = np.zeros(len(diff_df), dtype=np.float64)
    max_output_per_row = np.array([""] * len(diff_df), dtype=object)

    for col in numeric_columns:
        abs_col = pd.to_numeric(diff_df[f"{col}_abs_diff"], errors="coerce").fillna(0.0)
        abs_vals = abs_col.values
        threshold = _break_thresholds_for_output(col)
        break_mask = abs_vals > threshold
        break_rows.append(
            {
                "output": col,
                "threshold_abs_diff": threshold,
                "break_count": int(break_mask.sum()),
                "break_rate": float(break_mask.mean()),
            }
        )
        improve_mask = abs_vals > max_abs_per_row
        max_abs_per_row[improve_mask] = abs_vals[improve_mask]
        max_output_per_row[improve_mask] = col

    row_summary = merged[BASE_COLUMNS].copy()
    row_summary["max_abs_diff_any_output"] = max_abs_per_row
    row_summary["worst_output"] = max_output_per_row
    return pd.DataFrame(break_rows), row_summary


def _plot_output_ranges(summary_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary_df.sort_values("max_abs_diff", ascending=True)
    y = np.arange(len(plot_df))

    fig_h = max(4.5, 0.3 * len(plot_df))
    plt.figure(figsize=(10, fig_h))
    plt.hlines(y, 0, plot_df["max_abs_diff"], color="#1f77b4", linewidth=2.0, label="max")
    plt.hlines(
        y,
        0,
        plot_df["p99_abs_diff"],
        color="#ff7f0e",
        linewidth=2.0,
        alpha=0.9,
        label="p99",
    )
    plt.hlines(
        y,
        0,
        plot_df["p95_abs_diff"],
        color="#2ca02c",
        linewidth=2.0,
        alpha=0.9,
        label="p95",
    )
    plt.plot(plot_df["mean_abs_diff"], y, "o", color="#d62728", label="mean", markersize=4)
    plt.yticks(y, plot_df["output"])
    plt.xscale("log")
    plt.xlabel("Absolute difference (log scale)")
    plt.ylabel("Output")
    plt.title("TF vs PyTorch absolute difference ranges by output")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_row_max_hist(row_summary: pd.DataFrame, out_path: Path) -> None:
    vals = row_summary["max_abs_diff_any_output"].values
    bins = np.logspace(
        np.log10(max(vals.min(), 1e-12)),
        np.log10(max(vals.max(), 1e-12)) if vals.max() > 0 else -12,
        60,
    )
    bins = np.unique(bins)
    if bins.shape[0] < 2:
        bins = np.array([1e-12, 1e-11])

    plt.figure(figsize=(8, 4.5))
    plt.hist(vals, bins=bins, color="#1f77b4", alpha=0.85)
    plt.xscale("log")
    plt.xlabel("Max absolute difference across outputs (per pMHC)")
    plt.ylabel("Count")
    plt.title("Per-row worst-case difference distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_length_breakdown(
    diff_df: pd.DataFrame,
    target_output: str,
    out_path: Path,
) -> None:
    abs_col = f"{target_output}_abs_diff"
    if abs_col not in diff_df.columns:
        return

    grouped = (
        diff_df.groupby(diff_df["peptide"].str.len())[abs_col]
        .agg(["mean", "median", "max"])
        .reset_index()
        .rename(columns={"peptide": "peptide_length"})
    )
    grouped = grouped.sort_values("peptide_length")

    plt.figure(figsize=(8, 4.5))
    plt.plot(grouped["peptide_length"], grouped["mean"], marker="o", label="mean")
    plt.plot(grouped["peptide_length"], grouped["median"], marker="o", label="median")
    plt.plot(grouped["peptide_length"], grouped["max"], marker="o", label="max")
    plt.yscale("log")
    plt.xlabel("Peptide length")
    plt.ylabel(f"{target_output} abs diff (log)")
    plt.title(f"Difference vs peptide length: {target_output}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _safe_plot_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _plot_per_output_hists(
    diff_df: pd.DataFrame,
    numeric_columns: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for output in numeric_columns:
        abs_col = f"{output}_abs_diff"
        if abs_col not in diff_df.columns:
            continue
        vals = pd.to_numeric(diff_df[abs_col], errors="coerce").dropna().values.astype(float)
        if vals.size == 0:
            continue

        positive = vals[vals > 0]
        if positive.size > 0:
            epsilon = positive.min() / 10.0
        else:
            epsilon = 1e-18
        plot_vals = np.where(vals > 0, vals, epsilon)

        lo = float(plot_vals.min())
        hi = float(plot_vals.max())
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi > lo:
            bins = np.logspace(np.log10(lo), np.log10(hi), 60)
            bins = np.unique(bins)
            if bins.shape[0] < 2:
                bins = np.array([lo, hi + lo * 1e-6], dtype=float)
        else:
            bins = np.array([lo * 0.9, hi * 1.1], dtype=float)
            bins = np.where(bins <= 0, epsilon, bins)

        plt.figure(figsize=(8, 4.5))
        plt.hist(plot_vals, bins=bins, color="#1f77b4", alpha=0.85)
        plt.xscale("log")
        plt.xlabel(f"{output} absolute difference")
        plt.ylabel("Count")
        plt.title(
            f"{output} abs diff histogram (zero_count={(vals == 0).sum()}, max={vals.max():.3g})"
        )
        plt.tight_layout()
        plot_name = f"{_safe_plot_name(output)}.png"
        plt.savefig(out_dir / plot_name, dpi=180)
        plt.close()


def _write_break_report(
    out_path: Path,
    allele_count: int,
    peptide_count: int,
    pmhc_count: int,
    lengths: list[int],
    break_df: pd.DataFrame,
    row_summary: pd.DataFrame,
    top_rows: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Cross-allele TF vs PyTorch break analysis")
    lines.append(f"alleles: {allele_count}")
    lines.append(f"peptides: {peptide_count}")
    lines.append(f"pMHC rows: {pmhc_count}")
    lines.append(f"peptide lengths sampled: {lengths}")
    lines.append("")
    lines.append("Thresholded break counts by output:")
    for _, row in break_df.sort_values("break_rate", ascending=False).iterrows():
        lines.append(
            "  {output}: threshold={thr:.3g}, break_count={cnt}, break_rate={rate:.6g}".format(
                output=row["output"],
                thr=row["threshold_abs_diff"],
                cnt=int(row["break_count"]),
                rate=float(row["break_rate"]),
            )
        )
    lines.append("")
    lines.append(
        "Per-row max abs diff summary: mean={:.6g}, p95={:.6g}, p99={:.6g}, max={:.6g}".format(
            float(row_summary["max_abs_diff_any_output"].mean()),
            float(row_summary["max_abs_diff_any_output"].quantile(0.95)),
            float(row_summary["max_abs_diff_any_output"].quantile(0.99)),
            float(row_summary["max_abs_diff_any_output"].max()),
        )
    )
    lines.append("")
    lines.append("Top worst rows:")
    for _, row in top_rows.iterrows():
        lines.append(
            "  row_id={row_id} peptide={peptide} allele={allele} "
            "worst_output={worst_output} max_abs_diff={max_abs_diff_any_output:.6g}".format(
                **row.to_dict()
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf-repo-root", required=True)
    parser.add_argument("--pytorch-repo-root", default=str(_repo_root_default()))
    parser.add_argument("--num-peptides", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument(
        "--allele-panel",
        choices=["iedb_plus_animals", "all_hla", "all"],
        default="iedb_plus_animals",
    )
    parser.add_argument("--allele-regex", default=None)
    parser.add_argument(
        "--allow-missing-affinity-percentiles",
        action="store_true",
        help="Allow alleles that lack calibrated affinity percentiles.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--centrality-measure", default="mean")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out-dir", default="/tmp/mhcflurry-cross-allele-parity")
    parser.add_argument("--class1-models-dir", default=None)
    parser.add_argument("--presentation-models-dir", default=None)
    parser.add_argument("--processing-with-flanks-models-dir", default=None)
    parser.add_argument("--processing-without-flanks-models-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    presentation_models_dir = args.presentation_models_dir or pt_meta["presentation_models_dir"]
    processing_with_flanks_models_dir = (
        args.processing_with_flanks_models_dir
        or pt_meta["processing_with_flanks_models_dir"]
    )
    processing_without_flanks_models_dir = (
        args.processing_without_flanks_models_dir
        or pt_meta["processing_without_flanks_models_dir"]
    )

    alleles = _select_alleles(
        pt_meta=pt_meta,
        tf_meta=tf_meta,
        allele_panel=args.allele_panel,
        allele_regex=args.allele_regex,
        require_percentiles=(not args.allow_missing_affinity_percentiles),
    )

    pt_min_len, pt_max_len = pt_meta["affinity_supported_peptide_lengths"]
    tf_min_len, tf_max_len = tf_meta["affinity_supported_peptide_lengths"]
    lengths = _lengths_to_sample(
        min_supported=max(int(pt_min_len), int(tf_min_len)),
        max_supported=min(int(pt_max_len), int(tf_max_len)),
        min_requested=args.min_length,
        max_requested=args.max_length,
    )
    pt_with = pt_meta["with_flanks_lengths"]
    tf_with = tf_meta["with_flanks_lengths"]
    if pt_with is None or tf_with is None:
        raise ValueError("With-flanks processing models are required for this experiment.")
    n_flank_length = min(int(pt_with["n_flank"]), int(tf_with["n_flank"]))
    c_flank_length = min(int(pt_with["c_flank"]), int(tf_with["c_flank"]))

    peptides_df = _generate_uniform_peptides(
        num_peptides=args.num_peptides,
        lengths=lengths,
        n_flank_length=n_flank_length,
        c_flank_length=c_flank_length,
        seed=args.seed,
    )
    dataset = _cross_join_dataset(peptides_df=peptides_df, alleles=alleles)
    _pre_run_sanity_checks(peptides_df=peptides_df, dataset=dataset)

    dataset_path = out_dir / "dataset.csv.gz"
    peptides_path = out_dir / "peptides.csv.gz"
    alleles_path = out_dir / "alleles.txt"
    dataset.to_csv(dataset_path, index=False, compression="gzip")
    peptides_df.to_csv(peptides_path, index=False, compression="gzip")
    alleles_path.write_text("\n".join(alleles) + "\n")

    pt_predictions_path = out_dir / "pt_predictions.csv.gz"
    tf_predictions_path = out_dir / "tf_predictions.csv.gz"
    summary_path = out_dir / "diff_summary.json"
    report_path = out_dir / "diff_report.txt"
    outliers_path = out_dir / "top_outliers.csv.gz"

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
        )
    )

    pt = pd.read_csv(pt_predictions_path, keep_default_na=False)
    tf = pd.read_csv(tf_predictions_path, keep_default_na=False)
    pt_presentation_stats = _enforce_presentation_score_requirements(
        predictions=pt,
        label="PyTorch",
    )
    tf_presentation_stats = _enforce_presentation_score_requirements(
        predictions=tf,
        label="TensorFlow",
    )
    merged = pt.merge(tf, on=BASE_COLUMNS, suffixes=("_pt", "_tf"), how="inner")
    numeric_columns = _numeric_output_columns(pt)
    diff_df = _make_diff_frame(merged, numeric_columns=numeric_columns)
    output_summary_df = _per_output_summary(diff_df, numeric_columns=numeric_columns)
    break_df, row_summary = _break_analysis(
        merged=merged, diff_df=diff_df, numeric_columns=numeric_columns
    )

    output_summary_path = out_dir / "output_diff_summary.csv"
    break_summary_path = out_dir / "break_summary.csv"
    row_summary_path = out_dir / "row_worst_diff.csv.gz"
    top_rows_path = out_dir / "top_rows_by_any_output.csv.gz"
    output_summary_df.to_csv(output_summary_path, index=False)
    break_df.to_csv(break_summary_path, index=False)
    row_summary.to_csv(row_summary_path, index=False, compression="gzip")
    top_rows = row_summary.sort_values("max_abs_diff_any_output", ascending=False).head(args.top_k)
    top_rows.to_csv(top_rows_path, index=False, compression="gzip")

    _plot_output_ranges(output_summary_df, plots_dir / "output_abs_diff_ranges.png")
    _plot_row_max_hist(row_summary, plots_dir / "row_max_abs_diff_hist.png")
    if "pres_with_presentation_score" in numeric_columns:
        _plot_length_breakdown(
            diff_df,
            target_output="pres_with_presentation_score",
            out_path=plots_dir / "length_breakdown_pres_with_presentation_score.png",
        )
    _plot_per_output_hists(
        diff_df=diff_df,
        numeric_columns=numeric_columns,
        out_dir=plots_dir / "per_output_abs_diff_hist",
    )

    _write_break_report(
        out_path=out_dir / "break_analysis.txt",
        allele_count=len(alleles),
        peptide_count=len(peptides_df),
        pmhc_count=len(dataset),
        lengths=lengths,
        break_df=break_df,
        row_summary=row_summary,
        top_rows=top_rows,
    )

    metadata = {
        "allele_count": len(alleles),
        "peptide_count": int(len(peptides_df)),
        "pmhc_count": int(len(dataset)),
        "lengths": lengths,
        "n_flank_length": n_flank_length,
        "c_flank_length": c_flank_length,
        "allele_panel": args.allele_panel,
        "pytorch_repo_root": pt_repo_root,
        "tf_repo_root": tf_repo_root,
        "presentation_sanity": {
            "pytorch": pt_presentation_stats,
            "tensorflow": tf_presentation_stats,
        },
    }
    with open(out_dir / "run_metadata.json", "w") as out:
        json.dump(metadata, out, indent=2, sort_keys=True)

    print("Alleles:", len(alleles))
    print("Peptides:", len(peptides_df))
    print("pMHC rows:", len(dataset))
    print("Flank lengths:", {"n_flank": n_flank_length, "c_flank": c_flank_length})
    print("Presentation sanity (PyTorch):", pt_presentation_stats)
    print("Presentation sanity (TensorFlow):", tf_presentation_stats)
    print("Dataset:", dataset_path)
    print("PT predictions:", pt_predictions_path)
    print("TF predictions:", tf_predictions_path)
    print("Diff summary:", summary_path)
    print("Output diff summary:", output_summary_path)
    print("Break summary:", break_summary_path)
    print("Break analysis:", out_dir / "break_analysis.txt")
    print("Plots dir:", plots_dir)


if __name__ == "__main__":
    main()

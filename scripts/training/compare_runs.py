"""Compare two pan-allele training runs side-by-side.

Reads ``manifest.csv`` from each run's ``models.unselected.combined/``
to extract per-task wall-time, epoch counts, and final losses, plus
optionally each run's ``eval_comparison/`` outputs to compare per-allele
ROC-AUC / PR-AUC / PPV@N. Emits a markdown table to stdout and a
``compare_runs.csv`` to ``--out`` for downstream plots.

Usage:

    python scripts/training/compare_runs.py \\
        --baseline brev_runs/old_run/                 \\
        --candidate brev_runs/new_run/                \\
        --out results/compare_old_vs_new

Each ``--baseline`` / ``--candidate`` path can point at:
  * the run's top-level dir (auto-finds ``models.unselected.combined/``
    and ``eval_comparison/`` inside);
  * directly at ``models.unselected.combined/`` if there's no
    eval_comparison alongside.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Dict, List, Optional

import numpy
import pandas


def _parse_config_json(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(raw)


def _resolve_unselected_dir(path: str) -> str:
    candidate = os.path.join(path, "models.unselected.combined")
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(path) and os.path.isfile(
        os.path.join(path, "manifest.csv")
    ):
        return path
    raise FileNotFoundError(
        f"Couldn't find models.unselected.combined under {path!r}"
    )


def _resolve_eval_dir(path: str) -> Optional[str]:
    candidate = os.path.join(path, "eval_comparison")
    if os.path.isdir(candidate):
        return candidate
    return None


def _load_run_summary(unselected_dir: str) -> pandas.DataFrame:
    """One row per (model, phase) with time / epochs / final losses."""
    rows = []
    df = pandas.read_csv(os.path.join(unselected_dir, "manifest.csv"))
    for r in df.itertuples():
        cfg = _parse_config_json(r.config_json)
        layer_sizes = tuple(cfg["hyperparameters"].get("layer_sizes") or ())
        for fit_info in cfg.get("fit_info") or []:
            ti = fit_info.get("training_info", {})
            phase = ti.get("phase", "?")
            loss = fit_info.get("loss") or []
            val = fit_info.get("val_loss") or []
            rows.append({
                "model_name": r.model_name,
                "phase": phase,
                "layer_sizes": layer_sizes,
                "fold": ti.get("fold_num"),
                "n_epochs": len(loss),
                "wall_time_sec": fit_info.get("time"),
                "final_loss": loss[-1] if loss else float("nan"),
                "final_val_loss": val[-1] if val else float("nan"),
                "min_val_loss": min(val) if val else float("nan"),
            })
    return pandas.DataFrame(rows)


def _summarize(label: str, summary: pandas.DataFrame) -> Dict[str, float]:
    """Reduce a per-(model, phase) frame to single-line aggregates."""
    finetune = summary[summary.phase == "finetune"]
    pretrain = summary[summary.phase == "pretrain"]
    return {
        "label": label,
        "n_models": int(summary.model_name.nunique()),
        "finetune_total_wall_min": (
            finetune.wall_time_sec.sum() / 60 if len(finetune) else float("nan")
        ),
        "finetune_median_wall_min": (
            finetune.wall_time_sec.median() / 60
            if len(finetune) else float("nan")
        ),
        "finetune_max_wall_min": (
            finetune.wall_time_sec.max() / 60
            if len(finetune) else float("nan")
        ),
        "finetune_median_epochs": (
            finetune.n_epochs.median() if len(finetune) else float("nan")
        ),
        "finetune_max_epochs": (
            finetune.n_epochs.max() if len(finetune) else float("nan")
        ),
        "finetune_min_val_loss_p25": (
            finetune.min_val_loss.quantile(0.25)
            if len(finetune) else float("nan")
        ),
        "finetune_min_val_loss_median": (
            finetune.min_val_loss.median()
            if len(finetune) else float("nan")
        ),
        "pretrain_median_wall_sec": (
            pretrain.wall_time_sec.median() if len(pretrain) else float("nan")
        ),
    }


def _load_eval_comparison(eval_dir: Optional[str]) -> Optional[pandas.DataFrame]:
    """Return the per-allele metrics CSV emitted by compare_new_vs_public.py."""
    if eval_dir is None:
        return None
    for name in ("per_allele_metrics.csv", "metrics_per_allele.csv", "summary.csv"):
        path = os.path.join(eval_dir, name)
        if os.path.isfile(path):
            return pandas.read_csv(path)
    return None


def _compare_eval(
    baseline_eval: Optional[pandas.DataFrame],
    candidate_eval: Optional[pandas.DataFrame],
) -> Optional[pandas.DataFrame]:
    if baseline_eval is None or candidate_eval is None:
        return None
    on = "allele" if "allele" in baseline_eval.columns else baseline_eval.columns[0]
    merged = baseline_eval.merge(
        candidate_eval, on=on, suffixes=("_baseline", "_candidate")
    )
    return merged


def _print_markdown_table(rows: List[Dict[str, float]]) -> None:
    cols = list(rows[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if numpy.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional dir to write compare_runs.csv + per_allele_eval_compare.csv.",
    )
    args = parser.parse_args()

    base_unsel = _resolve_unselected_dir(args.baseline)
    cand_unsel = _resolve_unselected_dir(args.candidate)
    base_eval = _resolve_eval_dir(args.baseline)
    cand_eval = _resolve_eval_dir(args.candidate)

    base = _summarize(args.baseline_label, _load_run_summary(base_unsel))
    cand = _summarize(args.candidate_label, _load_run_summary(cand_unsel))

    print("\n## Per-task training summary\n")
    _print_markdown_table([base, cand])

    eval_compare = _compare_eval(
        _load_eval_comparison(base_eval),
        _load_eval_comparison(cand_eval),
    )
    if eval_compare is not None:
        # Aggregate any column whose name ends with _baseline / _candidate
        # to compute mean delta. Most useful: roc_auc, pr_auc, ppv_at_n.
        deltas = {}
        for col in eval_compare.columns:
            if col.endswith("_candidate"):
                base_col = col[: -len("_candidate")] + "_baseline"
                if base_col in eval_compare.columns:
                    metric = col[: -len("_candidate")]
                    deltas[metric] = {
                        "mean_baseline": eval_compare[base_col].mean(),
                        "mean_candidate": eval_compare[col].mean(),
                        "mean_delta": (
                            eval_compare[col] - eval_compare[base_col]
                        ).mean(),
                        "n_alleles": int(eval_compare[col].notna().sum()),
                    }
        if deltas:
            print("\n## Per-allele eval (mean over alleles)\n")
            print("| metric | baseline | candidate | delta | n |")
            print("|---|---|---|---|---|")
            for metric, d in sorted(deltas.items()):
                print(
                    "| {} | {:.4f} | {:.4f} | {:+.4f} | {} |".format(
                        metric,
                        d["mean_baseline"],
                        d["mean_candidate"],
                        d["mean_delta"],
                        d["n_alleles"],
                    )
                )
    else:
        print("\n(eval_comparison/ not found in one or both runs; "
              "skipping cross-run eval comparison)\n")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        pandas.DataFrame([base, cand]).to_csv(
            os.path.join(args.out, "compare_runs.csv"), index=False
        )
        if eval_compare is not None:
            eval_compare.to_csv(
                os.path.join(args.out, "per_allele_eval_compare.csv"),
                index=False,
            )


if __name__ == "__main__":
    main()

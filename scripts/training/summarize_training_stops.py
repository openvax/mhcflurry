#!/usr/bin/env python
"""Summarize convergence and early-stopping behavior from saved traces."""

import argparse
import json
from pathlib import Path

import numpy
import pandas


FIT_KEYS = (
    "manifest_path",
    "model_name",
    "fit_index",
    "phase",
    "fold",
    "architecture",
    "replicate",
)


def summarize(training_history):
    """Return per-fit details and condition-level early-stop summaries."""
    frame = pandas.read_csv(training_history)
    required = [*FIT_KEYS, "epoch", "val_loss"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(
            "Training history lacks required column(s): %s" %
            ", ".join(missing))
    frame["condition"] = frame.manifest_path.str.split("/").str[0]
    details = []
    for keys, group in frame.groupby(list(FIT_KEYS), dropna=False, sort=False):
        group = group.sort_values("epoch")
        finite = group[numpy.isfinite(group.val_loss)]
        if finite.empty:
            continue
        best_index = finite.val_loss.idxmin()
        best = group.loc[best_index]
        final = group.iloc[-1]
        row = dict(zip(FIT_KEYS, keys))
        row.update({
            "condition": str(group.condition.iloc[0]),
            "epoch_count": len(group),
            "best_epoch": int(best.epoch),
            "final_epoch": int(final.epoch),
            "epochs_after_best": int(final.epoch - best.epoch),
            "best_val_loss": float(best.val_loss),
            "final_val_loss": float(final.val_loss),
            "final_minus_best_val_loss": float(
                final.val_loss - best.val_loss),
            "relative_final_val_loss_gap": float(
                (final.val_loss - best.val_loss) / abs(best.val_loss)
                if best.val_loss else numpy.nan),
        })
        for source, target in (
            ("epoch_num_train_batches", "optimizer_steps"),
            ("epoch_num_train_rows", "training_rows_seen"),
            ("epoch_train_time", "summed_train_seconds"),
            ("epoch_total_time", "summed_epoch_seconds"),
        ):
            row[target] = (
                float(pandas.to_numeric(group[source], errors="coerce").sum())
                if source in group else numpy.nan)
        details.append(row)
    details = pandas.DataFrame(details)
    if details.empty:
        return details, pandas.DataFrame()
    summaries = []
    for (condition, phase), group in details.groupby(
            ["condition", "phase"], dropna=False, sort=False):
        summaries.append({
            "condition": condition,
            "phase": phase,
            "fit_count": len(group),
            "total_epochs": int(group.epoch_count.sum()),
            "median_epochs": float(group.epoch_count.median()),
            "median_best_epoch": float(group.best_epoch.median()),
            "median_epochs_after_best": float(
                group.epochs_after_best.median()),
            "total_optimizer_steps": float(group.optimizer_steps.sum()),
            "total_training_rows_seen": float(group.training_rows_seen.sum()),
            "summed_train_seconds": float(group.summed_train_seconds.sum()),
            "median_relative_final_val_loss_gap": float(
                group.relative_final_val_loss_gap.median()),
            "max_relative_final_val_loss_gap": float(
                group.relative_final_val_loss_gap.max()),
        })
    return details, pandas.DataFrame(summaries)


def main(argv=None):
    """Write per-fit and aggregate convergence tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_history")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details, summary = summarize(args.training_history)
    details.to_csv(out_dir / "early_stopping_per_fit.csv", index=False)
    summary.to_csv(out_dir / "early_stopping_summary.csv", index=False)
    payload = {
        "training_history": str(Path(args.training_history).resolve()),
        "fit_count": len(details),
        "conditions": summary.to_dict(orient="records"),
    }
    (out_dir / "early_stopping_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

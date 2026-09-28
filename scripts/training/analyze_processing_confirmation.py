"""Compare retained processing checkpoints and export a gated candidate recipe.

This is development selection, not release acceptance or final ensemble scoring.
The independent stopping partition selects epochs; this report compares recipes
on shared outer development folds. Use a fresh output directory per snapshot.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy
import pandas
import yaml

from mhcflurry.experiment_archive import sha256_file


METRICS = ["roc_auc", "pr_auc", "ppv_at_n"]
POLICIES = ["best", "best_ap", "terminal"]
BASELINE = "legacy_5aa__adam_keras__k11"
KEYS = ["checkpoint_policy", "fold_num", "sample_id"]


def collect_metrics(experiment):
    """Validate complete state/fold cohorts for every reported condition."""
    design = json.loads((experiment / "experiment.json").read_text())
    records = design["records"]
    if (design.get("design") != "processing-ranking-confirmation-v1" or len(records) != 6
            or len({row["condition"] for row in records}) != 6
            or any(row["fold_count"] != 4 or row["checkpoint_policy"] != "best_ap"
                   or len(row["hyperparameters"]) != 1 for row in records)):
        raise ValueError("Require the six-condition four-fold ranking-confirmation design")
    samples, micro, inputs, pending = [], [], [experiment / "experiment.json"], []
    reference = None
    for record in records:
        name = record["condition"]
        path = experiment / name / "checkpoint_per_sample.csv"
        micro_path = experiment / name / "checkpoint_micro_by_fold.csv"
        if not path.exists() or not micro_path.exists():
            pending.append(name)
            continue
        frame = pandas.read_csv(path, dtype={"sample_id": str})
        pooled = pandas.read_csv(micro_path)
        for table, keys in ((frame, KEYS), (pooled, KEYS[:2])):
            if (table.empty or set(table.condition) != {name} or table.duplicated(keys).any()
                    or table[keys].isna().any().any()
                    or not numpy.isfinite(table[METRICS + ["n", "n_pos"]]).all().all()
                    or not table[METRICS].ge(0).all().all() or not table[METRICS].le(1).all().all()
                    or not table.n_pos.gt(0).all() or not table.n.eq(2 * table.n_pos).all()
                    or set(table.checkpoint_policy) != set(POLICIES)
                    or any(set(group.fold_num) != {0, 1, 2, 3} for _, group in table.groupby("checkpoint_policy"))):
                raise ValueError("Require complete finite balanced checkpoint metrics: " + name)
        state_reference = None
        for policy in POLICIES:
            identity = frame.loc[frame.checkpoint_policy == policy].set_index(KEYS[1:])[ ["n", "n_pos"]].sort_index()
            if state_reference is None:
                state_reference = identity
            else:
                pandas.testing.assert_frame_equal(identity, state_reference, check_exact=True)
        if reference is None:
            reference = state_reference
        else:
            pandas.testing.assert_frame_equal(reference, state_reference, check_exact=True)
        counts = frame.groupby(KEYS[:2])[["n", "n_pos"]].sum().sort_index()
        pandas.testing.assert_frame_equal(counts, pooled.set_index(KEYS[:2])[["n", "n_pos"]].sort_index(), check_exact=True)
        samples.append(frame)
        micro.append(pooled)
        inputs.extend([path, micro_path])
    if not samples:
        raise ValueError("No completed checkpoint metrics yet")
    return design, pandas.concat(samples, ignore_index=True), pandas.concat(micro, ignore_index=True), pending, inputs


def compare_and_choose(design, samples, micro, pending, replicates=10000, seed=42):
    """Apply the declared macro/micro gate; incomplete screens cannot promote."""
    if not samples.condition.eq(BASELINE).any():
        raise ValueError("Baseline condition is not complete: " + BASELINE)
    means = samples.groupby(["condition", "checkpoint_policy", "sample_id"])[METRICS].mean()
    summary = means.groupby(["condition", "checkpoint_policy"]).mean().reset_index()
    baseline = means.loc[(BASELINE, "best")].sort_index()
    base_micro = micro.loc[(micro.condition == BASELINE) & (micro.checkpoint_policy == "best")].set_index("fold_num")[METRICS]
    indices = numpy.random.default_rng(seed).integers(len(baseline), size=(replicates, len(baseline)))
    comparisons, differences, draws = [], [], {}
    for (condition, policy), group in means.groupby(level=[0, 1]):
        group = group.droplevel([0, 1]).sort_index()
        pandas.testing.assert_index_equal(group.index, baseline.index)
        delta = group - baseline
        pooled = micro.loc[(micro.condition == condition) & (micro.checkpoint_policy == policy)].set_index("fold_num")[METRICS]
        pooled_delta = pooled - base_micro
        row = {"condition": condition, "checkpoint_policy": policy, "samples": len(baseline)}
        for metric in METRICS:
            values = delta[metric].to_numpy()
            bootstrap = values[indices].mean(axis=1)
            low, high = numpy.quantile(bootstrap, [0.025, 0.975])
            row.update({metric + "_delta": float(values.mean()), metric + "_low": float(low),
                        metric + "_high": float(high), metric + "_worst_fold_micro_delta": float(pooled_delta[metric].min())})
            draws[condition + "__" + policy + "__" + metric] = bootstrap
        row["development_gate"] = bool(
            row["pr_auc_delta"] > 0 and row["ppv_at_n_delta"] > 0
            and row["pr_auc_low"] > 0 and row["ppv_at_n_low"] > -0.002
            and all(row[m + "_worst_fold_micro_delta"] >= -0.002 for m in METRICS))
        comparisons.append(row)
        differences.append(delta.reset_index().assign(condition=condition, checkpoint_policy=policy))
    comparisons = pandas.DataFrame(comparisons)
    candidates = summary.merge(comparisons, on=["condition", "checkpoint_policy"])
    eligible = candidates.loc[(candidates.checkpoint_policy == "best_ap") & candidates.development_gate]
    winner = None
    if not pending and not eligible.empty:
        winner = eligible.sort_values(["pr_auc", "ppv_at_n", "condition"], ascending=[False, False, True]).iloc[0].condition
    decision = {"status": "incomplete" if pending else "candidate" if winner else "retain_control",
                "candidate_condition": winner, "pending": pending, "baseline": BASELINE + " / best loss",
                "checkpoint_selection": "inner sample-macro AP; earliest exact tie; loss-based patience",
                "gate": "Both macro deltas positive; AP 95% paired lower bound >0; PPV lower bound >-0.002; all four-fold pooled AUROC/AP/PPV deltas >=-0.002.",
                "limitations": "Exploratory repeated development screening; intervals conditional on fits, without multiple-screening or study-cluster correction. Not a fixed ensemble or presentation test.",
                "release_accepted": False}
    return summary, comparisons, pandas.concat(differences, ignore_index=True), draws, decision


def render(summary, comparisons, out, experiment=None):
    """Render policy-separated score maps and paired development intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    names = sorted(summary.condition.unique())
    labels = [name.replace("legacy_5aa__", "").replace("_keras", "").replace("_pytorch", "").replace("__", " / ") for name in names]
    with PdfPages(out / "confirmation.pdf") as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
        for axis, metric, title in zip(axes, METRICS, ["AUROC", "AUPRC", "PPV@N"]):
            for policy, label, marker in zip(POLICIES, ["Best loss", "Inner-best AP", "Terminal"], ["o", "s", "^"]):
                values = summary.loc[summary.checkpoint_policy == policy].set_index("condition").reindex(names)
                axis.plot(numpy.arange(len(names)), values[metric], marker=marker, label=label,
                          markerfacecolor="none" if policy == "best_ap" else None)
            axis.set(xticks=range(len(names)), xticklabels=labels, ylabel="Macro " + title)
            axis.tick_params(axis="x", rotation=45)
            axis.grid(alpha=0.2)
        axes[0].legend()
        fig.suptitle("Processing / 1:1 affinity-length-matched development\n%d samples; %d/6 conditions; four paired fits, not ensembles" %
                     (int(comparisons.samples.iloc[0]), len(names)))
        pdf.savefig(fig)
        fig.savefig(out / "scores.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        primary = comparisons.loc[comparisons.checkpoint_policy == "best_ap"].set_index("condition").reindex(names)
        for axis, metric, title in zip(axes, METRICS[1:], ["AUPRC", "PPV@N"]):
            delta = primary[metric + "_delta"].to_numpy()
            errors = numpy.array([delta - primary[metric + "_low"], primary[metric + "_high"] - delta])
            axis.errorbar(delta, numpy.arange(len(names)), xerr=errors, fmt="o", capsize=3)
            axis.axvline(0, color="gray", linestyle="--")
            axis.set(yticks=range(len(names)), yticklabels=labels, xlabel="Macro " + title + " difference")
        fig.suptitle("Inner-best AP versus Adam width 11 best-loss control\nExploratory 95% paired sample intervals; no screening/study correction")
        pdf.savefig(fig)
        fig.savefig(out / "paired-intervals.png", dpi=160)
        plt.close(fig)
        if experiment is not None:
            for name in names:
                path = experiment / name / "processing/models.unselected.short_flanks/manifest.csv"
                if not path.exists():
                    continue  # Metric-only snapshots remain independently usable.
                configs = pandas.read_csv(path).config_json.map(json.loads)
                fits = sorted((config["fit_info"][-1] for config in configs), key=lambda fit: fit["training_info"]["fold_num"])
                if [fit["training_info"]["fold_num"] for fit in fits] != [0, 1, 2, 3]:
                    raise ValueError("Epoch diagnostics require one fit in each held-out fold")
                fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharey="row", constrained_layout=True)
                for col, fit in enumerate(fits):
                    epochs = numpy.arange(1, len(fit["loss"]) + 1)
                    if any(len(fit[key]) != len(epochs) for key in ("val_loss", "val_macro_ap", "val_macro_ppv_at_n")):
                        raise ValueError("Incomplete epoch ranking/loss history")
                    axes[0, col].plot(epochs, fit["loss"], label="Training loss", color="0.65")
                    axes[0, col].plot(epochs, fit["val_loss"], label="Inner validation loss", color="0.15")
                    axes[1, col].plot(epochs, fit["val_macro_ap"], label="Inner macro AP")
                    axes[1, col].plot(epochs, fit["val_macro_ppv_at_n"], label="Inner macro PPV", color="tab:green")
                    for row in range(2):
                        axes[row, col].axvline(fit["best_epoch"], color="tab:blue", linestyle=":", label="Best loss epoch")
                        axes[row, col].axvline(fit["best_ranking_epoch"], color="tab:orange", linestyle="--", label="Best AP epoch")
                        axes[row, col].grid(alpha=0.15)
                    axes[0, col].set_title("Fold %d / %d inner samples" % (col, len(fit["ranking_validation_samples"])))
                    axes[1, col].set_xlabel("Epoch (trace ends at terminal)")
                axes[0, 0].set_ylabel("Binary cross-entropy")
                axes[1, 0].set_ylabel("Inner sample-macro ranking")
                axes[0, 0].legend(fontsize=7)
                axes[1, 0].legend(fontsize=7)
                fig.suptitle(name.replace("__", " / ") + "\nStopping-validation trajectories; outer scores never select epochs")
                pdf.savefig(fig)
                fig.savefig(out / (name + ".epochs.png"), dpi=160)
                plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.out.exists() or args.replicates < 100:
        raise ValueError("Require a new output directory and at least 100 bootstrap draws")
    design, samples, micro, pending, inputs = collect_metrics(args.experiment)
    summary, comparisons, differences, draws, decision = compare_and_choose(
        design, samples, micro, pending, args.replicates, args.seed)
    args.out.mkdir(parents=True)
    for name, frame in (("condition_means", summary), ("paired_comparisons", comparisons),
                        ("sample_differences", differences), ("per_fold_sample_metrics", samples), ("micro_by_fold", micro)):
        frame.to_csv(args.out / (name + ".csv"), index=False)
    numpy.savez_compressed(args.out / "bootstrap_deltas.npz", **draws)
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    if decision["candidate_condition"]:
        record = next(row for row in design["records"] if row["condition"] == decision["candidate_condition"])
        (args.out / "candidate_hyperparameters.yaml").write_text(yaml.safe_dump(record["hyperparameters"], sort_keys=True))
    render(summary, comparisons, args.out, args.experiment)
    inputs.extend(path for name in summary.condition.unique()
                  if (path := args.experiment / name / "processing/models.unselected.short_flanks/manifest.csv").exists())
    shutil.copyfile(__file__, args.out / "analysis_source.py")
    provenance = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "inputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in inputs],
                  "outputs": [{"path": path.name, "sha256": sha256_file(path)} for path in sorted(args.out.iterdir())]}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    main()

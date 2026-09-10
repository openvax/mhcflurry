"""Map a processing recipe factorial from saved paired development metrics.

No training, inference, model selection, or release-benchmark evaluation occurs.
"""

import argparse
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil

import numpy
import pandas

from paired_sample_metrics import paired_summary


METRICS = ["roc_auc", "pr_auc", "ppv_at_n"]
FACTORS = ["family", "optimizer_recipe", "initialization", "batch_size"]
INITIALIZATIONS = ["none", "orthogonal", "lsuv_pre", "lsuv_post"]
LIMITATIONS = (
    "Exploratory paired sample bootstrap, conditional on trained fits; no "
    "correction for repeated screening, multiple comparisons or study clustering. "
    "Per-member development metrics, not ensemble or presentation performance. "
    "Point-estimate Pareto membership is not a significance or release gate."
)


def validate_inputs(experiment, metrics):
    """Validate a single frozen factorial and exact fold/sample pairing.

    Entire unreported conditions are allowed; partially reported conditions,
    changing cohorts, duplicate units and nonfinite metrics are rejected.
    """
    if experiment.get("design") != "processing-training-recipe-v1":
        raise ValueError("Require a processing-training-recipe-v1 design")
    if not experiment.get("train_data_sha256"):
        raise ValueError("Missing frozen training table identity")
    records = pandas.DataFrame(experiment["records"])
    required = ["condition", "family", "optimizer", "optimizer_implementation",
                "initialization", "batch_size", "fold_count", "network_count",
                "kernel_width", "checkpoint_policy", "hyperparameters"]
    if set(required) - set(records):
        raise ValueError("Missing design fields")
    if records[required].isna().any().any() or records.condition.duplicated().any():
        raise ValueError("Nonmissing unique design identities required")
    records["optimizer_recipe"] = records.optimizer + "/" + records.optimizer_implementation
    if records.duplicated(FACTORS).any():
        raise ValueError("Duplicate factorial coordinates")
    if (not records.initialization.isin(INITIALIZATIONS).all() or
            not records.checkpoint_policy.eq("best").all() or
            records.fold_count.nunique() != 1 or
            not records.fold_count.eq(records.network_count).all() or
            records.kernel_width.nunique() != 1):
        raise ValueError("Require common width, best-loss policy and one member per paired fold")
    changing = {"optimizer", "optimizer_implementation", "initialization_method", "minibatch_size"}
    fixed_by_family = {}
    for record in records.to_dict("records"):
        if len(record["hyperparameters"]) != 1:
            raise ValueError("Require one architecture per condition")
        hp = record["hyperparameters"][0]
        expected = {"optimizer": record["optimizer"],
                    "optimizer_implementation": record["optimizer_implementation"],
                    "initialization_method": record["initialization"],
                    "minibatch_size": record["batch_size"],
                    "convolutional_kernel_size": record["kernel_width"],
                    "restore_best_weights": True}
        if any(hp.get(key) != value for key, value in expected.items()):
            raise ValueError("Design axes disagree with hyperparameters")
        fixed = {key: value for key, value in hp.items() if key not in changing}
        previous = fixed_by_family.setdefault(record["family"], fixed)
        if previous != fixed:
            raise ValueError("Uncontrolled hyperparameter change within a family")

    keys = ["condition", "fold_num", "sample_id"]
    fields = keys + ["n", "n_pos"] + METRICS
    if set(fields) - set(metrics) or metrics.empty:
        raise ValueError("Missing or empty per-fold/sample metric table")
    if metrics[fields].isna().any().any() or metrics.duplicated(keys).any():
        raise ValueError("Nonmissing unique fold/sample identities required")
    if set(metrics.condition) - set(records.condition):
        raise ValueError("Metrics contain conditions outside the frozen design")
    if "checkpoint_policy" in metrics and not metrics.checkpoint_policy.eq("best").all():
        raise ValueError("Metric checkpoint policy differs from the design")
    numeric = metrics[["fold_num", "n", "n_pos"] + METRICS].to_numpy(dtype=float)
    if not numpy.isfinite(numeric).all() or not metrics[METRICS].ge(0).all().all() or not metrics[METRICS].le(1).all().all():
        raise ValueError("Require finite metrics between zero and one")
    counts = metrics[["n", "n_pos", "fold_num"]].to_numpy(dtype=float)
    if not numpy.equal(counts, numpy.floor(counts)).all():
        raise ValueError("Counts and fold identifiers must be integers")
    if not metrics.n_pos.gt(0).all() or not metrics.n.eq(2 * metrics.n_pos).all():
        raise ValueError("This analysis requires one matched negative per positive")
    expected_folds = set(range(int(records.fold_count.iloc[0])))
    anchor = None
    for _, group in metrics.groupby("condition", sort=True):
        if set(group.fold_num) != expected_folds:
            raise ValueError("Incomplete condition: missing requested folds")
        cohort = group.set_index(["fold_num", "sample_id"])[["n", "n_pos"]].sort_index()
        if anchor is None:
            anchor = cohort
        elif not cohort.equals(anchor):
            raise ValueError("Unmatched fold/sample cohorts or counts")
    records["status"] = numpy.where(records.condition.isin(metrics.condition), "evaluated", "pending")
    return records


def analyze(experiment, metrics, *, replicates=10000, seed=42):
    """Return design, sample means, condition means and one-factor contrasts."""
    records = validate_inputs(experiment, metrics)
    # Repeated appearances of a sample are not independent bootstrap units.
    samples = metrics.groupby(["condition", "sample_id"], as_index=False)[METRICS + ["n", "n_pos"]].mean()
    means = samples.groupby("condition")[METRICS].mean().add_prefix("macro_").reset_index()
    means["samples"] = means.condition.map(samples.groupby("condition").size())
    means = means.merge(records.drop(columns="hyperparameters"), on="condition", validate="one_to_one")
    points = means[["macro_pr_auc", "macro_ppv_at_n"]].to_numpy()
    means["pareto_point_estimate"] = [not ((points >= point).all(axis=1) & (points > point).any(axis=1)).any() for point in points]
    complete = records.loc[records.status == "evaluated"].sort_values("condition")
    contrasts, differences = [], []
    for left, right in combinations(complete.to_dict("records"), 2):
        changed = [key for key in FACTORS if left[key] != right[key]]
        if len(changed) != 1:
            continue
        factor = changed[0]
        if factor == "initialization":
            reverse = INITIALIZATIONS.index(left[factor]) > INITIALIZATIONS.index(right[factor])
        else:
            reverse = left[factor] > right[factor]
        if reverse:
            left, right = right, left
        pair = samples.loc[samples.condition.isin([left["condition"], right["condition"]])]
        summary, delta, _ = paired_summary(
            pair, units=["sample_id"], condition="condition", metrics=METRICS,
            baseline=left["condition"], replicates=replicates, seed=seed)
        wide = delta.pivot(index="sample_id", columns="metric", values="delta")
        summary["joint_samples_improved"] = int((wide.pr_auc.gt(0) & wide.ppv_at_n.gt(0)).sum())
        metadata = {"factor": factor, "from_value": str(left[factor]), "to_value": str(right[factor]),
                    "fixed_settings": json.dumps({key: left[key] for key in FACTORS if key != factor}, sort_keys=True)}
        for key, value in metadata.items():
            summary[key] = value
            delta[key] = value
        delta["baseline"] = left["condition"]
        contrasts.append(summary)
        differences.append(delta)
    return (records, samples, means,
            pandas.concat(contrasts, ignore_index=True) if contrasts else pandas.DataFrame(),
            pandas.concat(differences, ignore_index=True) if differences else pandas.DataFrame())


def render_report(records, means, contrasts, out):
    """Render missing-aware maps and paginated paired-effect plots."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot
    from matplotlib.backends.backend_pdf import PdfPages

    title = "Processing / 1:1 affinity-length-matched development"
    subtitle = "Per-member scores; best-loss checkpoint; folds averaged within sample (not an ensemble)"
    colors = pyplot.get_cmap("viridis").with_extremes(bad="#eeeeee")
    with PdfPages(out / "recipe-analysis.pdf") as pdf:
        page = 0

        def save(fig):
            nonlocal page
            page += 1
            pdf.savefig(fig)
            fig.savefig(out / ("recipe-analysis-%02d.png" % page), dpi=160)
            pyplot.close(fig)

        for family in sorted(records.family.unique()):
            design = records.loc[records.family == family].copy()
            design["column"] = design.optimizer_recipe + " / " + design.batch_size.astype(str)
            columns = (design.sort_values(["optimizer_recipe", "batch_size"])
                       .column.drop_duplicates().tolist())
            group = design[["condition", "initialization", "column"]].merge(means, on=["condition", "initialization"], how="left")
            fig, axes = pyplot.subplots(1, 2, figsize=(12, 6))
            for axis, metric, label in zip(axes, ["pr_auc", "ppv_at_n"], ["AUPRC", "PPV@N"]):
                values = group.pivot(index="initialization", columns="column", values="macro_" + metric).reindex(index=INITIALIZATIONS, columns=columns)
                limits = means["macro_" + metric]
                lower, upper = limits.min(), limits.max()
                upper = max(upper, lower + 1e-6)
                plotted = axis.imshow(numpy.ma.masked_invalid(values.to_numpy()), cmap=colors,
                                      aspect="auto", vmin=lower, vmax=upper)
                axis.set(xticks=range(len(columns)), xticklabels=columns,
                         yticks=range(len(INITIALIZATIONS)), yticklabels=INITIALIZATIONS,
                         title="Macro " + label)
                axis.tick_params(axis="x", rotation=25, labelsize=8)
                for i in range(len(values)):
                    for j in range(len(columns)):
                        value = values.iloc[i, j]
                        color = "black" if not numpy.isfinite(value) or value > (lower + upper) / 2 else "white"
                        axis.text(j, i, "%.4f" % value if numpy.isfinite(value) else "pending",
                                  ha="center", va="center", color=color, fontsize=9)
                fig.colorbar(plotted, ax=axis, fraction=0.04)
            fig.suptitle(title + "\n" + family + " | " + subtitle +
                         "\n%d folds, %d held-out samples; none = Glorot" % (
                             int(records.fold_count.iloc[0]), int(means.samples.iloc[0])), fontsize=10)
            fig.text(0.5, 0.02, "Equal sample weighting. Gray cells are unreported, not zero. Common color scale across families.", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.09, 1, 0.9))
            save(fig)

        if contrasts.empty:
            return
        for factor, group in contrasts.groupby("factor", sort=True):
            identities = group[["condition", "baseline"]].drop_duplicates().to_records(index=False).tolist()
            for start in range(0, len(identities), 12):
                chunk = identities[start:start + 12]
                fig, axes = pyplot.subplots(1, 2, figsize=(14, max(4, 0.52 * len(chunk) + 2.5)))
                for axis, metric, label in zip(axes, ["pr_auc", "ppv_at_n"], ["AUPRC", "PPV@N"]):
                    rows = group.loc[group.metric == metric].set_index(["condition", "baseline"]).loc[chunk]
                    y = numpy.arange(len(rows))
                    axis.hlines(y, rows.ci_low, rows.ci_high, color="#286b8d", linewidth=2)
                    axis.scatter(rows.delta, y, color="#143b52", zorder=3)
                    labels = []
                    for row in rows.itertuples():
                        fixed = json.loads(row.fixed_settings)
                        context = ", ".join(str(fixed[key]) for key in FACTORS if key in fixed)
                        labels.append(row.to_value + " - " + row.from_value + "\n" + context)
                    axis.set(yticks=y, yticklabels=labels if axis is axes[0] else [],
                             title="Macro " + label, xlabel="Difference (95% paired sample interval)")
                    axis.tick_params(axis="y", labelsize=8)
                    axis.axvline(0, color="gray", linestyle="--", linewidth=1)
                    axis.invert_yaxis()
                fig.suptitle(title + "\nOne-factor contrasts: " + factor, fontsize=12)
                fig.text(0.5, 0.02, "Exploratory intervals; no multiple-screening or study-cluster correction. Other recipe settings held fixed.", ha="center", fontsize=9)
                fig.tight_layout(rect=(0, 0.07, 1, 0.9))
                save(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.environ.get("MHCFLURRY_CLI_PROG"), description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output directory for each immutable analysis snapshot")
    experiment = json.loads(args.experiment.read_text())
    metrics = pandas.read_csv(args.metrics, dtype={"sample_id": str}, float_precision="round_trip")
    records, samples, means, contrasts, differences = analyze(experiment, metrics, replicates=args.replicates, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("design_status", records.drop(columns="hyperparameters")), ("sample_means", samples),
                        ("condition_means", means), ("paired_contrasts", contrasts), ("sample_differences", differences)):
        frame.to_csv(args.out / (name + ".csv"), index=False)
    render_report(records, means, contrasts, args.out)
    provenance = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "inputs": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in (args.experiment, args.metrics, Path(__file__), Path(__file__).with_name("paired_sample_metrics.py"))],
        "train_data_sha256": experiment["train_data_sha256"],
        "evaluated_conditions": len(means), "planned_conditions": len(records),
        "method": "Average folds within sample, then paired sample percentile bootstrap; one-factor contrasts only.",
        "limitations": LIMITATIONS,
    }
    source_dir = args.out / "analysis_source"
    source_dir.mkdir()
    for path in (Path(__file__), Path(__file__).with_name("paired_sample_metrics.py")):
        shutil.copyfile(path, source_dir / path.name)
    provenance["outputs"] = [
        {"path": str(path.relative_to(args.out)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(args.out.rglob("*")) if path.is_file()]
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(means[["condition", "macro_pr_auc", "macro_ppv_at_n", "samples", "pareto_point_estimate"]].to_string(index=False))
    print("Evaluated %d/%d conditions. %s" % (len(means), len(records), LIMITATIONS))
    return 0


if __name__ == "__main__":
    main()

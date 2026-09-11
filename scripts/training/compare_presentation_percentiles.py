#!/usr/bin/env python
"""Reproducible compact-percentile comparison using cached model predictions.

Subcommands separate label-free calibration/selection from presentation
evaluation. No command loads neural-network weights or changes saved models.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mhcflurry.class1_presentation_predictor import presentation_percent_rank_bins
from mhcflurry.compact_percent_rank_transform import CompactPresentationPercentiles
from mhcflurry.percent_rank_transform import PercentRankTransform


CUTOFFS = np.array([.001, .003, .01, .03, .1, .3, 1., 3., 10., 30., 50., 90.])
SELECTION_CUTOFFS = [.03, .1, .3, 1., 3., 10.]
METRICS = ["auprc", "auroc", "ppv_expected_random_ties"]
# Resolve from this checkout (scripts/training/ is two levels below the root), not the cwd.
COMPACT_TRANSFORM_SOURCE = (
    Path(__file__).resolve().parents[2] / "mhcflurry" / "compact_percent_rank_transform.py")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def grouped_reference_split(peptides, seed):
    """Return 60/20/20 split IDs, with duplicate peptides kept together."""
    unique, inverse = np.unique(np.asarray(peptides), return_inverse=True)
    if len(unique) < 5:
        raise ValueError("At least five distinct peptides are required")
    order = np.random.default_rng(seed).permutation(len(unique))
    groups = np.full(len(unique), 2, dtype=np.int8)
    groups[order[:int(.6 * len(unique))]] = 0
    groups[order[int(.6 * len(unique)):int(.8 * len(unique))]] = 1
    return groups[inverse]


def calibration_accuracy(percentiles, cutoffs=CUTOFFS):
    """Compare predicted percentile thresholds with observed background mass."""
    values = np.asarray(percentiles)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite percentile vector")
    records = []
    for cutoff in cutoffs:
        count = int(np.count_nonzero(values <= cutoff))
        observed = 100.0 * count / len(values)
        smoothed = 100.0 * (count + .5) / (len(values) + 1)
        records.append(dict(cutoff=float(cutoff), count=count, rows=len(values),
                            observed_percent=observed, smoothed_percent=smoothed,
                            log10_error=float(np.log10(smoothed / cutoff))))
    return pd.DataFrame(records)


def select_compact_method(accuracy):
    """Choose the smallest curve near the best validation-only calibration error."""
    selected_rows = accuracy.loc[
        (accuracy.split == "validation") & accuracy.method.str.startswith("compact_") &
        accuracy.cutoff.isin(SELECTION_CUTOFFS)].copy()
    losses = selected_rows.groupby("method").log10_error.apply(
        lambda values: float(np.sqrt(np.mean(np.square(values)))))
    if losses.empty or not np.isfinite(losses).all():
        raise ValueError("Missing finite validation calibration errors")
    best = float(losses.min())
    tolerance = max(.1 * best, .01)
    eligible = losses[losses <= best + tolerance].index
    selected = min(eligible, key=lambda name: int(name.split("_")[-1]))
    return dict(selected=selected, validation_rms_log10_error=losses.to_dict(),
                best=best, tolerance=tolerance, selection_cutoffs=SELECTION_CUTOFFS)


def ranking_metrics(labels, scores):
    """Return AP/AUROC and analytic expected PPV when cutoff scores are tied."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.shape != scores.shape or labels.ndim != 1 or
            not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()):
        raise ValueError("Expected aligned binary labels and finite scores")
    n = int(labels.sum())
    if not 0 < n < len(labels):
        raise ValueError("Ranking evaluation needs both positive and negative rows")
    cutoff = np.partition(scores, len(scores) - n)[len(scores) - n]
    above, tied = scores > cutoff, scores == cutoff
    expected = labels[above].sum() + (n - above.sum()) * labels[tied].mean()
    return dict(auprc=float(average_precision_score(labels, scores)),
                auroc=float(roc_auc_score(labels, scores)),
                ppv_expected_random_ties=float(expected / n))


def fit_methods(reference, budgets):
    methods = {}
    for name, bins in [
            ("uniform_step", np.unique(np.quantile(reference, np.linspace(0, 1, 10001)))),
            ("tail_step", presentation_percent_rank_bins(reference))]:
        model = PercentRankTransform()
        model.fit(reference, bins=bins)
        methods[name] = model
    for budget in budgets:
        methods["compact_%d" % budget] = CompactPresentationPercentiles.fit(reference, budget)
    return methods


def percentiles(model, scores):
    if isinstance(model, CompactPresentationPercentiles):
        return model.transform(scores)
    return 100.0 - model.transform(scores)


def save_methods(methods, destination):
    destination.mkdir()
    records = []
    for name, model in methods.items():
        if isinstance(model, CompactPresentationPercentiles):
            path = destination / (name + ".json")
            write_json(path, model.to_dict())
            restored = CompactPresentationPercentiles.from_dict(json.loads(path.read_text()))
            probe = np.r_[np.linspace(0, 1, 10001), model.reference_bounds]
            np.testing.assert_array_equal(model.transform(probe), restored.transform(probe))
            count, knots = len(model.x) * 2 + 2, len(model.x)
        else:
            path = destination / (name + ".csv")
            model.to_series().to_csv(path)
            count, knots = len(model.cdf) + len(model.bin_edges), len(model.bin_edges)
        records.append(dict(method=name, knots=knots, numeric_scalars=count,
                            coordinate_bytes=count * 8, serialized_bytes=path.stat().st_size,
                            sha256=sha256(path)))
    return records


def load_methods(destination):
    methods = {}
    for path in sorted(Path(destination).glob("*")):
        if path.suffix == ".json":
            methods[path.stem] = CompactPresentationPercentiles.from_dict(json.loads(path.read_text()))
        elif path.suffix == ".csv":
            table = pd.read_csv(path, index_col=0, float_precision="round_trip")
            methods[path.stem] = PercentRankTransform.from_series(table.iloc[:, 0])
    return methods


def fit_command(args):
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(args.reference_dir)
    peptide_path = source / "reference_peptides.csv.bz2"
    peptides = pd.read_csv(peptide_path).peptide.to_numpy()
    paths = sorted(source.glob("reference_scores_*.npy"))
    if not paths:
        raise ValueError("No cached reference score files found")
    chunks = [np.load(path) for path in paths]
    if any(chunk.shape != (len(peptides),) for chunk in chunks):
        raise ValueError("Each allele must have one score per reference peptide")
    reference = np.concatenate(chunks)
    peptide_splits = grouped_reference_split(peptides, args.seed)
    splits = np.tile(peptide_splits, len(chunks))
    np.savez_compressed(out / "reference_split.npz", peptide_split=peptide_splits,
                        reference_split=splits, reference_scores=reference)
    print("Reference rows by split:", np.bincount(splits).tolist(), flush=True)
    methods = fit_methods(reference[splits == 0], args.knots)
    sizes = save_methods(methods, out / "split_fit")
    accuracy = []
    for split_id, split_name in [(1, "validation"), (2, "test")]:
        for name, model in methods.items():
            frame = calibration_accuracy(percentiles(model, reference[splits == split_id]))
            frame["method"], frame["split"] = name, split_name
            accuracy.append(frame)
    accuracy = pd.concat(accuracy, ignore_index=True)
    accuracy.to_csv(out / "calibration_accuracy.csv", index=False)
    selection = select_compact_method(accuracy)
    write_json(out / "selection.json", selection)
    print("Label-free selection:", selection, flush=True)
    full_methods = fit_methods(reference, args.knots)
    full_sizes = save_methods(full_methods, out / "full_fit")
    pd.DataFrame([dict(fit="split", **row) for row in sizes] +
                 [dict(fit="full", **row) for row in full_sizes]).to_csv(
                     out / "model_sizes.csv", index=False)
    provenance = dict(command=sys.argv, seed=args.seed, knots=args.knots,
                      reference_rows=len(reference), reference_peptides=len(peptides),
                      reference_hashes={str(path): sha256(path) for path in [peptide_path] + paths},
                      elapsed_seconds=time.perf_counter() - started,
                      source_hashes={str(path): sha256(path)
                                     for path in [Path(__file__), COMPACT_TRANSFORM_SOURCE]},
                      notes="Peptide-grouped split; no presentation labels used. Uniform-AA cached "
                      "reference differs from original release calibration policy.")
    for name in ("parameters.json", "provenance.json"):
        if (source / name).exists():
            provenance["reference_" + name.removesuffix(".json")] = json.loads((source / name).read_text())
    write_json(out / "provenance.json", provenance)


def evaluate_command(args):
    started = time.perf_counter()
    root = Path(args.calibration_dir)
    out = root / args.mode
    out.mkdir(exist_ok=False)
    methods = load_methods(root / "full_fit")
    if not methods:
        raise ValueError("No full-reference calibration curves found")
    columns = ["sample_id", "hla", "hit"] + [
        side + "_presentation_" + field
        for side in ("a", "b") for field in ("score", "percentile")]
    print("Reading frozen", args.mode, "prediction columns", flush=True)
    frame = pd.read_csv(args.predictions, usecols=columns, low_memory=False)
    if frame[columns].isna().any().any():
        raise ValueError("Evaluation table has missing values")
    labels = frame.hit.to_numpy(dtype=np.int8)
    raw = frame.a_presentation_score.to_numpy()
    groups = list(frame.groupby(["sample_id", "hla"], sort=False).indices.items())
    group_id = np.empty(len(frame), dtype=np.int32)
    group_records = []
    for index, ((sample_id, hla), rows) in enumerate(groups):
        group_id[rows] = index
        group_records.append(dict(group_id=index, sample_id=str(sample_id), hla=str(hla),
                                  rows=len(rows), hits=int(labels[rows].sum())))
    pd.DataFrame(group_records).to_csv(out / "groups.csv", index=False)
    print("Loaded", len(frame), "rows;", len(groups), "groups", flush=True)
    variants = dict(candidate_raw=raw, candidate_saved=-frame.a_presentation_percentile.to_numpy(),
                    public_raw=frame.b_presentation_score.to_numpy(),
                    public_saved=-frame.b_presentation_percentile.to_numpy())
    timings, audits = [], []
    raw_order = np.argsort(raw, kind="stable")
    raw_sorted = raw[raw_order]
    different = np.diff(raw_sorted) != 0
    raw_unique = int(np.count_nonzero(different) + 1)
    for name, model in methods.items():
        durations = []
        for _ in range(3):
            before = time.perf_counter()
            mapped = percentiles(model, raw)
            durations.append(time.perf_counter() - before)
        variants[name] = -mapped
        diffs = np.diff(mapped[raw_order])
        bounds = (model.reference_bounds if isinstance(model, CompactPresentationPercentiles)
                  else model.bin_edges[[0, -1]])
        audits.append(dict(method=name, raw_unique=raw_unique,
                           percentile_unique=int(np.unique(mapped).size),
                           distinct_adjacent_raw_pairs_tied=int(np.count_nonzero(different & (diffs == 0))),
                           inversions=int(np.count_nonzero(diffs > 0)),
                           below_reference=int(np.count_nonzero(raw < bounds[0])),
                           above_reference=int(np.count_nonzero(raw > bounds[1])),
                           zero_percentiles=int(np.count_nonzero(mapped == 0)),
                           full_percentiles=int(np.count_nonzero(mapped == 100))))
        timings.append(dict(method=name, rows=len(raw), median_seconds=float(np.median(durations)),
                            seconds_1=durations[0], seconds_2=durations[1], seconds_3=durations[2]))
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "ranking_audit.csv", index=False)
    records = []
    for name, scores in variants.items():
        records.append(dict(method=name, scope="micro", sample_id="all", hla="all",
                            **ranking_metrics(labels, scores)))
        for (sample_id, hla), rows in groups:
            records.append(dict(method=name, scope="sample", sample_id=str(sample_id), hla=str(hla),
                                **ranking_metrics(labels[rows], scores[rows])))
        print(args.mode, name, "metrics evaluated", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(out / "metrics.csv", index=False)
    macro = results.loc[results.scope == "sample"].groupby("method")[METRICS].mean()
    macro.to_csv(out / "macro.csv")
    np.savez_compressed(out / "evaluation_scores.npz", source_row=np.arange(len(frame)),
                        hit=labels, group_id=group_id, **variants)
    write_json(out / "provenance.json", dict(
        command=sys.argv, prediction_source=str(Path(args.predictions).resolve()),
        prediction_sha256=sha256(args.predictions), rows=len(frame), hits=int(labels.sum()),
        groups=len(groups), selection=json.loads((root / "selection.json").read_text()),
        elapsed_seconds=time.perf_counter() - started,
        source_sha256=sha256(__file__),
        score_convention="All saved score variants are higher-is-better; percentile variants are negated.",
        model_hashes={str(path): sha256(path) for path in (root / "full_fit").glob("*")}))
    print(macro.to_string(), flush=True)


def plot_command(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.calibration_dir)
    selection = json.loads((root / "selection.json").read_text())
    selected = selection["selected"]
    accuracy = pd.read_csv(root / "calibration_accuracy.csv")
    sizes = pd.read_csv(root / "model_sizes.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, split in zip(axes, ["validation", "test"]):
        for method, group in accuracy.loc[accuracy.split == split].groupby("method", sort=False):
            ax.loglog(group.cutoff, group.smoothed_percent, marker="o", markersize=3,
                      linewidth=2.5 if method == selected else 1, label=method)
        ax.plot([.001, 90], [.001, 90], "--", color="black", linewidth=1, label="ideal")
        ax.axvspan(.001, .03, color="grey", alpha=.08)
        ax.set(xlabel="Predicted percentile cutoff (%)", ylabel="Observed background mass (%)",
               title=split.capitalize() + " background peptides")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Compact percentile calibration: peptide-grouped 60/20/20 split\n"
                 "Shaded rare tail is descriptive; counts use +0.5 smoothing only for log plotting")
    for suffix in ("png", "svg"):
        fig.savefig(root / ("background_calibration." + suffix), dpi=180)
    plt.close(fig)

    modes = [mode for mode in ["with-flanks", "without-flanks"] if (root / mode / "macro.csv").exists()]
    if not modes:
        return
    fig, axes = plt.subplots(len(modes), 3, figsize=(15, 4.2 * len(modes)),
                             squeeze=False, constrained_layout=True)
    tables = []
    for row, mode in enumerate(modes):
        table = pd.read_csv(root / mode / "macro.csv", index_col="method")
        group_count = len(pd.read_csv(root / mode / "groups.csv"))
        order = ["public_raw", "public_saved", "candidate_raw", "candidate_saved", "uniform_step",
                 "tail_step"] + sorted([name for name in table.index if name.startswith("compact_")],
                                        key=lambda name: int(name.split("_")[-1]))
        table = table.loc[order]
        tables.append((mode, table))
        colors = ["#00876c" if name == selected else "#4c78a8" if name.startswith("compact_")
                  else "#999999" for name in order]
        for ax, metric in zip(axes[row], METRICS):
            ax.barh(order, table[metric], color=colors)
            ax.invert_yaxis()
            ax.axvline(table.loc["candidate_raw", metric], color="black", linestyle="--", linewidth=1)
            title = {"auprc": "AUPRC", "auroc": "AUROC",
                     "ppv_expected_random_ties": "PPV@N (tie-neutral)"}[metric]
            ax.set(title=mode + ": " + title,
                   xlabel="Macro metric (%d sample/HLA groups)" % group_count)
            for index, value in enumerate(table[metric]):
                ax.text(value, index, " %.6f" % value, va="center", fontsize=8)
            ax.set_xlim(0, table[metric].max() * 1.17)
    fig.suptitle("Frozen presentation predictions; dashed line = candidate raw ranking\n"
                 "Green = selected using background validation only; PPV@N is tie-neutral")
    for suffix in ("png", "svg"):
        fig.savefig(root / ("presentation_metrics." + suffix), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    full_sizes = sizes.loc[sizes.fit == "full"].set_index("method")
    for mode in modes:
        timing = pd.read_csv(root / mode / "timings.csv").set_index("method").loc[full_sizes.index]
        axes[1].plot(timing.index, timing.median_seconds * 1000 / timing.rows * 1e6,
                     "o-", label=mode)
    axes[0].bar(full_sizes.index, full_sizes.coordinate_bytes / 1024)
    axes[0].set(yscale="log", ylabel="Numeric coordinate storage (KiB)", title="Fixed calibration size")
    axes[1].set(ylabel="Milliseconds per million scores",
                title="Mapping only; median of 3 calls\nLocal timing, concurrent test workload")
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    for suffix in ("png", "svg"):
        fig.savefig(root / ("size_and_runtime." + suffix), dpi=180)
    plt.close(fig)

    lines = ["# Compact presentation-percentile comparison", "",
             "Selected without presentation labels: **%s**." % selected, "",
             "## Calibration selection", "", "Validation RMS log10 cutoff error:", ""]
    lines.extend("- %s: %.6f" % (name, value)
                 for name, value in selection["validation_rms_log10_error"].items())
    lines.extend(["", "Selection uses cutoffs 0.03–10%; the smallest model within the predefined",
                  "tolerance of the best validation error is retained. Test background peptides",
                  "were not used for fitting or selection. All allele scores for a peptide stay",
                  "in one split. Full-reference refits are used below.", ""])
    for mode, table in tables:
        lines.extend(["## " + mode, "", "| Method | Macro AUPRC | Macro AUROC | Macro PPV@N |",
                      "|---|---:|---:|---:|"])
        lines.extend("| %s | %.6f | %.6f | %.6f |" % (name, *values)
                     for name, values in table[METRICS].iterrows())
        lines.append("")
    lines.extend(["## Interpretation and limits", "",
                  "These methods approximate background percentiles, not probabilities of presentation.",
                  "No networks were retrained and no saved candidate/public calibration was changed.",
                  "The paired uniform/tail/compact comparison uses the same 400,000 cached scores.",
                  "Existing candidate/public percentile baselines have different calibration references.",
                  "The cached reference uses uniform amino acids, lengths 8–15, and eight single-allele",
                  "queries; it is not the original release calibration policy. With-flank evaluation",
                  "reuses this no-flank reference, as in the preceding controlled binning comparison.",
                  "Tail extrapolation is an explicit model assumption; scores outside reference support",
                  "are counted in ranking_audit.csv. Logit interpolation cannot undo pre-existing raw",
                  "score ties or eliminate all floating-point saturation. Ranking preservation and",
                  "absolute percentile accuracy are distinct gates; neither implies release readiness.",
                  "All evaluation score arrays are higher-is-better (percentiles are negated), include",
                  "source-row indices, labels and group IDs, and join to the hashed original CSV.",
                  "Per-sample/micro metrics, calibration counts, serialized curves, and timing repetitions",
                  "are preserved alongside PNG/SVG figures. PPV uses expected hits within cutoff ties.", ""])
    (root / "REPORT.md").write_text("\n".join(lines))
    plot_inputs = [root / "selection.json", root / "calibration_accuracy.csv", root / "model_sizes.csv"]
    plot_inputs.extend(root / mode / name for mode in modes
                       for name in ("macro.csv", "timings.csv", "groups.csv"))
    write_json(root / "plot_provenance.json", dict(
        command=sys.argv, script_sha256=sha256(__file__), matplotlib_version=matplotlib.__version__,
        input_hashes={str(path): sha256(path) for path in plot_inputs},
        notes="Plot-only refinements follow the numerical comparison; its original source is archived."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit and select using background scores only")
    fit.add_argument("--reference-dir", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--seed", type=int, default=403)
    fit.add_argument("--knots", nargs="+", type=int, default=[64, 128, 256])
    fit.set_defaults(function=fit_command)
    evaluate = commands.add_parser("evaluate", help="Score frozen presentation predictions")
    evaluate.add_argument("--calibration-dir", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--mode", required=True, choices=["with-flanks", "without-flanks"])
    evaluate.set_defaults(function=evaluate_command)
    plot = commands.add_parser("plot", help="Generate report and paper-ready figures")
    plot.add_argument("--calibration-dir", required=True)
    plot.set_defaults(function=plot_command)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

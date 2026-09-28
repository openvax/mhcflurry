#!/usr/bin/env python3
"""Compare saved MHCflurry scores with precomputed NetMHCpan and MixMHCpred.

Compare-models prediction tables are joined row for row with the NetMHCpan 4.0
BA, NetMHCpan 4.0 EL and MixMHCpred columns distributed in data_evaluation.
Per-sample, macro and pooled (micro) AUROC, AUPRC and PPV@N use the
compare-models metric code. By default each predictor uses its covered rows
and each pair uses rows both score. With --coverage common every metric and
figure uses the same intersection of covered rows. Paired intervals resample
whole samples. No predictor is run here.
"""

import argparse
from collections import namedtuple
import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy
import pandas

from mhcflurry.cli.compare_models import _metrics, _normalize_benchmark_genotype
from mhcflurry.experiment_archive import sha256_file


KEYS = ["protein_accession", "peptide", "sample_id", "n_flank", "c_flank", "hit", "hla"]
METRICS = ["roc_auc", "pr_auc", "ppv_at_n"]
METRIC_TITLES = {"roc_auc": "AUROC", "pr_auc": "AUPRC", "ppv_at_n": "PPV@N"}
EXTERNAL = {
    "netmhcpan4.el": ("NetMHCpan 4.0 EL", True),
    "netmhcpan4.ba": ("NetMHCpan 4.0 BA", False),
    "netmhcpan4.1.el": ("NetMHCpan 4.1 EL", True),
    "netmhcpan4.1.ba": ("NetMHCpan 4.1 BA", False),
    "netmhcpan4.2.el": ("NetMHCpan 4.2 EL", True),
    "netmhcpan4.2.ba": ("NetMHCpan 4.2 BA", False),
    "mixmhcpred": ("MixMHCpred", True),
}
# Color carries the predictor family; version is an ordinal step within it, so
# three NetMHCpan versions do not spend three unrelated hues. Both ramps pass
# the ordinal checks (monotone lightness, >=0.06 steps, one hue, light end
# clear of the surface). Five families cannot all separate under deuteranopia
# in one chart, so the eluted-ligand and binding-affinity ramps belong to
# separate facets; MixMHCpred and the reference series stay deliberately muted
# and always carry direct labels.
ROLE_COLORS = {"a": "#2a78d6", "b": "#eb6834",
               "netmhcpan4.el": "#34c191", "netmhcpan4.1.el": "#199a6d",
               "netmhcpan4.2.el": "#0c6b4b",
               "netmhcpan4.ba": "#a79ee2", "netmhcpan4.1.ba": "#6d5dc6",
               "netmhcpan4.2.ba": "#42328f",
               "mixmhcpred": "#898781", "baseline": "#5c5b55",
               "random": "#c3c2b7"}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TERMINAL_RESIDUES = 4
BASELINES = ("random", "terminal-logistic")
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE_INK = "#e1e0d9", "#c3c2b7"
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE_INK, "axes.linewidth": 0.8, "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": BASELINE_INK, "ytick.color": BASELINE_INK,
    "xtick.labelcolor": SECONDARY, "ytick.labelcolor": INK, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "font.size": 9, "text.color": INK,
    "legend.frameon": False,
}

Condition = namedtuple("Condition", ["label", "higher_is_better", "role"])


def row_identity(frame):
    """Hash benchmark row identity; NA-like strings compare consistently.

    compare-models saves canonical genotypes (sorted, homozygous copies
    removed), so both sides pass through its idempotent normalizer first.
    """
    hit = pandas.to_numeric(frame["hit"], errors="raise")
    if not hit.isin([0, 1]).all():
        raise ValueError("Benchmark hit values must be 0 or 1")
    keys = pandas.DataFrame({
        name: hit.astype("int64") if name == "hit" else frame[name].fillna("").astype(str)
        for name in KEYS})
    genotypes = {value: _normalize_benchmark_genotype(value) for value in pandas.unique(keys["hla"])}
    keys["hla"] = keys["hla"].map(genotypes)
    return pandas.util.hash_pandas_object(keys, index=False).to_numpy()


def read_predictions(path, score_columns):
    """Read one compare-models table with identity, source file and scores."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing saved predictions: %s" % path)
    required = set(KEYS) | {"source_file"} | set(score_columns)
    frame = pandas.read_csv(
        path, usecols=lambda name: name in required or name in EXTERNAL,
        dtype={name: str for name in KEYS if name != "hit"}, low_memory=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("%s lacks columns: %s" % (path, ", ".join(missing)))
    frame["_identity"] = row_identity(frame)
    frame["hit"] = frame["hit"].astype("int64")
    return frame


def load_saved_scores(comparison_dir, cohort, a_label, b_label):
    """Return saved rows, named score conditions and their input files."""
    comparison_dir = Path(comparison_dir)
    conditions = {}
    if cohort == "multiallelic":
        score_columns = ["a_presentation_score", "b_presentation_score",
                         "a_presentation_percentile", "b_presentation_percentile"]
        paths = {mode: comparison_dir / "presentation" / ("predictions_%s.csv.bz2" % mode)
                 for mode in ("with_flanks", "without_flanks")}
        tables = {mode: read_predictions(path, score_columns) for mode, path in paths.items()}
        with_flanks, without_flanks = tables["with_flanks"], tables["without_flanks"]
        if (len(with_flanks) != len(without_flanks)
                or not numpy.array_equal(with_flanks["_identity"], without_flanks["_identity"])
                or not with_flanks.source_file.equals(without_flanks.source_file)):
            raise ValueError("With- and without-flank predictions must describe identical rows in one order")
        keep = ["sample_id", "source_file", "hit", "peptide", "_identity"] + [
            name for name in EXTERNAL if name in with_flanks]
        frame = with_flanks[keep].copy()
        for mode, table in tables.items():
            mode_label = mode.replace("_", " ")
            for side, label in (("a", a_label), ("b", b_label)):
                frame["%s_%s" % (side, mode)] = table[side + "_presentation_score"].to_numpy(dtype=float)
                conditions["%s_%s" % (side, mode)] = Condition("%s, %s" % (label, mode_label), True, side)
                name = "%s_%s_percentile" % (side, mode)
                frame[name] = table[side + "_presentation_percentile"].to_numpy(dtype=float)
                conditions[name] = Condition("%s, %s, percentile" % (label, mode_label), False, side)
        return frame, conditions, list(paths.values())
    path = comparison_dir / "affinity" / "predictions.csv.bz2"
    table = read_predictions(path, ["a_pred", "b_pred"])
    frame = table[["sample_id", "source_file", "hit", "peptide", "_identity"] + [
        name for name in EXTERNAL if name in table]].copy()
    for side, label in (("a", a_label), ("b", b_label)):
        frame[side + "_affinity"] = table[side + "_pred"].to_numpy(dtype=float)
        conditions[side + "_affinity"] = Condition("%s affinity" % label, False, side)
    return frame, conditions, [path]


def external_path(directories, cohort, predictor, source_file):
    """Find one predictor's file for a saved benchmark file, compressed or not."""
    marker = ".train_excluded."
    if not source_file.startswith("benchmark.%s." % cohort) or marker not in source_file:
        raise ValueError("Unrecognized %s benchmark source file: %s" % (cohort, source_file))
    name = "benchmark.%s.%s%s%s" % (cohort, predictor, marker, source_file.split(marker, 1)[1])
    names = [name, name[:-len(".bz2")]] if name.endswith(".bz2") else [name]
    for directory in directories:
        for candidate in names:
            path = Path(directory) / candidate
            if path.is_file():
                return path
    raise FileNotFoundError("No %s file for %s in: %s" % (
        predictor, source_file, ", ".join(str(directory) for directory in directories)))


def attach_external(frame, cohort, directories, predictors):
    """Attach precomputed predictor columns by source file and exact row identity.

    Saved rows are an ordered subset of each benchmark file, so the k-th saved
    occurrence of an identity matches the k-th external occurrence. Every saved
    row must match exactly once; a saved copy of a predictor column must agree.
    """
    frame = frame.copy()
    frame["_row"] = numpy.arange(len(frame))
    frame["_occurrence"] = frame.groupby(["source_file", "_identity"], sort=False).cumcount()
    pieces, inputs = [], []
    for source_file, sub in frame.groupby("source_file", sort=True):
        for predictor in predictors:
            path = external_path(directories, cohort, predictor, source_file)
            raw = pandas.read_csv(path, usecols=KEYS + [predictor],
                                  dtype={name: str for name in KEYS if name != "hit"},
                                  low_memory=False)
            external = pandas.DataFrame({
                "_identity": row_identity(raw),
                predictor: pandas.to_numeric(raw[predictor], errors="coerce").to_numpy()})
            external["_occurrence"] = external.groupby("_identity", sort=False).cumcount()
            existing = sub.pop(predictor).to_numpy(dtype=float) if predictor in sub else None
            merged = sub.merge(external, on=["_identity", "_occurrence"], how="left",
                               validate="one_to_one", indicator=True)
            unmatched = int((merged["_merge"] != "both").sum())
            if unmatched:
                raise ValueError("%d saved rows from %s lack a matching %s row" % (
                    unmatched, source_file, predictor))
            if existing is not None and not numpy.allclose(
                    existing, merged[predictor].to_numpy(dtype=float),
                    rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError("Saved %s values disagree with %s" % (predictor, path.name))
            sub = merged.drop(columns="_merge")
            inputs.append(path)
        pieces.append(sub)
    joined = pandas.concat(pieces, ignore_index=True).sort_values("_row", kind="stable")
    return joined.drop(columns="_occurrence").reset_index(drop=True), inputs


def condition_scores(frame, name, condition):
    """Finite-row mask and higher-is-better scores for one condition."""
    values = frame[name].to_numpy(dtype=float)
    return numpy.isfinite(values), values if condition.higher_is_better else -values


def per_sample_metrics(frame, names, conditions, mask):
    """compare-models metrics per sample for named conditions on masked rows."""
    rows = numpy.flatnonzero(mask)
    hits = frame.hit.to_numpy()[rows]
    positions = pandas.Series(numpy.arange(len(rows))).groupby(
        frame.sample_id.to_numpy()[rows], sort=True).indices
    scores = {name: condition_scores(frame, name, conditions[name])[1][rows] for name in names}
    records = []
    for sample in sorted(positions):
        index = positions[sample]
        for name in names:
            records.append(dict(sample_id=sample, condition=name, label=conditions[name].label,
                                **_metrics(hits[index], scores[name][index])))
    result = pandas.DataFrame(records)
    if (result.empty or result[METRICS].isna().any().any()
            or result.sample_id.nunique() != frame.sample_id.nunique()):
        raise ValueError("Every sample needs scored hits and decoys for: %s" % ", ".join(names))
    return result


def score_conditions(frame, conditions):
    """Each predictor on the rows it scores: per-sample, macro and pooled metrics."""
    coverage = pandas.DataFrame({"sample_id": frame.sample_id.to_numpy(), "rows": 1})
    per_sample, summary = [], []
    hits = frame.hit.to_numpy()
    for name, condition in conditions.items():
        mask, scores = condition_scores(frame, name, condition)
        coverage[name + "_unscored"] = ~mask
        metrics = per_sample_metrics(frame, [name], conditions, mask)
        pooled = _metrics(hits[mask], scores[mask])
        summary.append(dict(condition=name, label=condition.label,
                            samples=metrics.sample_id.nunique(), rows=pooled["n"],
                            hits=pooled["n_pos"],
                            **{"macro_" + metric: metrics[metric].mean() for metric in METRICS},
                            **{"micro_" + metric: pooled[metric] for metric in METRICS}))
        per_sample.append(metrics)
    coverage = coverage.groupby("sample_id", sort=True).sum().reset_index()
    return pandas.concat(per_sample, ignore_index=True), pandas.DataFrame(summary), coverage


def common_coverage(frame, conditions):
    """Retain identical scored rows for every model, preserving every sample."""
    mask = numpy.logical_and.reduce([
        condition_scores(frame, name, condition)[0] for name, condition in conditions.items()])
    result = frame.loc[mask].copy()
    if set(result.sample_id) != set(frame.sample_id):
        raise ValueError("Common coverage would drop an entire evaluation sample")
    return result


def terminal_residue_features(peptides):
    """One-hot the first and last four residues; other symbols share one slot."""
    import scipy.sparse
    series = pandas.Series(peptides, dtype=object).fillna("").astype(str).str.upper()
    lookup = numpy.full(256, len(AMINO_ACIDS), dtype=numpy.int64)
    for index, residue in enumerate(AMINO_ACIDS):
        lookup[ord(residue)] = index
    width, alphabet = TERMINAL_RESIDUES, len(AMINO_ACIDS) + 1
    blocks = []
    for part in (series.str[:width].str.ljust(width, "X"),
                 series.str[-width:].str.rjust(width, "X")):
        raw = numpy.frombuffer("".join(part).encode("ascii", "replace"), dtype=numpy.uint8)
        blocks.append(lookup[raw].reshape(len(series), width))
    columns = numpy.hstack(blocks) + numpy.arange(2 * width) * alphabet
    rows = numpy.repeat(numpy.arange(len(series)), 2 * width)
    return scipy.sparse.csr_matrix(
        (numpy.ones(rows.size, dtype=numpy.float32), (rows, columns.ravel())),
        shape=(len(series), 2 * width * alphabet))


def leave_one_sample_out_logistic(frame):
    """Score each sample with a logistic regression fitted on every other sample.

    Features are the one-hot first and last four residues only, with no MHC or
    flank input, so this measures how much generic peptide-terminus
    composition separates hits from decoys. A sample's own labels never reach
    its scores.
    """
    from sklearn.linear_model import LogisticRegression
    features = terminal_residue_features(frame.peptide.to_numpy())
    hits = frame.hit.to_numpy()
    samples = frame.sample_id.to_numpy()
    scores = numpy.full(len(frame), numpy.nan)
    for sample in numpy.unique(samples):
        test, train = numpy.flatnonzero(samples == sample), numpy.flatnonzero(samples != sample)
        if len(numpy.unique(hits[train])) < 2:
            raise ValueError("Leave-one-sample-out baseline needs hits and decoys outside %s" % sample)
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(features[train], hits[train])
        scores[test] = model.decision_function(features[test])
    return scores


def comparison_pairs(cohort, conditions):
    """Candidate/reference pairs reported with paired intervals."""
    candidate, reference = (("a_with_flanks", "b_with_flanks")
                            if cohort == "multiallelic" else ("a_affinity", "b_affinity"))
    pairs = [(candidate, reference)]
    # Derive external pairs from the supported registry, including every
    # requested version and both score types. Availability is filtered below.
    pairs.extend((candidate, name) for name in EXTERNAL)
    pairs.append((candidate, "terminal_logistic"))
    if cohort == "multiallelic":
        pairs.extend([("a_without_flanks", "b_without_flanks"),
                      ("a_with_flanks_percentile", "b_with_flanks_percentile")])
    pairs.extend((reference, name) for name in EXTERNAL)
    return [pair for pair in pairs if set(pair) <= set(conditions)]


def paired_comparisons(frame, conditions, per_sample, pairs, replicates, seed):
    """Paired sample bootstrap per pair on rows both predictors score.

    One seed gives every pair the same resampled samples.
    """
    path = Path(__file__).resolve().parent / "paired_sample_metrics.py"
    spec = importlib.util.spec_from_file_location("paired_sample_metrics", path)
    paired_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paired_module)
    summaries, differences, draws = [], [], {}
    for candidate, reference in pairs:
        candidate_mask = condition_scores(frame, candidate, conditions[candidate])[0]
        reference_mask = condition_scores(frame, reference, conditions[reference])[0]
        joint = candidate_mask & reference_mask
        if joint.sum() == candidate_mask.sum() == reference_mask.sum():
            subset = per_sample.loc[per_sample.condition.isin([candidate, reference])]
        else:
            subset = per_sample_metrics(frame, [candidate, reference], conditions, joint)
        summary, delta, bootstrap = paired_module.paired_summary(
            subset[["sample_id", "condition", "n", "n_pos"] + METRICS], units=["sample_id"],
            condition="condition", metrics=METRICS, baseline=reference,
            replicates=replicates, seed=seed)
        summaries.append(summary.rename(columns={"condition": "candidate", "baseline": "reference"})
                         .assign(rows_excluded=int(len(frame) - joint.sum())))
        differences.append(delta.rename(columns={"condition": "candidate"}).assign(reference=reference))
        for key, values in bootstrap.items():
            draws["%s-vs-%s:%s" % (candidate, reference, key.split(":", 1)[1])] = values
    return pandas.concat(summaries, ignore_index=True), pandas.concat(differences, ignore_index=True), draws


def color_for(name, conditions):
    return ROLE_COLORS[conditions[name].role]


def spread(count):
    return numpy.linspace(-0.18, 0.18, count) if count > 1 else numpy.zeros(count)


def plot_metric_strips(names, conditions, per_sample, summary, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.8 + 0.45 * len(names)), sharey=True,
                             layout="constrained")
    macro = summary.set_index("condition")
    for axis, metric in zip(axes, METRICS):
        for row, name in enumerate(names):
            values = per_sample.loc[per_sample.condition == name, metric].to_numpy()
            color = color_for(name, conditions)
            axis.scatter(values, row + spread(len(values)), s=12, color=color, alpha=0.4,
                         linewidths=0, zorder=2)
            mean = macro.loc[name, "macro_" + metric]
            axis.scatter([mean], [row], s=64, color=color, edgecolors=SURFACE, linewidths=1.5,
                         zorder=3)
            axis.annotate("%.3f" % mean, (mean, row), xytext=(0, 9), textcoords="offset points",
                          ha="center", fontsize=8, color=SECONDARY)
        axis.set_title("Macro " + METRIC_TITLES[metric], loc="left")
        axis.grid(axis="x")
        axis.set_axisbelow(True)
        axis.set_xlabel("Small dots: samples. Large dots: macro mean.")
    axes[0].set_yticks(numpy.arange(len(names)), labels=[conditions[name].label for name in names])
    axes[0].set_ylim(len(names) - 0.5, -0.6)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return fig


def plot_paired(pairs, conditions, paired, differences, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.8 + 0.62 * len(pairs)), sharey=True,
                             layout="constrained")
    for axis, metric in zip(axes, METRICS):
        for row, (candidate, reference) in enumerate(pairs):
            match = (paired.candidate == candidate) & (paired.reference == reference)
            summary = paired.loc[match & (paired.metric == metric)].iloc[0]
            deltas = differences.loc[(differences.candidate == candidate)
                                     & (differences.reference == reference)
                                     & (differences.metric == metric), "delta"].to_numpy()
            color = color_for(reference, conditions)
            axis.scatter(deltas, row + spread(len(deltas)), s=12, color=MUTED, alpha=0.55,
                         linewidths=0, zorder=2)
            axis.hlines(row, summary.ci_low, summary.ci_high, color=color, linewidth=2, zorder=3)
            axis.scatter([summary.delta], [row], s=64, color=color, edgecolors=SURFACE,
                         linewidths=1.5, zorder=4)
        axis.axvline(0, color=BASELINE_INK, linewidth=1, zorder=1)
        axis.set_title(METRIC_TITLES[metric] + " difference", loc="left")
        axis.grid(axis="x")
        axis.set_axisbelow(True)
        axis.set_xlabel("Dots: samples. Bar: 95% paired sample interval.")
    axes[0].set_yticks(numpy.arange(len(pairs)), labels=[
        "%s\nminus %s" % (conditions[candidate].label, conditions[reference].label)
        for candidate, reference in pairs])
    axes[0].set_ylim(len(pairs) - 0.5, -0.6)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return fig


def plot_precision_recall(names, conditions, summary, frame, title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, axis = plt.subplots(figsize=(7.5, 5.4), layout="constrained")
    micro = summary.set_index("condition")["micro_pr_auc"]
    hits = frame.hit.to_numpy()
    for name in names:
        mask, scores = condition_scores(frame, name, conditions[name])
        precision, recall, _ = precision_recall_curve(hits[mask], scores[mask])
        step = max(1, len(recall) // 4000)
        index = numpy.unique(numpy.r_[numpy.arange(0, len(recall), step), len(recall) - 1])
        axis.plot(recall[index], precision[index], color=color_for(name, conditions), linewidth=2,
                  solid_joinstyle="round", solid_capstyle="round",
                  label="%s (pooled AUPRC %.3f)" % (conditions[name].label, micro[name]))
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    axis.grid(True)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return fig


def render(out, cohort, conditions, frame, per_sample, summary, paired, differences, pairs):
    """Write the comparison pages as one PDF and separate PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    samples = int(summary.samples.iloc[0])
    cohort_title = ("%d held-out multiallelic presentation samples" % samples
                    if cohort == "multiallelic" else "%d monoallelic samples" % samples)
    raw = [name for name in conditions if not name.endswith("_percentile")]
    curve_order = (["a_with_flanks", "b_with_flanks"] if cohort == "multiallelic"
                   else ["a_affinity", "b_affinity"])
    curve_order += [name for name in EXTERNAL if name in conditions]
    # Candidate-versus-reference differences are an order of magnitude smaller
    # than external gaps; separate pages keep both legible.
    internal = [pair for pair in pairs if conditions[pair[1]].role in ("a", "b")]
    external = [pair for pair in pairs if conditions[pair[1]].role not in ("a", "b")]
    with plt.rc_context(STYLE), PdfPages(out / "external_comparison.pdf") as pdf:
        pages = [("macro_metrics", lambda: plot_metric_strips(
            raw, conditions, per_sample, summary, "Ranking metrics per predictor, " + cohort_title))]
        if internal:
            pages.append(("paired_differences_mhcflurry", lambda: plot_paired(
                internal, conditions, paired, differences,
                "MHCflurry candidate versus MHCflurry reference, " + cohort_title)))
        if external:
            pages.append(("paired_differences_external", lambda: plot_paired(
                external, conditions, paired, differences,
                "MHCflurry versus NetMHCpan and MixMHCpred, " + cohort_title)))
        pages.append(("precision_recall", lambda: plot_precision_recall(
            [name for name in curve_order if name in conditions], conditions, summary, frame,
            "Pooled precision-recall, " + cohort_title)))
        for name, make in pages:
            fig = make()
            pdf.savefig(fig)
            fig.savefig(out / (name + ".png"), dpi=180)
            plt.close(fig)


def markdown(frame, columns, headers, formats):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" if fmt is None else "---:" for fmt in formats) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(
            str(row[column]) if fmt is None else fmt % row[column]
            for column, fmt in zip(columns, formats)) + " |")
    return "\n".join(lines)


def write_summary_markdown(out, args, summary, paired, coverage):
    paired = paired.assign(interval=[
        "[%+.4f, %+.4f]" % (low, high) for low, high in zip(paired.ci_low, paired.ci_high)],
        improved=["%d/%d" % (up, n) for up, n in zip(paired.samples_improved, paired.samples)])
    labels = summary.set_index("condition")["label"]
    paired = paired.assign(candidate_label=paired.candidate.map(labels),
                           reference_label=paired.reference.map(labels),
                           metric_label=paired.metric.map(METRIC_TITLES))
    unscored = ["%s %d" % (labels[column[:-len("_unscored")]], int(coverage[column].sum()))
                for column in coverage.columns
                if column.endswith("_unscored") and coverage[column].sum()]
    coverage_description = (
        "Every predictor, paired comparison and figure uses the same %d rows "
        "(%d rows excluded from every predictor)." % (
            int(summary.rows.iloc[0]), int(coverage.common_excluded.sum()))
        if args.coverage == "common" else
        "Each predictor is scored on the rows it covers and each paired "
        "comparison on rows both predictors score.")
    text = [
        "# External predictor comparison (%s)" % args.cohort, "",
        "%d samples, %d original benchmark rows. %s Rows without a score before "
        "filtering: %s. Macro metrics average samples; micro metrics pool rows. Intervals "
        "are paired sample bootstraps (%d draws, seed %d): exploratory, conditional on "
        "the trained models, and uncorrected for multiple comparisons." % (
            int(summary.samples.iloc[0]), int(coverage.rows.sum()),
            coverage_description, ", ".join(unscored) or "none", args.replicates, args.seed), "",
        "## Metrics", "",
        markdown(summary, ["label", "rows", "macro_roc_auc", "macro_pr_auc", "macro_ppv_at_n",
                           "micro_roc_auc", "micro_pr_auc", "micro_ppv_at_n"],
                 ["Predictor", "Rows", "Macro AUROC", "Macro AUPRC", "Macro PPV@N",
                  "Micro AUROC", "Micro AUPRC", "Micro PPV@N"],
                 [None, "%d"] + ["%.4f"] * 6), "",
        "## Paired differences", "",
        markdown(paired, ["candidate_label", "reference_label", "metric_label", "delta",
                          "interval", "improved", "rows_excluded"],
                 ["Candidate", "Reference", "Metric", "Difference", "95% interval",
                  "Samples improved", "Rows excluded"],
                 [None, None, None, "%+.4f", None, None, "%d"]), ""]
    (out / "summary.md").write_text("\n".join(text))


def json_argument(value):
    """Paths, including repeatable path options, are recorded as strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [str(item) if isinstance(item, Path) else item for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.environ.get("MHCFLURRY_CLI_PROG"), description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True,
                        help="compare-models output directory with saved prediction tables.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="data_evaluation download containing precomputed predictor files.")
    parser.add_argument("--external-dir", type=Path, action="append", default=[],
                        help="Extra directory of predictor files, repeatable. Searched after "
                        "--data-dir; files may be plain CSV. Use for locally generated scores.")
    parser.add_argument("--cohort", choices=("multiallelic", "monoallelic"), default="multiallelic",
                        help="multiallelic: saved presentation scores; monoallelic: saved affinities.")
    parser.add_argument("--external", default=",".join(EXTERNAL),
                        help="Comma-separated precomputed predictors (default: %(default)s).")
    parser.add_argument("--baselines", default=",".join(BASELINES),
                        help="Reference baselines: random, terminal-logistic, or none "
                        "(default: %(default)s).")
    parser.add_argument("--a-label", default="MHCflurry candidate")
    parser.add_argument("--b-label", default="MHCflurry 2.2")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage", choices=("available", "common"), default="available",
                        help="Use each predictor's covered rows, or identical rows for every metric and figure.")
    parser.add_argument("--skip-joined-table", action="store_true",
                        help="Do not write joined_scores.csv.gz (large monoallelic cohorts).")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory.")
    args = parser.parse_args(argv)
    predictors = [name.strip() for name in args.external.split(",") if name.strip()]
    if not predictors or set(predictors) - set(EXTERNAL):
        raise ValueError("--external must name predictors from: %s" % ", ".join(EXTERNAL))
    baselines = [] if args.baselines.strip().lower() == "none" else [
        name.strip() for name in args.baselines.split(",") if name.strip()]
    if set(baselines) - set(BASELINES):
        raise ValueError("--baselines must name: %s, or none" % ", ".join(BASELINES))
    if args.replicates < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Output directory must be new or empty: %s" % args.out)
    frame, conditions, inputs = load_saved_scores(
        args.comparison_dir, args.cohort, args.a_label, args.b_label)
    frame, external_inputs = attach_external(
        frame, args.cohort, [args.data_dir] + list(args.external_dir), predictors)
    for predictor in predictors:
        label, higher_is_better = EXTERNAL[predictor]
        conditions[predictor] = Condition(label, higher_is_better, predictor)
    if "random" in baselines:
        frame["random"] = numpy.random.default_rng(args.seed).random(len(frame))
        conditions["random"] = Condition("Random scores", True, "random")
    if "terminal-logistic" in baselines:
        frame["terminal_logistic"] = leave_one_sample_out_logistic(frame)
        conditions["terminal_logistic"] = Condition(
            "Terminal 4-residue logistic regression, no MHC", True, "baseline")
    original_rows, original_hits = len(frame), int(frame.hit.sum())
    original_coverage = None
    if args.coverage == "common":
        common = common_coverage(frame, conditions)
        original_coverage = pandas.DataFrame({
            "sample_id": frame.sample_id, "rows": 1,
            "common_excluded": ~frame.index.isin(common.index),
            **{name + "_unscored": ~condition_scores(frame, name, condition)[0]
               for name, condition in conditions.items()},
        }).groupby("sample_id", sort=True).sum().reset_index()
        frame = common
    per_sample, summary, coverage = score_conditions(frame, conditions)
    if original_coverage is not None:
        coverage = original_coverage
    pairs = comparison_pairs(args.cohort, conditions)
    paired, differences, draws = paired_comparisons(
        frame, conditions, per_sample, pairs, args.replicates, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(args.out / "per_sample_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    coverage.to_csv(args.out / "coverage.csv", index=False)
    paired.to_csv(args.out / "paired_differences.csv", index=False)
    differences.to_csv(args.out / "sample_differences.csv", index=False)
    numpy.savez_compressed(args.out / "bootstrap_deltas.npz", **draws)
    if not args.skip_joined_table:
        columns = ["_row", "source_file", "sample_id", "hit"] + list(conditions)
        frame[columns].rename(columns={"_row": "prediction_row"}).to_csv(
            args.out / "joined_scores.csv.gz", index=False,
            compression={"method": "gzip", "compresslevel": 1})
    render(args.out, args.cohort, conditions, frame, per_sample, summary, paired, differences, pairs)
    write_summary_markdown(args.out, args, summary, paired, coverage)
    shutil.copyfile(__file__, args.out / "analysis_source.py")
    provenance = {
        "arguments": {key: json_argument(value) for key, value in vars(args).items()},
        "conditions": {name: condition._asdict() for name, condition in conditions.items()},
        "pairs": pairs,
        "coverage": {"policy": args.coverage, "original_rows": original_rows,
                     "original_hits": original_hits, "scored_rows": len(frame),
                     "scored_hits": int(frame.hit.sum()),
                     "excluded_rows": original_rows - len(frame)},
        "method": ("compare-models _metrics per sample; " + (
            "all predictors and pairs on identical common rows; " if args.coverage == "common" else
            "each predictor on its scored rows, each pair on rows both score; ") +
            "paired sample percentile bootstrap of equal-sample macro means"),
        "baselines": {
            "random": "uniform scores from numpy default_rng(seed)",
            "terminal-logistic": "one-hot first and last four residues (21 symbols each), "
                                 "sklearn LogisticRegression(C=1), fitted leave-one-sample-out; "
                                 "no MHC or flank input"},
        "inputs": [{"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
                   for path in inputs + external_inputs],
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print((args.out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

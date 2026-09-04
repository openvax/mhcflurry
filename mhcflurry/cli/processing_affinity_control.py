# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Evaluate processing scores on affinity-controlled hit/decoy risk sets."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy
import pandas

from .compare_models import _metrics
from ..experiment_archive import sha256_file


IDENTITY_COLUMNS = (
    "sample_id", "peptide", "hit", "n_flank", "c_flank",
)
AFFINITY_COLUMN = "mhcflurry_production_affinity"


def make_parser(prog="mhcflurry eval processing-affinity-control"):
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--score", action="append", required=True, metavar="NAME=PATH:COLUMN",
        help=(
            "Named score column from a saved processing prediction table. "
            "Repeat for every model to compare."),
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Score name used as the paired comparison baseline.",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="data_evaluation directory containing cached production affinity.",
    )
    parser.add_argument(
        "--existing",
        help=(
            "Existing processing-affinity-control output to extend. Its "
            "ordered cohort and risk-set assignments are verified and reused, "
            "so only the new --score tables are read."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--decoys-per-hit", type=int, default=10)
    parser.add_argument(
        "--same-protein-caliper", type=float, default=0.25,
        help=(
            "Maximum absolute log10-affinity distance for preferred "
            "same-protein decoys. Remaining slots use the same sample and "
            "peptide length. Default: 0.25."),
    )
    return parser


def _parse_score_spec(value):
    try:
        name, source = value.split("=", 1)
        path, column = source.rsplit(":", 1)
    except ValueError as error:
        raise ValueError(
            "Invalid --score %r; expected NAME=PATH:COLUMN" % value
        ) from error
    if not name or not path or not column:
        raise ValueError(
            "Invalid --score %r; expected NAME=PATH:COLUMN" % value)
    return name, os.path.abspath(path), column


def _cohort_hash(frame):
    normalized = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
    normalized["hit"] = pandas.to_numeric(
        normalized["hit"], errors="raise").astype("int8")
    for column in ("sample_id", "peptide", "n_flank", "c_flank"):
        normalized[column] = normalized[column].fillna("").astype(str)
    return pandas.util.hash_pandas_object(
        normalized, index=False).to_numpy(dtype="uint64")


def _load_scores(specs):
    cohort = None
    cohort_hash = None
    sources = []
    seen = set()
    specs_by_path = {}
    for spec in specs:
        name, path, column = _parse_score_spec(spec)
        if name in seen:
            raise ValueError("Duplicate score name: %s" % name)
        seen.add(name)
        if not os.path.isfile(path):
            raise ValueError("Score table does not exist: %s" % path)
        specs_by_path.setdefault(path, []).append((name, column))

    for path, path_specs in specs_by_path.items():
        score_columns = list(dict.fromkeys(
            column for _, column in path_specs))
        usecols = list(IDENTITY_COLUMNS) + score_columns
        frame = pandas.read_csv(path, usecols=usecols)
        current_hash = _cohort_hash(frame)
        if cohort is None:
            cohort = frame.loc[:, IDENTITY_COLUMNS].copy()
            cohort_hash = current_hash
        elif (
                len(current_hash) != len(cohort_hash) or
                not numpy.array_equal(current_hash, cohort_hash)):
            raise ValueError(
                "Scores from %s do not use the same ordered benchmark cohort" %
                path)
        digest = sha256_file(path)
        for name, column in path_specs:
            values = pandas.to_numeric(frame[column], errors="coerce")
            if not numpy.isfinite(values).all():
                raise ValueError(
                    "%s has %d non-finite scores" % (
                        name, int((~numpy.isfinite(values)).sum())))
            cohort[name] = values.to_numpy(dtype="float64")
            sources.append({
                "name": name,
                "path": path,
                "column": column,
                "sha256": digest,
            })
    return cohort, sources


def _extend_existing(existing_dir, new_cohort, new_sources):
    """Attach new score columns to verified saved cohort and risk sets."""
    existing_dir = Path(existing_dir)
    experiment_path = existing_dir / "experiment.json"
    heldout_path = existing_dir / "heldout_predictions.csv.bz2"
    matched_path = existing_dir / "matched_predictions.csv.bz2"
    for path in (experiment_path, heldout_path, matched_path):
        if not path.is_file():
            raise ValueError("Existing affinity-control artifact missing: %s" % path)

    configuration = json.loads(experiment_path.read_text())
    old_sources = configuration.get("score_sources", [])
    old_names = [source["name"] for source in old_sources]
    new_names = [source["name"] for source in new_sources]
    duplicates = set(old_names).intersection(new_names)
    if duplicates:
        raise ValueError(
            "New score names already exist: %s" % sorted(duplicates))

    heldout = pandas.read_csv(heldout_path)
    if (
            len(heldout) != len(new_cohort) or
            not numpy.array_equal(
                _cohort_hash(heldout), _cohort_hash(new_cohort))):
        raise ValueError(
            "New scores do not use the existing ordered benchmark cohort")
    for name in new_names:
        heldout[name] = new_cohort[name].to_numpy(dtype="float64")

    matched = pandas.read_csv(matched_path)
    if "source_row" not in matched:
        raise ValueError("Existing matched predictions lack source_row")
    source_rows = pandas.to_numeric(
        matched.source_row, errors="raise").to_numpy(dtype="int64")
    if (
            len(source_rows) and
            (source_rows.min() < 0 or source_rows.max() >= len(heldout))):
        raise ValueError("Existing matched predictions have invalid source_row")
    source_identity = heldout.iloc[source_rows].reset_index(drop=True)
    if not numpy.array_equal(
            _cohort_hash(source_identity), _cohort_hash(matched)):
        raise ValueError(
            "Existing matched predictions do not agree with source_row")
    for name in new_names:
        matched[name] = heldout[name].to_numpy()[source_rows]

    return {
        "heldout": heldout,
        "matched": matched,
        "score_sources": old_sources + new_sources,
        "affinity_sources": configuration.get("affinity_sources", []),
        "diagnostics": configuration.get("diagnostics", {}),
        "existing_experiment": str(experiment_path.resolve()),
        "existing_experiment_sha256": sha256_file(experiment_path),
    }


def _production_paths_by_sample(data_dir):
    pattern = os.path.join(
        data_dir,
        "benchmark.multiallelic.production.train_excluded.*.csv.bz2",
    )
    result = {}
    for path in sorted(glob.glob(pattern)):
        sample = pandas.read_csv(path, usecols=["sample_id"], nrows=1)
        if sample.empty:
            raise ValueError("Cached production-affinity file is empty: %s" % path)
        sample_id = str(sample.sample_id.iloc[0])
        if sample_id in result:
            raise ValueError(
                "Multiple cached production-affinity files for sample %s" %
                sample_id)
        result[sample_id] = path
    return result


def _attach_affinity(cohort, data_dir):
    frames = []
    paths = []
    paths_by_sample = _production_paths_by_sample(data_dir)
    usecols = list(IDENTITY_COLUMNS) + [
        "protein_accession", AFFINITY_COLUMN,
    ]
    for sample_id in cohort.sample_id.drop_duplicates():
        path = paths_by_sample.get(str(sample_id))
        if path is None:
            raise ValueError(
                "No cached production-affinity file for sample %s" % sample_id)
        frame = pandas.read_csv(path, usecols=usecols)
        frame["n_flank"] = frame.n_flank.fillna("")
        frame["c_flank"] = frame.c_flank.fillna("")
        frames.append(frame)
        paths.append({"path": path, "sha256": sha256_file(path)})
    affinity = pandas.concat(frames, ignore_index=True)
    if (
            len(affinity) != len(cohort) or
            not numpy.array_equal(_cohort_hash(affinity), _cohort_hash(cohort))):
        raise ValueError(
            "Cached affinity rows do not exactly match the ordered prediction "
            "cohort")
    values = pandas.to_numeric(affinity[AFFINITY_COLUMN], errors="coerce")
    if not numpy.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Cached affinity must be finite and positive")
    result = cohort.copy()
    result["protein_accession"] = affinity.protein_accession.to_numpy()
    result[AFFINITY_COLUMN] = values.to_numpy(dtype="float64")
    result["log10_affinity"] = numpy.log10(result[AFFINITY_COLUMN])
    result["peptide_len"] = result.peptide.str.len().astype("int8")
    result["hit"] = pandas.to_numeric(
        result.hit, errors="raise").astype("int8")
    return result, paths


def _sorted_pool(frame, indices):
    indices = numpy.asarray(indices, dtype="int64")
    order = numpy.argsort(
        frame.log10_affinity.to_numpy()[indices], kind="stable")
    indices = indices[order]
    return (
        indices,
        frame.log10_affinity.to_numpy()[indices],
    )


def _nearest(pool, target, count, excluded=(), max_distance=None):
    indices, values = pool
    if count <= 0 or not len(indices):
        return []
    position = int(numpy.searchsorted(values, target))
    radius = min(len(indices), max(count * 3, count + len(excluded)))
    start = max(0, position - radius)
    end = min(len(indices), position + radius)
    candidates = indices[start:end]
    candidate_values = values[start:end]
    order = numpy.argsort(numpy.abs(candidate_values - target), kind="stable")
    excluded = set(excluded)
    selected = []
    for offset in order:
        index = int(candidates[offset])
        if (
                max_distance is not None and
                abs(float(candidate_values[offset]) - target) > max_distance):
            continue
        if index not in excluded:
            selected.append(index)
            if len(selected) == count:
                break
    if len(selected) < count and len(candidates) < len(indices):
        # A very dense set of excluded near-neighbors can exhaust the local
        # slice. The full stable ordering is the deterministic fallback.
        order = numpy.argsort(numpy.abs(values - target), kind="stable")
        for offset in order:
            index = int(indices[offset])
            if (
                    max_distance is not None and
                    abs(float(values[offset]) - target) > max_distance):
                continue
            if index not in excluded and index not in selected:
                selected.append(index)
                if len(selected) == count:
                    break
    return selected


def make_affinity_controlled_risk_sets(
        frame, decoys_per_hit=10, same_protein_caliper=0.25):
    """Return hit-centered risk sets with nearest-affinity decoys."""
    if decoys_per_hit < 1:
        raise ValueError("decoys_per_hit must be positive")
    if same_protein_caliper is not None and same_protein_caliper < 0:
        raise ValueError("same_protein_caliper must be nonnegative")
    negatives = frame.index[frame.hit == 0].to_numpy(dtype="int64")
    global_pools = {}
    protein_pools = {}
    for key, group in frame.loc[negatives].groupby(
            ["sample_id", "peptide_len"], sort=False):
        global_pools[key] = _sorted_pool(frame, group.index)
    for key, group in frame.loc[negatives].groupby(
            ["sample_id", "peptide_len", "protein_accession"],
            sort=False, dropna=False):
        protein_pools[key] = _sorted_pool(frame, group.index)

    row_indices = []
    risk_ids = []
    match_ranks = []
    same_protein = []
    distances = []
    fallback_count = 0
    incomplete = 0
    for risk_id, (hit_index, hit) in enumerate(
            frame.loc[frame.hit == 1].iterrows()):
        target = float(hit.log10_affinity)
        protein_key = (
            hit.sample_id, hit.peptide_len, hit.protein_accession)
        selected = _nearest(
            protein_pools.get(protein_key, (numpy.array([], dtype="int64"),
                                            numpy.array([], dtype="float64"))),
            target,
            decoys_per_hit,
            max_distance=same_protein_caliper,
        )
        selected_same_protein = [True] * len(selected)
        if len(selected) < decoys_per_hit:
            needed = decoys_per_hit - len(selected)
            fallback = _nearest(
                global_pools[(hit.sample_id, hit.peptide_len)],
                target,
                needed,
                excluded=selected,
            )
            fallback_count += len(fallback)
            selected.extend(fallback)
            selected_same_protein.extend([False] * len(fallback))
        if len(selected) != decoys_per_hit:
            incomplete += 1
            continue

        row_indices.append(int(hit_index))
        risk_ids.append(risk_id)
        match_ranks.append(0)
        same_protein.append(True)
        distances.append(0.0)
        for rank, (index, is_same) in enumerate(
                zip(selected, selected_same_protein), 1):
            row_indices.append(index)
            risk_ids.append(risk_id)
            match_ranks.append(rank)
            same_protein.append(is_same)
            distances.append(abs(
                float(frame.at[index, "log10_affinity"]) - target))

    if incomplete:
        raise ValueError(
            "%d hits could not be assigned %d affinity-matched decoys" % (
                incomplete, decoys_per_hit))
    result = frame.loc[row_indices].copy().reset_index().rename(
        columns={"index": "source_row"})
    result["risk_set_id"] = numpy.asarray(risk_ids, dtype="int64")
    result["match_rank"] = numpy.asarray(match_ranks, dtype="int16")
    result["same_protein_match"] = numpy.asarray(same_protein, dtype=bool)
    result["log10_affinity_distance"] = numpy.asarray(
        distances, dtype="float64")
    diagnostics = {
        "risk_sets": int(result.risk_set_id.nunique()),
        "rows": int(len(result)),
        "decoys_per_hit": int(decoys_per_hit),
        "same_protein_caliper": same_protein_caliper,
        "fallback_decoys": int(fallback_count),
        "same_protein_decoy_fraction": float(
            result.loc[result.hit == 0, "same_protein_match"].mean()),
        "median_log10_affinity_distance": float(
            result.loc[result.hit == 0, "log10_affinity_distance"].median()),
        "p95_log10_affinity_distance": float(
            result.loc[result.hit == 0, "log10_affinity_distance"].quantile(.95)),
    }
    return result, diagnostics


def _concordance(frame, score_column):
    values = []
    for _, group in frame.groupby("risk_set_id", sort=False):
        hit_score = float(group.loc[group.hit == 1, score_column].iloc[0])
        decoy_scores = group.loc[group.hit == 0, score_column].to_numpy()
        values.append(float(numpy.mean(
            (hit_score > decoy_scores) + 0.5 * (hit_score == decoy_scores))))
    return float(numpy.mean(values))


def _metric_record(frame, score, scope, group):
    metrics = _metrics(frame.hit.to_numpy(), frame[score].to_numpy())
    return {
        "scope": scope,
        "group": str(group),
        "score": score,
        **metrics,
        "concordance": _concordance(frame, score),
    }


def score_risk_sets(frame, score_columns):
    """Return overall, sample, and peptide-length metrics."""
    records = []
    for score in score_columns:
        records.append(_metric_record(frame, score, "overall", "all"))
        for sample_id, group in frame.groupby("sample_id", sort=False):
            records.append(_metric_record(
                group, score, "sample", sample_id))
        for length, group in frame.groupby("peptide_len", sort=True):
            records.append(_metric_record(group, score, "length", int(length)))
    return pandas.DataFrame(records)


def summarize_metrics(metrics, baseline):
    """Return score summaries and paired deltas from the baseline."""
    overall = metrics.loc[metrics.scope == "overall"].set_index("score")
    samples = metrics.loc[metrics.scope == "sample"]
    macro = samples.groupby("score")[
        ["roc_auc", "pr_auc", "ppv_at_n", "concordance"]
    ].mean().add_prefix("macro_")
    summary = overall[[
        "n", "n_pos", "roc_auc", "pr_auc", "ppv_at_n", "concordance",
    ]].join(macro).reset_index()
    if baseline not in set(summary.score):
        raise ValueError("Unknown baseline score: %s" % baseline)
    baseline_row = summary.set_index("score").loc[baseline]
    comparisons = summary.copy()
    for column in (
            "roc_auc", "pr_auc", "ppv_at_n", "concordance",
            "macro_roc_auc", "macro_pr_auc", "macro_ppv_at_n",
            "macro_concordance"):
        comparisons[column + "_diff"] = (
            comparisons[column] - baseline_row[column])
    comparisons.insert(1, "baseline", baseline)
    return summary, comparisons


def run(args):
    """Run affinity-controlled processing evaluation."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cohort, new_score_sources = _load_scores(args.score)
    if args.existing:
        extended = _extend_existing(
            args.existing, cohort, new_score_sources)
        attached = extended["heldout"]
        matched = extended["matched"]
        score_sources = extended["score_sources"]
        affinity_sources = extended["affinity_sources"]
        diagnostics = extended["diagnostics"]
    else:
        score_sources = new_score_sources
    if args.baseline not in set(source["name"] for source in score_sources):
        raise ValueError("--baseline is not one of the named --score values")
    if not args.existing:
        attached, affinity_sources = _attach_affinity(cohort, args.data_dir)
        matched, diagnostics = make_affinity_controlled_risk_sets(
            attached,
            decoys_per_hit=args.decoys_per_hit,
            same_protein_caliper=args.same_protein_caliper,
        )
    score_columns = [source["name"] for source in score_sources]
    metrics = score_risk_sets(matched, score_columns)
    summary, comparisons = summarize_metrics(metrics, args.baseline)

    # Preserve the complete, ordered held-out cohort as the canonical join
    # surface for external predictors. The matched table below intentionally
    # repeats decoys across hit-centered risk sets and therefore cannot serve
    # that purpose on its own.
    attached.to_csv(out / "heldout_predictions.csv.bz2", index=False)
    matched.to_csv(out / "matched_predictions.csv.bz2", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    comparisons.to_csv(out / "comparisons.csv", index=False)
    configuration = {
        "design": "processing-affinity-controlled-risk-sets-v1",
        "score_sources": score_sources,
        "affinity_sources": affinity_sources,
        "baseline": args.baseline,
        "data_dir": os.path.abspath(args.data_dir),
        "diagnostics": diagnostics,
    }
    if args.existing:
        configuration.update({
            "extended_from": extended["existing_experiment"],
            "extended_from_sha256": extended[
                "existing_experiment_sha256"],
        })
    (out / "experiment.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print("Wrote %s" % out)
    return 0


def run_argv(argv=None, prog="mhcflurry eval processing-affinity-control"):
    """Parse arguments and run the command."""
    return run(make_parser(prog).parse_args(argv))

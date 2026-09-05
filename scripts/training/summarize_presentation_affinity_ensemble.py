#!/usr/bin/env python3
"""Fit and evaluate a fixed public/new affinity score ensemble."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from mhcflurry import Class1PresentationPredictor
from mhcflurry.common import configure_random_seed
from mhcflurry.regression_target import from_ic50, to_ic50


METRICS = ("AUROC", "AUPRC", "PPV@N")
CONDITIONS = (
    "public_2_2",
    "public_affinity__new_processing",
    "new_affinity__new_processing",
    "hybrid_affinity__new_processing",
    "dual_affinity_features__new_processing",
)


def sha256_file(path):
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_values(targets, scores):
    """Calculate release metrics with deterministic PPV@N ties."""
    targets = numpy.asarray(targets)
    scores = numpy.asarray(scores)
    if not numpy.isfinite(scores).all():
        raise ValueError("Non-finite score encountered")
    num_positives = int(targets.sum())
    if not num_positives or num_positives == len(targets):
        return {name: numpy.nan for name in METRICS}
    order = numpy.argsort(-scores, kind="stable")[:num_positives]
    return {
        "AUROC": float(roc_auc_score(targets, scores)),
        "AUPRC": float(average_precision_score(targets, scores)),
        "PPV@N": float(targets[order].sum()) / num_positives,
    }


def logistic_score(intercept, affinity_weight, processing_weight,
                   affinity_score, processing_score):
    """Evaluate the published two-feature logistic presentation combiner."""
    logits = (
        float(intercept)
        + float(affinity_weight) * numpy.asarray(affinity_score)
        + float(processing_weight) * numpy.asarray(processing_score)
    )
    return 1.0 / (1.0 + numpy.exp(-numpy.clip(logits, -40.0, 40.0)))


def summarize_scores(frame, mode):
    """Return overall, per-sample, and per-length metric tables."""
    per_sample_rows = []
    for (sample_id, hla), group in frame.groupby(
            ["sample_id", "hla"], dropna=False, sort=False):
        for condition in CONDITIONS:
            values = metric_values(group.hit.values, group[condition].values)
            per_sample_rows.append({
                "flank_mode": mode,
                "sample_id": sample_id,
                "hla": hla,
                "condition": condition,
                "n": int(len(group)),
                "n_pos": int(group.hit.sum()),
                **values,
            })
    per_sample = pandas.DataFrame(per_sample_rows)

    overall_rows = []
    for condition in CONDITIONS:
        micro = metric_values(frame.hit.values, frame[condition].values)
        condition_samples = per_sample.loc[
            per_sample.condition == condition]
        for metric in METRICS:
            overall_rows.extend([
                {
                    "flank_mode": mode,
                    "condition": condition,
                    "average": "Micro",
                    "metric": metric,
                    "value": micro[metric],
                },
                {
                    "flank_mode": mode,
                    "condition": condition,
                    "average": "Macro",
                    "metric": metric,
                    "value": float(condition_samples[metric].mean()),
                },
            ])

    per_length_rows = []
    for length, group in frame.groupby("peptide_len", sort=True):
        for condition in CONDITIONS:
            micro = metric_values(group.hit.values, group[condition].values)
            sample_metrics = []
            for _, sample_group in group.groupby(
                    ["sample_id", "hla"], dropna=False, sort=False):
                sample_metrics.append(metric_values(
                    sample_group.hit.values,
                    sample_group[condition].values,
                ))
            for metric in METRICS:
                per_length_rows.extend([
                    {
                        "flank_mode": mode,
                        "length": int(length),
                        "condition": condition,
                        "average": "Micro",
                        "metric": metric,
                        "value": micro[metric],
                    },
                    {
                        "flank_mode": mode,
                        "length": int(length),
                        "condition": condition,
                        "average": "Macro",
                        "metric": metric,
                        "value": float(numpy.nanmean([
                            values[metric] for values in sample_metrics
                        ])),
                    },
                ])
    return (
        pandas.DataFrame(overall_rows),
        per_sample,
        pandas.DataFrame(per_length_rows),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog=os.environ.get("MHCFLURRY_CLI_PROG"), description=__doc__)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--public-affinity-cache", required=True)
    parser.add_argument("--new-affinity-cache", required=True)
    parser.add_argument("--processing-with-cache", required=True)
    parser.add_argument("--processing-without-cache", required=True)
    parser.add_argument("--with-flanks-predictions", required=True)
    parser.add_argument("--without-flanks-predictions", required=True)
    parser.add_argument("--public-new-processing-weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--public-model-count", type=int, default=10)
    parser.add_argument("--new-model-count", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_random_seed(
        args.random_seed, name="presentation-affinity-ensemble-followup")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    data = pandas.read_csv(
        args.training_data, dtype={"sample_id": str}, low_memory=False)
    data = data.loc[data.peptide.str.len().between(8, 15)].copy()
    data["experiment_id"] = data.hla
    data = data.sort_values(
        "experiment_id", kind="stable").reset_index(drop=True)

    public_affinity = numpy.load(args.public_affinity_cache)
    new_affinity = numpy.load(args.new_affinity_cache)
    processing_with = numpy.load(args.processing_with_cache)
    processing_without = numpy.load(args.processing_without_cache)
    arrays = {
        "public affinity": public_affinity,
        "new affinity": new_affinity,
        "processing with flanks": processing_with,
        "processing without flanks": processing_without,
    }
    for name, values in arrays.items():
        if len(values) != len(data):
            raise ValueError(
                "%s cache has %d rows; expected %d" % (
                    name, len(values), len(data)))
        if not numpy.isfinite(values).all():
            raise ValueError("%s cache contains non-finite values" % name)

    total_models = args.public_model_count + args.new_model_count
    public_weight = args.public_model_count / float(total_models)
    new_weight = args.new_model_count / float(total_models)
    hybrid_affinity_score = (
        public_weight * from_ic50(public_affinity)
        + new_weight * from_ic50(new_affinity)
    )
    hybrid_affinity = to_ic50(hybrid_affinity_score)

    predictor = Class1PresentationPredictor()
    predictor.fit_from_scores(
        targets=data.hit.values,
        affinities=hybrid_affinity,
        processing_scores_by_model={
            "with_flanks": processing_with,
            "without_flanks": processing_without,
        },
        verbose=1,
    )
    predictor.weights_dataframe.to_csv(out / "weights.csv")

    dual_models = {}
    dual_weight_rows = []
    public_affinity_score = from_ic50(public_affinity)
    new_affinity_score = from_ic50(new_affinity)
    for mode, processing_score in (
            ("with_flanks", processing_with),
            ("without_flanks", processing_without)):
        model = LogisticRegression(solver="lbfgs")
        model.fit(
            numpy.column_stack([
                public_affinity_score,
                new_affinity_score,
                processing_score,
            ]),
            data.hit.astype(float).values,
        )
        dual_models[mode] = model
        dual_weight_rows.append({
            "flank_mode": mode,
            "intercept": float(model.intercept_[0]),
            "public_affinity_score": float(model.coef_[0, 0]),
            "new_affinity_score": float(model.coef_[0, 1]),
            "processing_score": float(model.coef_[0, 2]),
        })
    pandas.DataFrame(dual_weight_rows).to_csv(
        out / "dual_feature_weights.csv", index=False)

    public_new_weights = pandas.read_csv(
        args.public_new_processing_weights, index_col=0)
    all_overall = []
    correlation_rows = []
    for mode, path in (
            ("with_flanks", args.with_flanks_predictions),
            ("without_flanks", args.without_flanks_predictions)):
        frame = pandas.read_csv(path, low_memory=False)
        required = {
            "hit", "sample_id", "hla", "peptide_len", "a_affinity",
            "b_affinity", "a_processing_score", "a_presentation_score",
            "b_presentation_score",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("%s is missing columns: %s" % (
                path, ", ".join(missing)))

        new_score = from_ic50(frame.a_affinity.values)
        public_score = from_ic50(frame.b_affinity.values)
        hybrid_score = public_weight * public_score + new_weight * new_score
        processing_score = frame.a_processing_score.values

        hybrid_model = predictor.get_model(mode)
        hybrid_presentation = hybrid_model.predict_proba(
            numpy.column_stack([hybrid_score, processing_score]))[:, 1]
        row = public_new_weights.loc[mode]
        public_new_presentation = logistic_score(
            row.intercept,
            row.affinity_score,
            row.processing_score,
            public_score,
            processing_score,
        )

        scored = frame[[
            "protein_accession", "peptide", "sample_id", "n_flank",
            "c_flank", "hit", "hla", "source_file", "peptide_len",
        ]].copy()
        scored["public_affinity_score"] = public_score
        scored["new_affinity_score"] = new_score
        scored["hybrid_affinity_score"] = hybrid_score
        scored["new_processing_score"] = processing_score
        scored["public_2_2"] = frame.b_presentation_score.values
        scored["public_affinity__new_processing"] = public_new_presentation
        scored["new_affinity__new_processing"] = (
            frame.a_presentation_score.values)
        scored["hybrid_affinity__new_processing"] = hybrid_presentation
        scored["dual_affinity_features__new_processing"] = (
            dual_models[mode].predict_proba(numpy.column_stack([
                public_score,
                new_score,
                processing_score,
            ]))[:, 1]
        )

        output_path = out / ("predictions_%s.csv.bz2" % mode)
        scored.to_csv(output_path, index=False, compression="bz2")
        overall, per_sample, per_length = summarize_scores(scored, mode)
        overall.to_csv(out / ("summary_%s.csv" % mode), index=False)
        per_sample.to_csv(out / ("per_sample_%s.csv" % mode), index=False)
        per_length.to_csv(out / ("per_length_%s.csv" % mode), index=False)
        all_overall.append(overall)

        for subset, mask in (
                ("all", numpy.ones(len(scored), dtype=bool)),
                ("hits", scored.hit.values == 1),
                ("decoys", scored.hit.values == 0)):
            values = scored.loc[mask, [
                "public_affinity_score", "new_affinity_score",
                "new_processing_score",
            ]]
            matrix = values.corr(method="pearson")
            for left in matrix.columns:
                for right in matrix.columns:
                    if left < right:
                        correlation_rows.append({
                            "flank_mode": mode,
                            "subset": subset,
                            "left": left,
                            "right": right,
                            "pearson": float(matrix.loc[left, right]),
                            "n": int(len(values)),
                        })

    overall = pandas.concat(all_overall, ignore_index=True)
    reference_rows = []
    for reference in (
            "public_2_2", "public_affinity__new_processing"):
        reference_values = overall.loc[
            overall.condition == reference,
            ["flank_mode", "average", "metric", "value"],
        ].rename(columns={"value": "reference_value"})
        comparison = overall.merge(
            reference_values,
            on=["flank_mode", "average", "metric"],
            how="left",
        )
        comparison["reference"] = reference
        comparison["diff"] = comparison.value - comparison.reference_value
        comparison["pct_change"] = (
            100.0 * comparison["diff"] / comparison.reference_value)
        reference_rows.append(comparison)
    release_summary = pandas.concat(reference_rows, ignore_index=True)
    release_summary.to_csv(out / "release_summary.csv", index=False)
    pandas.DataFrame(correlation_rows).to_csv(
        out / "score_correlations.csv", index=False)

    input_paths = {
        key: Path(value).resolve()
        for key, value in {
            "training_data": args.training_data,
            "public_affinity_cache": args.public_affinity_cache,
            "new_affinity_cache": args.new_affinity_cache,
            "processing_with_cache": args.processing_with_cache,
            "processing_without_cache": args.processing_without_cache,
            "with_flanks_predictions": args.with_flanks_predictions,
            "without_flanks_predictions": args.without_flanks_predictions,
            "public_new_processing_weights": (
                args.public_new_processing_weights),
        }.items()
    }
    provenance = {
        "format": 1,
        "random_seed": args.random_seed,
        "blend": {
            "level": "presentation affinity_score after best-allele selection",
            "public_model_count": args.public_model_count,
            "new_model_count": args.new_model_count,
            "public_weight": public_weight,
            "new_weight": new_weight,
            "ratio_was_tuned": False,
            "network_merge_avoided": (
                "source predictors use different allele pseudosequence tables"
            ),
        },
        "rows": {
            "presentation_training": int(len(data)),
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in input_paths.items()
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "provenance.json"
        },
    }
    with open(out / "provenance.json", "w") as fd:
        json.dump(provenance, fd, indent=2, sort_keys=True)
        fd.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

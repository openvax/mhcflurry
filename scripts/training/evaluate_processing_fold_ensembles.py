"""Score specified ensembles within shared held-out folds from cached predictions.

This is development evaluation of fold-specific ensembles, not final model
selection or an estimate for one ensemble combining members across folds.
"""

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy
import pandas

from mhcflurry.cli.compare_models import _metrics
from mhcflurry.experiment_archive import sha256_file


KEYS = ["fold_num", "validation_row_index"]
SCORE_COLUMNS = ["condition", "model_name", "checkpoint_policy", "processing_score"]
SCOPE = "per-member held-out fold, retained best and terminal states"
RANKING_SCOPE = "per-member held-out fold, retained best-loss, best-AP and terminal states"


def load_predictions(path, policies):
    """Verify a checkpoint cache and its exact row identity across states."""
    marker_path = path.with_name(path.name + ".json")
    marker = json.loads(marker_path.read_text())
    identity = marker["identity"]
    if marker["prediction_sha256"] != sha256_file(path) or identity.get("scope") not in (SCOPE, RANKING_SCOPE):
        raise ValueError("Changed checkpoint cache or incompatible prediction scope")
    frame = pandas.read_csv(path, dtype={"sample_id": str}, float_precision="round_trip")
    required = KEYS + SCORE_COLUMNS + ["sample_id", "hit", "peptide", "n_flank", "c_flank", "risk_set_id"]
    if set(required) - set(frame) or frame.empty:
        raise ValueError("Missing identified checkpoint predictions")
    if set(frame.condition) != {identity["condition"]}:
        raise ValueError("Checkpoint condition differs from its marker")
    if (frame[KEYS + ["sample_id", "model_name", "checkpoint_policy"]].isna().any().any()
            or frame.duplicated(KEYS + ["checkpoint_policy"]).any()
            or not numpy.isfinite(frame.processing_score).all()
            or not frame.hit.isin([0, 1]).all()):
        raise ValueError("Require unique finite held-out predictions and binary labels")
    if not frame.groupby("fold_num").model_name.nunique().eq(1).all():
        raise ValueError("Require exactly one member per condition and fold")
    mapping = frame[["fold_num", "model_name"]].drop_duplicates()
    if mapping.model_name.duplicated().any():
        raise ValueError("A model cannot represent multiple held-out folds")
    values, reference = {}, None
    for policy in policies:
        selected = frame.loc[frame.checkpoint_policy == policy].set_index(KEYS).sort_index()
        if selected.empty:
            raise ValueError("Missing requested checkpoint policy: " + policy)
        metadata = selected.drop(columns=SCORE_COLUMNS)
        if reference is None:
            reference = metadata
        elif not reference.equals(metadata):
            raise ValueError("Checkpoint states do not share identical held-out rows")
        values[policy] = selected.processing_score.to_numpy(dtype="float32")
    fingerprint = {"path": str(path.resolve()), "sha256": marker["prediction_sha256"],
                   "marker_sha256": sha256_file(marker_path), "identity": identity}
    return reference, values, mapping, fingerprint


def score_ensembles(definitions, base, policies):
    """Average only row-aligned, same-fold members and preserve every score."""
    if not definitions or not policies or len(set(policies)) != len(policies):
        raise ValueError("Require ensembles and distinct checkpoint policies")
    reference, training_hash = None, None
    scores, members, inputs = {}, [], []
    for name, paths in definitions.items():
        if not name or not paths:
            raise ValueError("Require nonempty ensemble names and members")
        sums = {}
        seen_conditions, seen_models = set(), set()
        for index, filename in enumerate(paths):
            path = (base / filename).resolve()
            print("Verifying %s member %d/%d: %s" % (name, index + 1, len(paths), path.name), flush=True)
            metadata, values, mapping, fingerprint = load_predictions(path, policies)
            current_hash = fingerprint["identity"].get("training_data_sha256")
            if not current_hash:
                raise ValueError("Missing training-table identity")
            if reference is None:
                reference, training_hash = metadata, current_hash
            elif training_hash != current_hash or not reference.equals(metadata):
                raise ValueError("Members do not share frozen data and identical held-out fold rows")
            condition = fingerprint["identity"]["condition"]
            if condition in seen_conditions or set(mapping.model_name) & seen_models:
                raise ValueError("Duplicate ensemble member")
            seen_conditions.add(condition)
            seen_models.update(mapping.model_name)
            for row in mapping.itertuples(index=False):
                members.append({"ensemble": name, "member_index": index, "condition": condition,
                                "fold_num": row.fold_num, "model_name": row.model_name})
            for policy in policies:
                if policy not in sums:
                    sums[policy] = numpy.zeros(len(metadata), dtype="float32")
                sums[policy] += values[policy]
            inputs.append(fingerprint)
        for policy, total in sums.items():
            column = name + "__" + policy
            if column in reference.columns or column in scores:
                raise ValueError("Generated score name collision")
            scores[column] = total / numpy.float32(len(paths))
    result = reference.reset_index()
    for column, values in scores.items():
        result[column] = values
    rows, micro = [], []
    for name in scores:
        for (fold, sample), group in result.groupby(["fold_num", "sample_id"]):
            rows.append({"condition": name, "fold_num": fold, "sample_id": sample,
                         **_metrics(group.hit, group[name])})
        for fold, group in result.groupby("fold_num"):
            micro.append({"condition": name, "fold_num": fold, **_metrics(group.hit, group[name])})
    per_sample = pandas.DataFrame(rows)
    metrics = ["roc_auc", "pr_auc", "ppv_at_n", "n", "n_pos"]
    sample_means = per_sample.groupby(["condition", "sample_id"], as_index=False)[metrics].mean()
    summary = sample_means.groupby("condition")[metrics[:3]].mean().add_prefix("macro_").reset_index()
    return result, pandas.DataFrame(members), per_sample, sample_means, pandas.DataFrame(micro), summary, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.environ.get("MHCFLURRY_CLI_PROG"), description=__doc__)
    parser.add_argument("--ensembles", required=True, type=Path,
                        help="JSON object mapping ensemble names to ordered checkpoint-cache paths, relative to this JSON.")
    parser.add_argument("--checkpoint-policy", nargs="+", choices=["best", "best_ap", "terminal"], default=["best", "terminal"])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    definitions = json.loads(args.ensembles.read_text())
    result, members, metrics, sample_means, micro, summary, inputs = score_ensembles(
        definitions, args.ensembles.parent, args.checkpoint_policy)
    args.out.mkdir(parents=True)
    for name, frame in [("members", members), ("per_fold_sample_metrics", metrics),
                        ("sample_means", sample_means), ("micro_by_fold", micro), ("summary", summary)]:
        frame.to_csv(args.out / (name + ".csv"), index=False)
    print(summary.to_string(index=False), flush=True)
    print("Writing %d joinable prediction rows with gzip level 1" % len(result), flush=True)
    temporary = args.out / "fold_ensemble_predictions.tmp.gz"
    result.to_csv(temporary, index=False, compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
    temporary.replace(args.out / "fold_ensemble_predictions.csv.gz")
    shutil.copyfile(__file__, args.out / "analysis_source.py")
    provenance = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "definitions": definitions, "definitions_sha256": sha256_file(args.ensembles), "inputs": inputs,
        "arithmetic": "Sequential float32 summation in specified member order, then float32 division.",
        "scope": "Each evaluated row uses only that fold's models; average fold metrics within sample, then macro over samples.",
        "limitations": "Exploratory development diagnostic of fold-specific ensembles, not one fixed ensemble across folds; no selection or release acceptance.",
        "training": False, "inference": False, "release_accepted": False,
        "outputs": [{"path": path.name, "sha256": sha256_file(path)} for path in sorted(args.out.iterdir())],
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    main()

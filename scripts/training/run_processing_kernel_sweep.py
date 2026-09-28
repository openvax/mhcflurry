#!/usr/bin/env python3
"""Train paired width or training-recipe sweeps on frozen matched processing data."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy
import pandas
import yaml

from generate_processing_cleavage_boundaries import build_kernel_conditions
from run_exact_public_processing import Driver, cached_scores, utc_now, write_json
from mhcflurry.cli.compare_models import _metrics
from mhcflurry.experiment_archive import sha256_file
from mhcflurry.processing_matching import validate_matched_training_data
from mhcflurry.release_holdout import load_excluded_samples


def import_completed_conditions(previous, out, config):
    """Copy verified completed conditions and frozen folds into a new run.

    Failed/partial conditions are left untouched in the source run and are not
    imported. Retention-only changes may be added for new fits; missing old
    terminal states are recorded, never synthesized from best weights.
    """
    from mhcflurry import Class1ProcessingPredictor

    previous, out = Path(previous).resolve(), Path(out).resolve()
    if previous == out or previous in out.parents or out in previous.parents:
        raise ValueError("Recovery source and destination must be separate directories")
    prior_manifest = previous / "experiment.json"
    prior = json.loads(prior_manifest.read_text())
    for key in ("random_seed", "train_data_sha256", "holdout_sha256"):
        if prior[key] != config[key]:
            raise ValueError("Recovery input mismatch: " + key)
    expected_records = {r["condition"]: r for r in config["records"]}
    for record in prior["records"]:
        expected = expected_records.get(record["condition"])
        if expected is None:
            raise ValueError("Recovery contains an unknown condition")
        for actual_hp, expected_hp in zip(record["hyperparameters"], expected["hyperparameters"]):
            left, right = dict(actual_hp), dict(expected_hp)
            left.pop("save_all_checkpoints", None)
            right.pop("save_all_checkpoints", None)
            if left != right:
                raise ValueError("Recovery hyperparameters differ: " + record["condition"])
        if len(record["hyperparameters"]) != len(expected["hyperparameters"]):
            raise ValueError("Recovery architecture count differs")
    source_folds = previous / "inputs/training_with_folds.csv.bz2"
    destination_folds = out / "inputs/training_with_folds.csv.bz2"
    if not source_folds.exists():
        raise ValueError("Recovery source has no frozen folds")
    if destination_folds.exists():
        if sha256_file(destination_folds) != sha256_file(source_folds):
            raise ValueError("Recovery frozen folds changed")
    else:
        shutil.copy2(source_folds, destination_folds)
    imported = []
    files = {}
    for record in prior["records"]:
        name = record["condition"]
        source = previous / name
        selected = source / "processing/models.selected.short_flanks"
        predictions = selected / "model_selection_predictions.csv.bz2"
        if not predictions.exists() or not (source / "validation_metrics.csv").exists():
            continue
        predictor = Class1ProcessingPredictor.load(str(selected))
        folds = [m.fit_info[-1]["training_info"]["fold_num"] for m in predictor.models]
        if sorted(folds) != list(range(record["fold_count"])):
            raise ValueError("Recovery is not one completed model per fold: " + name)
        source_data = source / "processing/models.unselected.short_flanks/train_data.csv.bz2"
        if sha256_file(source_data) != sha256_file(source_folds):
            raise ValueError("Recovery condition has different frozen folds: " + name)
        validation_metrics(predictions, name)
        destination = out / name
        inventory = {str(p.relative_to(previous)): sha256_file(p)
                     for p in source.rglob("*") if p.is_file()}
        if destination.exists():
            for relative, digest in inventory.items():
                copied = out / relative
                if not copied.exists() or sha256_file(copied) != digest:
                    raise ValueError("Recovered artifact changed: " + relative)
        else:
            temporary = out / (name + ".importing")
            if temporary.exists():
                raise ValueError("Incomplete recovery copy requires inspection: " + str(temporary))
            shutil.copytree(source, temporary)
            temporary.rename(destination)
        files.update(inventory)
        imported.append(name)
    write_json(out / "recovery.json", {
        "source": str(previous), "source_manifest_sha256": sha256_file(prior_manifest),
        "source_commit": prior["source_commit"], "completed_conditions": imported,
        "frozen_folds_sha256": sha256_file(source_folds), "copied_files_sha256": files,
        "checkpoint_note": "Imported legacy fits may lack terminal states; only new fits retain both."})
    return set(imported)


def validation_metrics(path, condition):
    """Sample-balanced held-out metrics; repeated sample folds remain explicit."""
    frame = pandas.read_csv(path, dtype={"sample_id": str})
    if frame.empty or not numpy.isfinite(frame.processing_score).all():
        raise ValueError("Incomplete validation predictions: " + str(path))
    records = []
    for (fold, sample), group in frame.groupby(["fold_num", "sample_id"]):
        records.append({"condition": condition, "fold_num": fold, "sample_id": sample,
                        **_metrics(group.hit, group.processing_score)})
    return pandas.DataFrame(records)


def save_checkpoint_validation_predictions(model_dir, destination, condition):
    """Save context-joinable, genuinely held-out predictions for every state."""
    from mhcflurry import Class1ProcessingPredictor
    from mhcflurry.training_folds import read_processing_training_data
    from compose_processing_ensemble import fingerprint_directory

    model_dir, destination = Path(model_dir), Path(destination)
    marker = destination.with_name(destination.name + ".json")
    identity = {"condition": condition, "predictor": fingerprint_directory(model_dir),
                "training_data_sha256": sha256_file(model_dir / "train_data.csv.bz2"),
                "scope": "per-member held-out fold, retained best and terminal states"}
    configs = pandas.read_csv(model_dir / "manifest.csv", usecols=["config_json"]).config_json.map(json.loads)
    if any(config.get("hyperparameters", {}).get("monitor_validation_ranking") or
           config.get("hyperparameters", {}).get("checkpoint_metric") == "val_macro_ap" for config in configs):
        identity["scope"] = "per-member held-out fold, retained best-loss, best-AP and terminal states"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved["identity"] != identity or not destination.exists() or
                saved["prediction_sha256"] != sha256_file(destination)):
            raise ValueError("Changed checkpoint prediction cache: " + str(destination))
        return
    predictor = Class1ProcessingPredictor.load(str(model_dir))
    data = read_processing_training_data(model_dir / "train_data.csv.bz2")
    result = []
    for row in predictor.manifest_df.itertuples():
        model = row.model
        info = model.fit_info[-1]
        fold = info["training_info"]["fold_num"]
        membership = data["fold_%d" % fold].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False})
        if membership.isna().any():
            raise ValueError("Invalid validation fold membership")
        held_out = data.loc[~membership].copy()
        if held_out.empty:
            raise ValueError("Empty validation fold")
        held_out = held_out.drop(columns=[c for c in held_out if c.startswith("fold_")])
        held_out.insert(0, "validation_row_index", held_out.index)
        original = model.get_weights()
        policies = dict(getattr(model, "checkpoint_weights", {}))
        primary = info.get("restored_checkpoint_policy", "best" if info.get("restored_best_weights") else "terminal")
        policies.setdefault(primary, original)
        try:
            for policy, weights in sorted(policies.items()):
                model.network().set_weights_list(weights, auto_convert_keras=False)
                frame = held_out.copy()
                frame["processing_score"] = model.predict(
                    frame.peptide.tolist(), frame.n_flank.fillna("").tolist(),
                    frame.c_flank.fillna("").tolist())
                if not numpy.isfinite(frame.processing_score).all():
                    raise ValueError("Non-finite checkpoint validation predictions")
                frame["condition"] = condition
                frame["model_name"] = row.model_name
                frame["fold_num"] = fold
                frame["checkpoint_policy"] = policy
                result.append(frame)
        finally:
            model.network().set_weights_list(original, auto_convert_keras=False)
    compression = ({"method": "gzip", "compresslevel": 1, "mtime": 0}
                   if destination.suffix == ".gz" else "bz2")
    temporary = destination.with_name(destination.name + ".tmp")
    pandas.concat(result, ignore_index=True).to_csv(temporary, index=False, compression=compression)
    temporary.replace(destination)
    write_json(marker, {"identity": identity, "prediction_sha256": sha256_file(destination)})


def prepare_evaluation(args):
    """Freeze the release cohort once, using the standard strict matching policy."""
    from mhcflurry.cli.compare_models import _load_presentation_benchmark_for_component
    from mhcflurry.cli.processing_affinity_control import _attach_affinity
    from mhcflurry.processing_matching import make_affinity_controlled_risk_sets

    directory = args.out / "evaluation"
    directory.mkdir(exist_ok=True)
    path = directory / "matching_assignments.csv.bz2"
    marker = directory / "cohort.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if info["assignments_sha256"] != sha256_file(path):
            raise ValueError("Changed saved evaluation risk sets")
        return pandas.read_csv(path, dtype={"sample_id": str})
    options = argparse.Namespace(release_holdout_dir=str(args.release_holdout_dir), limit_files=None)
    cohort = _load_presentation_benchmark_for_component(
        str(args.public_root / "data_evaluation"), options, "processing")
    cohort, sources = _attach_affinity(cohort, str(args.public_root / "data_evaluation"))
    cohort.to_csv(directory / "candidate_pool.csv.bz2", index=False)
    matched, diagnostics = make_affinity_controlled_risk_sets(cohort)
    matched.to_csv(path, index=False)
    write_json(marker, {"diagnostics": diagnostics, "affinity_sources": sources,
                       "assignments_sha256": sha256_file(path),
                       "candidate_pool_sha256": sha256_file(directory / "candidate_pool.csv.bz2"),
                       "role": "exploratory release holdout; not an untouched final test"})
    return matched


def evaluate_predictor(out, name, model_dir, matched):
    """Cache each member's predictions as well as the unmixed ensemble mean."""
    from mhcflurry import Class1ProcessingPredictor
    from mhcflurry.cli.processing_affinity_control import score_risk_sets

    directory = out / "evaluation"
    unique = matched.drop_duplicates("source_row").sort_values("source_row")
    input_hash = sha256_file(directory / "matching_assignments.csv.bz2")
    predictor = Class1ProcessingPredictor.load(str(model_dir))
    scores = unique[["source_row", "sample_id", "peptide", "hit", "n_flank", "c_flank"]].copy()
    columns = []
    for index, model in enumerate(predictor.models):
        column = "member_%02d" % index
        scores[column] = cached_scores(directory, name + "__" + column, unique,
            input_hash, model_dir, lambda model=model: model.predict(
                unique.peptide.tolist(), unique.n_flank.fillna("").tolist(),
                unique.c_flank.fillna("").tolist(), batch_size=4096),
            ordering="unique matched evaluation source_row ascending")
        columns.append(column)
    # Match the public predictor's ordered float32 summation, including ties.
    ensemble = numpy.zeros(len(scores), dtype="float32")
    for column in columns:
        ensemble += scores[column].to_numpy(dtype="float32")
    scores[name] = ensemble / numpy.float32(len(columns))
    scores.to_csv(directory / (name + ".predictions.csv.bz2"), index=False)
    scored = matched.copy()
    scored[name] = scored.source_row.map(scores.set_index("source_row")[name])
    metrics = score_risk_sets(scored, [name])
    metrics.to_csv(directory / (name + ".metrics.csv"), index=False)
    return metrics


def render_summary(out, records):
    """Refresh plot-ready paired validation and release-evaluation summaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = pandas.DataFrame([dict(condition=name, **axes) for name, _, axes in records])
    tables = []
    for name, _, _ in records:
        path = out / name / "validation_metrics.csv"
        if path.exists():
            tables.append(pandas.read_csv(path, dtype={"sample_id": str}))
    if not tables:
        return
    per_sample = pandas.concat(tables, ignore_index=True)
    per_sample.to_csv(out / "validation_per_sample.csv", index=False)
    # Average folds within each sample before giving each sample equal weight.
    metrics = ["roc_auc", "pr_auc", "ppv_at_n"]
    summary = per_sample.groupby(["condition", "sample_id"])[metrics].mean().groupby("condition").mean()
    summary = summary.add_prefix("macro_").reset_index().merge(axes, on="condition")
    summary.to_csv(out / "validation_summary.csv", index=False)
    confirmation = any(axes.get("checkpoint_policy") == "best_ap" for _, _, axes in records)
    if confirmation:
        fig, panels = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for panel, metric, label in zip(panels, metrics, ["AUROC", "AUPRC", "PPV@N"]):
            for optimizer, group in summary.groupby("optimizer"):
                group = group.sort_values("kernel_width")
                panel.plot(group.kernel_width, group["macro_" + metric], "o-", label=optimizer)
            panel.set(xlabel="Kernel width", ylabel="Macro " + label, xticks=[11, 13, 15])
            panel.grid(alpha=0.2)
        panels[0].legend()
        fig.suptitle("Processing / 1:1 matched development / four folds / inner-best AP checkpoints")
        fig.savefig(out / "validation_ranking_confirmation.png", dpi=200)
        fig.savefig(out / "validation_ranking_confirmation.pdf")
        plt.close(fig)
        return
    if "initialization" in summary:
        families = sorted(summary.family.unique())
        fig, panels = plt.subplots(len(families), 3, figsize=(13, 4 * len(families)),
                                   squeeze=False, constrained_layout=True)
        order = ["none", "orthogonal", "lsuv_pre", "lsuv_post"]
        for row, family in enumerate(families):
            group = summary.loc[summary.family == family].copy()
            group["optimizer_batch"] = group.optimizer + " / " + group.batch_size.astype(str)
            for col, (metric, label) in enumerate(zip(metrics, ["AUROC", "AUPRC", "PPV@N"])):
                values = group.pivot(index="initialization", columns="optimizer_batch",
                                     values="macro_" + metric).reindex(order)
                panel = panels[row, col]
                plotted = panel.imshow(values.to_numpy(), aspect="auto", cmap="viridis")
                panel.set(xticks=range(len(values.columns)), xticklabels=values.columns,
                          yticks=range(len(order)), yticklabels=order, title=family + " / " + label)
                panel.tick_params(axis="x", rotation=45)
                for i in range(len(order)):
                    for j in range(len(values.columns)):
                        value = values.iloc[i, j]
                        if numpy.isfinite(value):
                            panel.text(j, i, "%.4f" % value, ha="center", va="center", color="white")
                fig.colorbar(plotted, ax=panel)
        fig.suptitle("Processing recipe screen: sample-balanced, two paired folds, one matched decoy per hit")
        fig.savefig(out / "validation_training_recipe.png", dpi=200)
        fig.savefig(out / "validation_training_recipe.pdf")
        plt.close(fig)
        return
    fig, panels = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for panel, metric, label in zip(panels, metrics, ["AUROC", "AUPRC", "PPV@N"]):
        for family, group in summary.groupby("family"):
            group = group.sort_values("kernel_width")
            panel.plot(group.kernel_width, group["macro_" + metric], "o-", label=family)
        panel.set(xlabel="Kernel width (residues)", ylabel="Macro " + label, xticks=[5, 7, 9, 11, 13, 15])
        panel.grid(alpha=0.2)
    panels[0].legend(fontsize=8)
    fig.suptitle("Matched processing validation — sample-balanced, four paired folds")
    fig.savefig(out / "validation_kernel_width.png", dpi=200)
    fig.savefig(out / "validation_kernel_width.pdf")
    plt.close(fig)
    evaluation = list((out / "evaluation").glob("*.metrics.csv"))
    if evaluation:
        from mhcflurry.cli.processing_affinity_control import summarize_metrics
        combined = pandas.concat([pandas.read_csv(path) for path in evaluation])
        combined.to_csv(out / "evaluation/metrics.csv", index=False)
        summary, deltas = summarize_metrics(combined, "public_2_1_x")
        summary.to_csv(out / "evaluation/summary.csv", index=False)
        deltas.to_csv(out / "evaluation/comparisons.csv", index=False)
        plotted = summary.merge(axes, left_on="score", right_on="condition")
        public = summary.set_index("score").loc["public_2_1_x"]
        fig, panels = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for panel, metric, label in zip(panels, metrics, ["AUROC", "AUPRC", "PPV@N"]):
            for family, group in plotted.groupby("family"):
                group = group.sort_values("kernel_width")
                panel.plot(group.kernel_width, group["macro_" + metric], "o-", label=family + " (4)")
            panel.axhline(public["macro_" + metric], color="black", linestyle="--", label="public (8)")
            panel.set(xlabel="Kernel width (residues)", ylabel="Macro " + label, xticks=[5, 7, 9, 11, 13, 15])
            panel.grid(alpha=0.2)
        panels[0].legend(fontsize=8)
        fig.suptitle("Exploratory release holdout - 10 affinity/length-matched decoys per hit")
        fig.savefig(out / "evaluation/kernel_width.png", dpi=200)
        fig.savefig(out / "evaluation/kernel_width.pdf")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--release-holdout-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--design", choices=("kernel-width", "training-recipe", "ranking-confirmation"), default="kernel-width")
    parser.add_argument("--folds-from", type=Path,
                        help="Frozen fold table; use the number of folds declared by the selected design.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluation", choices=("all", "none"), default="all")
    parser.add_argument("--resume-from", type=Path,
                        help="Read-only prior width run; import completed conditions and frozen folds.")
    parser.add_argument("--save-all-checkpoints", action="store_true",
                        help="Retain terminal and best weights for newly trained models.")
    args = parser.parse_args(argv)
    for key in ("out", "train_data", "public_root", "release_holdout_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    from mhcflurry.training_folds import read_processing_training_data
    frame = read_processing_training_data(args.train_data)
    validate_matched_training_data(frame)
    excluded = set(load_excluded_samples(str(args.release_holdout_dir / "processing_samples.csv")))
    if excluded.intersection(frame.sample_id):
        raise ValueError("Processing training overlaps release holdout samples")
    if args.design == "training-recipe":
        from generate_processing_recipe import build_processing_recipe_conditions
        records = build_processing_recipe_conditions()
    elif args.design == "ranking-confirmation":
        from generate_processing_recipe import build_processing_confirmation_conditions
        records = build_processing_confirmation_conditions()
        if not args.folds_from or args.evaluation != "none":
            raise ValueError("Ranking confirmation requires --folds-from and --evaluation none")
    else:
        records = build_kernel_conditions()
    fold_count = records[0][2]["fold_count"]
    if args.save_all_checkpoints:
        for _, grid, _ in records:
            for hp in grid:
                hp["save_all_checkpoints"] = True
    design_names = {"kernel-width": "processing-kernel-sweep-v1", "training-recipe": "processing-training-recipe-v1",
                    "ranking-confirmation": "processing-ranking-confirmation-v1"}
    config = {"design": design_names[args.design],
              "networks": sum(axes["network_count"] for _, _, axes in records),
              "random_seed": args.random_seed, "source_commit": args.source_commit,
              "train_data_sha256": sha256_file(args.train_data),
              "training_validation": {"policy": "matched", "rows": len(frame),
                                      "samples": int(frame.sample_id.nunique())},
              "public_root": str(args.public_root), "evaluation": args.evaluation,
              "holdout_sha256": {p.name: sha256_file(p) for p in args.release_holdout_dir.glob("*") if p.is_file()},
              "records": [dict(condition=name, hyperparameters=grid, **axes) for name, grid, axes in records]}
    if args.resume_from:
        if args.design != "kernel-width":
            raise ValueError("--resume-from currently supports the width design only")
        config["resume_from"] = str(args.resume_from.resolve())
    if args.folds_from:
        config["folds_from_sha256"] = sha256_file(args.folds_from)
    manifest = args.out / "experiment.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Refusing to change inputs/design in an existing experiment")
    write_json(manifest, config)
    inputs = args.out / "inputs"
    inputs.mkdir(exist_ok=True)
    if args.folds_from:
        original = read_processing_training_data(args.folds_from)
        selected_columns = ["fold_%d" % i for i in range(fold_count)]
        if not set(selected_columns) <= set(original):
            raise ValueError("Missing requested frozen folds")
        compared = original.drop(columns=[c for c in original if c.startswith("fold_")])
        # Use the frozen table itself as --train-data when inheriting old folds.
        # Do not forgive floating-point drift or check only neural input columns.
        expected = frame.drop(columns=[c for c in frame if c.startswith("fold_")])
        pandas.testing.assert_frame_equal(
            compared, expected, check_exact=True)
        table = original.drop(columns=[c for c in original if c.startswith("fold_") and c not in selected_columns])
        validate_matched_training_data(table)
        path = inputs / "training_with_folds.csv.bz2"
        if path.exists():
            pandas.testing.assert_frame_equal(read_processing_training_data(path), table, check_exact=True)
        else:
            table.to_csv(path, index=False)
    for name, grid, _ in records:
        (inputs / (name + ".yaml")).write_text(yaml.safe_dump(grid, sort_keys=True))
    if args.prepare_only:
        return 0
    imported = import_completed_conditions(args.resume_from, args.out, config) if args.resume_from else set()
    os.environ.update(MHCFLURRY_TORCH_COMPILE="0", MHCFLURRY_TORCH_COMPILE_LOSS="0",
                      MHCFLURRY_MATMUL_PRECISION="highest")
    driver = Driver(args.out)
    parallel = ["--gpus", args.gpus, "--num-jobs", args.num_jobs, "--max-workers-per-gpu", 1,
                "--torch-compile", 0, "--matmul-precision", "highest", "--dataloader-num-workers", 1]
    matched = None
    if args.evaluation == "all":
        from mhcflurry.common import configure_pytorch
        configure_pytorch(backend="gpu" if args.gpus else "cpu", num_threads=1)
        matched = prepare_evaluation(args)
        evaluate_predictor(args.out, "public_2_1_x",
            args.public_root / "models_class1_processing/models.selected.short_flanks", matched)
    folded = inputs / "training_with_folds.csv.bz2"
    for name, _, _ in records:
        processing = args.out / name / "processing"
        processing.mkdir(parents=True, exist_ok=True)
        unselected = processing / "models.unselected.short_flanks"
        selected = processing / "models.selected.short_flanks"
        if name not in imported and not (unselected / "training_init_info.pkl").exists():
            data_args = ["--data", folded, "--reuse-folds"] if folded.exists() else [
                "--data", args.train_data, "--held-out-samples", 10]
            driver.run(name + "-initialize", ["mhcflurry", "class1-train-processing-models",
                *data_args, "--num-folds", fold_count, "--random-seed", args.random_seed,
                "--hyperparameters", inputs / (name + ".yaml"), "--out-models-dir", unselected,
                "--only-initialize", *parallel])
        generated = unselected / "train_data.csv.bz2"
        if not folded.exists():
            shutil.copy2(generated, folded)
        else:
            expected = pandas.read_csv(folded, dtype=str, keep_default_na=False)
            observed = pandas.read_csv(generated, dtype=str, keep_default_na=False)
            pandas.testing.assert_frame_equal(expected, observed)
        if name not in imported:
            driver.run(name + "-train", ["mhcflurry", "class1-train-processing-models",
                "--out-models-dir", unselected, "--continue-incomplete", *parallel])
            driver.run(name + "-select", ["mhcflurry", "class1-select-processing-models",
                "--data", generated, "--models-dir", unselected, "--out-models-dir", selected,
                "--min-models-per-fold", 1, "--max-models-per-fold", 1,
                "--save-validation-predictions", *parallel])
        if name not in imported:
            validation_metrics(selected / "model_selection_predictions.csv.bz2", name).to_csv(
                args.out / name / "validation_metrics.csv", index=False)
            validation_predictions = pandas.read_csv(selected / "model_selection_predictions.csv.bz2")
            pandas.DataFrame([dict(fold_num=fold, **_metrics(group.hit, group.processing_score))
                              for fold, group in validation_predictions.groupby("fold_num")]).to_csv(
                args.out / name / "validation_micro_by_fold.csv", index=False)
            driver.run(name + "-loss-plots", ["mhcflurry", "train", "plot-loss-curves",
                "--selected-dir", selected, "--unselected-dir", unselected,
                "--out", args.out / name / "loss_plots"])
        checkpoint_predictions = args.out / "checkpoint_predictions"
        checkpoint_predictions.mkdir(exist_ok=True)
        suffix = ".csv.gz" if args.design == "ranking-confirmation" else ".csv.bz2"
        prediction_path = checkpoint_predictions / (name + ".validation_predictions" + suffix)
        save_checkpoint_validation_predictions(unselected, prediction_path, name)
        if args.design == "ranking-confirmation":
            checkpoint_frame = pandas.read_csv(prediction_path, dtype={"sample_id": str}, float_precision="round_trip")
            checkpoint_metrics = []
            checkpoint_micro = []
            for (policy, fold, sample), group in checkpoint_frame.groupby(["checkpoint_policy", "fold_num", "sample_id"]):
                checkpoint_metrics.append(dict(condition=name, checkpoint_policy=policy, fold_num=fold, sample_id=sample,
                                               **_metrics(group.hit, group.processing_score)))
            for (policy, fold), group in checkpoint_frame.groupby(["checkpoint_policy", "fold_num"]):
                checkpoint_micro.append(dict(condition=name, checkpoint_policy=policy, fold_num=fold,
                                             **_metrics(group.hit, group.processing_score)))
            pandas.DataFrame(checkpoint_metrics).to_csv(args.out / name / "checkpoint_per_sample.csv", index=False)
            pandas.DataFrame(checkpoint_micro).to_csv(args.out / name / "checkpoint_micro_by_fold.csv", index=False)
            from mhcflurry.experiment_archive import export_training_tables
            export_training_tables(unselected, args.out / name / "training_tables")
        if matched is not None:
            evaluate_predictor(args.out, name, selected, matched)
        render_summary(args.out, records)
        write_json(args.out / "progress.json", {"last_completed_condition": name, "at": utc_now()})
    write_json(args.out / "completed.json", {"at": utc_now(), "release_accepted": False})
    return 0


if __name__ == "__main__":
    sys.exit(main())

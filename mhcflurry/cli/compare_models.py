"""Compare model ensembles on the data_evaluation benchmarks.

Combines the three legacy ``scripts/training/compare_*.py`` tools into one
command. ``--a`` and ``--b`` may each be a training-run directory, the
literal ``public`` (resolves to the currently-installed public release),
or ``public:<release_name>`` (pin a non-default release). ``--b`` defaults
to ``public``.

Runs whichever components are available on both sides:

* ``training_stats`` — per-task wall-time, epoch-count, final-loss
  deltas from each side's ``manifest.csv``. Skipped when either side is
  public (no manifest).
* ``affinity`` — per-allele ROC-AUC / PR-AUC / PPV@N on the
  monoallelic hit/decoy benchmark.
* ``presentation`` — per-sample + per-length micro/macro metrics on
  the multiallelic hit/decoy benchmark, with-flanks and without-flanks.

Writes CSVs + JSON only. ``mhcflurry plot-model-comparison`` consumes
the CSVs to render plots.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import time
import warnings
from functools import partial
from typing import Optional

import numpy
import pandas
from sklearn.metrics import average_precision_score, roc_auc_score

from ..local_parallelism import (
    add_prediction_parallelism_args,
    call_wrapped_kwargs,
    chunk_ranges_for_local_parallelism,
    worker_pool_with_gpu_assignments_from_args,
)
from ..pseudosequences import LEGACY_ALLELE_SEQUENCES_FILENAME
from ..workload_planning import (
    WORKLOAD_AFFINITY_INFERENCE,
    WORKLOAD_PRESENTATION_INFERENCE,
)


_METRIC_NAMES = ("roc_auc", "pr_auc", "ppv_at_n")
_PRESENTATION_SCORE_KINDS = ("presentation_score", "presentation_percentile")
_PRESENTATION_MODES = ("with_flanks", "without_flanks")
_COMPONENT_NAMES = ("training_stats", "affinity", "presentation")

_T0 = time.time()


def _stamp(msg):
    print("[+%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


# ---------------------------------------------------------------------------
# argparse + dispatch
# ---------------------------------------------------------------------------


def make_parser():
    """Return a standalone parser for documentation tooling (autoprogram)."""
    parser = argparse.ArgumentParser(prog="mhcflurry compare-models")
    register_subparser(parser)
    return parser


def run_argv(argv):
    """Entry point for the lazy ``mhcflurry compare-models`` dispatcher."""
    return run(make_parser().parse_args(argv))


def register_subparser(parser):
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument(
        "--a", required=True, dest="a",
        help=(
            "Side A: a training-run directory, 'public', or "
            "'public:<release_name>'."
        ),
    )
    parser.add_argument(
        "--b", default="public", dest="b",
        help=(
            "Side B: same forms as --a. Defaults to 'public' (the most "
            "recently installed mhcflurry release)."
        ),
    )
    parser.add_argument("--a-label", default=None,
                        help="Display label for side A (default: derived).")
    parser.add_argument("--b-label", default=None,
                        help="Display label for side B (default: derived).")
    for letter in ("a", "b"):
        for role in ("affinity", "processing", "presentation", "training"):
            parser.add_argument(
                "--%s-%s-dir" % (letter, role),
                default=None,
                dest="%s_%s_dir" % (letter, role),
                help=(
                    "Override the auto-probed %s path for side %s."
                    % (role, letter.upper())
                ),
            )
    parser.add_argument(
        "--out", required=True,
        help="Output directory. Subdirs per component are created here.",
    )
    parser.add_argument(
        "--include",
        default="auto",
        help=(
            "Comma-separated subset of {training_stats, affinity, "
            "presentation}; default 'auto' runs whichever components are "
            "available on both sides."
        ),
    )
    parser.add_argument(
        "--data-dir", default=None,
        help=(
            "data_evaluation directory. Defaults to the currently-installed "
            "data_evaluation download."
        ),
    )
    parser.add_argument(
        "--limit-files", type=int, default=None,
        help="Smoke-test: only read first N benchmark files.",
    )
    parser.add_argument(
        "--affinity-source", choices=["mixmhcpred", "netmhcpan4", "both"],
        default="mixmhcpred",
        help="Which monoallelic benchmark source to use for affinity eval.",
    )
    parser.add_argument(
        "--presentation-modes",
        default=",".join(_PRESENTATION_MODES),
        help=(
            "Comma-separated subset of {with_flanks, without_flanks} for "
            "the presentation component."
        ),
    )
    add_prediction_parallelism_args(parser)
    return parser


def run(args):
    os.makedirs(args.out, exist_ok=True)
    side_a = _resolve_side("a", args.a, args.a_label, args)
    side_b = _resolve_side("b", args.b, args.b_label, args)

    with open(os.path.join(args.out, "side_a.json"), "w") as fd:
        json.dump(_side_to_json(side_a), fd, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "side_b.json"), "w") as fd:
        json.dump(_side_to_json(side_b), fd, indent=2, sort_keys=True)

    components = _resolve_components(args.include, side_a, side_b)
    _stamp("running components: %s" % (", ".join(components) or "(none)"))

    headline = {"side_a": side_a["label"], "side_b": side_b["label"]}

    if "training_stats" in components:
        headline["training_stats"] = _run_training_stats(side_a, side_b, args.out)
    if "affinity" in components:
        headline["affinity"] = _run_affinity(side_a, side_b, args)
    if "presentation" in components:
        headline["presentation"] = _run_presentation(side_a, side_b, args)

    _write_summary_markdown(headline, side_a, side_b, args.out, components)
    return 0


# ---------------------------------------------------------------------------
# Side resolution
# ---------------------------------------------------------------------------


def _resolve_side(letter, spec, label, args):
    """Resolve a CLI ``--a`` / ``--b`` spec to per-role paths.

    ``spec`` may be a filesystem path, ``"public"``, or
    ``"public:<release_name>"``. Per-role CLI overrides
    (``--a-affinity-dir`` etc) take precedence over the auto-probe.
    """
    overrides = {
        "affinity": getattr(args, "%s_affinity_dir" % letter),
        "processing": getattr(args, "%s_processing_dir" % letter),
        "presentation": getattr(args, "%s_presentation_dir" % letter),
        "training": getattr(args, "%s_training_dir" % letter),
    }
    # Match the public sentinel exactly so user-named directories like
    # ``public_models/`` or ``publication_data/`` still resolve as run dirs.
    is_public = isinstance(spec, str) and (
        spec == "public" or spec.startswith("public:")
    )
    release_pin = None
    if is_public and ":" in spec:
        _, release_pin = spec.split(":", 1)

    paths = {}
    for role in ("affinity", "processing", "presentation", "training"):
        if overrides[role]:
            paths[role] = overrides[role]
        elif is_public:
            paths[role] = _public_path_for_role(role, release_pin)
        else:
            paths[role] = _probe_run_dir(spec, role)

    if label is None:
        if is_public:
            label = "public" if release_pin is None else "public:%s" % release_pin
        else:
            label = os.path.basename(os.path.normpath(spec)) or spec
    return {"letter": letter, "spec": spec, "label": label, "paths": paths}


def _public_path_for_role(role, release_pin):
    """Resolve a public-install models dir for ``role`` if available."""
    from .. import downloads
    role_to_download = {
        "affinity": ("models_class1_pan", "models.combined"),
        "processing": (
            "models_class1_processing", "models.selected.with_flanks"),
        "presentation": ("models_class1_presentation", "models.combined"),
        "training": (None, None),
    }
    download_name, sub = role_to_download.get(role, (None, None))
    if download_name is None:
        return None
    release_env = "MHCFLURRY_DOWNLOADS_CURRENT_RELEASE"
    missing = object()
    previous_release = os.environ.get(release_env, missing)
    try:
        if release_pin is not None:
            os.environ[release_env] = release_pin
            try:
                downloads.configure()
            except KeyError:
                # Unknown release name — no path resolvable for any role.
                return None
        return downloads.get_path(download_name, sub)
    except (RuntimeError, OSError, TypeError):
        # No downloads installed (TypeError when downloads dir is None) or
        # the specific archive isn't present (RuntimeError from get_path).
        return None
    finally:
        if release_pin is not None:
            if previous_release is missing:
                os.environ.pop(release_env, None)
            else:
                os.environ[release_env] = previous_release
            downloads.configure()


def _probe_run_dir(spec, role):
    """Probe a run directory for the canonical ``role`` subdirectory."""
    if not os.path.isdir(spec):
        return None
    candidates = []
    if role == "affinity":
        candidates = [
            os.path.join(spec, "affinity", "models.combined"),
            os.path.join(spec, "models.combined"),
            spec,
        ]
        return _first_match(candidates, _looks_like_affinity_dir)
    if role == "presentation":
        candidates = [
            os.path.join(spec, "presentation", "models.combined"),
            os.path.join(spec, "presentation"),
            spec,
        ]
        return _first_match(candidates, _looks_like_presentation_dir)
    if role == "processing":
        candidates = [
            os.path.join(spec, "processing", "models.selected.with_flanks"),
            os.path.join(spec, "processing"),
        ]
        return _first_match(candidates, os.path.isdir)
    if role == "training":
        candidates = [
            os.path.join(spec, "affinity", "models.unselected.combined"),
            os.path.join(spec, "models.unselected.combined"),
            spec,
        ]
        return _first_match(
            candidates,
            lambda p: os.path.isfile(os.path.join(p, "manifest.csv")),
        )
    return None


def _first_match(paths, predicate):
    for path in paths:
        if path and predicate(path):
            return path
    return None


def _looks_like_affinity_dir(path):
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, LEGACY_ALLELE_SEQUENCES_FILENAME)):
        return True
    return any(
        f.startswith("manifest") for f in os.listdir(path)
    )


def _looks_like_presentation_dir(path):
    return os.path.isdir(path) and os.path.isfile(
        os.path.join(path, "weights.csv")
    )


def _side_to_json(side):
    return {
        "letter": side["letter"],
        "spec": side["spec"],
        "label": side["label"],
        "paths": side["paths"],
    }


def _resolve_components(include_arg, side_a, side_b):
    available = []
    if side_a["paths"]["training"] and side_b["paths"]["training"]:
        available.append("training_stats")
    if side_a["paths"]["affinity"] and side_b["paths"]["affinity"]:
        available.append("affinity")
    if side_a["paths"]["presentation"] and side_b["paths"]["presentation"]:
        available.append("presentation")

    if include_arg == "auto":
        return available
    requested = [p.strip() for p in include_arg.split(",") if p.strip()]
    bad = [p for p in requested if p not in _COMPONENT_NAMES]
    if bad:
        raise SystemExit("Unknown --include components: %s" % ", ".join(bad))
    missing = [p for p in requested if p not in available]
    if missing:
        for p in missing:
            _stamp("WARNING: %s requested but unavailable on both sides" % p)
    return [p for p in requested if p in available]


# ---------------------------------------------------------------------------
# Metric helpers (shared)
# ---------------------------------------------------------------------------


def _ppv_at_n(y_true, y_score, n):
    # Stable sort so tied scores break deterministically (by original index),
    # making PPV@N reproducible run-to-run. Applied symmetrically to sides A
    # and B, so it does not bias the comparison.
    order = numpy.argsort(-y_score, kind="stable")
    top = order[:n]
    return float(y_true[top].sum()) / float(n) if n > 0 else numpy.nan


def _metrics(y_true, y_score):
    y_true = numpy.asarray(y_true)
    y_score = numpy.asarray(y_score)
    mask = ~numpy.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return dict(
            n=int(len(y_true)), n_pos=n_pos,
            roc_auc=numpy.nan, pr_auc=numpy.nan, ppv_at_n=numpy.nan,
        )
    return dict(
        n=int(len(y_true)),
        n_pos=n_pos,
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        ppv_at_n=_ppv_at_n(y_true, y_score, n_pos),
    )


def _add_diffs(df, metric_names, a_prefix="a", b_prefix="b"):
    """Add ``<metric>_diff = a - b`` columns in-place and return df."""
    for metric in metric_names:
        a_col = "%s_%s" % (a_prefix, metric)
        b_col = "%s_%s" % (b_prefix, metric)
        if a_col in df.columns and b_col in df.columns:
            df["%s_diff" % metric] = df[a_col] - df[b_col]
    return df


def _metric_table_columns(id_columns, metric_names=_METRIC_NAMES):
    columns = list(id_columns) + ["n", "n_pos"]
    for metric in metric_names:
        columns.extend(["a_%s" % metric, "b_%s" % metric, "%s_diff" % metric])
    return columns


def _metric_table(rows, id_columns, metric_names=_METRIC_NAMES):
    df = pandas.DataFrame(rows, columns=_metric_table_columns(
        id_columns, metric_names))
    return _add_diffs(df, metric_names)


def _per_length_columns(metric_names=_METRIC_NAMES):
    columns = ["length", "n", "n_pos", "n_alleles_reported"]
    for metric in metric_names:
        columns.extend([
            "a_micro_%s" % metric,
            "b_micro_%s" % metric,
            "a_macro_%s" % metric,
            "b_macro_%s" % metric,
            "micro_%s_diff" % metric,
            "macro_%s_diff" % metric,
        ])
    return columns


def _presentation_per_length_columns(metric_names=_METRIC_NAMES):
    columns = ["length", "n", "n_pos", "n_samples_reported"]
    for metric in metric_names:
        columns.extend([
            "a_micro_%s" % metric,
            "b_micro_%s" % metric,
            "micro_%s_diff" % metric,
            "a_macro_%s" % metric,
            "b_macro_%s" % metric,
            "macro_%s_diff" % metric,
        ])
    return columns


# ---------------------------------------------------------------------------
# Component: training_stats
# ---------------------------------------------------------------------------


def _run_training_stats(side_a, side_b, out_dir):
    component_dir = os.path.join(out_dir, "training_stats")
    os.makedirs(component_dir, exist_ok=True)

    a_summary = _load_training_summary(side_a["paths"]["training"])
    b_summary = _load_training_summary(side_b["paths"]["training"])
    a_summary["side"] = side_a["label"]
    b_summary["side"] = side_b["label"]
    per_task = pandas.concat([a_summary, b_summary], ignore_index=True)
    per_task.to_csv(os.path.join(component_dir, "per_task.csv"), index=False)

    # ``agg`` is built in (side_a, side_b) order, so positional .iloc lookups
    # are robust even when the two labels happen to collide.
    agg = pandas.DataFrame([
        _aggregate_training_summary(side_a["label"], a_summary),
        _aggregate_training_summary(side_b["label"], b_summary),
    ])
    agg.to_csv(os.path.join(component_dir, "summary.csv"), index=False)
    _stamp("wrote training_stats per_task.csv + summary.csv")
    return {
        "side_a_finetune_total_wall_min": float(
            agg.iloc[0]["finetune_total_wall_min"]),
        "side_b_finetune_total_wall_min": float(
            agg.iloc[1]["finetune_total_wall_min"]),
        "side_a_n_models": int(agg.iloc[0]["n_models"]),
        "side_b_n_models": int(agg.iloc[1]["n_models"]),
    }


def _parse_config_json(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(raw)


def _load_training_summary(training_dir):
    rows = []
    manifest_path = os.path.join(training_dir, "manifest.csv")
    df = pandas.read_csv(manifest_path)
    required = {"model_name", "config_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "manifest %s missing required columns: %s"
            % (manifest_path, sorted(missing))
        )
    for r in df.itertuples():
        cfg = _parse_config_json(r.config_json)
        layer_sizes = tuple(cfg["hyperparameters"].get("layer_sizes") or ())
        for fit_info in cfg.get("fit_info") or []:
            ti = fit_info.get("training_info", {})
            loss = fit_info.get("loss") or []
            val = fit_info.get("val_loss") or []
            rows.append({
                "model_name": r.model_name,
                "phase": ti.get("phase", "?"),
                "layer_sizes": str(layer_sizes),
                "fold": ti.get("fold_num"),
                "n_epochs": len(loss),
                "wall_time_sec": fit_info.get("time"),
                "final_loss": loss[-1] if loss else float("nan"),
                "final_val_loss": val[-1] if val else float("nan"),
                "min_val_loss": min(val) if val else float("nan"),
            })
    return pandas.DataFrame(rows)


def _aggregate_training_summary(label, summary):
    finetune = summary[summary.phase == "finetune"]
    pretrain = summary[summary.phase == "pretrain"]
    return {
        "side": label,
        "n_models": int(summary.model_name.nunique()),
        "finetune_total_wall_min": (
            finetune.wall_time_sec.sum() / 60 if len(finetune) else float("nan")
        ),
        "finetune_median_wall_min": (
            finetune.wall_time_sec.median() / 60 if len(finetune) else float("nan")
        ),
        "finetune_max_wall_min": (
            finetune.wall_time_sec.max() / 60 if len(finetune) else float("nan")
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
            pretrain.wall_time_sec.median()
            if len(pretrain) else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Component: affinity
# ---------------------------------------------------------------------------


def _predict_affinity_chunk(predictor_dir, peptides, alleles, chunk_num):
    """Worker entry: load affinity predictor, score one chunk."""
    from .. import Class1AffinityPredictor
    predictor = Class1AffinityPredictor.load(predictor_dir)
    return chunk_num, numpy.asarray(predictor.predict(
        peptides=peptides, alleles=alleles, throw=False))


def _parallel_affinity_predict(args, predictor_dir, peptides, alleles):
    if len(peptides) == 0:
        return numpy.asarray([], dtype=float)
    worker_pool = worker_pool_with_gpu_assignments_from_args(
        args,
        workload_name=WORKLOAD_AFFINITY_INFERENCE,
        workload_hints={"prediction_rows": len(peptides)},
        start_method="spawn",
    )
    _stamp(
        "      prediction plan: jobs=%d, gpus=%d, workers/gpu=%d, backend=%s"
        % (
            int(args.num_jobs),
            int(args.gpus or 0),
            int(args.max_workers_per_gpu),
            args.backend,
        )
    )
    if worker_pool is None:
        _, predictions = _predict_affinity_chunk(
            predictor_dir, peptides, alleles, chunk_num=0)
        return predictions

    work_items = []
    for (chunk_num, start, end) in chunk_ranges_for_local_parallelism(
            len(peptides), args.num_jobs):
        work_items.append({
            "chunk_num": chunk_num,
            "predictor_dir": predictor_dir,
            "peptides": peptides[start:end],
            "alleles": alleles[start:end],
        })
    try:
        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, _predict_affinity_chunk),
            work_items,
            chunksize=1,
        )
        chunks = [result for result in results]
        worker_pool.close()
        worker_pool.join()
        worker_pool = None
    finally:
        # On failure mid-iteration, terminate() rather than close()/join()
        # (which can hang on a wedged worker) and leave non-daemon workers
        # behind. Mirrors the predict / predict-scan teardown.
        if worker_pool is not None:
            worker_pool.terminate()
            worker_pool.join()
    # chunk_num is unique per work item, so sorted() never compares ndarrays.
    return numpy.concatenate([values for (_, values) in sorted(chunks)])


def _read_supported_alleles(predictor_dir):
    path = os.path.join(predictor_dir, LEGACY_ALLELE_SEQUENCES_FILENAME)
    if not os.path.exists(path):
        return set()
    df = pandas.read_csv(path)
    col = "normalized_allele" if "normalized_allele" in df.columns else df.columns[0]
    return set(df[col].astype(str).tolist())


def _load_affinity_benchmark(data_dir, source, limit_files):
    if source == "both":
        patterns = ["mixmhcpred", "netmhcpan4"]
    else:
        patterns = [source]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(
            data_dir,
            "benchmark.monoallelic.%s.train_excluded.*.csv.bz2" % pat,
        ))))
    if limit_files:
        files = files[:limit_files]
    _stamp("affinity benchmark: %d files" % len(files))
    if not files:
        raise SystemExit("No affinity benchmark files in %s" % data_dir)
    dfs = []
    for i, f in enumerate(files):
        df = pandas.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
        if (i + 1) % 50 == 0:
            _stamp("  loaded %d/%d" % (i + 1, len(files)))
    return pandas.concat(dfs, ignore_index=True)


def _run_affinity(side_a, side_b, args):
    component_dir = os.path.join(args.out, "affinity")
    os.makedirs(component_dir, exist_ok=True)

    data_dir = args.data_dir or _default_data_evaluation_dir()
    test = _load_affinity_benchmark(
        data_dir, args.affinity_source, args.limit_files)

    a_alleles = _read_supported_alleles(side_a["paths"]["affinity"])
    b_alleles = _read_supported_alleles(side_b["paths"]["affinity"])
    if not a_alleles or not b_alleles:
        _stamp(
            "WARNING: supported-allele set empty for %s%s%s; skipping "
            "allele-intersection filter -- the two models may be scored on "
            "different allele supports" % (
                "side A" if not a_alleles else "",
                " and " if not a_alleles and not b_alleles else "",
                "side B" if not b_alleles else "",
            )
        )
    both = a_alleles & b_alleles if (a_alleles and b_alleles) else None
    if both is not None:
        before = len(test)
        test = test[test.hla.isin(both)].copy()
        if len(test) < before:
            _stamp(
                "  dropped %d rows outside the %d-allele intersect"
                % (before - len(test), len(both))
            )
    test["peptide_len"] = test["peptide"].str.len()
    test = test[(test.peptide_len >= 8) & (test.peptide_len <= 15)].copy()
    _stamp("  evaluable rows: %d" % len(test))

    _stamp("predicting side A affinity...")
    test["a_pred"] = _parallel_affinity_predict(
        args, side_a["paths"]["affinity"],
        test.peptide.values, test.hla.values,
    )
    _stamp("predicting side B affinity...")
    test["b_pred"] = _parallel_affinity_predict(
        args, side_b["paths"]["affinity"],
        test.peptide.values, test.hla.values,
    )
    test = test.dropna(subset=["a_pred", "b_pred"])
    test["a_score"] = -numpy.log10(numpy.clip(test.a_pred, 1e-3, 1e8))
    test["b_score"] = -numpy.log10(numpy.clip(test.b_pred, 1e-3, 1e8))
    test.to_csv(
        os.path.join(component_dir, "predictions.csv.bz2"), index=False)
    _stamp("  wrote predictions.csv.bz2 (%d rows)" % len(test))

    per_allele = _affinity_per_allele(test)
    per_allele.to_csv(
        os.path.join(component_dir, "per_allele.csv"), index=False)
    _stamp("  wrote per_allele.csv (%d alleles)" % len(per_allele))

    per_length, per_length_per_allele = _affinity_per_length(test)
    per_length.to_csv(
        os.path.join(component_dir, "per_length.csv"), index=False)
    if not per_length_per_allele.empty:
        per_length_per_allele.to_csv(
            os.path.join(component_dir, "per_length_per_allele.csv"),
            index=False,
        )

    summary = _affinity_summary(test, per_allele, per_length)
    with open(os.path.join(component_dir, "summary.json"), "w") as fd:
        json.dump(summary, fd, indent=2, sort_keys=True)
    _stamp("  wrote summary.json")
    return summary


def _affinity_per_allele(test):
    rows = []
    for allele, group in test.groupby("hla"):
        if len(group) < 30 or group.hit.sum() == 0:
            continue
        m_a = _metrics(group.hit.values, group.a_score.values)
        m_b = _metrics(group.hit.values, group.b_score.values)
        row = {"allele": allele, "n": m_a["n"], "n_pos": m_a["n_pos"]}
        for metric in _METRIC_NAMES:
            row["a_%s" % metric] = m_a[metric]
            row["b_%s" % metric] = m_b[metric]
        rows.append(row)
    return _metric_table(rows, ["allele"]).sort_values("n", ascending=False)


def _affinity_per_length(test):
    lengths = sorted(set(int(L) for L in test.peptide_len.unique()))
    rows = []
    per_allele_rows = []
    for L in lengths:
        sub = test[test.peptide_len == L]
        if len(sub) == 0:
            continue
        m_a_L = _metrics(sub.hit.values, sub.a_score.values)
        m_b_L = _metrics(sub.hit.values, sub.b_score.values)
        per_allele_L = []
        for allele, group in sub.groupby("hla"):
            if len(group) < 30 or group.hit.sum() == 0:
                continue
            ma_a = _metrics(group.hit.values, group.a_score.values)
            ma_b = _metrics(group.hit.values, group.b_score.values)
            per_allele_L.append({
                "allele": allele, "length": L,
                "n": ma_a["n"], "n_pos": ma_a["n_pos"],
                **{"a_%s" % m: ma_a[m] for m in _METRIC_NAMES},
                **{"b_%s" % m: ma_b[m] for m in _METRIC_NAMES},
            })
        per_allele_rows.extend(per_allele_L)
        row = {
            "length": L,
            "n": m_a_L["n"], "n_pos": m_a_L["n_pos"],
            "n_alleles_reported": len(per_allele_L),
        }
        for metric in _METRIC_NAMES:
            row["a_micro_%s" % metric] = m_a_L[metric]
            row["b_micro_%s" % metric] = m_b_L[metric]
            row["a_macro_%s" % metric] = (
                float(numpy.nanmean([r["a_%s" % metric] for r in per_allele_L]))
                if per_allele_L else float("nan")
            )
            row["b_macro_%s" % metric] = (
                float(numpy.nanmean([r["b_%s" % metric] for r in per_allele_L]))
                if per_allele_L else float("nan")
            )
            row["micro_%s_diff" % metric] = (
                row["a_micro_%s" % metric] - row["b_micro_%s" % metric]
            )
            row["macro_%s_diff" % metric] = (
                row["a_macro_%s" % metric] - row["b_macro_%s" % metric]
            )
        rows.append(row)
    per_length = pandas.DataFrame(
        rows, columns=_per_length_columns()).sort_values("length")
    per_length_per_allele = _metric_table(
        per_allele_rows, ["allele", "length"],
    ).sort_values(["length", "n"], ascending=[True, False])
    return per_length, per_length_per_allele


def _affinity_summary(test, per_allele, per_length):
    m_a_all = _metrics(test.hit.values, test.a_score.values)
    m_b_all = _metrics(test.hit.values, test.b_score.values)
    return {
        "n_rows": int(len(test)),
        "n_hits": int(test.hit.sum()),
        "n_alleles_reported": int(len(per_allele)),
        "micro_pooled": {"a": m_a_all, "b": m_b_all},
        "macro_mean_over_alleles": {
            metric: {
                "a": float(per_allele["a_%s" % metric].mean()),
                "b": float(per_allele["b_%s" % metric].mean()),
            }
            for metric in _METRIC_NAMES
        },
        "allele_count": {
            "a_better_%s" % metric: int((per_allele["%s_diff" % metric] > 0).sum())
            for metric in _METRIC_NAMES
        } | {
            "b_better_%s" % metric: int((per_allele["%s_diff" % metric] < 0).sum())
            for metric in _METRIC_NAMES
        },
        "per_length": per_length.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Component: presentation
# ---------------------------------------------------------------------------


def _predict_presentation_chunk(predictor_dir, rows, mode, chunk_num):
    """Worker entry: load presentation predictor, score one chunk."""
    from .. import Class1PresentationPredictor
    predictor = Class1PresentationPredictor.load(predictor_dir)

    df = pandas.DataFrame(rows)
    sample_to_alleles = (
        df.drop_duplicates("sample_id")
        .set_index("sample_id")
        .hla.str.split()
        .to_dict()
    )
    kwargs = dict(
        peptides=df.peptide.values,
        sample_names=df.sample_id.values,
        alleles=sample_to_alleles,
        verbose=0,
        throw=False,
    )
    if mode == "with_flanks":
        kwargs["n_flanks"] = df.n_flank.values
        kwargs["c_flanks"] = df.c_flank.values
    elif mode != "without_flanks":
        raise ValueError("Unexpected presentation mode: %s" % mode)
    pred = predictor.predict(**kwargs)
    if len(pred) != len(df):
        raise ValueError(
            "Predictor returned %d rows for %d inputs" % (len(pred), len(df))
        )
    out_cols = [
        "presentation_score", "presentation_percentile",
        "affinity", "processing_score",
    ]
    out = pandas.DataFrame(index=df.index)
    for col in out_cols:
        out[col] = pred[col].values if col in pred else numpy.nan
    return chunk_num, out


def _parallel_presentation_predict(args, predictor_dir, df, mode, label):
    if len(df) == 0:
        return pandas.DataFrame()
    worker_pool = worker_pool_with_gpu_assignments_from_args(
        args,
        workload_name=WORKLOAD_PRESENTATION_INFERENCE,
        workload_hints={"prediction_rows": len(df)},
        start_method="spawn",
    )
    _stamp("predicting %s presentation (%s, %d rows)" % (label, mode, len(df)))
    if worker_pool is None:
        _, frame = _predict_presentation_chunk(
            predictor_dir, df.to_dict("list"), mode, chunk_num=0)
        return frame.reset_index(drop=True)

    work_items = []
    for (chunk_num, start, end) in chunk_ranges_for_local_parallelism(
            len(df), args.num_jobs):
        work_items.append({
            "chunk_num": chunk_num,
            "predictor_dir": predictor_dir,
            "rows": df.iloc[start:end].to_dict("list"),
            "mode": mode,
        })
    try:
        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, _predict_presentation_chunk),
            work_items,
            chunksize=1,
        )
        chunks = [result for result in results]
        worker_pool.close()
        worker_pool.join()
        worker_pool = None
    finally:
        # On failure mid-iteration, terminate() rather than close()/join()
        # (which can hang on a wedged worker) and leave non-daemon workers
        # behind. Mirrors the predict / predict-scan teardown.
        if worker_pool is not None:
            worker_pool.terminate()
            worker_pool.join()
    return pandas.concat(
        [frame for (_, frame) in sorted(chunks, key=lambda t: t[0])],
        ignore_index=True,
    )


def _load_presentation_benchmark(data_dir, limit_files):
    files = sorted(glob.glob(os.path.join(
        data_dir,
        "benchmark.multiallelic.train_excluded.*.csv.bz2",
    )))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise SystemExit(
            "No presentation benchmark files in %s "
            "(benchmark.multiallelic.train_excluded.*.csv.bz2)" % data_dir)
    _stamp("presentation benchmark: %d files" % len(files))
    dfs = []
    for i, path in enumerate(files):
        df = pandas.read_csv(path)
        df["source_file"] = os.path.basename(path)
        dfs.append(df)
        if (i + 1) % 25 == 0:
            _stamp("  loaded %d/%d" % (i + 1, len(files)))
    result = pandas.concat(dfs, ignore_index=True)
    required = {"peptide", "sample_id", "hla", "hit"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError("Presentation benchmark missing columns: %s" % missing)
    result = result.dropna(subset=["peptide", "sample_id", "hla", "hit"]).copy()
    result["hit"] = result["hit"].astype(int)
    result["peptide_len"] = result.peptide.str.len()
    result = result[
        (result.peptide_len >= 8) & (result.peptide_len <= 15)
    ].reset_index(drop=True)
    for col in ("n_flank", "c_flank"):
        if col not in result:
            result[col] = ""
        result[col] = result[col].fillna("")
    _stamp(
        "  benchmark rows after filtering: %d (samples=%d, hits=%d)" % (
            len(result),
            result.sample_id.nunique(),
            int(result.hit.sum()),
        )
    )
    return result


def _score_values(df, prefix, score_kind):
    """Higher = better for the score we feed sklearn.

    Affinity is already ``-log10(nM)``; ``presentation_score`` is already
    higher-better; ``presentation_percentile`` is lower-better so we
    negate.
    """
    if score_kind == "presentation_score":
        return df["%s_presentation_score" % prefix].values
    if score_kind == "presentation_percentile":
        return -df["%s_presentation_percentile" % prefix].values
    raise ValueError("Unknown score kind: %s" % score_kind)


def _presentation_per_sample(scored, score_kind):
    # NOTE: unlike _affinity_per_allele (which skips groups with <30 rows or
    # zero hits before entering the macro), this per-sample macro applies no
    # min-N / class-balance floor -- every (sample_id, hla) group is included,
    # and only fully-degenerate groups (all-hit or all-decoy) drop out via the
    # NaN that the downstream nanmean skips. This asymmetry is intentional (the
    # two macros were never defined to share a threshold), but it does mean the
    # presentation macro can be pulled around by small, noisy samples.
    rows = []
    for (sample_id, hla), group in scored.groupby(
            ["sample_id", "hla"], dropna=False):
        m_a = _metrics(group.hit.values, _score_values(group, "a", score_kind))
        m_b = _metrics(group.hit.values, _score_values(group, "b", score_kind))
        row = {
            "sample_id": sample_id, "hla": hla,
            "n": m_a["n"], "n_pos": m_a["n_pos"],
        }
        for metric in _METRIC_NAMES:
            row["a_%s" % metric] = m_a[metric]
            row["b_%s" % metric] = m_b[metric]
        rows.append(row)
    return _metric_table(rows, ["sample_id", "hla"]).sort_values(
        "n", ascending=False)


def _presentation_per_length(scored, score_kind):
    rows = []
    per_length_per_sample = []
    for length, group in scored.groupby("peptide_len"):
        m_a = _metrics(group.hit.values, _score_values(group, "a", score_kind))
        m_b = _metrics(group.hit.values, _score_values(group, "b", score_kind))
        sub_sample = _presentation_per_sample(group, score_kind)
        sub_sample["length"] = int(length)
        per_length_per_sample.append(sub_sample)
        row = {
            "length": int(length),
            "n": m_a["n"], "n_pos": m_a["n_pos"],
            "n_samples_reported": int(len(sub_sample)),
        }
        for metric in _METRIC_NAMES:
            row["a_micro_%s" % metric] = m_a[metric]
            row["b_micro_%s" % metric] = m_b[metric]
            row["micro_%s_diff" % metric] = m_a[metric] - m_b[metric]
            with warnings.catch_warnings():
                # All-NaN slices emit a RuntimeWarning; nan is the intended
                # result here (matches the silent pandas .mean() macro above).
                warnings.simplefilter("ignore", category=RuntimeWarning)
                macro_a = float(numpy.nanmean(sub_sample["a_%s" % metric]))
                macro_b = float(numpy.nanmean(sub_sample["b_%s" % metric]))
            row["a_macro_%s" % metric] = macro_a
            row["b_macro_%s" % metric] = macro_b
            row["macro_%s_diff" % metric] = macro_a - macro_b
        rows.append(row)
    per_length = pandas.DataFrame(
        rows,
        columns=_presentation_per_length_columns(),
    ).sort_values("length")
    if per_length_per_sample:
        per_length_per_sample = pandas.concat(
            per_length_per_sample, ignore_index=True)
    else:
        per_length_per_sample = pandas.DataFrame(
            columns=_metric_table_columns(["sample_id", "hla", "length"]))
    return per_length, per_length_per_sample


def _presentation_mode_summary(scored, per_sample, per_length, mode, score_kind):
    m_a = _metrics(scored.hit.values, _score_values(scored, "a", score_kind))
    m_b = _metrics(scored.hit.values, _score_values(scored, "b", score_kind))
    return {
        "mode": mode,
        "score_kind": score_kind,
        "n_rows": int(m_a["n"]),
        "n_hits": int(m_a["n_pos"]),
        "n_samples_reported": int(len(per_sample)),
        "micro_pooled": {"a": m_a, "b": m_b},
        "macro_mean_over_samples": {
            metric: {
                "a": float(numpy.nanmean(per_sample["a_%s" % metric])),
                "b": float(numpy.nanmean(per_sample["b_%s" % metric])),
            }
            for metric in _METRIC_NAMES
        },
        "sample_count": {
            "a_better_%s" % m: int((per_sample["%s_diff" % m] > 0).sum())
            for m in _METRIC_NAMES
        } | {
            "b_better_%s" % m: int((per_sample["%s_diff" % m] < 0).sum())
            for m in _METRIC_NAMES
        },
        "per_length": per_length.to_dict(orient="records"),
    }


def _presentation_summary_row(summary):
    row = {
        "mode": summary["mode"],
        "score_kind": summary["score_kind"],
        "n_rows": summary["n_rows"],
        "n_hits": summary["n_hits"],
        "n_samples_reported": summary["n_samples_reported"],
    }
    for metric in _METRIC_NAMES:
        row["a_micro_%s" % metric] = summary["micro_pooled"]["a"][metric]
        row["b_micro_%s" % metric] = summary["micro_pooled"]["b"][metric]
        row["micro_%s_diff" % metric] = (
            row["a_micro_%s" % metric] - row["b_micro_%s" % metric]
        )
        row["a_macro_%s" % metric] = (
            summary["macro_mean_over_samples"][metric]["a"]
        )
        row["b_macro_%s" % metric] = (
            summary["macro_mean_over_samples"][metric]["b"]
        )
        row["macro_%s_diff" % metric] = (
            row["a_macro_%s" % metric] - row["b_macro_%s" % metric]
        )
    return row


def _run_presentation(side_a, side_b, args):
    component_dir = os.path.join(args.out, "presentation")
    os.makedirs(component_dir, exist_ok=True)

    data_dir = args.data_dir or _default_data_evaluation_dir()
    requested_modes = [m.strip() for m in args.presentation_modes.split(",") if m]
    bad_modes = [m for m in requested_modes if m not in _PRESENTATION_MODES]
    if bad_modes:
        raise SystemExit(
            "Unknown presentation modes: %s" % ", ".join(bad_modes))

    benchmark = _load_presentation_benchmark(data_dir, args.limit_files)
    summaries = {}
    summary_rows = []
    for mode in requested_modes:
        _stamp("=== presentation mode: %s ===" % mode)
        scored = benchmark.copy()
        a_pred = _parallel_presentation_predict(
            args, side_a["paths"]["presentation"],
            benchmark, mode, label="A",
        )
        b_pred = _parallel_presentation_predict(
            args, side_b["paths"]["presentation"],
            benchmark, mode, label="B",
        )
        for prefix, pred in (("a", a_pred), ("b", b_pred)):
            for col in [
                "presentation_score", "presentation_percentile",
                "affinity", "processing_score",
            ]:
                scored["%s_%s" % (prefix, col)] = pred[col].values
        pred_path = os.path.join(
            component_dir, "predictions_%s.csv.bz2" % mode)
        scored.to_csv(pred_path, index=False)
        _stamp("  wrote %s" % pred_path)

        summaries[mode] = {}
        for score_kind in _PRESENTATION_SCORE_KINDS:
            per_sample = _presentation_per_sample(scored, score_kind)
            per_sample.to_csv(
                os.path.join(
                    component_dir,
                    "per_sample_%s_%s.csv" % (mode, score_kind),
                ),
                index=False,
            )
            per_length, per_length_per_sample = _presentation_per_length(
                scored, score_kind)
            per_length.to_csv(
                os.path.join(
                    component_dir,
                    "per_length_%s_%s.csv" % (mode, score_kind),
                ),
                index=False,
            )
            if not per_length_per_sample.empty:
                per_length_per_sample.to_csv(
                    os.path.join(
                        component_dir,
                        "per_length_per_sample_%s_%s.csv" % (mode, score_kind),
                    ),
                    index=False,
                )
            summary = _presentation_mode_summary(
                scored, per_sample, per_length, mode, score_kind)
            summaries[mode][score_kind] = summary
            summary_rows.append(_presentation_summary_row(summary))

    with open(os.path.join(component_dir, "summary.json"), "w") as fd:
        json.dump(summaries, fd, indent=2, sort_keys=True)
    summary_table = pandas.DataFrame(summary_rows)
    summary_table.to_csv(
        os.path.join(component_dir, "summary_table.csv"), index=False)
    _stamp("  wrote summary.json + summary_table.csv")
    return {
        "modes": requested_modes,
        "summaries": summaries,
    }


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------


def _write_summary_markdown(headline, side_a, side_b, out_dir, components):
    lines = []
    lines.append("# compare-models summary\n")
    lines.append("- side A: `%s` (%s)" % (side_a["label"], side_a["spec"]))
    lines.append("- side B: `%s` (%s)" % (side_b["label"], side_b["spec"]))
    lines.append("")

    if "training_stats" in components:
        ts = headline["training_stats"]
        lines.append("## training_stats")
        lines.append(
            "- %s: %d models, %.1f min total finetune wall-time" % (
                side_a["label"],
                ts["side_a_n_models"],
                ts["side_a_finetune_total_wall_min"],
            )
        )
        lines.append(
            "- %s: %d models, %.1f min total finetune wall-time" % (
                side_b["label"],
                ts["side_b_n_models"],
                ts["side_b_finetune_total_wall_min"],
            )
        )
        lines.append("- Details: `training_stats/per_task.csv`, `training_stats/summary.csv`")
        lines.append("")

    if "affinity" in components:
        s = headline["affinity"]
        lines.append("## affinity")
        for metric in _METRIC_NAMES:
            macro = s["macro_mean_over_alleles"][metric]
            lines.append(
                "- macro %s: A=%.4f, B=%.4f, diff=%+.4f" % (
                    metric, macro["a"], macro["b"], macro["a"] - macro["b"],
                )
            )
        lines.append(
            "- alleles reported: %d (A-better roc_auc: %d, B-better: %d)" % (
                s["n_alleles_reported"],
                s["allele_count"]["a_better_roc_auc"],
                s["allele_count"]["b_better_roc_auc"],
            )
        )
        lines.append("- Details: `affinity/per_allele.csv`, `affinity/summary.json`")
        lines.append("")

    if "presentation" in components:
        s = headline["presentation"]
        lines.append("## presentation")
        for mode in s["modes"]:
            for score_kind in _PRESENTATION_SCORE_KINDS:
                msum = s["summaries"][mode][score_kind]
                pooled_a = msum["micro_pooled"]["a"]["roc_auc"]
                pooled_b = msum["micro_pooled"]["b"]["roc_auc"]
                lines.append(
                    "- %s / %s: micro ROC-AUC A=%.4f, B=%.4f, diff=%+.4f "
                    "(%d samples reported)" % (
                        mode, score_kind, pooled_a, pooled_b,
                        pooled_a - pooled_b, msum["n_samples_reported"],
                    )
                )
        lines.append("- Details: `presentation/summary_table.csv`, `presentation/summary.json`")
        lines.append("")

    with open(os.path.join(out_dir, "summary.md"), "w") as fd:
        fd.write("\n".join(lines))
    _stamp("wrote summary.md")


def _default_data_evaluation_dir() -> Optional[str]:
    from .. import downloads
    try:
        return downloads.get_path("data_evaluation")
    except RuntimeError:
        return None


# Module-level parser for sphinx autoprogram; behaves like the legacy
# ``mhcflurry-*`` command modules.
parser = make_parser()

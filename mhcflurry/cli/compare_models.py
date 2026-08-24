# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
* ``processing`` — per-sample + per-length metrics on the multiallelic
  hit/decoy benchmark for the requested processing flank variants.
* ``presentation`` — per-sample + per-length micro/macro metrics on
  the multiallelic hit/decoy benchmark, with-flanks and without-flanks.

Writes detailed CSV/JSON artifacts plus release-summary CSV/Markdown tables.
``mhcflurry plot-model-comparison`` consumes the CSVs to render plots.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import shutil
import time
import warnings
from functools import partial
from typing import Optional

import numpy
import pandas
from sklearn.metrics import average_precision_score, roc_auc_score

from ..common import (
    normalize_class1_genotype,
    normalize_allele_name,
    normalize_sequence_resolved_allele_name,
    positive_int_arg,
)
from .model_comparison_constants import (
    METRIC_NAMES,
    PRESENTATION_MODES,
    PRESENTATION_SCORE_KINDS,
    PROCESSING_MODES,
)
from ..parallelism import (
    add_prediction_parallelism_args,
    call_wrapped_kwargs,
    chunk_ranges_for_local_parallelism,
    worker_pool_with_gpu_assignments_from_args,
)
from ..pytorch_sizing import default_prediction_batch_is_auto
from ..pseudosequences import LEGACY_ALLELE_SEQUENCES_FILENAME
from ..release_holdout import canonical_allele_mapping, load_excluded_samples
from ..workload_planning import (
    WORKLOAD_AFFINITY_INFERENCE,
    WORKLOAD_PROCESSING_INFERENCE,
    WORKLOAD_PRESENTATION_INFERENCE,
    model_artifact_size_bytes,
)


METRIC_SCORE_KINDS = PRESENTATION_SCORE_KINDS + ("processing_score",)

_COMPONENT_NAMES = ("training_stats", "affinity", "processing", "presentation")

_T0 = time.time()


def _stamp(msg):
    print("[+%7.1fs] %s" % (time.time() - _T0, msg), flush=True)


def _num_jobs_override_arg(value):
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected 'auto' or an integer >= 0, got %r" % value
        )
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "expected 'auto' or an integer >= 0, got %r" % value
        )
    return parsed


def _max_workers_per_gpu_override_arg(value):
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected 'auto' or an integer >= 1, got %r" % value
        )
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "expected 'auto' or an integer >= 1, got %r" % value
        )
    return parsed


def _parallelism_args_for_component(args, component):
    component_args = argparse.Namespace(**vars(args))
    component_args._local_parallelism_args_resolved = False
    if hasattr(component_args, "workload_plan"):
        del component_args.workload_plan

    if component != "presentation":
        return component_args

    overrides = {
        "num_jobs": args.presentation_num_jobs,
        "max_workers_per_gpu": args.presentation_max_workers_per_gpu,
        "max_tasks_per_worker": args.presentation_max_tasks_per_worker,
        "torch_compile": args.presentation_torch_compile,
    }
    overrides = {
        name: value for (name, value) in overrides.items()
        if value is not None
    }
    for name, value in overrides.items():
        setattr(component_args, name, value)
    return component_args


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
            "Comma-separated subset of {training_stats, affinity, processing, "
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
        "--release-holdout-dir",
        help=(
            "Release holdout manifest directory. When specified, affinity, "
            "processing, and presentation benchmarks are restricted to their "
            "frozen evaluation sample manifests."
        ),
    )
    parser.add_argument(
        "--limit-files", type=positive_int_arg, default=None,
        help="Smoke-test: only read first N benchmark files.",
    )
    parser.add_argument(
        "--affinity-source",
        choices=["mixmhcpred", "netmhcpan4", "no_additional_ms", "both"],
        default="mixmhcpred",
        help=(
            "Which monoallelic benchmark source to use for affinity eval. "
            "The no_additional_ms source is train-excluded for the matching "
            "models_class1_pan_variants/models.no_additional_ms predictor."
        ),
    )
    parser.add_argument(
        "--affinity-training-overlap-policy",
        choices=["exclude", "audit"],
        default="exclude",
        help=(
            "For frozen release affinity evaluation, either exclude the union "
            "of both predictors' recorded training pMHCs or audit/report that "
            "overlap without changing the score set."
        ),
    )
    parser.add_argument(
        "--processing-modes",
        default=",".join(PROCESSING_MODES),
        help=(
            "Comma-separated subset of {with_flanks, no_flank, short_flanks} "
            "for the processing component."
        ),
    )
    parser.add_argument(
        "--presentation-modes",
        default=",".join(PRESENTATION_MODES),
        help=(
            "Comma-separated subset of {with_flanks, without_flanks} for "
            "the presentation component."
        ),
    )
    add_prediction_parallelism_args(parser)
    parser.add_argument(
        "--presentation-num-jobs",
        type=_num_jobs_override_arg,
        default=None,
        help=(
            "Override --num-jobs for presentation inference. This is useful "
            "because presentation prediction has a larger per-worker GPU "
            "footprint than affinity or processing."
        ),
    )
    parser.add_argument(
        "--presentation-max-workers-per-gpu",
        type=_max_workers_per_gpu_override_arg,
        default=None,
        help="Override --max-workers-per-gpu for presentation inference.",
    )
    parser.add_argument(
        "--presentation-max-tasks-per-worker",
        type=positive_int_arg,
        default=None,
        help="Override --max-tasks-per-worker for presentation inference.",
    )
    parser.add_argument(
        "--presentation-torch-compile",
        choices=("auto", "0", "1"),
        default=None,
        help="Override --torch-compile for presentation inference.",
    )
    return parser


def run(args):
    _validate_comparison_output_location(args)
    side_a = _resolve_side("a", args.a, args.a_label, args)
    side_b = _resolve_side("b", args.b, args.b_label, args)
    components = _resolve_components(args.include, side_a, side_b)
    _validate_component_configuration(args, components, side_a, side_b)

    # Resolve and validate every requested input before invalidating a previous
    # comparison in this output directory. This keeps a typo or an incomplete
    # candidate artifact from destroying the last usable review packet.
    _reset_comparison_outputs(args.out)
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "side_a.json"), "w") as fd:
        json.dump(_side_to_json(side_a), fd, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "side_b.json"), "w") as fd:
        json.dump(_side_to_json(side_b), fd, indent=2, sort_keys=True)

    _stamp("running components: %s" % (", ".join(components) or "(none)"))

    headline = {"side_a": side_a["label"], "side_b": side_b["label"]}

    if "training_stats" in components:
        headline["training_stats"] = _run_training_stats(side_a, side_b, args.out)
    if "affinity" in components:
        headline["affinity"] = _run_affinity(side_a, side_b, args)
    if "processing" in components:
        headline["processing"] = _run_processing(side_a, side_b, args)
    if "presentation" in components:
        headline["presentation"] = _run_presentation(side_a, side_b, args)

    _write_release_summary_tables(headline, side_a, side_b, args.out, components)
    _write_summary_markdown(headline, side_a, side_b, args.out, components)
    return 0


def _validate_comparison_output_location(args):
    """Refuse output paths that contain model or benchmark inputs."""
    out = os.path.realpath(args.out)
    inputs = [("--data-dir", args.data_dir)] if args.data_dir else []
    for letter in ("a", "b"):
        spec = getattr(args, letter)
        if isinstance(spec, str) and not (
                spec == "public" or spec.startswith("public:")):
            inputs.append(("--%s" % letter, spec))
        for role in ("affinity", "processing", "presentation", "training"):
            value = getattr(args, "%s_%s_dir" % (letter, role))
            if value:
                inputs.append(("--%s-%s-dir" % (letter, role), value))
    for option, value in inputs:
        path = os.path.realpath(value)
        try:
            output_contains_input = os.path.commonpath([out, path]) == out
        except ValueError:
            output_contains_input = False
        if output_contains_input:
            raise ValueError(
                "Comparison output directory cannot contain an input path: "
                "%s contains %s=%s" % (args.out, option, value))


def _reset_comparison_outputs(out_dir):
    """Remove outputs owned by compare-models before starting a new run.

    A comparison directory is commonly reused for release retries. Leaving a
    component directory or plot packet from the previous invocation can make a
    narrower or failed retry look successful, so all derived outputs are
    invalidated together before new side metadata is written.
    """
    for name in (
            "training_stats", "affinity", "processing", "presentation",
            "plots", "worker_logs"):
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.unlink(path)
    for name in (
            "side_a.json", "side_b.json", "release_summary.csv",
            "release_summary.md", "summary.md", "summary.pdf"):
        path = os.path.join(out_dir, name)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


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
            "models_class1_processing", ""),
        "presentation": ("models_class1_presentation", "models"),
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
            os.path.join(spec, "presentation", "models"),
            os.path.join(spec, "presentation", "models.combined"),
            os.path.join(spec, "presentation"),
            spec,
        ]
        return _first_match(candidates, _looks_like_presentation_dir)
    if role == "processing":
        candidates = [
            os.path.join(spec, "processing"),
            spec,
        ]
        return _first_match(candidates, _looks_like_processing_dir)
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


def _looks_like_processing_dir(path):
    if not os.path.isdir(path):
        return False
    basename = os.path.basename(os.path.normpath(path))
    if basename.startswith("models.selected."):
        return True
    return any(
        os.path.isdir(os.path.join(path, "models.selected.%s" % mode))
        for mode in PROCESSING_MODES
    )


def _side_to_json(side):
    result = {
        "letter": side["letter"],
        "spec": side["spec"],
        "label": side["label"],
        "paths": side["paths"],
        "model_package_versions": _model_package_versions(side["paths"]),
    }
    provenance_path = os.path.join(
        side["spec"], "release_provenance.json") if os.path.isdir(
            side["spec"]) else None
    if provenance_path and os.path.isfile(provenance_path):
        with open(provenance_path) as fd:
            result["release_provenance"] = json.load(fd)
    return result


def _model_package_versions(paths):
    """Return package versions recorded by each resolved model artifact."""
    info_paths = {}
    for role in ("affinity", "presentation"):
        root = paths.get(role)
        if root:
            info_paths[role] = [os.path.join(root, "info.txt")]
    processing = paths.get("processing")
    if processing:
        if os.path.basename(os.path.normpath(processing)).startswith(
                "models.selected."):
            info_paths["processing"] = [os.path.join(processing, "info.txt")]
        else:
            info_paths["processing"] = sorted(glob.glob(os.path.join(
                processing, "models.selected.*", "info.txt")))

    result = {}
    for role, candidates in info_paths.items():
        versions = []
        for path in candidates:
            if not os.path.isfile(path):
                continue
            with open(path) as fd:
                for line in fd:
                    fields = line.split()
                    if fields[:2] == ["package", "mhcflurry"] and len(fields) > 2:
                        versions.append(fields[2])
                        break
        if versions:
            result[role] = sorted(set(versions))
    return result


def _resolve_components(include_arg, side_a, side_b):
    available = []
    if side_a["paths"]["training"] and side_b["paths"]["training"]:
        available.append("training_stats")
    if side_a["paths"]["affinity"] and side_b["paths"]["affinity"]:
        available.append("affinity")
    if side_a["paths"]["processing"] and side_b["paths"]["processing"]:
        available.append("processing")
    if side_a["paths"]["presentation"] and side_b["paths"]["presentation"]:
        available.append("presentation")

    if include_arg == "auto":
        if not available:
            raise SystemExit(
                "No comparable model components are available on both sides."
            )
        return available
    pieces = [p.strip() for p in include_arg.split(",")]
    if any(not piece for piece in pieces):
        raise SystemExit("--include contains an empty component")
    requested = pieces
    if not requested:
        raise SystemExit("--include must name at least one component or be 'auto'")
    bad = [p for p in requested if p not in _COMPONENT_NAMES]
    if bad:
        raise SystemExit("Unknown --include components: %s" % ", ".join(bad))
    duplicates = sorted({p for p in requested if requested.count(p) > 1})
    if duplicates:
        raise SystemExit("Duplicate --include components: %s" % (
            ", ".join(duplicates)))
    missing = [p for p in requested if p not in available]
    if missing:
        raise SystemExit(
            "Requested comparison component(s) are unavailable on both sides: "
            "%s. Every explicit --include component must exist on both the "
            "candidate and baseline." % ", ".join(missing)
        )
    return requested


def _requested_modes(value, allowed, option):
    """Parse and validate a comma-separated comparison-mode option."""
    requested = [mode.strip() for mode in value.split(",")]
    if not requested or any(not mode for mode in requested):
        raise SystemExit("%s contains an empty mode" % option)
    bad = [mode for mode in requested if mode not in allowed]
    if bad:
        raise SystemExit("Unknown %s modes: %s" % (
            option.removeprefix("--"), ", ".join(bad)))
    duplicates = sorted({
        mode for mode in requested if requested.count(mode) > 1
    })
    if duplicates:
        raise SystemExit("Duplicate %s modes: %s" % (
            option.removeprefix("--"), ", ".join(duplicates)))
    return requested


def _processing_model_dirs(side_a, side_b, requested_modes):
    """Resolve requested processing variants or fail with both missing sides."""
    model_dirs = {}
    missing = []
    for mode in requested_modes:
        a_model_dir = _processing_model_dir(
            side_a["paths"]["processing"], mode)
        b_model_dir = _processing_model_dir(
            side_b["paths"]["processing"], mode)
        missing_sides = []
        if not a_model_dir:
            missing_sides.append("A (%s)" % side_a["label"])
        if not b_model_dir:
            missing_sides.append("B (%s)" % side_b["label"])
        if missing_sides:
            missing.append("%s: side %s" % (mode, " and ".join(missing_sides)))
        else:
            model_dirs[mode] = (a_model_dir, b_model_dir)
    if missing:
        raise SystemExit(
            "Requested processing mode model(s) are unavailable: %s. "
            "Every --processing-modes entry must exist on both sides." %
            "; ".join(missing)
        )
    return model_dirs


def _validate_component_configuration(args, components, side_a, side_b):
    """Validate component-specific options before output cleanup begins."""
    if "processing" in components:
        modes = _requested_modes(
            args.processing_modes, PROCESSING_MODES, "--processing-modes")
        _processing_model_dirs(side_a, side_b, modes)
    if "presentation" in components:
        _requested_modes(
            args.presentation_modes, PRESENTATION_MODES,
            "--presentation-modes")


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
    mask = numpy.isfinite(y_score)
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


def _require_binary_comparison_rows(df, context):
    """Require at least one positive and one negative evaluation row."""
    if "hit" not in df:
        raise ValueError("%s is missing the hit column" % context)
    hits = pandas.to_numeric(df["hit"], errors="coerce")
    valid_binary = numpy.isfinite(hits.values) & hits.isin([0, 1]).values
    if not valid_binary.all():
        raise ValueError(
            "%s contains %d non-binary or missing hit value(s); expected "
            "only 0 and 1" % (context, int((~valid_binary).sum()))
        )
    n_rows = int(len(hits))
    n_pos = int((hits == 1).sum())
    n_neg = int((hits == 0).sum())
    if n_rows == 0 or n_pos == 0 or n_neg == 0:
        raise ValueError(
            "%s has no valid binary comparison set after shared-row "
            "filtering (rows=%d, positives=%d, negatives=%d)" % (
                context, n_rows, n_pos, n_neg
            )
        )


def _require_complete_benchmark_rows(df, columns, context):
    """Reject missing/blank required inputs instead of silently dropping rows."""
    missing_columns = sorted(set(columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "%s missing required column(s): %s" % (
                context, ", ".join(missing_columns))
        )
    failures = []
    for column in columns:
        values = df[column]
        invalid = values.isnull()
        # pandas 3 defaults text columns to ``StringDtype`` rather than
        # ``object``. Normalize every scalar column to its nullable string view
        # so blank detection is independent of the pandas dtype policy.
        invalid |= values.astype("string").str.strip().eq("").fillna(False)
        if invalid.any():
            failures.append("%s=%d" % (column, int(invalid.sum())))
    if failures:
        raise ValueError(
            "%s contains missing or blank required values (%s). Benchmark "
            "rows must not be silently discarded." % (
                context, ", ".join(failures))
        )


def _add_diffs(df, metric_names, a_prefix="a", b_prefix="b"):
    """Add ``<metric>_diff = a - b`` columns in-place and return df."""
    for metric in metric_names:
        a_col = "%s_%s" % (a_prefix, metric)
        b_col = "%s_%s" % (b_prefix, metric)
        if a_col in df.columns and b_col in df.columns:
            df["%s_diff" % metric] = df[a_col] - df[b_col]
    return df


def _metric_table_columns(id_columns, metric_names=METRIC_NAMES):
    columns = list(id_columns) + ["n", "n_pos"]
    for metric in metric_names:
        columns.extend(["a_%s" % metric, "b_%s" % metric, "%s_diff" % metric])
    return columns


def _metric_table(rows, id_columns, metric_names=METRIC_NAMES):
    df = pandas.DataFrame(rows, columns=_metric_table_columns(
        id_columns, metric_names))
    return _add_diffs(df, metric_names)


def _per_length_columns(metric_names=METRIC_NAMES):
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


def _presentation_per_length_columns(metric_names=METRIC_NAMES):
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


def _parallel_affinity_predict(
        args, predictor_dir, peptides, alleles, model_bytes=None):
    if len(peptides) == 0:
        return numpy.asarray([], dtype=float)
    worker_pool = worker_pool_with_gpu_assignments_from_args(
        args,
        workload_name=WORKLOAD_AFFINITY_INFERENCE,
        workload_hints={
            "elastic_batch": default_prediction_batch_is_auto(),
            "model_bytes": (
                model_bytes or model_artifact_size_bytes(predictor_dir)),
            "prediction_rows": len(peptides),
        },
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
        raise ValueError(
            "Affinity predictor is missing its allele sequence registry: %s" %
            path
        )
    df = pandas.read_csv(path)
    if df.empty:
        raise ValueError("Affinity allele sequence registry is empty: %s" % path)
    col = "normalized_allele" if "normalized_allele" in df.columns else df.columns[0]
    sequence_col = next(
        (name for name in df.columns if name != col), None)
    result = set()
    skipped = []
    for _, row in df.iterrows():
        raw = row[col]
        sequence = str(row[sequence_col]) if sequence_col else ""
        normalized = normalize_allele_name(
            raw, raise_on_error=False, use_allele_aliases=False)
        if normalized is None or "X" in sequence:
            alias_normalized = normalize_allele_name(
                raw, raise_on_error=False, use_allele_aliases=True)
            if alias_normalized is not None:
                normalized = alias_normalized
        if normalized is None:
            skipped.append(str(raw))
            continue
        result.add(normalized)
    if skipped:
        _stamp(
            "WARNING: ignored %d pseudosequence registry key(s) that the "
            "predictor API cannot canonicalize: %s%s" % (
                len(skipped),
                ", ".join(skipped[:3]),
                " ..." if len(skipped) > 3 else "",
            )
        )
    if not result:
        raise ValueError(
            "Affinity allele sequence registry has no canonicalizable "
            "class-I alleles: %s" % path
        )
    return result


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


def _affinity_training_data_path(predictor_dir):
    """Return recorded training rows for an affinity predictor, if present."""
    for filename in ("train_data.csv.bz2", "train_data.csv"):
        path = os.path.join(predictor_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def _exclude_affinity_training_overlap(
        test, side_a, side_b, policy="exclude"):
    """Audit, and optionally drop, side A/B training pMHC overlap."""
    if policy not in ("exclude", "audit"):
        raise ValueError("Unknown affinity training-overlap policy: %s" % policy)
    benchmark_index = pandas.MultiIndex.from_frame(test[["hla", "peptide"]])
    union_mask = numpy.zeros(len(test), dtype=bool)
    report = {
        "policy": policy,
        "policy_description": (
            "drop union of side A and side B affinity training pMHCs"
            if policy == "exclude"
            else "audit side A and side B affinity training pMHCs only"
        ),
        "rows_before": int(len(test)),
        "hits_before": int(test.hit.sum()),
        "sides": {},
    }
    for side in (side_a, side_b):
        predictor_dir = side["paths"]["affinity"]
        training_path = _affinity_training_data_path(predictor_dir)
        if training_path is None:
            raise ValueError(
                "Release affinity comparison cannot audit training overlap "
                "for %s: missing train_data.csv[.bz2] in %s" % (
                    side["label"], predictor_dir)
            )
        training = pandas.read_csv(
            training_path, usecols=["allele", "peptide"])
        allele_map = canonical_allele_mapping(training.allele)
        training["allele"] = training.allele.astype(str).map(allele_map)
        training = training.loc[training.allele.notna()]
        training_index = pandas.MultiIndex.from_frame(
            training[["allele", "peptide"]]).unique()
        overlap_mask = benchmark_index.isin(training_index)
        union_mask |= overlap_mask
        report["sides"][side["letter"]] = {
            "label": side["label"],
            "predictor_dir": predictor_dir,
            "training_data": training_path,
            "training_rows": int(len(training)),
            "training_unique_pmhcs": int(len(training_index)),
            "overlap_rows": int(overlap_mask.sum()),
            "overlap_hits": int(test.hit.loc[overlap_mask].sum()),
            "overlap_unique_pmhcs": int(
                benchmark_index[overlap_mask].nunique()),
        }
    retained_mask = ~union_mask if policy == "exclude" else numpy.ones(
        len(test), dtype=bool)
    report.update({
        "exclusion_applied": policy == "exclude",
        "union_overlap_rows": int(union_mask.sum()),
        "union_overlap_hits": int(test.hit.loc[union_mask].sum()),
        "union_overlap_unique_pmhcs": int(
            benchmark_index[union_mask].nunique()),
        "rows_after": int(retained_mask.sum()),
        "hits_after": int(test.hit.loc[retained_mask].sum()),
    })
    _stamp(
        "  release training-overlap %s: %d rows / %d hits overlap; "
        "%d rows / %d hits scored" % (
            policy,
            report["union_overlap_rows"],
            report["union_overlap_hits"],
            report["rows_after"],
            report["hits_after"],
        )
    )
    return test.loc[retained_mask].copy(), report


def _filter_release_holdout_samples(frame, args, component):
    """Restrict a benchmark to its frozen release-evaluation samples."""
    if not args.release_holdout_dir:
        return frame
    filenames = {
        "affinity": "affinity_samples.csv",
        "processing": "processing_samples.csv",
        "presentation": "presentation_samples.csv",
    }
    path = os.path.join(args.release_holdout_dir, filenames[component])
    samples = load_excluded_samples(path)
    sample_ids = frame.sample_id.astype(str)
    result = frame.loc[sample_ids.isin(samples)].copy()
    if result.empty:
        raise ValueError(
            "Release holdout selected no %s benchmark rows using %s" % (
                component, path))
    _stamp(
        "  release holdout %s: %d rows, %d samples" % (
            component, len(result), result.sample_id.nunique()))
    if getattr(args, "limit_files", None):
        selected_files = result.source_file.drop_duplicates().iloc[
            :args.limit_files
        ]
        result = result.loc[result.source_file.isin(selected_files)].copy()
        _stamp(
            "  release holdout %s file limit: %d files, %d rows" % (
                component,
                len(selected_files),
                len(result),
            )
        )
    return result


def _load_presentation_benchmark_for_component(data_dir, args, component):
    """Load, holdout-filter, then file-limit a presentation benchmark."""
    initial_file_limit = None if args.release_holdout_dir else args.limit_files
    row_filter = None
    if args.release_holdout_dir:
        row_filter = partial(
            _filter_release_holdout_samples,
            args=args,
            component=component,
        )
    return _load_presentation_benchmark(
        data_dir,
        initial_file_limit,
        row_filter=row_filter,
    )


def _run_affinity(side_a, side_b, args):
    component_dir = os.path.join(args.out, "affinity")
    os.makedirs(component_dir, exist_ok=True)
    affinity_args = _parallelism_args_for_component(args, "affinity")

    data_dir = args.data_dir or _default_data_evaluation_dir()
    initial_file_limit = None if args.release_holdout_dir else args.limit_files
    test = _load_affinity_benchmark(
        data_dir, args.affinity_source, initial_file_limit)
    test = _filter_release_holdout_samples(test, args, "affinity")
    _require_complete_benchmark_rows(
        test, ("peptide", "hla", "hit"), "Affinity benchmark")
    try:
        test["hla"] = test["hla"].map(
            normalize_sequence_resolved_allele_name)
    except ValueError as error:
        raise ValueError(
            "Affinity benchmark contains an invalid class-I allele: %s" % error
        ) from error

    a_alleles = _read_supported_alleles(side_a["paths"]["affinity"])
    b_alleles = _read_supported_alleles(side_b["paths"]["affinity"])
    both = a_alleles & b_alleles
    if not both:
        raise ValueError(
            "Affinity predictors have no shared sequence-resolved alleles; "
            "a fair comparison cannot be computed."
        )
    before = len(test)
    test = test[test.hla.isin(both)].copy()
    if len(test) < before:
        _stamp(
            "  dropped %d rows outside the %d-allele intersect"
            % (before - len(test), len(both))
        )
    test["peptide_len"] = test["peptide"].str.len()
    test = test[(test.peptide_len >= 8) & (test.peptide_len <= 15)].copy()
    test["hit"] = pandas.to_numeric(test["hit"], errors="coerce")
    _require_binary_comparison_rows(test, "Affinity benchmark")
    training_overlap = None
    if args.release_holdout_dir:
        test, training_overlap = _exclude_affinity_training_overlap(
            test,
            side_a,
            side_b,
            policy=args.affinity_training_overlap_policy,
        )
        _require_binary_comparison_rows(
            test, "Train-excluded release affinity benchmark")
        with open(
                os.path.join(component_dir, "training_overlap.json"),
                "w") as fd:
            json.dump(training_overlap, fd, indent=2, sort_keys=True)
    _stamp("  evaluable rows: %d" % len(test))

    comparison_model_bytes = max(
        model_artifact_size_bytes(side_a["paths"]["affinity"]) or 0,
        model_artifact_size_bytes(side_b["paths"]["affinity"]) or 0,
    ) or None
    _stamp("predicting side A affinity...")
    test["a_pred"] = _parallel_affinity_predict(
        affinity_args, side_a["paths"]["affinity"],
        test.peptide.values, test.hla.values,
        model_bytes=comparison_model_bytes,
    )
    _stamp("predicting side B affinity...")
    test["b_pred"] = _parallel_affinity_predict(
        affinity_args, side_b["paths"]["affinity"],
        test.peptide.values, test.hla.values,
        model_bytes=comparison_model_bytes,
    )
    _require_valid_affinity_predictions(
        test,
        labels=(side_a["label"], side_b["label"]),
    )
    _require_binary_comparison_rows(
        test, "Affinity comparison for %s versus %s" % (
            side_a["label"], side_b["label"]
        ))
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
    if training_overlap is not None:
        summary["training_overlap"] = training_overlap
    with open(os.path.join(component_dir, "summary.json"), "w") as fd:
        json.dump(summary, fd, indent=2, sort_keys=True)
    _stamp("  wrote summary.json")
    return summary


def _affinity_per_allele(test):
    rows = []
    for allele, group in test.groupby("hla"):
        if len(group) < 30 or group.hit.nunique() < 2:
            continue
        m_a = _metrics(group.hit.values, group.a_score.values)
        m_b = _metrics(group.hit.values, group.b_score.values)
        row = {"allele": allele, "n": m_a["n"], "n_pos": m_a["n_pos"]}
        for metric in METRIC_NAMES:
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
                **{"a_%s" % m: ma_a[m] for m in METRIC_NAMES},
                **{"b_%s" % m: ma_b[m] for m in METRIC_NAMES},
            })
        per_allele_rows.extend(per_allele_L)
        row = {
            "length": L,
            "n": m_a_L["n"], "n_pos": m_a_L["n_pos"],
            "n_alleles_reported": len(per_allele_L),
        }
        for metric in METRIC_NAMES:
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
            for metric in METRIC_NAMES
        },
        "allele_count": {
            "a_better_%s" % metric: int((per_allele["%s_diff" % metric] > 0).sum())
            for metric in METRIC_NAMES
        } | {
            "b_better_%s" % metric: int((per_allele["%s_diff" % metric] < 0).sum())
            for metric in METRIC_NAMES
        },
        "per_length": per_length.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Component: processing
# ---------------------------------------------------------------------------


def _processing_model_dir(processing_root, mode):
    """Return the model directory for one processing flank ``mode``."""
    if not processing_root:
        return None
    candidates = [
        os.path.join(processing_root, "models.selected.%s" % mode),
        processing_root,
    ]
    expected_basename = "models.selected.%s" % mode
    for path in candidates:
        if (
                os.path.isdir(path) and
                os.path.basename(os.path.normpath(path)) == expected_basename):
            return path
    return None


def _predict_processing_chunk(predictor_dir, rows, mode, chunk_num):
    """Worker entry: load processing predictor, score one chunk."""
    from .. import Class1ProcessingPredictor
    predictor = Class1ProcessingPredictor.load(predictor_dir)

    df = pandas.DataFrame(rows)
    kwargs = {
        "peptides": df.peptide.values,
        "batch_size": "auto",
        "throw": False,
    }
    if mode in ("with_flanks", "short_flanks"):
        kwargs["n_flanks"] = df.n_flank.values
        kwargs["c_flanks"] = df.c_flank.values
    elif mode != "no_flank":
        raise ValueError("Unexpected processing mode: %s" % mode)
    predictions = numpy.asarray(predictor.predict(**kwargs))
    if len(predictions) != len(df):
        raise ValueError(
            "Predictor returned %d rows for %d inputs" % (
                len(predictions), len(df))
        )
    return chunk_num, predictions


def _parallel_processing_predict(
        args, predictor_dir, df, mode, label, model_bytes=None):
    if len(df) == 0:
        return numpy.asarray([], dtype=float)
    worker_pool = worker_pool_with_gpu_assignments_from_args(
        args,
        workload_name=WORKLOAD_PROCESSING_INFERENCE,
        workload_hints={
            "elastic_batch": True,
            "model_bytes": (
                model_bytes or model_artifact_size_bytes(predictor_dir)),
            "prediction_rows": len(df),
        },
        start_method="spawn",
    )
    _stamp("predicting %s processing (%s, %d rows)" % (label, mode, len(df)))
    if worker_pool is None:
        _, predictions = _predict_processing_chunk(
            predictor_dir, df.to_dict("list"), mode, chunk_num=0)
        return predictions

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
            partial(call_wrapped_kwargs, _predict_processing_chunk),
            work_items,
            chunksize=1,
        )
        chunks = [result for result in results]
        worker_pool.close()
        worker_pool.join()
        worker_pool = None
    finally:
        if worker_pool is not None:
            worker_pool.terminate()
            worker_pool.join()
    return numpy.concatenate([
        values for (_, values) in sorted(chunks, key=lambda t: t[0])
    ])


def _run_processing(side_a, side_b, args):
    requested_modes = _requested_modes(
        args.processing_modes, PROCESSING_MODES, "--processing-modes")
    model_dirs = _processing_model_dirs(side_a, side_b, requested_modes)

    component_dir = os.path.join(args.out, "processing")
    os.makedirs(component_dir, exist_ok=True)
    processing_args = _parallelism_args_for_component(args, "processing")
    data_dir = args.data_dir or _default_data_evaluation_dir()
    benchmark = _load_presentation_benchmark_for_component(
        data_dir, args, "processing")
    summaries = {}
    summary_rows = []
    for mode in requested_modes:
        a_model_dir, b_model_dir = model_dirs[mode]
        _stamp("=== processing mode: %s ===" % mode)
        scored = benchmark[[
            "peptide", "sample_id", "hla", "hit", "peptide_len",
            "n_flank", "c_flank",
        ]].copy()
        comparison_model_bytes = max(
            model_artifact_size_bytes(a_model_dir) or 0,
            model_artifact_size_bytes(b_model_dir) or 0,
        ) or None
        scored["a_processing_score"] = _parallel_processing_predict(
            processing_args, a_model_dir, benchmark, mode, label="A",
            model_bytes=comparison_model_bytes)
        scored["b_processing_score"] = _parallel_processing_predict(
            processing_args, b_model_dir, benchmark, mode, label="B",
            model_bytes=comparison_model_bytes)

        _require_finite_processing_scores(
            scored,
            mode=mode,
            labels=(side_a["label"], side_b["label"]),
        )

        pred_path = os.path.join(
            component_dir, "predictions_%s.csv.bz2" % mode)
        scored.to_csv(pred_path, index=False)
        _stamp("  wrote %s" % pred_path)

        shared_scored = _shared_score_rows(scored, "processing_score")
        per_sample = _presentation_per_sample(shared_scored, "processing_score")
        per_sample.to_csv(
            os.path.join(
                component_dir,
                "per_sample_%s_processing_score.csv" % mode,
            ),
            index=False,
        )
        per_length, per_length_per_sample = _presentation_per_length(
            shared_scored, "processing_score")
        per_length.to_csv(
            os.path.join(
                component_dir,
                "per_length_%s_processing_score.csv" % mode,
            ),
            index=False,
        )
        if not per_length_per_sample.empty:
            per_length_per_sample.to_csv(
                os.path.join(
                    component_dir,
                    "per_length_per_sample_%s_processing_score.csv" % mode,
                ),
                index=False,
            )
        summary = _presentation_mode_summary(
            shared_scored, per_sample, per_length, mode, "processing_score")
        summaries[mode] = {"processing_score": summary}
        summary_rows.append(_presentation_summary_row(summary))

    with open(os.path.join(component_dir, "summary.json"), "w") as fd:
        json.dump(summaries, fd, indent=2, sort_keys=True)
    summary_table = pandas.DataFrame(
        summary_rows, columns=_component_summary_table_columns())
    summary_table.to_csv(
        os.path.join(component_dir, "summary_table.csv"), index=False)
    _stamp("  wrote processing summary.json + summary_table.csv")
    return {
        "modes": [row["mode"] for row in summary_rows],
        "summaries": summaries,
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


def _parallel_presentation_predict(
        args, predictor_dir, df, mode, label, model_bytes=None):
    if len(df) == 0:
        return pandas.DataFrame()
    worker_pool = worker_pool_with_gpu_assignments_from_args(
        args,
        workload_name=WORKLOAD_PRESENTATION_INFERENCE,
        workload_hints={
            "elastic_batch": default_prediction_batch_is_auto(),
            "model_bytes": (
                model_bytes or model_artifact_size_bytes(predictor_dir)),
            "prediction_rows": len(df),
        },
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


def _load_presentation_benchmark(data_dir, limit_files, row_filter=None):
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
    _require_complete_benchmark_rows(
        result,
        ("peptide", "sample_id", "hla", "hit"),
        "Presentation benchmark",
    )
    result = result.copy()
    result["hit"] = pandas.to_numeric(result["hit"], errors="coerce")
    _require_binary_comparison_rows(result, "Presentation benchmark")
    if row_filter is not None:
        # Genotype normalization dominates runtime for the full benchmark.
        # Release evaluation only needs its frozen samples, so select those
        # rows after whole-input integrity checks but before parsing genotypes.
        result = row_filter(result)
    result["hla"] = result["hla"].map(_normalize_benchmark_genotype)
    genotype_counts = result.groupby("sample_id", dropna=False).hla.nunique()
    inconsistent_samples = genotype_counts[genotype_counts > 1]
    if not inconsistent_samples.empty:
        examples = ", ".join(
            "%s (%d genotypes)" % (sample_id, count)
            for sample_id, count in inconsistent_samples.head(3).items()
        )
        raise ValueError(
            "Presentation benchmark maps %d sample_id value(s) to multiple "
            "HLA genotypes; examples: %s" % (
                len(inconsistent_samples), examples)
        )
    result["hit"] = result["hit"].astype(int)
    result["peptide_len"] = result.peptide.str.len()
    result = result[
        (result.peptide_len >= 8) & (result.peptide_len <= 15)
    ].reset_index(drop=True)
    _require_binary_comparison_rows(
        result, "Presentation benchmark after peptide-length filtering")
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


def _normalize_benchmark_genotype(value):
    """Canonicalize a whitespace-delimited class-I genotype deterministically."""
    try:
        normalized = sorted(normalize_class1_genotype(value))
    except ValueError as exc:
        raise ValueError(
            "Invalid presentation benchmark HLA genotype %r: %s" % (
                value, exc)
        ) from exc
    return " ".join(normalized)


def _score_values(df, prefix, score_kind):
    """Higher = better for the score we feed sklearn.

    ``presentation_score`` and ``processing_score`` are already higher-better;
    ``presentation_percentile`` is lower-better so we negate it.
    """
    if score_kind == "presentation_score":
        return df["%s_presentation_score" % prefix].values
    if score_kind == "presentation_percentile":
        return -df["%s_presentation_percentile" % prefix].values
    if score_kind == "processing_score":
        return df["%s_processing_score" % prefix].values
    raise ValueError("Unknown score kind: %s" % score_kind)


def _score_pair_columns(score_kind):
    if score_kind not in METRIC_SCORE_KINDS:
        raise ValueError("Unknown score kind: %s" % score_kind)
    return ("a_%s" % score_kind, "b_%s" % score_kind)


def _shared_score_rows(scored, score_kind):
    """Return rows where both sides have a score for ``score_kind``.

    A/B release metrics are only meaningful when both sides are scored on the
    same examples. Predictors run with ``throw=False`` can emit non-finite
    values for unsupported peptides, so filter on both side-specific score
    columns before computing support counts or metric differences.
    """
    columns = list(_score_pair_columns(score_kind))
    finite = numpy.ones(len(scored), dtype=bool)
    for column in columns:
        finite &= numpy.isfinite(pandas.to_numeric(
            scored[column], errors="coerce").values)
    return scored.loc[finite].copy()


def _require_finite_processing_scores(scored, mode, labels):
    """Fail a release comparison when either model misses benchmark rows."""
    failures = []
    for side, label in zip(("a", "b"), labels):
        column = "%s_processing_score" % side
        values = pandas.to_numeric(scored[column], errors="coerce").values
        bad = ~numpy.isfinite(values)
        if not bad.any():
            continue
        examples = scored.loc[bad, "peptide"].astype(str).head(3).tolist()
        failures.append(
            "%s (%s): %d non-finite score(s), examples: %s" % (
                side.upper(), label, int(bad.sum()), ", ".join(examples)
            )
        )
    if failures:
        raise ValueError(
            "Processing comparison mode %s requires every benchmark peptide "
            "to be scored by both models. %s. Check peptide validity and each "
            "model's supported peptide-length range." % (
                mode, "; ".join(failures)
            )
        )


def _require_valid_affinity_predictions(scored, labels):
    """Fail when either affinity model omits or corrupts a benchmark row."""
    failures = []
    for side, label in zip(("a", "b"), labels):
        column = "%s_pred" % side
        values = pandas.to_numeric(scored[column], errors="coerce").values
        bad = ~numpy.isfinite(values) | (values <= 0)
        if not bad.any():
            continue
        examples = [
            "%s:%s" % (row.hla, row.peptide)
            for row in scored.loc[bad, ["hla", "peptide"]].head(3).itertuples()
        ]
        failures.append(
            "%s (%s): %d invalid IC50 prediction(s), examples: %s" % (
                side.upper(), label, int(bad.sum()), ", ".join(examples)
            )
        )
    if failures:
        raise ValueError(
            "Affinity comparison requires a finite, positive IC50 prediction "
            "for every benchmark row on both sides. %s. Check allele support, "
            "peptide validity, and model output integrity." % "; ".join(failures)
        )


def _require_finite_presentation_scores(scored, mode, labels):
    """Fail when either presentation model omits a release benchmark row."""
    failures = []
    for side, label in zip(("a", "b"), labels):
        for score_kind in PRESENTATION_SCORE_KINDS:
            column = "%s_%s" % (side, score_kind)
            values = pandas.to_numeric(scored[column], errors="coerce").values
            bad = ~numpy.isfinite(values)
            if not bad.any():
                continue
            examples = scored.loc[bad, "peptide"].astype(str).head(3).tolist()
            failures.append(
                "%s (%s) %s: %d non-finite score(s), examples: %s" % (
                    side.upper(), label, score_kind, int(bad.sum()),
                    ", ".join(examples),
                )
            )
    if failures:
        raise ValueError(
            "Presentation comparison mode %s requires every benchmark row to "
            "have both a presentation score and percentile on both sides. %s. "
            "Check allele support, peptide validity, and percentile "
            "calibration." % (mode, "; ".join(failures))
        )


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
        for metric in METRIC_NAMES:
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
        for metric in METRIC_NAMES:
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
            for metric in METRIC_NAMES
        },
        "sample_count": {
            "a_better_%s" % m: int((per_sample["%s_diff" % m] > 0).sum())
            for m in METRIC_NAMES
        } | {
            "b_better_%s" % m: int((per_sample["%s_diff" % m] < 0).sum())
            for m in METRIC_NAMES
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
    for metric in METRIC_NAMES:
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


def _component_summary_table_columns():
    columns = ["mode", "score_kind", "n_rows", "n_hits", "n_samples_reported"]
    for metric in METRIC_NAMES:
        columns.extend([
            "a_micro_%s" % metric,
            "b_micro_%s" % metric,
            "micro_%s_diff" % metric,
            "a_macro_%s" % metric,
            "b_macro_%s" % metric,
            "macro_%s_diff" % metric,
        ])
    return columns


def _run_presentation(side_a, side_b, args):
    component_dir = os.path.join(args.out, "presentation")
    os.makedirs(component_dir, exist_ok=True)

    data_dir = args.data_dir or _default_data_evaluation_dir()
    requested_modes = _requested_modes(
        args.presentation_modes, PRESENTATION_MODES, "--presentation-modes")

    benchmark = _load_presentation_benchmark_for_component(
        data_dir, args, "presentation")
    summaries = {}
    summary_rows = []
    presentation_args = _parallelism_args_for_component(args, "presentation")
    comparison_model_bytes = max(
        model_artifact_size_bytes(side_a["paths"]["presentation"]) or 0,
        model_artifact_size_bytes(side_b["paths"]["presentation"]) or 0,
    ) or None
    for mode in requested_modes:
        _stamp("=== presentation mode: %s ===" % mode)
        scored = benchmark.copy()
        a_pred = _parallel_presentation_predict(
            presentation_args, side_a["paths"]["presentation"],
            benchmark, mode, label="A", model_bytes=comparison_model_bytes,
        )
        b_pred = _parallel_presentation_predict(
            presentation_args, side_b["paths"]["presentation"],
            benchmark, mode, label="B", model_bytes=comparison_model_bytes,
        )
        for prefix, pred in (("a", a_pred), ("b", b_pred)):
            for col in [
                "presentation_score", "presentation_percentile",
                "affinity", "processing_score",
            ]:
                scored["%s_%s" % (prefix, col)] = pred[col].values
        _require_finite_presentation_scores(
            scored,
            mode=mode,
            labels=(side_a["label"], side_b["label"]),
        )
        pred_path = os.path.join(
            component_dir, "predictions_%s.csv.bz2" % mode)
        scored.to_csv(pred_path, index=False)
        _stamp("  wrote %s" % pred_path)

        summaries[mode] = {}
        for score_kind in PRESENTATION_SCORE_KINDS:
            shared_scored = _shared_score_rows(scored, score_kind)
            _require_binary_comparison_rows(
                shared_scored,
                "Presentation comparison mode %s score %s" % (
                    mode, score_kind),
            )
            per_sample = _presentation_per_sample(shared_scored, score_kind)
            per_sample.to_csv(
                os.path.join(
                    component_dir,
                    "per_sample_%s_%s.csv" % (mode, score_kind),
                ),
                index=False,
            )
            per_length, per_length_per_sample = _presentation_per_length(
                shared_scored, score_kind)
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
                shared_scored, per_sample, per_length, mode, score_kind)
            summaries[mode][score_kind] = summary
            summary_rows.append(_presentation_summary_row(summary))

    with open(os.path.join(component_dir, "summary.json"), "w") as fd:
        json.dump(summaries, fd, indent=2, sort_keys=True)
    summary_table = pandas.DataFrame(
        summary_rows, columns=_component_summary_table_columns())
    summary_table.to_csv(
        os.path.join(component_dir, "summary_table.csv"), index=False)
    _stamp("  wrote summary.json + summary_table.csv")
    return {
        "modes": requested_modes,
        "summaries": summaries,
    }


# ---------------------------------------------------------------------------
# Release-gate summary tables
# ---------------------------------------------------------------------------


_METRIC_DISPLAY_NAMES = {
    "roc_auc": "AUROC",
    "pr_auc": "AUPRC",
    "ppv_at_n": "PPV@N",
}


def _pct_change(a_value, b_value):
    if b_value is None or numpy.isnan(b_value) or b_value == 0:
        return numpy.nan
    return 100.0 * (a_value - b_value) / b_value


def _append_release_metric_row(rows, component, group_key, group_value,
                               metric, average, a_value, b_value):
    rows.append({
        "component": component,
        group_key: group_value,
        "metric": _METRIC_DISPLAY_NAMES[metric],
        "average": average,
        "side_a": a_value,
        "side_b": b_value,
        "diff": a_value - b_value,
        "pct_change": _pct_change(a_value, b_value),
    })


def _release_summary_rows(headline, components):
    rows = []
    if "affinity" in components and "affinity" in headline:
        summary = headline["affinity"]
        for metric in METRIC_NAMES:
            macro = summary["macro_mean_over_alleles"][metric]
            _append_release_metric_row(
                rows, "affinity", "eval", "affinity",
                metric, "Macro", macro["a"], macro["b"])
        for metric in METRIC_NAMES:
            micro_a = summary["micro_pooled"]["a"][metric]
            micro_b = summary["micro_pooled"]["b"][metric]
            _append_release_metric_row(
                rows, "affinity", "eval", "affinity",
                metric, "Micro", micro_a, micro_b)

    for component, score_kind in (
            ("processing", "processing_score"),
            ("presentation", "presentation_score")):
        if component not in components or component not in headline:
            continue
        for mode in headline[component]["modes"]:
            summary = headline[component]["summaries"].get(mode, {}).get(score_kind)
            if summary is None:
                continue
            for metric in METRIC_NAMES:
                macro = summary["macro_mean_over_samples"][metric]
                _append_release_metric_row(
                    rows, component, "flank_mode", mode,
                    metric, "Macro", macro["a"], macro["b"])
            for metric in METRIC_NAMES:
                micro_a = summary["micro_pooled"]["a"][metric]
                micro_b = summary["micro_pooled"]["b"][metric]
                _append_release_metric_row(
                    rows, component, "flank_mode", mode,
                    metric, "Micro", micro_a, micro_b)
    return rows


def _format_metric(value, signed=False, pct=False):
    if value is None or numpy.isnan(value):
        return "nan"
    if pct:
        return "%+.2f%%" % value
    if signed:
        return "%+.4f" % value
    return "%.4f" % value


def _label_heading(label):
    return str(label).replace("_", " ").title()


def _write_release_summary_tables(headline, side_a, side_b, out_dir, components):
    rows = _release_summary_rows(headline, components)
    if not rows:
        return
    csv_path = os.path.join(out_dir, "release_summary.csv")
    pandas.DataFrame(rows).to_csv(csv_path, index=False)

    side_a_heading = _label_heading(side_a["label"])
    side_b_heading = _label_heading(side_b["label"])
    lines = ["# release summary", ""]
    for component, title, group_key, group_label in (
            ("affinity", "Affinity", "eval", "Eval"),
            ("presentation", "Presentation", "flank_mode", "Flank mode"),
            ("processing", "Processing", "flank_mode", "Flank mode")):
        component_rows = [row for row in rows if row["component"] == component]
        if not component_rows:
            continue
        lines.extend([
            "## %s" % title,
            "",
            "| %s | Metric | Average | %s | %s | Diff | %% change |"
            % (group_label, side_a_heading, side_b_heading),
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in component_rows:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s |" % (
                    row.get(group_key, ""),
                    row["metric"],
                    row["average"],
                    _format_metric(row["side_a"]),
                    _format_metric(row["side_b"]),
                    _format_metric(row["diff"], signed=True),
                    _format_metric(row["pct_change"], pct=True),
                )
            )
        lines.append("")

    with open(os.path.join(out_dir, "release_summary.md"), "w") as fd:
        fd.write("\n".join(lines))
    _stamp("wrote release_summary.csv + release_summary.md")


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
        for metric in METRIC_NAMES:
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
        overlap = s.get("training_overlap")
        if overlap:
            lines.append(
                "- training-overlap policy `%s`: %d rows / %d hits overlap; "
                "%d rows / %d hits scored" % (
                    overlap["policy"],
                    overlap["union_overlap_rows"],
                    overlap["union_overlap_hits"],
                    overlap["rows_after"],
                    overlap["hits_after"],
                )
            )
        lines.append("- Details: `affinity/per_allele.csv`, `affinity/summary.json`")
        lines.append("")

    if "processing" in components:
        s = headline["processing"]
        lines.append("## processing")
        for mode in s["modes"]:
            msum = s["summaries"][mode]["processing_score"]
            pooled_a = msum["micro_pooled"]["a"]["roc_auc"]
            pooled_b = msum["micro_pooled"]["b"]["roc_auc"]
            lines.append(
                "- %s / processing_score: micro ROC-AUC A=%.4f, "
                "B=%.4f, diff=%+.4f (%d samples reported)" % (
                    mode, pooled_a, pooled_b, pooled_a - pooled_b,
                    msum["n_samples_reported"],
                )
            )
        lines.append("- Details: `processing/summary_table.csv`, `processing/summary.json`")
        lines.append("")

    if "presentation" in components:
        s = headline["presentation"]
        lines.append("## presentation")
        for mode in s["modes"]:
            for score_kind in PRESENTATION_SCORE_KINDS:
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

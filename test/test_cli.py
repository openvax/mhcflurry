"""Tests for the unified ``mhcflurry`` CLI.

Covers dispatch, side resolution, training_stats end-to-end on
synthetic manifests, and presence of help text. Predict-running paths
(affinity + presentation) need real models on disk and live in
integration suites, not here.
"""

import argparse
import json

import pandas
import pytest

from mhcflurry.cli import compare_models, main as cli_main
from mhcflurry.cli import plot_model_comparison


def test_top_level_parser_lists_subcommands():
    parser = cli_main.build_parser()
    help_text = parser.format_help()
    assert "compare-models" in help_text
    assert "plot-model-comparison" in help_text


def test_compare_models_help_runs(capsys):
    """The compare-models help text exposes the documented flags.

    Goes through ``main()`` rather than ``build_parser`` because subparsers
    are name-only (lazy import); per-subcommand args are only built when
    the legacy module is actually invoked.
    """
    with pytest.raises(SystemExit):
        cli_main.main(["compare-models", "--help"])
    captured = capsys.readouterr().out
    for flag in ["--a", "--b", "--include", "--out", "--data-dir",
                 "--num-jobs", "--gpus", "--max-workers-per-gpu",
                 "--presentation-modes"]:
        assert flag in captured, "missing flag in help: %s" % flag


def test_plot_help_runs(capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["plot-model-comparison", "--help"])
    captured = capsys.readouterr().out
    assert "--input" in captured


def test_unknown_subcommand_errors():
    parser = cli_main.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["does-not-exist"])


def test_all_legacy_commands_registered():
    """All 12 legacy mhcflurry-* commands are reachable as subcommands."""
    expected = {
        "predict", "predict-scan", "downloads",
        "calibrate-percentile-ranks",
        "class1-train-allele-specific-models",
        "class1-select-allele-specific-models",
        "class1-train-pan-allele-models",
        "class1-select-pan-allele-models",
        "class1-train-processing-models",
        "class1-select-processing-models",
        "class1-train-presentation-models",
        "pseudosequences",
    }
    assert expected.issubset(set(cli_main._SUBCOMMANDS))


def test_main_help_does_not_import_predict_command():
    """``mhcflurry --help`` must not pay the cost of importing every
    legacy command module. Lazy-import is the whole reason build_parser
    only registers subcommand names."""
    import subprocess
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, "-c",
         "import sys; from mhcflurry.cli.main import build_parser; "
         "build_parser(); "
         "print(int('mhcflurry.predict_command' in sys.modules))"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "0", (
        "predict_command was imported by build_parser(); should be lazy: %s"
        % result.stdout
    )


def test_main_dispatches_pseudosequences_list(capsys):
    """End-to-end: ``mhcflurry pseudosequences list`` runs the legacy
    module's main(argv) and prints the registry rows."""
    cli_main.main(["pseudosequences", "list"])
    out = capsys.readouterr().out
    assert "netmhcpan" in out
    assert "pseudosequences.mhcflurry.39aa.csv" in out


def test_main_unknown_subcommand_exits():
    with pytest.raises(SystemExit):
        cli_main.main(["does-not-exist"])


# ---------------------------------------------------------------------------
# Side resolution
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    defaults = dict(
        a=None, b=None, a_label=None, b_label=None, out=None,
        include="auto", data_dir=None, limit_files=None,
        affinity_source="mixmhcpred",
        presentation_modes="with_flanks,without_flanks",
    )
    for letter in ("a", "b"):
        for role in ("affinity", "processing", "presentation", "training"):
            defaults["%s_%s_dir" % (letter, role)] = None
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_side_label_defaults_to_basename(tmp_path):
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    side = compare_models._resolve_side(
        "a", str(run_dir), label=None, args=_make_args())
    assert side["label"] == "my_run"
    assert side["spec"] == str(run_dir)


def test_resolve_side_public_label():
    side = compare_models._resolve_side(
        "b", "public", label=None, args=_make_args())
    assert side["label"] == "public"


def test_resolve_side_public_pinned_release_label():
    side = compare_models._resolve_side(
        "b", "public:4-pre-2.2.0", label=None, args=_make_args())
    assert side["label"] == "public:4-pre-2.2.0"


def test_resolve_side_explicit_label_overrides_default():
    side = compare_models._resolve_side(
        "a", "public", label="baseline", args=_make_args())
    assert side["label"] == "baseline"


def test_resolve_side_publicy_path_is_not_public_sentinel(tmp_path):
    """A user-named directory like ``public_data/`` must not be mistaken
    for the public-install sentinel."""
    run_dir = tmp_path / "public_data"
    run_dir.mkdir()
    side = compare_models._resolve_side(
        "a", str(run_dir), label=None, args=_make_args())
    # Label derives from basename, not the literal "public" sentinel.
    assert side["label"] == "public_data"
    # And no role paths were resolved through the public-download lookup.
    for role, path in side["paths"].items():
        if path is not None:
            assert path.startswith(str(tmp_path)), (role, path)


def test_resolve_side_override_paths_win(tmp_path):
    override = tmp_path / "overridden_affinity"
    override.mkdir()
    args = _make_args(a_affinity_dir=str(override))
    side = compare_models._resolve_side("a", "public", None, args)
    assert side["paths"]["affinity"] == str(override)


def test_probe_run_dir_finds_training_via_manifest(tmp_path):
    """A run dir with manifest.csv inside models.unselected.combined is
    picked up for the training role even when nested under affinity/."""
    target = tmp_path / "run" / "affinity" / "models.unselected.combined"
    target.mkdir(parents=True)
    (target / "manifest.csv").write_text(
        "model_name,config_json\nmodel_a,\"{}\"\n")
    resolved = compare_models._probe_run_dir(
        str(tmp_path / "run"), "training")
    assert resolved == str(target)


def test_probe_run_dir_finds_affinity_via_allele_sequences(tmp_path):
    target = tmp_path / "run" / "affinity" / "models.combined"
    target.mkdir(parents=True)
    # The presence of allele_sequences.csv is one of the affinity probes.
    (target / "allele_sequences.csv").write_text("allele,sequence\n")
    resolved = compare_models._probe_run_dir(
        str(tmp_path / "run"), "affinity")
    assert resolved == str(target)


def test_resolve_components_auto_picks_available(tmp_path):
    a_train = tmp_path / "a_train"
    a_train.mkdir()
    b_train = tmp_path / "b_train"
    b_train.mkdir()
    side_a = {
        "label": "a", "letter": "a", "spec": "a",
        "paths": {"training": str(a_train), "affinity": None,
                  "presentation": None, "processing": None},
    }
    side_b = {
        "label": "b", "letter": "b", "spec": "b",
        "paths": {"training": str(b_train), "affinity": None,
                  "presentation": None, "processing": None},
    }
    components = compare_models._resolve_components("auto", side_a, side_b)
    assert components == ["training_stats"]


def test_resolve_components_explicit_drops_unavailable(capsys):
    side_a = {
        "label": "a", "letter": "a", "spec": "a",
        "paths": {"training": "/tmp/x", "affinity": None,
                  "presentation": None, "processing": None},
    }
    side_b = {
        "label": "b", "letter": "b", "spec": "b",
        "paths": {"training": "/tmp/y", "affinity": None,
                  "presentation": None, "processing": None},
    }
    # affinity requested but unavailable -> dropped with warning.
    components = compare_models._resolve_components(
        "affinity,training_stats", side_a, side_b)
    assert "training_stats" in components
    assert "affinity" not in components


def test_resolve_components_bad_name_raises():
    side = {"label": "a", "letter": "a", "spec": "a",
            "paths": {k: None for k in
                      ("training", "affinity", "processing", "presentation")}}
    with pytest.raises(SystemExit):
        compare_models._resolve_components("nope", side, side)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def test_metrics_handles_only_positives_or_only_negatives():
    only_pos = compare_models._metrics([1, 1, 1], [0.1, 0.2, 0.3])
    assert pandas.isna(only_pos["roc_auc"])
    only_neg = compare_models._metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert pandas.isna(only_neg["roc_auc"])


def test_metrics_ppv_at_n_basic():
    # 4 hits, 4 misses. Scores rank hits first → PPV@4 = 1.0
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4]
    m = compare_models._metrics(y, s)
    assert m["ppv_at_n"] == 1.0
    assert m["roc_auc"] == 1.0


def test_metrics_ignores_nans_in_scores():
    import numpy as np
    y = [1, 1, 0, 0]
    s = [0.9, np.nan, 0.1, np.nan]
    m = compare_models._metrics(y, s)
    assert m["n"] == 2


# ---------------------------------------------------------------------------
# training_stats end-to-end
# ---------------------------------------------------------------------------


def _write_synthetic_manifest(target_dir, model_name, wall_time_sec,
                              n_finetune_epochs):
    target_dir.mkdir(parents=True, exist_ok=True)
    fit_info = [{
        "training_info": {"phase": "finetune", "fold_num": 0},
        "time": wall_time_sec,
        "loss": [0.5] * n_finetune_epochs,
        "val_loss": [0.6] * n_finetune_epochs,
    }]
    config_json = json.dumps({
        "hyperparameters": {"layer_sizes": [32]},
        "fit_info": fit_info,
    })
    df = pandas.DataFrame([{
        "model_name": model_name, "config_json": config_json,
    }])
    df.to_csv(target_dir / "manifest.csv", index=False)


def test_training_stats_component_end_to_end(tmp_path):
    a_dir = tmp_path / "run_a" / "models.unselected.combined"
    _write_synthetic_manifest(a_dir, "model_a", wall_time_sec=120.0,
                              n_finetune_epochs=10)
    b_dir = tmp_path / "run_b" / "models.unselected.combined"
    _write_synthetic_manifest(b_dir, "model_b", wall_time_sec=60.0,
                              n_finetune_epochs=20)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    side_a = compare_models._resolve_side(
        "a", str(tmp_path / "run_a"), "a", _make_args())
    side_b = compare_models._resolve_side(
        "b", str(tmp_path / "run_b"), "b", _make_args())
    assert side_a["paths"]["training"].endswith("models.unselected.combined")

    headline = compare_models._run_training_stats(
        side_a, side_b, str(out_dir))
    per_task = pandas.read_csv(out_dir / "training_stats" / "per_task.csv")
    summary = pandas.read_csv(out_dir / "training_stats" / "summary.csv")
    assert set(per_task["side"]) == {"a", "b"}
    assert set(summary["side"]) == {"a", "b"}
    # Side A wall-time was double B's.
    assert headline["side_a_finetune_total_wall_min"] == pytest.approx(2.0)
    assert headline["side_b_finetune_total_wall_min"] == pytest.approx(1.0)


def test_training_stats_handles_colliding_labels(tmp_path):
    """If --a-label and --b-label collide, positional indexing keeps the
    headline pointing at the correct side."""
    a_dir = tmp_path / "run_a" / "models.unselected.combined"
    _write_synthetic_manifest(a_dir, "model_a", wall_time_sec=120.0,
                              n_finetune_epochs=10)
    b_dir = tmp_path / "run_b" / "models.unselected.combined"
    _write_synthetic_manifest(b_dir, "model_b", wall_time_sec=60.0,
                              n_finetune_epochs=20)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    side_a = compare_models._resolve_side(
        "a", str(tmp_path / "run_a"), "collision", _make_args())
    side_b = compare_models._resolve_side(
        "b", str(tmp_path / "run_b"), "collision", _make_args())
    headline = compare_models._run_training_stats(
        side_a, side_b, str(out_dir))
    # 2.0 vs 1.0 must map to A vs B by position, not by label lookup.
    assert headline["side_a_finetune_total_wall_min"] == pytest.approx(2.0)
    assert headline["side_b_finetune_total_wall_min"] == pytest.approx(1.0)


def test_load_training_summary_rejects_bad_manifest_schema(tmp_path):
    """A manifest missing required columns should fail loudly with the
    missing column names + the manifest path, not AttributeError."""
    bad_dir = tmp_path / "run"
    bad_dir.mkdir()
    # Manifest is missing 'config_json'.
    (bad_dir / "manifest.csv").write_text("model_name,something_else\nx,1\n")
    with pytest.raises(ValueError, match="config_json"):
        compare_models._load_training_summary(str(bad_dir))


def test_run_orchestrator_training_stats_only(tmp_path):
    """End-to-end smoke for run(): training_stats only, both sides
    synthetic. Catches regressions in is_public, _stamp, and the
    headline-by-label bugs all at once."""
    for letter in ("a", "b"):
        target = tmp_path / ("run_" + letter) / "models.unselected.combined"
        _write_synthetic_manifest(
            target, "model_" + letter,
            wall_time_sec=180.0 if letter == "a" else 90.0,
            n_finetune_epochs=10,
        )
    out_dir = tmp_path / "out"
    args = _make_args(
        a=str(tmp_path / "run_a"),
        b=str(tmp_path / "run_b"),
        a_label="candidate",
        b_label="baseline",
        out=str(out_dir),
        include="training_stats",
    )
    assert compare_models.run(args) == 0
    # Side files written.
    assert json.loads((out_dir / "side_a.json").read_text())["label"] == "candidate"
    assert json.loads((out_dir / "side_b.json").read_text())["label"] == "baseline"
    # Top-level summary mentions both labels.
    summary_md = (out_dir / "summary.md").read_text()
    assert "candidate" in summary_md
    assert "baseline" in summary_md
    assert "training_stats" in summary_md
    # Component CSVs landed.
    per_task = pandas.read_csv(out_dir / "training_stats" / "per_task.csv")
    assert set(per_task["side"]) == {"candidate", "baseline"}


# ---------------------------------------------------------------------------
# plot_model_comparison detection
# ---------------------------------------------------------------------------


def test_detect_available_components_empty(tmp_path):
    assert plot_model_comparison._detect_available_components(str(tmp_path)) == []


def test_detect_available_components_finds_affinity(tmp_path):
    aff = tmp_path / "affinity"
    aff.mkdir()
    (aff / "predictions.csv.bz2").write_text("hit,a_score,b_score\n")
    assert "affinity" in plot_model_comparison._detect_available_components(
        str(tmp_path))


def test_detect_available_components_finds_presentation(tmp_path):
    (tmp_path / "presentation").mkdir()
    assert "presentation" in plot_model_comparison._detect_available_components(
        str(tmp_path))


def test_load_side_labels_falls_back_when_missing(tmp_path):
    labels = plot_model_comparison._load_side_labels(str(tmp_path))
    assert labels == {"a": "a", "b": "b"}


def test_load_side_labels_reads_json(tmp_path):
    (tmp_path / "side_a.json").write_text(json.dumps({"label": "candidate"}))
    (tmp_path / "side_b.json").write_text(json.dumps({"label": "baseline"}))
    labels = plot_model_comparison._load_side_labels(str(tmp_path))
    assert labels == {"a": "candidate", "b": "baseline"}

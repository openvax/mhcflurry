"""Model deployment accepts and packages compact JSON percentile calibrations."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from mhcflurry.version import __version__


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = REPO_ROOT / "scripts/release/deploy_trained_models.sh"
CALIBRATED_DIRS = ("affinity/models.combined", "presentation/models")
COMPACT_PERCENT_RANKS = json.dumps(
    {"format": "mhcflurry-percent-ranks-v1", "transforms": {}}, indent=2) + "\n"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json_calibrated_run(run_dir):
    """Write a minimal run calibrated with the default compact method."""
    model_dirs = (
        "affinity/models.combined",
        "processing/models.selected.no_flank",
        "processing/models.selected.with_flanks",
        "presentation/models",
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    for relative in model_dirs:
        (run_dir / relative).mkdir(parents=True)
        (run_dir / relative / "info.txt").write_text(
            "package\tmhcflurry %s\ngit commit\t%s\nworkflow id\ttest-workflow\n" % (
                __version__, commit))
    for relative in model_dirs[:3]:
        (run_dir / relative / "manifest.csv").write_text("model_name\nmodel\n")
        (run_dir / relative / "weights_model.npz").write_bytes(b"selected test weights")
    (run_dir / "presentation/models/weights.csv").write_text("model_name\nmodel\n")
    for relative in CALIBRATED_DIRS:
        (run_dir / relative / "percent_ranks.json").write_text(COMPACT_PERCENT_RANKS)
    holdout = run_dir / "release_holdout"
    holdout.mkdir()
    records = {}
    for name in ("affinity_pmhcs.csv", "affinity_samples.csv",
                 "processing_samples.csv", "presentation_samples.csv"):
        (holdout / name).write_text("sample_id\n")
        records[name] = {"rows": 0, "sha256": _sha256(holdout / name)}
    (holdout / "policy.json").write_text(
        json.dumps({"schema_version": 1, "holdout_files": records}))
    (holdout / "validation.json").write_text(json.dumps({
        "schema_version": 1,
        "policy_sha256": _sha256(holdout / "policy.json"),
        "holdout_files": records,
        "affinity_overlap_rows": 0,
        "processing_overlap_rows": 0,
        "presentation_overlap_rows": 0,
    }))


def _run_deploy(run_dir, *args, env=None):
    return subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "--run-dir", str(run_dir),
         "--release", __version__, "--repo", str(REPO_ROOT),
         "--allow-dirty-repo", *args],
        capture_output=True, text=True, env=env, cwd=run_dir.parent)


def test_deploy_archives_json_only_percent_ranks_unchanged(tmp_path):
    run_dir = tmp_path / "release-run"
    _write_json_calibrated_run(run_dir)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "gh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (fake_bin / "gh").chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"])
    assets = tmp_path / "assets"

    result = _run_deploy(
        run_dir, "--assets-dir", str(assets), "--date", "20260911", "--draft", env=env)

    assert result.returncode == 0, result.stderr
    for asset, member in (
            ("models_class1_pan.selected.20260911.tar.bz2",
             "models.combined/percent_ranks.json"),
            ("models_class1_presentation.20260911.tar.bz2",
             "models/percent_ranks.json")):
        with tarfile.open(assets / asset, "r:bz2") as archive:
            assert archive.extractfile(member).read().decode() == COMPACT_PERCENT_RANKS
        assert asset in (assets / "SHA256SUMS").read_text()


@pytest.mark.parametrize("calibrated_dir", CALIBRATED_DIRS)
def test_deploy_still_requires_a_percent_rank_calibration(tmp_path, calibrated_dir):
    run_dir = tmp_path / "release-run"
    _write_json_calibrated_run(run_dir)
    (run_dir / calibrated_dir / "percent_ranks.json").unlink()

    result = _run_deploy(run_dir, "--dry-run")

    assert result.returncode != 0
    assert str(run_dir.resolve() / calibrated_dir / "percent_ranks.json") in result.stderr
    assert "tar -C" not in result.stdout + result.stderr


def test_export_selected_weights_without_checkpoint_sidecars(tmp_path):
    import importlib.util
    import pandas

    spec = importlib.util.spec_from_file_location(
        "export_inference_models", REPO_ROOT / "scripts/release/export_inference_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    manifest = pandas.DataFrame({
        "model_name": ["selected"], "config_json": ["x" * 200000],
        "checkpoint_best_weights": ["checkpoints/best.npz"],
    })
    manifest.to_csv(source / "manifest.csv", index=False)
    (source / "weights_selected.npz").write_bytes(b"selected")
    (source / "weights_orphan.npz").write_bytes(b"orphan")
    (source / "checkpoints").mkdir()
    (source / "checkpoints/best.npz").write_bytes(b"best")
    (source / "train_data.csv").write_text("peptide\nSIINFEKL\n")
    before = _sha256(source / "manifest.csv")
    dest = tmp_path / "export"
    module.export_models(source, dest)
    assert _sha256(source / "manifest.csv") == before
    assert (dest / "weights_selected.npz").read_bytes() == b"selected"
    assert not (dest / "weights_orphan.npz").exists()
    assert not (dest / "checkpoints").exists()
    assert (dest / "train_data.csv").read_bytes() == (source / "train_data.csv").read_bytes()
    pandas.testing.assert_frame_equal(
        pandas.read_csv(dest / "manifest.csv"), manifest.drop(columns="checkpoint_best_weights"))
    assert json.loads((dest / "inference_export.json").read_text())["manifests"][0]["source_sha256"] == before
    with pytest.raises(ValueError, match="already exists"):
        module.export_models(source, dest)
    (source / "weights_selected.npz").unlink()
    with pytest.raises(ValueError, match="Missing selected weights"):
        module.export_models(source, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()

"""Regression tests for count-matched processing ensemble evaluation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy
import pandas
import pytest

from .test_processing_affinity_control import _cohort
from mhcflurry.processing_matching import make_affinity_controlled_risk_sets


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/training/evaluate_processing_subsets.py"
    spec = importlib.util.spec_from_file_location("evaluate_processing_subsets", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_subsets_exhaustive_not_selected():
    result = module().subsets(8, 4)
    assert len(result) == len(set(result)) == 70
    assert result[0] == (0, 1, 2, 3)
    assert result[-1] == (4, 5, 6, 7)
    assert all(sum(i in subset for subset in result) == 35 for i in range(8))


@pytest.mark.parametrize("count,size,limit", [(0, 4, 100), (8, 0, 100), (8, 9, 100), (8, 4, 69)])
def test_invalid_subset_requests(count, size, limit):
    with pytest.raises(ValueError):
        module().subsets(count, size, limit)


def test_unique_rows_validate_reused_decoy_identity():
    frame = _cohort().assign(source_row=numpy.arange(6))
    frame = pandas.concat([frame, frame.iloc[[1]]], ignore_index=True)
    assert len(module().unique_inputs(frame)) == 6
    frame.loc[6, "peptide"] = "YYYYYYYY"
    with pytest.raises(ValueError, match="Inconsistent"):
        module().unique_inputs(frame)


@pytest.mark.parametrize("value", [-1, 0.5, numpy.inf, numpy.nan])
def test_invalid_source_rows(value):
    frame = _cohort().assign(source_row=numpy.arange(6, dtype=float))
    frame.loc[0, "source_row"] = value
    with pytest.raises(ValueError):
        module().unique_inputs(frame)


def setup_run(tmp_path, monkeypatch, mismatch=False):
    import mhcflurry
    command = module()
    frame, _ = make_affinity_controlled_risk_sets(_cohort(), decoys_per_hit=2)
    # Four genuinely different prediction vectors; the mean must agree exactly.
    predictions = numpy.array([[.9, .6, .2, .7, .5, .1],
                               [.8, .5, .1, .9, .6, .2],
                               [.7, .4, .3, .8, .7, .2],
                               [.9, .3, .2, .6, .4, .3]]).T
    frame["reference"] = predictions.mean(axis=1)[frame.source_row]
    if mismatch:
        frame["reference"] += .01
    path = tmp_path / "input.csv.bz2"
    frame.to_csv(path, index=False)
    models = tmp_path / "models"
    models.mkdir()
    pandas.DataFrame({"model_name": ["model%d" % i for i in range(4)],
                      "config_json": ["{}"] * 4}).to_csv(models / "manifest.csv", index=False)
    for i in range(4):
        (models / ("weights_model%d.npz" % i)).write_bytes(bytes([i]))
    loaded = SimpleNamespace(models=[SimpleNamespace(
        predict_encoded=lambda sequences, batch_size, i=i: predictions[:, i]) for i in range(4)])
    monkeypatch.setattr(mhcflurry.Class1ProcessingPredictor, "load", lambda *a: loaded)
    monkeypatch.setattr(command, "configure_pytorch", lambda **kwargs: None)
    args = command.make_parser().parse_args([
        "--input", str(path), "--models-dir", str(models), "--subset-size", "2",
        "--reference-score", "reference", "--comparison-score", "candidate",
        "--out", str(tmp_path / "result")])
    return command, args, predictions


def test_end_to_end_preserves_all_predictions_and_membership(tmp_path, monkeypatch):
    command, args, matrix = setup_run(tmp_path, monkeypatch)
    assert command.run(args) == 0
    saved = pandas.read_csv(args.out / "matched_predictions.csv.bz2")
    membership = json.loads((args.out / "subset_membership.json").read_text())
    assert len(membership) == 6
    for member in membership:
        expected = matrix[:, member["indices"]].mean(axis=1)[saved.source_row]
        numpy.testing.assert_allclose(saved[member["score"]], expected)
    numpy.testing.assert_allclose(numpy.load(args.out / "member_predictions.npz")["predictions"], matrix)
    assert len(list(args.out.glob("member_*_predictions.csv.bz2"))) == 4
    distribution = pandas.read_csv(args.out / "subset_distribution.csv")
    assert distribution.subsets.eq(6).all()
    assert (distribution["min"] <= distribution["median"]).all()
    assert (distribution["median"] <= distribution["max"]).all()
    assert (args.out / "completed.json").exists()
    with pytest.raises(ValueError, match="fresh"):
        command.run(args)


def test_mismatched_full_reference_fails_before_reporting_subsets(tmp_path, monkeypatch):
    command, args, _ = setup_run(tmp_path, monkeypatch, mismatch=True)
    with pytest.raises(ValueError, match="Reconstructed"):
        command.run(args)
    assert (args.out / "verification.json").exists()
    assert not (args.out / "completed.json").exists()
    assert not (args.out / "summary.csv").exists()


def test_cache_reuse_and_identity_guards(tmp_path, monkeypatch):
    command, args, _ = setup_run(tmp_path, monkeypatch)
    assert command.run(args) == 0
    args.member_cache_dir = args.out
    args.out = tmp_path / "reused"
    assert command.run(args) == 0
    original = pandas.read_csv(args.member_cache_dir / "summary.csv")
    pandas.testing.assert_frame_equal(original, pandas.read_csv(args.out / "summary.csv"))
    args.out = tmp_path / "changed"
    args.backend = "mps"
    with pytest.raises(ValueError, match="execution mismatch"):
        command.run(args)
    args.out = tmp_path / "weights-changed"
    args.backend = "cpu"
    (args.models_dir / "weights_model0.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        command.run(args)

"""Cached ensembles must never average models across incompatible folds."""

import importlib.util
import json
from pathlib import Path

import numpy
import pandas
import pytest

from mhcflurry.experiment_archive import sha256_file


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/training/evaluate_processing_fold_ensembles.py"
    spec = importlib.util.spec_from_file_location("fold_ensembles", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_cache(directory, condition, frame=None, training_hash="frozen", scope=None):
    command = module()
    if frame is None:
        rows = []
        for policy, offset in (("best", 0), ("terminal", 0.05)):
            for fold, samples in ((0, ("a", "b")), (1, ("a", "c"))):
                for sample_index, sample in enumerate(samples):
                    for j in range(4):
                        score = [0.8, 0.4, 0.7, 0.1][j] + offset
                        if condition == "second":
                            score = [0.6, 0.5, 0.4, 0.3][j] + offset
                        rows.append(dict(condition=condition, model_name=condition + "-f%d" % fold,
                                         checkpoint_policy=policy, fold_num=fold, validation_row_index=sample_index * 4 + j,
                                         sample_id=sample, hit=int(j % 2 == 0), peptide="AAAAAAAA", n_flank="NN", c_flank="CC",
                                         risk_set_id="%s-%d" % (sample, j // 2), processing_score=score))
        frame = pandas.DataFrame(rows)
    path = directory / (condition + ".csv.bz2")
    frame.to_csv(path, index=False)
    marker = {"identity": {"scope": scope or command.SCOPE, "condition": condition,
                           "training_data_sha256": training_hash}, "prediction_sha256": sha256_file(path)}
    path.with_name(path.name + ".json").write_text(json.dumps(marker))
    return path, frame


def test_fold_ensemble_arithmetic_identity_and_macro_weighting(tmp_path):
    command = module()
    first, _ = write_cache(tmp_path, "first")
    second, _ = write_cache(tmp_path, "second")
    result, members, metrics, sample_means, micro, summary, _ = command.score_ensembles(
        {"two": [first.name, second.name]}, tmp_path, ["best", "terminal"])
    expected = (numpy.array([0.8, 0.4, 0.7, 0.1], dtype="float32") +
                numpy.array([0.6, 0.5, 0.4, 0.3], dtype="float32")) / numpy.float32(2)
    numpy.testing.assert_array_equal(result.two__best, numpy.tile(expected, 4))
    assert len(members) == 4
    assert members.groupby("fold_num").size().tolist() == [2, 2]
    assert len(metrics) == 8 and len(sample_means) == 6 and len(micro) == 4
    assert summary.macro_pr_auc.eq(1).all()
    assert not result.duplicated(command.KEYS).any()


@pytest.mark.parametrize("problem", ["sample", "label", "flank", "row", "fold", "hash", "duplicate", "two_models", "model_two_folds", "nan", "missing_policy", "scope"])
def test_fold_guards_reject_invalid_caches(tmp_path, problem):
    command = module()
    first, _ = write_cache(tmp_path, "first")
    second, frame = write_cache(tmp_path, "second")
    if problem == "sample":
        frame.loc[0, "sample_id"] = "changed"
    elif problem == "label":
        frame.loc[0, "hit"] = 0
    elif problem == "flank":
        frame.loc[0, "n_flank"] = "YY"
    elif problem == "row":
        frame = frame.drop(index=[0, 16])
    elif problem == "fold":
        frame.loc[frame.fold_num.eq(1), "fold_num"] = 2
    elif problem == "duplicate":
        frame = pandas.concat([frame, frame.iloc[:1]])
    elif problem == "two_models":
        frame.loc[0, "model_name"] = "unexpected"
    elif problem == "model_two_folds":
        frame.model_name = "shared-model"
    elif problem == "nan":
        frame.loc[0, "processing_score"] = numpy.nan
    elif problem == "missing_policy":
        frame = frame.loc[frame.checkpoint_policy.eq("best")]
    second, _ = write_cache(tmp_path, "second", frame,
                            training_hash="different" if problem == "hash" else "frozen",
                            scope="in-sample" if problem == "scope" else None)
    with pytest.raises(ValueError):
        command.score_ensembles({"two": [first.name, second.name]}, tmp_path, ["best", "terminal"])


def test_corrupt_and_duplicate_members_are_rejected(tmp_path):
    command = module()
    path, _ = write_cache(tmp_path, "first")
    with pytest.raises(ValueError, match="Duplicate ensemble member"):
        command.score_ensembles({"bad": [path.name, path.name]}, tmp_path, ["best"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed checkpoint cache"):
        command.load_predictions(path, ["best"])


def test_command_preserves_scores_and_provenance_without_overwrite(tmp_path):
    command = module()
    first, _ = write_cache(tmp_path, "first")
    second, _ = write_cache(tmp_path, "second")
    definition = tmp_path / "ensembles.json"
    definition.write_text(json.dumps({"two": [first.name, second.name]}))
    out = tmp_path / "out"
    args = ["--ensembles", str(definition), "--out", str(out)]
    assert command.main(args) == 0
    saved = pandas.read_csv(out / "fold_ensemble_predictions.csv.gz")
    assert {"two__best", "two__terminal", "fold_num", "validation_row_index", "n_flank", "c_flank"} <= set(saved)
    provenance = json.loads((out / "provenance.json").read_text())
    assert not provenance["training"] and not provenance["inference"] and not provenance["release_accepted"]
    assert len(provenance["inputs"]) == 2
    assert not (out / "fold_ensemble_predictions.tmp.gz").exists()
    for record in provenance["outputs"]:
        assert sha256_file(out / record["path"]) == record["sha256"]
    with pytest.raises(ValueError, match="fresh"):
        command.main(args)

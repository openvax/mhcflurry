import importlib.util
from pathlib import Path

import numpy
import pandas
import pytest


@pytest.fixture
def driver_module(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts/training"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("exact_public", directory / "run_exact_public_processing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_public_design_is_eight_matched_networks(driver_module):
    records = [record for record in driver_module.build_conditions(
        architectures=["large_relu"], peptide_context_lengths=[5])
        if record[0] in driver_module.CONDITIONS]
    assert len(records) == 2
    assert all(len(record[1]) == 1 for record in records)
    legacy, boundary = (record[1][0] for record in records)
    changes = {key for key in set(legacy) | set(boundary) if legacy.get(key) != boundary.get(key)}
    assert changes == {"flanking_averages", "cleavage_boundary_flank_length",
                       "cleavage_boundary_peptide_length", "cleavage_boundary_hidden_size",
                       "cleavage_boundary_context_dropout"}
    assert boundary["cleavage_boundary_peptide_length"] == 5
    assert legacy["minibatch_size"] == boundary["minibatch_size"] == 512


def test_score_cache_rejects_changed_inputs_weights_and_corruption(tmp_path, driver_module):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"initial")
    data = pandas.DataFrame({"peptide": ["A", "B"]})
    calls = []

    def calculate():
        calls.append(True)
        return [0.1, 0.2]

    def cached(digest="input1"):
        return driver_module.cached_scores(tmp_path, "test", data, digest, model, calculate)

    numpy.testing.assert_array_equal(cached(), cached())
    assert len(calls) == 1
    cached("input2")
    assert len(calls) == 2
    weights.write_bytes(b"changed")
    cached("input2")
    assert len(calls) == 3
    (tmp_path / "component_scores/test.npy").write_bytes(b"corrupt")
    cached("input2")
    assert len(calls) == 4


def test_public_input_gate_checks_hashes_and_holdout(tmp_path, driver_module, monkeypatch):
    processing = tmp_path / "processing.csv"
    presentation = tmp_path / "presentation.csv"
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pandas.DataFrame({"sample_id": ["heldout"]}).to_csv(holdout / "processing_samples.csv", index=False)
    pandas.DataFrame({"sample_id": ["heldout"]}).to_csv(holdout / "presentation_samples.csv", index=False)
    data = pandas.DataFrame({"sample_id": ["a", "b"], **{
        "fold_%d" % i: [True, False] for i in range(4)}})
    data.to_csv(processing, index=False)
    data.to_csv(presentation, index=False)
    for name, path in (("PROCESSING", processing), ("PRESENTATION", presentation)):
        monkeypatch.setattr(driver_module, "PUBLIC_%s_SHA256" % name, driver_module.sha256_file(path))
    result = driver_module.verify_public_inputs(processing, presentation, holdout)
    assert result["processing"]["rows"] == 2
    pandas.DataFrame({"sample_id": ["a"]}).to_csv(holdout / "processing_samples.csv", index=False)
    with pytest.raises(ValueError, match="overlaps"):
        driver_module.verify_public_inputs(processing, presentation, holdout)
    monkeypatch.setattr(driver_module, "PUBLIC_PROCESSING_SHA256", "bad")
    with pytest.raises(ValueError, match="Not the archived"):
        driver_module.verify_public_inputs(processing, presentation, holdout)


def test_completed_stages_refuse_changed_commands(tmp_path, driver_module):
    stage = tmp_path / "stages/test.json"
    driver_module.write_json(stage, {"command": ["same"]})
    runner = driver_module.Driver(tmp_path)
    runner.run("test", ["same"])
    with pytest.raises(ValueError, match="changed command"):
        runner.run("test", ["different"])

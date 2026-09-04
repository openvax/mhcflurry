"""Tests for the cleavage-boundary processing experiment design."""

import importlib.util
from pathlib import Path

import yaml


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "training"
    / "generate_processing_cleavage_boundaries.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "generate_processing_cleavage_boundaries", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleavage_boundary_design_is_paired_and_minimal(tmp_path):
    module = _module()
    conditions = module.build_conditions()

    assert len(conditions) == 8
    assert {axes["architecture"] for _, _, axes in conditions} == {
        "small_tanh", "large_relu",
    }
    assert {axes["model_kind"] for _, _, axes in conditions} == {
        "legacy_5aa", "legacy_no_flank", "cleavage_boundary",
    }
    boundary_conditions = [
        record for record in conditions
        if record[2]["model_kind"] == "cleavage_boundary"
    ]
    assert {axes["peptide_context_length"] for _, _, axes in
            boundary_conditions} == {2, 5}
    for condition, grid, axes in conditions:
        assert len(grid) == 1
        item = grid[0]
        assert item["minibatch_size"] == 512
        assert item["optimizer_implementation"] == "keras"
        assert item["init"] == "glorot_uniform"
        assert axes["baseline_5aa_condition"].endswith("__legacy_5aa")
        assert axes["baseline_no_flank_condition"].endswith(
            "__legacy_no_flank")
        if axes["model_kind"] == "cleavage_boundary":
            assert item["n_flank_length"] == 5
            assert item["c_flank_length"] == 5
            assert item["flanking_averages"] is False
            assert item["cleavage_boundary_flank_length"] == 5
            assert item["cleavage_boundary_peptide_length"] == axes[
                "peptide_context_length"]
            assert item["cleavage_boundary_context_dropout"] == 0.25
        elif axes["model_kind"] == "legacy_5aa":
            assert condition.endswith("__legacy_5aa")
            assert item["n_flank_length"] == 5
            assert item["c_flank_length"] == 5
            assert item["flanking_averages"] is True
            assert "cleavage_boundary_flank_length" not in item
        else:
            assert condition.endswith("__legacy_no_flank")
            assert item["n_flank_length"] == 0
            assert item["c_flank_length"] == 0
            assert "cleavage_boundary_flank_length" not in item

    manifest = module.write_conditions(tmp_path)
    assert manifest["design"] == "processing-cleavage-boundaries"
    assert len(manifest["records"]) == 8
    assert manifest["network_budget"]["total_networks"] == 32
    for record in manifest["records"]:
        path = tmp_path / record["hyperparameters_path"]
        assert len(yaml.safe_load(path.read_text())) == 1

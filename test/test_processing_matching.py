"""Shared processing training/evaluation matching contract."""

import json
from types import SimpleNamespace

import numpy
import pandas
import pytest

from mhcflurry.processing_matching import (
    make_affinity_controlled_risk_sets, matched_training_data,
    validate_matched_training_data, sample_validation_mask)
from .test_processing_affinity_control import _cohort


def training_frame():
    frame = _cohort().rename(columns={"mhcflurry_production_affinity": "affinity_prediction"})
    return matched_training_data(frame, {"sha256": "0" * 64})[0]


def test_training_and_evaluation_share_assignments():
    frame = training_frame()
    expected, _ = make_affinity_controlled_risk_sets(_cohort(), decoys_per_hit=1)
    assert frame.source_row.tolist() == expected.source_row.tolist()
    validate_matched_training_data(frame)
    assert frame.hit.sum() == 2
    assert len(frame) == 4


@pytest.mark.parametrize("problem", ["length", "affinity", "ratio", "metadata", "fold"])
def test_matched_training_rejects_corrupted_or_unmatched_data(problem):
    frame = training_frame()
    if problem == "length":
        frame.loc[1, "peptide"] += "A"
    elif problem == "affinity":
        frame.loc[1, "affinity_prediction"] = 50000
    elif problem == "ratio":
        frame = frame.iloc[1:]
    elif problem == "metadata":
        frame = frame.drop(columns="processing_matching_policy")
    else:
        frame["fold_0"] = [True, False, True, False]
    with pytest.raises(ValueError):
        validate_matched_training_data(frame)


def test_unmatched_data_requires_explicit_legacy_policy():
    with pytest.raises(ValueError, match="requires matched"):
        validate_matched_training_data(_cohort())
    validate_matched_training_data(_cohort(), policy="legacy")


def test_global_fallback_cannot_violate_affinity_or_length_caliper():
    frame = _cohort()
    frame.loc[1:2, "log10_affinity"] = 4
    with pytest.raises(ValueError, match="no unmatched fallback"):
        make_affinity_controlled_risk_sets(frame, decoys_per_hit=1)
    frame = _cohort()
    frame.loc[1:2, "peptide"] += "A"
    frame.loc[1:2, "peptide_len"] = 9
    with pytest.raises(ValueError, match="no unmatched fallback"):
        make_affinity_controlled_risk_sets(frame, decoys_per_hit=1)


def test_same_sequence_cannot_supply_multiple_decoys_in_one_risk_set():
    frame = _cohort().iloc[:3].copy()
    frame.loc[2, "peptide"] = frame.loc[1, "peptide"]
    with pytest.raises(ValueError, match="no unmatched fallback"):
        make_affinity_controlled_risk_sets(frame, decoys_per_hit=2)


def test_validation_groups_keep_reused_rows_and_samples_together():
    frame = pandas.concat([training_frame().assign(sample_id=str(i)) for i in range(5)], ignore_index=True)
    mask = sample_validation_mask(frame, 0.2, seed=42)
    assert mask.sum() == 4
    assert frame.loc[mask].sample_id.nunique() == 1
    assert not set(frame.loc[mask].sample_id) & set(frame.loc[~mask].sample_id)
    numpy.testing.assert_array_equal(mask, sample_validation_mask(frame, 0.2, seed=42))


def test_validation_requires_two_samples_unless_disabled():
    with pytest.raises(ValueError, match="at least two"):
        sample_validation_mask(training_frame(), 0.1, 42)
    assert not sample_validation_mask(training_frame(), 0, 42).any()


def test_validate_cached_training_command(tmp_path):
    from mhcflurry.cli.train_command import run_argv
    path = tmp_path / "training.csv.bz2"
    frame = training_frame().assign(matching_affinity_reference_sha256="a" * 64)
    frame.to_csv(path, index=False)
    assert run_argv(["validate-processing-data", "--data", str(path)]) == 0
    frame.loc[1, "affinity_prediction"] = 50000
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="caliper"):
        run_argv(["validate-processing-data", "--data", str(path)])


def test_default_processing_comparison_scores_only_matched_risk_sets(tmp_path, monkeypatch):
    from mhcflurry.cli import compare_models as command
    from mhcflurry.cli.processing_affinity_control import AFFINITY_COLUMN
    frame = pandas.DataFrame({
        "peptide": ["A" * 8, "Y" * 9] + [c + "A" * 7 for c in "CDEFGHIKLM"] +
                   [c + "Y" * 8 for c in "CDEFGHIKLM"] + ["F" * 10, "W" * 8],
        "hit": [1, 1] + [0] * 22,
        AFFINITY_COLUMN: [10, 100] + [11] * 10 + [101] * 10 + [10, 50000],
        "sample_id": "sample", "hla": "HLA-A*02:01",
        "n_flank": "N" * 5, "c_flank": "C" * 5, "protein_accession": "p1"})
    frame["peptide_len"] = frame.peptide.str.len()
    frame.to_csv(tmp_path / "benchmark.multiallelic.production.train_excluded.sample.0.csv.bz2", index=False)
    monkeypatch.setattr(command, "_processing_model_dirs", lambda *a: {"short_flanks": ("a", "b")})
    monkeypatch.setattr(command, "_load_presentation_benchmark_for_component", lambda *a: frame)
    monkeypatch.setattr(command, "_parallelism_args_for_component", lambda *a: None)
    monkeypatch.setattr(command, "model_artifact_size_bytes", lambda *a: 1)
    monkeypatch.setattr(command, "_parallel_processing_predict", lambda *a, **k: numpy.arange(len(frame)) / len(frame))
    args = command.make_parser().parse_args([
        "--a", "a", "--out", str(tmp_path / "out"), "--data-dir", str(tmp_path),
        "--processing-modes", "short_flanks"])
    result = command._run_processing({"label": "a"}, {"label": "b"}, args)
    assert result["negative_policy"] == "matched"
    out = tmp_path / "out" / "processing"
    matched = pandas.read_csv(out / "predictions_short_flanks.csv.bz2")
    assert len(matched) == 22
    assert matched.groupby("risk_set_id").peptide_len.nunique().eq(1).all()
    assert matched.log10_affinity_distance.max() < .25
    assert len(pandas.read_csv(out / "heldout_predictions_short_flanks.csv.bz2")) == len(frame)
    info = json.loads((out / "cohort.json").read_text())
    assert info["processing_release_eligible"] is True
    assert len(info["affinity_sources"][0]["sha256"]) == 64

    def fail_prediction(*a, **kw):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(command, "_parallel_processing_predict", fail_prediction)
    with pytest.raises(RuntimeError, match="prediction failed"):
        command._run_processing({"label": "a"}, {"label": "b"}, args)
    info = json.loads((out / "cohort.json").read_text())
    assert info["policy"] == "incomplete"
    assert info["processing_release_eligible"] is False


def test_unmatched_plot_guard_preserves_existing_figures(tmp_path):
    from mhcflurry.cli import plot_model_comparison as command
    (tmp_path / "processing").mkdir()
    (tmp_path / "processing" / "summary_table.csv").write_text("mode,a_macro_pr_auc\nshort_flanks,0.3\n")
    (tmp_path / "plots").mkdir()
    existing = tmp_path / "plots" / "keep.txt"
    existing.write_text("historical figure")
    args = command.make_parser().parse_args(["--input", str(tmp_path), "--components", "processing"])
    with pytest.raises(ValueError, match="matched evaluation"):
        command.run(args)
    assert existing.read_text() == "historical figure"


def test_paper_processing_plot_does_not_mix_presentation_cohort(tmp_path):
    from mhcflurry.cli.paper_figures import _plot_current_ap_vs_summary
    import matplotlib.pyplot as plt
    for name in ("processing", "presentation"):
        (tmp_path / name).mkdir()
        pandas.DataFrame([dict(mode="short_flanks", a_macro_roc_auc=.8,
                               a_macro_pr_auc=.4, a_macro_ppv_at_n=.43)]).to_csv(
            tmp_path / name / "summary_table.csv", index=False)
    inputs = SimpleNamespace(comparison_dir=str(tmp_path))
    figures = []
    writer = SimpleNamespace(save=lambda fig, *a, **kw: figures.append(fig))
    assert _plot_current_ap_vs_summary(inputs, writer) is False
    (tmp_path / "processing" / "cohort.json").write_text(json.dumps({"policy": "matched"}))
    assert _plot_current_ap_vs_summary(inputs, writer) is True
    assert all(len(ax.patches) == 1 for ax in figures[0].axes)
    plt.close(figures[0])

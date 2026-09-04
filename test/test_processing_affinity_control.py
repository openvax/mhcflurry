"""Tests for affinity-controlled processing evaluation."""

import json

import pandas

from mhcflurry.cli import processing_affinity_control as command


def _cohort():
    return pandas.DataFrame({
        "sample_id": ["sample"] * 6,
        "peptide": [
            "AAAAAAAA", "BBBBBBBB", "CCCCCCCC",
            "DDDDDDDD", "EEEEEEEE", "FFFFFFFF",
        ],
        "hit": [1, 0, 0, 1, 0, 0],
        "n_flank": ["NNNNN"] * 6,
        "c_flank": ["CCCCC"] * 6,
        "protein_accession": ["p1", "p1", "p2", "p2", "p2", "p3"],
        "mhcflurry_production_affinity": [10, 11, 12, 100, 101, 102],
        "log10_affinity": [
            1.0, 1.0413927, 1.0791812, 2.0, 2.0043214, 2.0086002,
        ],
        "peptide_len": [8] * 6,
        "candidate": [.9, .8, .1, .7, .6, .2],
        "baseline": [.8, .7, .2, .6, .5, .3],
    })


def test_affinity_risk_sets_prefer_same_protein():
    matched, diagnostics = command.make_affinity_controlled_risk_sets(
        _cohort(), decoys_per_hit=2)

    assert len(matched) == 6
    assert matched.groupby("risk_set_id").hit.agg(["sum", "count"]).values.tolist() == [
        [1, 3], [1, 3],
    ]
    first = matched.loc[matched.risk_set_id == 0]
    assert first.loc[first.match_rank == 1, "protein_accession"].item() == "p1"
    assert diagnostics["risk_sets"] == 2
    assert diagnostics["fallback_decoys"] == 2


def test_score_risk_sets_reports_paired_summary():
    matched, _ = command.make_affinity_controlled_risk_sets(
        _cohort(), decoys_per_hit=2)
    metrics = command.score_risk_sets(
        matched, ["candidate", "baseline"])
    summary, comparisons = command.summarize_metrics(metrics, "baseline")

    assert set(metrics.scope) == {"overall", "sample", "length"}
    assert set(summary.score) == {"candidate", "baseline"}
    baseline = comparisons.set_index("score").loc["baseline"]
    assert baseline.macro_pr_auc_diff == 0
    assert baseline.macro_ppv_at_n_diff == 0


def test_processing_affinity_control_end_to_end(tmp_path):
    cohort = _cohort()
    identity = list(command.IDENTITY_COLUMNS)
    predictions = cohort[identity].copy()
    predictions["a_processing_score"] = cohort.candidate
    predictions["b_processing_score"] = cohort.baseline
    prediction_path = tmp_path / "predictions.csv.bz2"
    predictions.to_csv(prediction_path, index=False)

    production = cohort[identity + [
        "protein_accession", "mhcflurry_production_affinity",
    ]]
    # The command keys files by their contents, not sanitized filename text.
    production_path = tmp_path / (
        "benchmark.multiallelic.production.train_excluded.not-the-id.0.csv.bz2")
    production.to_csv(production_path, index=False)

    out = tmp_path / "out"
    result = command.run_argv([
        "--score", "candidate=%s:a_processing_score" % prediction_path,
        "--score", "baseline=%s:b_processing_score" % prediction_path,
        "--baseline", "baseline",
        "--data-dir", str(tmp_path),
        "--decoys-per-hit", "2",
        "--out", str(out),
    ])

    assert result == 0
    heldout = pandas.read_csv(out / "heldout_predictions.csv.bz2")
    assert len(heldout) == len(cohort)
    assert {"candidate", "baseline", "protein_accession",
            "mhcflurry_production_affinity"}.issubset(heldout.columns)
    assert (out / "matched_predictions.csv.bz2").exists()
    assert set(pandas.read_csv(out / "summary.csv").score) == {
        "candidate", "baseline",
    }
    experiment = json.loads((out / "experiment.json").read_text())
    assert experiment["diagnostics"]["risk_sets"] == 2

    added = predictions[identity].copy()
    added["radius3"] = [0.95, 0.7, 0.1, 0.8, 0.4, 0.2]
    added["radius4"] = [0.9, 0.6, 0.2, 0.75, 0.5, 0.1]
    added_path = tmp_path / "added.csv.bz2"
    added.to_csv(added_path, index=False)
    extended_out = tmp_path / "extended"
    result = command.run_argv([
        "--score", "radius3=%s:radius3" % added_path,
        "--score", "radius4=%s:radius4" % added_path,
        "--baseline", "baseline",
        "--data-dir", str(tmp_path),
        "--existing", str(out),
        "--out", str(extended_out),
    ])

    assert result == 0
    extended = pandas.read_csv(
        extended_out / "heldout_predictions.csv.bz2")
    assert extended.radius3.tolist() == added.radius3.tolist()
    matched = pandas.read_csv(
        extended_out / "matched_predictions.csv.bz2")
    assert matched.radius4.tolist() == extended.radius4.iloc[
        matched.source_row].tolist()
    assert set(pandas.read_csv(extended_out / "summary.csv").score) == {
        "candidate", "baseline", "radius3", "radius4",
    }
    extended_experiment = json.loads(
        (extended_out / "experiment.json").read_text())
    assert extended_experiment["extended_from"].endswith(
        "out/experiment.json")

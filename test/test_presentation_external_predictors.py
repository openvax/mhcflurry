"""External predictor comparisons join exact benchmark rows and orient scores."""

import importlib.util
import json
from pathlib import Path

import numpy
import pandas
import pytest

from mhcflurry.cli.compare_models import _metrics


KEYS = ["protein_accession", "peptide", "sample_id", "n_flank", "c_flank", "hit", "hla"]
EXTERNAL = ("netmhcpan4.el", "netmhcpan4.ba", "mixmhcpred")


@pytest.fixture
def module():
    path = (Path(__file__).resolve().parents[1]
            / "scripts/training/compare_presentation_external_predictors.py")
    spec = importlib.util.spec_from_file_location("presentation_external_predictors", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def benchmark_rows(sample, seed, n=300, n_hits=30):
    rng = numpy.random.default_rng(seed)
    hit = numpy.r_[numpy.ones(n_hits, dtype=int), numpy.zeros(n - n_hits, dtype=int)]
    peptides = ["PEPTIDE%04d" % index for index in range(n)]
    peptides[1] = peptides[0]  # identical identities must keep their file order
    frame = pandas.DataFrame({
        "protein_accession": "P1", "peptide": peptides, "sample_id": sample,
        "n_flank": "NA", "c_flank": "AC", "hit": hit, "hla": "HLA-B*07:02 HLA-A*02:01 HLA-A*02:01"})
    frame["netmhcpan4.el"] = numpy.round(0.3 * hit + rng.uniform(0, 1, n), 6)
    frame["netmhcpan4.ba"] = numpy.round(numpy.where(
        hit == 1, rng.uniform(5, 800, n), rng.uniform(50, 50000, n)), 3)
    frame["mixmhcpred"] = numpy.round(0.2 * hit + rng.uniform(0, 1, n), 6)
    return frame


def write_multiallelic(tmp_path, drop_external_row=False):
    data = tmp_path / "data_evaluation"
    data.mkdir()
    presentation = tmp_path / "comparison" / "presentation"
    presentation.mkdir(parents=True)
    saved = []
    for index, sample in enumerate(["S1", "S/2"]):
        rows = benchmark_rows(sample, index)
        suffix = "%s.0.csv.bz2" % sample.replace("/", "")
        base = "benchmark.multiallelic.train_excluded." + suffix
        rows[KEYS].to_csv(data / base, index=False)
        if index == 0:
            rows.loc[[3, 50], "netmhcpan4.el"] = numpy.nan  # a hit and a decoy left unscored
        for predictor in EXTERNAL:
            external = rows[KEYS + [predictor]].assign(**{predictor + "_best_allele": "HLA-A*02:01"})
            if drop_external_row and predictor == "netmhcpan4.el" and index == 0:
                external = external.drop(index=10)
            external.to_csv(data / ("benchmark.multiallelic.%s.train_excluded.%s" % (predictor, suffix)),
                            index=False)
        kept = rows.drop(index=5)  # a length-filtered row
        # compare-models saves the canonical genotype, not the raw benchmark string.
        saved.append(kept.assign(source_file=base, peptide_len=kept.peptide.str.len(),
                                 hla="HLA-A*02:01 HLA-B*07:02"))
    saved = pandas.concat(saved[::-1], ignore_index=True)  # not sorted by file name
    rng = numpy.random.default_rng(7)
    for side, weight in (("a", 0.6), ("b", 0.4)):
        score = numpy.clip(weight * saved.hit + rng.uniform(0, 0.6, len(saved)), 0, 1)
        saved[side + "_presentation_score"] = score
        saved[side + "_presentation_percentile"] = 100 * (1 - score)
    saved.drop(columns=list(EXTERNAL)).to_csv(presentation / "predictions_with_flanks.csv.bz2", index=False)
    without = saved.drop(columns=list(EXTERNAL))
    without["a_presentation_score"] = numpy.clip(0.5 * saved.hit + rng.uniform(0, 0.6, len(saved)), 0, 1)
    without.to_csv(presentation / "predictions_without_flanks.csv.bz2", index=False)
    return data, tmp_path / "comparison", saved


def arguments(data, comparison, out, *extra):
    return ["--comparison-dir", str(comparison), "--data-dir", str(data), "--out", str(out),
            "--replicates", "100", *extra]


def test_joins_exact_rows_orients_scores_and_writes_figures(tmp_path, module):
    data, comparison, saved = write_multiallelic(tmp_path)
    out = tmp_path / "out"
    assert module.main(arguments(data, comparison, out, "--a-label", "Candidate")) == 0
    joined = pandas.read_csv(out / "joined_scores.csv.gz")
    assert joined.prediction_row.tolist() == list(range(len(saved)))
    for predictor in EXTERNAL:
        numpy.testing.assert_allclose(joined[predictor], saved[predictor], equal_nan=True)
    per_sample = pandas.read_csv(out / "per_sample_metrics.csv")
    s1 = saved.loc[saved.sample_id == "S1"]

    def row(condition):
        return per_sample.loc[(per_sample.sample_id == "S1") & (per_sample.condition == condition)].iloc[0]

    ba = _metrics(s1.hit.to_numpy(), -s1["netmhcpan4.ba"].to_numpy())
    assert row("netmhcpan4.ba").roc_auc == pytest.approx(ba["roc_auc"]) and ba["roc_auc"] > 0.5
    # MHCflurry keeps every row even where an external tool lacks a score.
    candidate = _metrics(s1.hit.to_numpy(), s1.a_presentation_score.to_numpy())
    assert row("a_with_flanks").n == len(s1) and row("netmhcpan4.el").n == len(s1) - 2
    assert row("a_with_flanks").pr_auc == pytest.approx(candidate["pr_auc"])
    assert row("a_with_flanks").ppv_at_n == pytest.approx(candidate["ppv_at_n"])
    percentile = _metrics(s1.hit.to_numpy(), -s1.a_presentation_percentile.to_numpy())
    assert row("a_with_flanks_percentile").roc_auc == pytest.approx(percentile["roc_auc"])
    paired = pandas.read_csv(out / "paired_differences.csv")
    assert {("a_with_flanks", "b_with_flanks"), ("a_with_flanks", "netmhcpan4.el"),
            ("a_with_flanks", "netmhcpan4.ba")} <= set(zip(paired.candidate, paired.reference))
    excluded = paired.groupby(["candidate", "reference"]).rows_excluded.first()
    assert excluded[("a_with_flanks", "b_with_flanks")] == 0
    assert excluded[("a_with_flanks", "netmhcpan4.el")] == 2
    joint = s1.loc[s1["netmhcpan4.el"].notna()]
    expected = _metrics(joint.hit.to_numpy(), joint.a_presentation_score.to_numpy())["pr_auc"] - \
        _metrics(joint.hit.to_numpy(), joint["netmhcpan4.el"].to_numpy())["pr_auc"]
    differences = pandas.read_csv(out / "sample_differences.csv", dtype={"sample_id": str})
    got = differences.loc[(differences.sample_id == "S1") & (differences.candidate == "a_with_flanks")
                          & (differences.reference == "netmhcpan4.el") & (differences.metric == "pr_auc"), "delta"]
    assert got.item() == pytest.approx(expected)
    assert (out / "external_comparison.pdf").stat().st_size > 1000
    for name in ("macro_metrics.png", "paired_differences_mhcflurry.png",
                 "paired_differences_external.png", "precision_recall.png",
                 "summary.md", "coverage.csv", "analysis_source.py"):
        assert (out / name).is_file()
    coverage = pandas.read_csv(out / "coverage.csv")
    assert coverage["netmhcpan4.el_unscored"].sum() == 2 and coverage["a_with_flanks_unscored"].sum() == 0
    provenance = json.loads((out / "provenance.json").read_text())
    assert any(item["path"].endswith("benchmark.multiallelic.netmhcpan4.ba.train_excluded.S2.0.csv.bz2")
               for item in provenance["inputs"])
    with pytest.raises(ValueError, match="new or empty"):
        module.main(arguments(data, comparison, out))


def test_unmatched_saved_row_fails(tmp_path, module):
    data, comparison, _ = write_multiallelic(tmp_path, drop_external_row=True)
    with pytest.raises(ValueError, match="lack a matching netmhcpan4.el row"):
        module.main(arguments(data, comparison, tmp_path / "out"))


def test_flank_modes_must_share_rows(tmp_path, module):
    data, comparison, _ = write_multiallelic(tmp_path)
    path = comparison / "presentation" / "predictions_without_flanks.csv.bz2"
    pandas.read_csv(path).iloc[::-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="identical rows"):
        module.main(arguments(data, comparison, tmp_path / "out"))


def test_monoallelic_affinity_verifies_saved_external_column(tmp_path, module):
    data = tmp_path / "data_evaluation"
    data.mkdir()
    affinity = tmp_path / "comparison" / "affinity"
    affinity.mkdir(parents=True)
    saved = []
    for index, sample in enumerate(["A0201", "B0702"]):
        rows = benchmark_rows(sample, index + 3).assign(hla="HLA-A*02:01")
        suffix = sample + ".0.csv.bz2"
        for predictor in EXTERNAL:
            rows[KEYS + [predictor]].to_csv(
                data / ("benchmark.monoallelic.%s.train_excluded.%s" % (predictor, suffix)), index=False)
        saved.append(rows.assign(source_file="benchmark.monoallelic.mixmhcpred.train_excluded." + suffix))
    saved = pandas.concat(saved, ignore_index=True)
    rng = numpy.random.default_rng(11)
    saved["a_pred"] = numpy.where(saved.hit == 1, rng.uniform(5, 500, len(saved)), rng.uniform(100, 40000, len(saved)))
    saved["b_pred"] = numpy.where(saved.hit == 1, rng.uniform(10, 2000, len(saved)), rng.uniform(100, 40000, len(saved)))
    columns = KEYS + ["source_file", "mixmhcpred", "a_pred", "b_pred"]
    saved[columns].to_csv(affinity / "predictions.csv.bz2", index=False)
    out = tmp_path / "out"
    assert module.main(arguments(data, tmp_path / "comparison", out, "--cohort", "monoallelic",
                                 "--skip-joined-table")) == 0
    summary = pandas.read_csv(out / "summary.csv")
    assert set(summary.condition) == {"a_affinity", "b_affinity", *EXTERNAL}
    assert not (out / "joined_scores.csv.gz").exists()
    bad = saved[columns].copy()
    bad.loc[3, "mixmhcpred"] += 1
    bad.to_csv(affinity / "predictions.csv.bz2", index=False)
    with pytest.raises(ValueError, match="disagree"):
        module.main(arguments(data, tmp_path / "comparison", tmp_path / "out2", "--cohort", "monoallelic"))

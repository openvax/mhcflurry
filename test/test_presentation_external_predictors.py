"""External predictor comparisons join exact benchmark rows and orient scores."""

import importlib.util
import json
from pathlib import Path

import numpy
import pandas
import pytest

from mhcflurry.cli.compare_models import _metrics


KEYS = ["protein_accession", "peptide", "sample_id", "n_flank", "c_flank", "hit", "hla"]
PRECOMPUTED = ("netmhcpan4.el", "netmhcpan4.ba", "mixmhcpred")
LOCAL = ("netmhcpan4.1.el", "netmhcpan4.1.ba", "netmhcpan4.2.el", "netmhcpan4.2.ba")


@pytest.fixture
def module():
    path = (Path(__file__).resolve().parents[1]
            / "scripts/training/compare_presentation_external_predictors.py")
    spec = importlib.util.spec_from_file_location("presentation_external_predictors", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize("cohort,candidate,reference", [
    ("monoallelic", "a_affinity", "b_affinity"),
    ("multiallelic", "a_with_flanks", "b_with_flanks")])
def test_every_external_version_gets_paired_intervals(module, cohort, candidate, reference):
    conditions = dict.fromkeys([candidate, reference] + list(PRECOMPUTED + LOCAL))
    pairs = module.comparison_pairs(cohort, conditions)
    assert {(candidate, name) for name in PRECOMPUTED + LOCAL} <= set(pairs)
    assert {(reference, name) for name in PRECOMPUTED + LOCAL} <= set(pairs)
    assert len(pairs) == len(set(pairs))


def test_common_coverage_uses_one_row_set_and_never_drops_a_sample(module):
    frame = pandas.DataFrame({"sample_id": ["a", "a", "b"],
                              "first": [1., 2., 3.], "second": [numpy.nan, 2., 3.]})
    conditions = {name: module.Condition(name, True, name) for name in ("first", "second")}
    result = module.common_coverage(frame, conditions)
    assert result.index.tolist() == [1, 2]
    frame.loc[2, "second"] = numpy.nan
    with pytest.raises(ValueError, match="entire evaluation sample"):
        module.common_coverage(frame, conditions)


def benchmark_rows(sample, seed, n=300, n_hits=30):
    rng = numpy.random.default_rng(seed)
    hit = numpy.r_[numpy.ones(n_hits, dtype=int), numpy.zeros(n - n_hits, dtype=int)]
    peptides = ["PEPTIDE%04d" % index for index in range(n)]
    peptides[1] = peptides[0]  # identical identities must keep their file order
    frame = pandas.DataFrame({
        "protein_accession": "P1", "peptide": peptides, "sample_id": sample,
        "n_flank": "NA", "c_flank": "AC", "hit": hit,
        "hla": "HLA-B*07:02 HLA-A*02:01 HLA-A*02:01"})
    for name, weight in (("netmhcpan4.el", 0.3), ("mixmhcpred", 0.2),
                         ("netmhcpan4.1.el", 0.32), ("netmhcpan4.2.el", 0.35)):
        frame[name] = numpy.round(weight * hit + rng.uniform(0, 1, n), 6)
    for name in ("netmhcpan4.ba", "netmhcpan4.1.ba", "netmhcpan4.2.ba"):
        frame[name] = numpy.round(numpy.where(
            hit == 1, rng.uniform(5, 800, n), rng.uniform(50, 50000, n)), 3)
    return frame


def write_multiallelic(tmp_path, drop_external_row=False):
    data = tmp_path / "data_evaluation"
    data.mkdir()
    extra = tmp_path / "local_scores"
    extra.mkdir()
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
        for predictor in PRECOMPUTED:
            external = rows[KEYS + [predictor]].assign(**{predictor + "_best_allele": "HLA-A*02:01"})
            if drop_external_row and predictor == "netmhcpan4.el" and index == 0:
                external = external.drop(index=10)
            external.to_csv(data / ("benchmark.multiallelic.%s.train_excluded.%s" % (predictor, suffix)),
                            index=False)
        for predictor in LOCAL:
            # Locally generated scores: plain CSV in a directory of their own.
            rows[KEYS + [predictor]].to_csv(
                extra / ("benchmark.multiallelic.%s.train_excluded.%s" % (
                    predictor, suffix[:-len(".bz2")])), index=False)
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
    columns = list(PRECOMPUTED) + list(LOCAL)
    saved.drop(columns=columns).to_csv(presentation / "predictions_with_flanks.csv.bz2", index=False)
    without = saved.drop(columns=columns)
    without["a_presentation_score"] = numpy.clip(0.5 * saved.hit + rng.uniform(0, 0.6, len(saved)), 0, 1)
    without.to_csv(presentation / "predictions_without_flanks.csv.bz2", index=False)
    return data, extra, tmp_path / "comparison", saved


def arguments(data, extra, comparison, out, *rest):
    return ["--comparison-dir", str(comparison), "--data-dir", str(data),
            "--external-dir", str(extra), "--out", str(out), "--replicates", "100", *rest]


def test_joins_exact_rows_orients_scores_and_writes_figures(tmp_path, module):
    data, extra, comparison, saved = write_multiallelic(tmp_path)
    out = tmp_path / "out"
    assert module.main(arguments(data, extra, comparison, out, "--a-label", "Candidate")) == 0
    joined = pandas.read_csv(out / "joined_scores.csv.gz")
    assert joined.prediction_row.tolist() == list(range(len(saved)))
    for predictor in PRECOMPUTED + LOCAL:
        numpy.testing.assert_allclose(joined[predictor], saved[predictor], equal_nan=True)
    per_sample = pandas.read_csv(out / "per_sample_metrics.csv")
    s1 = saved.loc[saved.sample_id == "S1"]

    def row(condition):
        return per_sample.loc[(per_sample.sample_id == "S1") & (per_sample.condition == condition)].iloc[0]

    for predictor in ("netmhcpan4.ba", "netmhcpan4.2.ba"):
        expected = _metrics(s1.hit.to_numpy(), -s1[predictor].to_numpy())
        assert row(predictor).roc_auc == pytest.approx(expected["roc_auc"]) and expected["roc_auc"] > 0.5
    # MHCflurry keeps every row even where an external tool lacks a score.
    candidate = _metrics(s1.hit.to_numpy(), s1.a_presentation_score.to_numpy())
    assert row("a_with_flanks").n == len(s1) and row("netmhcpan4.el").n == len(s1) - 2
    assert row("netmhcpan4.2.el").n == len(s1)
    assert row("a_with_flanks").pr_auc == pytest.approx(candidate["pr_auc"])
    assert row("a_with_flanks").ppv_at_n == pytest.approx(candidate["ppv_at_n"])
    percentile = _metrics(s1.hit.to_numpy(), -s1.a_presentation_percentile.to_numpy())
    assert row("a_with_flanks_percentile").roc_auc == pytest.approx(percentile["roc_auc"])
    paired = pandas.read_csv(out / "paired_differences.csv")
    assert {("a_with_flanks", "b_with_flanks"), ("a_with_flanks", "netmhcpan4.el"),
            ("a_with_flanks", "netmhcpan4.2.el"), ("a_with_flanks", "netmhcpan4.2.ba"),
            ("a_with_flanks", "netmhcpan4.ba")} <= set(zip(paired.candidate, paired.reference))
    excluded = paired.groupby(["candidate", "reference"]).rows_excluded.first()
    assert excluded[("a_with_flanks", "b_with_flanks")] == 0
    assert excluded[("a_with_flanks", "netmhcpan4.el")] == 2
    assert excluded[("a_with_flanks", "netmhcpan4.2.el")] == 0
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
    assert coverage["netmhcpan4.2.el_unscored"].sum() == 0
    provenance = json.loads((out / "provenance.json").read_text())
    paths = [item["path"] for item in provenance["inputs"]]
    assert any(path.endswith("benchmark.multiallelic.netmhcpan4.ba.train_excluded.S2.0.csv.bz2") for path in paths)
    assert any(path.endswith("benchmark.multiallelic.netmhcpan4.2.el.train_excluded.S2.0.csv") for path in paths)
    with pytest.raises(ValueError, match="new or empty"):
        module.main(arguments(data, extra, comparison, out))


def test_unmatched_saved_row_fails(tmp_path, module):
    data, extra, comparison, _ = write_multiallelic(tmp_path, drop_external_row=True)
    with pytest.raises(ValueError, match="lack a matching netmhcpan4.el row"):
        module.main(arguments(data, extra, comparison, tmp_path / "out"))


def test_common_coverage_cli_reports_identical_denominators(tmp_path, module):
    data, extra, comparison, saved = write_multiallelic(tmp_path)
    out = tmp_path / "out"
    assert module.main(arguments(data, extra, comparison, out,
                                 "--coverage", "common", "--baselines", "none")) == 0
    summary = pandas.read_csv(out / "summary.csv")
    assert summary.rows.nunique() == 1 and summary.hits.nunique() == 1
    assert summary.rows.iloc[0] == len(saved) - 2
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["coverage"]["excluded_rows"] == 2
    coverage = pandas.read_csv(out / "coverage.csv")
    assert coverage.common_excluded.sum() == coverage["netmhcpan4.el_unscored"].sum() == 2
    assert pandas.read_csv(out / "paired_differences.csv").rows_excluded.eq(0).all()


def test_missing_external_file_names_the_searched_directories(tmp_path, module):
    data, extra, comparison, _ = write_multiallelic(tmp_path)
    for path in extra.glob("*netmhcpan4.2.el*"):
        path.rename(path.with_suffix(".moved"))
    with pytest.raises(FileNotFoundError, match="netmhcpan4.2.el"):
        module.main(arguments(data, extra, comparison, tmp_path / "out"))


def test_flank_modes_must_share_rows(tmp_path, module):
    data, extra, comparison, _ = write_multiallelic(tmp_path)
    path = comparison / "presentation" / "predictions_without_flanks.csv.bz2"
    pandas.read_csv(path).iloc[::-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="identical rows"):
        module.main(arguments(data, extra, comparison, tmp_path / "out"))


def test_monoallelic_affinity_verifies_saved_external_column(tmp_path, module):
    data = tmp_path / "data_evaluation"
    data.mkdir()
    extra = tmp_path / "local_scores"
    extra.mkdir()
    affinity = tmp_path / "comparison" / "affinity"
    affinity.mkdir(parents=True)
    saved = []
    for index, sample in enumerate(["A0201", "B0702"]):
        rows = benchmark_rows(sample, index + 3).assign(hla="HLA-A*02:01")
        suffix = sample + ".0.csv.bz2"
        for predictor in PRECOMPUTED + LOCAL:
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
    assert module.main(arguments(data, extra, tmp_path / "comparison", out, "--cohort", "monoallelic",
                                 "--skip-joined-table")) == 0
    summary = pandas.read_csv(out / "summary.csv")
    assert set(summary.condition) == {"a_affinity", "b_affinity", *PRECOMPUTED, *LOCAL,
                                      "random", "terminal_logistic"}
    assert not (out / "joined_scores.csv.gz").exists()
    bad = saved[columns].copy()
    bad.loc[3, "mixmhcpred"] += 1
    bad.to_csv(affinity / "predictions.csv.bz2", index=False)
    with pytest.raises(ValueError, match="disagree"):
        module.main(arguments(data, extra, tmp_path / "comparison", tmp_path / "out2",
                              "--cohort", "monoallelic"))


def test_reference_baselines_are_scored_without_leaking_labels(tmp_path, module):
    data, extra, comparison, _ = write_multiallelic(tmp_path)
    out = tmp_path / "out"
    assert module.main(arguments(data, extra, comparison, out)) == 0
    summary = pandas.read_csv(out / "summary.csv").set_index("condition")
    assert 0.3 < summary.loc["random", "macro_roc_auc"] < 0.7
    assert "terminal_logistic" in summary.index
    paired = pandas.read_csv(out / "paired_differences.csv")
    assert ("a_with_flanks", "terminal_logistic") in set(zip(paired.candidate, paired.reference))
    frame = pandas.DataFrame({
        "peptide": ["SIINFEKL", "GILGFVFT", "NLVPMVAT", "AAAAAAAA"] * 30,
        "hit": [1, 0, 0, 0] * 30,
        "sample_id": ["A"] * 60 + ["B"] * 60})
    before = module.leave_one_sample_out_logistic(frame)
    flipped = frame.copy()
    flipped.loc[flipped.sample_id == "A", "hit"] = [0, 1, 0, 0] * 15
    after = module.leave_one_sample_out_logistic(flipped)
    in_a = (frame.sample_id == "A").to_numpy()
    numpy.testing.assert_allclose(after[in_a], before[in_a])
    assert not numpy.allclose(after[~in_a], before[~in_a])


def test_terminal_features_encode_both_ends_and_unknown_residues(module):
    features = module.terminal_residue_features(["ACDEFGHIK", "AC"]).toarray()
    assert features.shape == (2, 8 * 21) and (features.sum(axis=1) == 8).all()
    assert features[0, 0 * 21 + 0] == 1 and features[0, 7 * 21 + module.AMINO_ACIDS.index("K")] == 1
    assert features[1, 2 * 21 + 20] == 1  # short peptide padded with the unknown symbol


def test_reference_baselines_can_be_disabled(tmp_path, module):
    data, extra, comparison, _ = write_multiallelic(tmp_path)
    out = tmp_path / "out"
    assert module.main(arguments(data, extra, comparison, out, "--baselines", "none")) == 0
    assert not {"random", "terminal_logistic"} & set(pandas.read_csv(out / "summary.csv").condition)

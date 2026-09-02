"""Tests for common-cohort affinity candidate figure inputs."""

import json

import numpy
import pandas
import pytest

from mhcflurry.cli import affinity_candidate_figures
from mhcflurry.cli import merge_external_predictions
from mhcflurry.cli import paper_figures


def _write_comparison(root, condition, baseline, identity, offset):
    if condition == baseline:
        comparison = root / "baseline-vs-public-no-additional-ms"
    else:
        comparison = (
            root / condition / "comparison-vs-public-no-additional-ms")
    affinity = comparison / "affinity"
    affinity.mkdir(parents=True)
    rows = 40
    hit = numpy.arange(rows) % 2
    public_score = numpy.where(hit, 0.8, 0.2)
    frame = pandas.DataFrame({
        "source_file": "benchmark.csv",
        "protein_accession": "P12345",
        "sample_id": "sample-1",
        "hla": "HLA-A*02:01",
        "peptide": ["PEPTIDE%02d" % index for index in range(rows)],
        "n_flank": "NNNNN",
        "c_flank": "CCCCC",
        "hit": hit,
        "peptide_len": 9,
        "a_score": public_score + offset,
        "b_score": public_score,
        "netmhcpan4.ba": numpy.where(hit, 50.0, 5000.0),
        "netmhcpan4.el": numpy.where(hit, 0.9, 0.1),
    })
    frame.to_csv(affinity / "predictions.csv.bz2", index=False)
    (affinity / "summary.json").write_text(json.dumps({
        "n_rows": rows,
        "n_hits": int(hit.sum()),
        "benchmark_identity": identity,
    }))


def test_candidate_figure_inputs_include_public_and_external_predictions(
        tmp_path):
    factorial = tmp_path / "factorial"
    factorial.mkdir()
    baseline = "keras_128"
    candidate = "native_1024"
    (factorial / "manifest.json").write_text(json.dumps({
        "baseline_condition": baseline,
        "records": [
            {"condition": baseline},
            {"condition": candidate},
        ],
    }))
    identity = {
        "algorithm": "test",
        "columns": ["source_file", "hla", "peptide", "hit"],
        "ordered_rows": True,
        "row_count": 40,
        "sha256": "a" * 64,
    }
    _write_comparison(factorial, baseline, baseline, identity, 0.01)
    _write_comparison(factorial, candidate, baseline, identity, 0.02)

    out = tmp_path / "figures"
    provenance = affinity_candidate_figures.build_candidate_figure_inputs(
        factorial,
        out,
        [baseline, candidate],
        "public_2_2",
    )

    predictions = pandas.read_csv(out / "benchmark.monoallelic.csv.bz2")
    assert {
        baseline, candidate, "public_2_2", "netmhcpan4.ba", "netmhcpan4.el",
    }.issubset(predictions.columns)
    numpy.testing.assert_allclose(
        predictions.public_2_2,
        numpy.where(predictions.hit, 0.8, 0.2),
    )
    info = pandas.read_csv(out / "predictor_info.csv")
    assert set(info.predictor) == {
        baseline, candidate, "public_2_2", "netmhcpan4.ba", "netmhcpan4.el",
    }
    scores = pandas.read_csv(out / "accuracy_scores.monoallelic.csv")
    assert set(scores.predictor) == set(info.predictor)
    assert provenance["benchmark_identity"]["sha256"] == "a" * 64
    assert provenance["external_predictors_included"] == [
        "netmhcpan4.ba", "netmhcpan4.el"]


def test_candidate_figure_inputs_join_external_prediction_table(tmp_path):
    factorial = tmp_path / "factorial"
    factorial.mkdir()
    baseline = "keras_128"
    (factorial / "manifest.json").write_text(json.dumps({
        "baseline_condition": baseline,
        "records": [{"condition": baseline}],
    }))
    identity = {
        "algorithm": "test",
        "columns": ["source_file", "hla", "peptide", "hit"],
        "ordered_rows": True,
        "row_count": 40,
        "sha256": "b" * 64,
    }
    _write_comparison(factorial, baseline, baseline, identity, 0.01)
    source = pandas.read_csv(
        factorial / "baseline-vs-public-no-additional-ms" /
        "affinity" / "predictions.csv.bz2")
    external = tmp_path / "external.csv.bz2"
    source[[
        "protein_accession", "sample_id", "hla", "peptide", "n_flank",
        "c_flank", "hit",
    ]].assign(
        mixmhcpred=numpy.where(source.hit, 0.9, 0.1),
    ).to_csv(external, index=False)

    out = tmp_path / "figures"
    provenance = affinity_candidate_figures.build_candidate_figure_inputs(
        factorial,
        out,
        [baseline],
        "public_2_2",
        [external],
    )

    predictions = pandas.read_csv(out / "benchmark.monoallelic.csv.bz2")
    assert "mixmhcpred" in predictions
    assert predictions.mixmhcpred.notnull().all()
    assert provenance["external_predictors_included"] == [
        "netmhcpan4.ba", "netmhcpan4.el", "mixmhcpred"]
    record = provenance["external_prediction_sources"][0]
    assert record["matched_candidate_rows"] == 40
    assert record["finite_prediction_rows"] == {"mixmhcpred": 40}

    source.iloc[:-1][[
        "protein_accession", "sample_id", "hla", "peptide", "n_flank",
        "c_flank", "hit",
    ]].assign(mixmhcpred=0.5).to_csv(external, index=False)
    with pytest.raises(ValueError, match="failed to cover 1 of 40"):
        affinity_candidate_figures.build_candidate_figure_inputs(
            factorial,
            tmp_path / "incomplete",
            [baseline],
            "public_2_2",
            [external],
        )


def test_candidate_figure_render_includes_every_candidate_row(
        tmp_path, monkeypatch):
    conditions = ["keras_128", "native_1024"]
    captured = {}
    monkeypatch.setattr(
        affinity_candidate_figures,
        "build_candidate_figure_inputs",
        lambda *_args, **_kwargs: {
            "outputs": {"predictions": str(tmp_path / "predictions.csv")},
            "external_predictors_included": ["netmhcpan4.ba"],
        },
    )

    def fake_render(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(paper_figures, "run", fake_render)
    args = affinity_candidate_figures.make_parser().parse_args([
        "--factorial-dir", str(tmp_path / "factorial"),
        "--out", str(tmp_path / "figures"),
        "--condition", conditions[0],
        "--condition", conditions[1],
    ])

    assert affinity_candidate_figures.run(args) == 0
    rendered = captured["args"]
    assert rendered.candidate_predictor == conditions[0]
    assert rendered.monoallelic_panel_predictors == ",".join(conditions)
    assert rendered.external_baselines == (
        "mhcflurry_public_2_2,netmhcpan4.ba")


def test_merge_external_prediction_groups(tmp_path):
    metadata = pandas.DataFrame({
        "protein_accession": ["P1", "P2"],
        "peptide": ["AAAAAAAA", "BBBBBBBB"],
        "sample_id": ["sample-1", "sample-1"],
        "n_flank": ["NNNNN", "NNNNN"],
        "c_flank": ["CCCCC", "CCCCC"],
        "hit": [1, 0],
        "hla": ["HLA-A*02:01", "HLA-A*02:01"],
    })
    specs = []
    for predictor, values in [
            ("netmhcpan4.ba", [20.0, 2000.0]),
            ("netmhcpan4.el", [0.9, 0.1]),
            ("mixmhcpred", [0.8, 0.2])]:
        member_name = (
            "benchmark.monoallelic.%s.train_excluded.sample-1.csv.bz2" %
            predictor)
        member = tmp_path / member_name
        metadata.assign(**{
            predictor: values,
            "%s_best_allele" % predictor: "HLA-A*02:01",
        }).to_csv(member, index=False)
        manifest = tmp_path / ("group.%s.csv" % predictor)
        pandas.DataFrame({"filename": [member_name]}).to_csv(
            manifest, index=False)
        specs.append("%s=%s" % (predictor, manifest))

    out = tmp_path / "external.csv.bz2"
    provenance = merge_external_predictions.merge_external_prediction_groups(
        specs, out)
    result = pandas.read_csv(out)

    assert list(result.columns) == list(metadata.columns) + [
        "netmhcpan4.ba", "netmhcpan4.el", "mixmhcpred"]
    assert provenance["member_count"] == 1
    assert provenance["row_count"] == 2
    assert provenance["output"]["sha256"]
    assert (tmp_path / "external.csv.bz2.provenance.json").is_file()

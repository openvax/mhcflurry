"""Tests for common-cohort affinity candidate figure inputs."""

import json

import numpy
import pandas

from mhcflurry.cli import affinity_candidate_figures


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
        "hla": "HLA-A*02:01",
        "peptide": ["PEPTIDE%02d" % index for index in range(rows)],
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

import numpy
import pandas
import pytest

from mhcflurry.cli import compare_models


def test_uncalibrated_combiner_requires_explicit_score_only_mode():
    frame = pandas.DataFrame({"peptide": ["SIINFEKL"],
        "a_presentation_score": [0.2], "b_presentation_score": [0.3],
        "a_presentation_percentile": [numpy.nan], "b_presentation_percentile": [1.0]})
    with pytest.raises(ValueError, match="presentation_percentile"):
        compare_models._require_finite_presentation_scores(frame, "with_flanks", ("new", "public"))
    compare_models._require_finite_presentation_scores(
        frame, "with_flanks", ("new", "public"), score_kinds=("presentation_score",))
    frame.loc[0, "a_presentation_score"] = numpy.nan
    with pytest.raises(ValueError, match="a_presentation_score|presentation_score"):
        compare_models._require_finite_presentation_scores(
            frame, "with_flanks", ("new", "public"), score_kinds=("presentation_score",))


def test_default_comparison_still_requires_scores_and_percentiles():
    args = compare_models.make_parser().parse_args(["--a", "new", "--out", "eval"])
    assert args.presentation_score_kinds == "presentation_score,presentation_percentile"


def test_summary_markdown_accepts_explicit_score_only_comparison(tmp_path):
    headline = {"presentation": {"modes": ["with_flanks"], "summaries": {
        "with_flanks": {"presentation_score": {"micro_pooled": {
            "a": {"roc_auc": 0.9}, "b": {"roc_auc": 0.8}}, "n_samples_reported": 1}}}}}
    compare_models._write_summary_markdown(
        headline, {"label": "new", "spec": "new"}, {"label": "public", "spec": "public"},
        str(tmp_path), ["presentation"])
    result = (tmp_path / "summary.md").read_text()
    assert "with_flanks / presentation_score" in result
    assert "with_flanks / presentation_percentile" not in result


def test_score_only_component_persists_predictions_without_fabricated_percentiles(tmp_path, monkeypatch):
    benchmark = pandas.DataFrame({"peptide": ["SIINFEKL", "SLYNTVATL"], "hit": [0, 1],
        "sample_id": ["sample"] * 2, "hla": ["HLA-A*02:01"] * 2, "peptide_len": [8, 9]})
    monkeypatch.setattr(compare_models, "_load_presentation_benchmark_for_component", lambda *a: benchmark)
    monkeypatch.setattr(compare_models, "model_artifact_size_bytes", lambda *a: 1)
    prediction = pandas.DataFrame({"presentation_score": [0.1, 0.9],
        "presentation_percentile": [numpy.nan] * 2, "affinity": [1000, 10], "processing_score": [0.1, 0.9]})
    monkeypatch.setattr(compare_models, "_parallel_presentation_predict", lambda *a, **k: prediction)
    args = compare_models.make_parser().parse_args([
        "--a", "new", "--out", str(tmp_path), "--data-dir", str(tmp_path),
        "--presentation-modes", "with_flanks", "--presentation-score-kinds", "presentation_score"])
    side = {"label": "uncalibrated", "paths": {"presentation": str(tmp_path)}}
    result = compare_models._run_presentation(side, side, args)
    assert result["score_kinds"] == ["presentation_score"]
    saved = pandas.read_csv(tmp_path / "presentation/predictions_with_flanks.csv.bz2")
    assert saved.a_presentation_percentile.isna().all()
    assert saved.a_presentation_score.tolist() == [0.1, 0.9]

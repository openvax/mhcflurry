"""Expanded processing evaluation preserves all hits and one shared cohort."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy
import pandas
import pytest

from mhcflurry.processing_evaluation import load_processing_cohort, verify_hits
from mhcflurry.processing_matching import IncompleteProcessingMatches, validate_matched_training_data


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[1] / "scripts/training/prepare_processing_evaluation.py"
    spec = importlib.util.spec_from_file_location("prepare_processing_evaluation", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def pool(length=8):
    return pandas.DataFrame({
        "sample_id": "s", "peptide": [letter * length for letter in "ACDE"],
        "protein_accession": "p", "hit": [1, 1, 0, 0], "n_flank": "NNN", "c_flank": "CCC",
        "hla": "HLA-A*02:01", "affinity_prediction": 100.0})


def arguments(tmp_path):
    args = argparse.Namespace(out=tmp_path / "matched.csv", random_seed=42,
        matching_reference={"sha256": "a" * 64}, decoys_per_hit=10,
        max_affinity_distance=.25, max_expansion_rounds=3,
        expansion_candidates_per_length=100, resume_matching_dir=None)
    Path(str(args.out) + ".matching").mkdir()
    return args


@pytest.mark.parametrize("length", [8, 13, 15])
def test_expand_real_proteome_windows_keeps_all_hits_and_unique_decoys(tmp_path, module, length):
    args, original = arguments(tmp_path), pool(length)
    scored = []

    def score(peptides):
        scored.extend(peptides)
        return numpy.full(len(peptides), 100.0)

    sequences = {"p": "ACDEFGHIKLMNPQRSTVWY" * 9}
    result = module.expand_sample(args, original, sequences, score)
    verify_hits(original, result)
    validate_matched_training_data(result)
    assert result.hit.sum() == 2 and len(result) == 22
    assert not result.loc[result.hit.eq(0), "peptide"].duplicated().any()
    assert set(scored).isdisjoint(original.peptide)
    assert result.peptide.str.len().eq(length).all()
    assert result.hla.eq("HLA-A*02:01").all()
    scored.clear()
    resumed = module.expand_sample(args, original, sequences, score)
    assert not scored
    pandas.testing.assert_frame_equal(result, resumed, check_dtype=False)


def test_exhaustion_preserves_failure_instead_of_reusing_negatives(tmp_path, module):
    args = arguments(tmp_path)
    with pytest.raises(IncompleteProcessingMatches, match="No unscored candidates"):
        module.expand_sample(args, pool(), {"p": "AAAAAAAA"},
                             lambda peptides: numpy.full(len(peptides), 100.0))
    assert list(Path(str(args.out) + ".matching").glob("*.failure.json"))


def test_cli_writes_checked_cohort_and_rejects_corruption(tmp_path, module, monkeypatch):
    import mhcflurry
    original = pool()
    source = tmp_path / "cached.csv"
    original.to_csv(source, index=False)
    reference = tmp_path / "reference.csv"
    pandas.DataFrame({"accession": ["p"], "seq": ["ACDEFGHIKLMNPQRSTVWY" * 9]}).to_csv(reference, index=False)
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pandas.DataFrame({"sample_id": ["s"]}).to_csv(holdout / "processing_samples.csv", index=False)
    monkeypatch.setattr(module, "affinity_reference_fingerprint", lambda path: {"sha256": "a" * 64})
    monkeypatch.setattr(module, "_load_presentation_benchmark_for_component",
                        lambda *args: original.assign(source_file=source.name))
    monkeypatch.setattr(module, "_attach_affinity", lambda frame, data_dir: (
        frame.assign(**{module.AFFINITY_COLUMN: 100.0}),
        [{"path": str(source), "sha256": module.sha256_file(source)}]))
    monkeypatch.setattr(mhcflurry.Class1AffinityPredictor, "load", lambda path: None)

    class Predictor:
        def __init__(self, **kwargs):
            pass

        def predict_affinity(self, peptides, **kwargs):
            return pandas.DataFrame({"affinity": numpy.full(len(peptides), 100.0)})

    monkeypatch.setattr(mhcflurry, "Class1PresentationPredictor", Predictor)
    out = tmp_path / "out"
    argv = ["--data-dir", str(tmp_path), "--release-holdout-dir", str(holdout),
            "--proteome-reference-csv", str(reference), "--affinity-predictor", "public",
            "--expansion-candidates-per-length", "100", "--out", str(out)]
    assert module.main(argv) == 0
    frame, metadata = load_processing_cohort(out, original)
    assert len(frame) == 22 and metadata["hits"] == 2
    # Exercise the maintained comparison consumer too: both models must see
    # exactly these ordered rows, without rematching the original short pool.
    from mhcflurry.cli import compare_models
    compare_args = compare_models.make_parser().parse_args([
        "--a", "candidate", "--b", "baseline", "--out", str(tmp_path / "comparison"),
        "--data-dir", str(tmp_path), "--processing-modes", "no_flank",
        "--processing-matched-cohort", str(out)])
    monkeypatch.setattr(compare_models, "_processing_model_dirs", lambda *args: {"no_flank": ("a", "b")})
    monkeypatch.setattr(compare_models, "_load_presentation_benchmark_for_component", lambda *args: original.copy())
    seen = []

    def predict(args, path, rows, mode, label, model_bytes=None):
        seen.append(rows.peptide.tolist())
        return rows.hit.to_numpy() * .8 + .1

    monkeypatch.setattr(compare_models, "_parallel_processing_predict", predict)
    summary = compare_models._run_processing({"label": "a"}, {"label": "b"}, compare_args)
    assert summary["negative_policy"] == "matched"
    assert seen == [frame.peptide.tolist(), frame.peptide.tolist()]
    assert module.main(argv + ["--resume"]) == 0
    with pytest.raises(ValueError, match="changed held-out hits"):
        load_processing_cohort(out, original.iloc[1:].copy())
    metadata["seed"] += 1
    (out / "cohort.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata disagrees"):
        load_processing_cohort(out, original)
    (out / "matched.csv.bz2").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_processing_cohort(out, original)

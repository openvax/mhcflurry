"""Adaptive matching, cache integrity and numeric-position sampling contracts."""

import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy
import pandas
import pytest

from mhcflurry.processing_matching import (
    IncompleteProcessingMatches, initialize_matching_artifacts,
    validate_preparation_provenance, validate_matched_training_data)
from mhcflurry.processing_preparation import SamplePreparation, prepare_sample
from mhcflurry.proteome_decoys import (
    make_peptide_frame_for_accessions, sample_peptide_frame_for_accessions)


def inputs(tmp_path, sample="s"):
    args = SimpleNamespace(out=str(tmp_path / "train.csv"),
        matching_reference={"sha256": "a" * 64}, decoys_per_hit=1,
        max_affinity_distance=.25, max_expansion_rounds=2,
        expansion_candidates_per_length=10, resume_matching_dir=None)
    (tmp_path / "train.csv.matching").mkdir(parents=True)
    hits = pandas.DataFrame({"sample_id": sample, "peptide": ["A" * 8, "C" * 9],
                             "protein_accession": "p", "n_flank": "NN", "c_flank": "CC", "hit": 1})
    pool = pandas.concat([hits, hits.assign(peptide=["D" * 8, "E" * 9], hit=0)], ignore_index=True)
    return args, hits, pool


def no_call(*args, **kwargs):
    raise AssertionError("A cached sample must not be sampled or scored again")


def score(peptides):
    return numpy.array([500.0 if p == "D" * 8 else 50.0 for p in peptides])


def additional(length, excluded, seed):
    assert length == 8
    assert {"A" * 8, "D" * 8} <= excluded
    assert isinstance(seed, int)
    return pandas.DataFrame({"sample_id": ["s"], "peptide": ["F" * 8], "hit": [0],
                             "protein_accession": ["p"], "n_flank": ["NN"], "c_flank": ["CC"]})


def test_expansion_only_scores_new_unresolved_length_and_resumes(tmp_path):
    args, hits, pool = inputs(tmp_path)
    calls = []

    def predict(peptides):
        calls.append(list(peptides))
        return score(peptides)

    first = prepare_sample(args, "s", hits, 42, lambda: pool.copy(), additional, predict)
    assert calls == [pool.peptide.tolist(), ["F" * 8]]
    validate_matched_training_data(first)
    assert set(first.loc[first.hit == 0, "peptide"]) == {"E" * 9, "F" * 8}
    second = prepare_sample(args, "s", hits, 42, no_call, no_call, no_call)
    pandas.testing.assert_frame_equal(first, second, check_dtype=False)
    rounds = sorted((tmp_path / "train.csv.matching").glob("*.round-*.json"))
    assert len(rounds) == 2
    record = json.loads(rounds[1].read_text())
    assert record["lengths"] == [8]
    assert record["rows"] == 1
    assert "8" in record["seeds_by_length"]
    assert len(record["sha256"]) == 64


@pytest.mark.parametrize("interrupt_expansion", [False, True])
def test_numeric_expansion_filters_committed_peptides_and_resumes(tmp_path, interrupt_expansion):
    from mhcflurry.numeric_proteome import ProcessingProteome, ProteinWindows, NumericCandidatePool
    args, hits, _ = inputs(tmp_path)
    index = ProcessingProteome({"p": "D" * 9 + "E" * 10 + "F" * 9})
    initial = NumericCandidatePool(hits, ProteinWindows(index, [0, 0], [0, 9], [8, 9]), "s", 2)
    calls = []

    def expand(length, excluded, seed):
        assert length == 8
        assert {"A" * 8, "D" * 8} <= excluded
        # Even a sampler returning a previously committed peptide must not
        # cause repeat inference; the workflow enforces numeric exclusions.
        return NumericCandidatePool(hits.iloc[:0],
            ProteinWindows(index, [0, 0], [0, 19], [8, 8]), "s", 2)

    def predict(request):
        assert isinstance(request, NumericCandidatePool)
        peptides = request.prediction_input("cpu").to_strings().tolist()
        calls.append(peptides)
        if interrupt_expansion and len(calls) == 2:
            raise RuntimeError("interrupt expansion scoring")
        return score(peptides)

    if interrupt_expansion:
        with pytest.raises(RuntimeError, match="interrupt expansion"):
            prepare_sample(args, "s", hits, 42, lambda: initial, expand, predict)
        assert not list((tmp_path / "train.csv.matching").glob("*.complete.json"))
        result = prepare_sample(args, "s", hits, 42, no_call, expand, predict)
        assert calls == [["A" * 8, "C" * 9, "D" * 8, "E" * 9], ["F" * 8], ["F" * 8]]
    else:
        result = prepare_sample(args, "s", hits, 42, lambda: initial, expand, predict)
        assert calls == [["A" * 8, "C" * 9, "D" * 8, "E" * 9], ["F" * 8]]
    validate_matched_training_data(result)
    assert set(result.loc[result.hit == 0, "peptide"]) == {"E" * 9, "F" * 8}
    cached = prepare_sample(args, "s", hits, 42, no_call, no_call, no_call)
    pandas.testing.assert_frame_equal(result, cached, check_dtype=False)


def test_expansion_failure_is_pickleable_and_bounded(tmp_path):
    args, hits, pool = inputs(tmp_path)
    args.max_expansion_rounds = 1

    def bad_additional(*a):
        return additional(*a).assign(peptide="G" * 8)

    with pytest.raises(IncompleteProcessingMatches, match="budget exhausted") as failure:
        prepare_sample(args, "s", hits, 42, lambda: pool.copy(), bad_additional,
                       lambda peptides: [50 if p[0] in "ACE" else 500 for p in peptides])
    cloned = pickle.loads(pickle.dumps(failure.value))
    assert cloned.failures == failure.value.failures
    assert cloned.failures[0]["peptide_length"] == 8
    assert not list((tmp_path / "train.csv.matching").glob("*.complete.json"))


def test_interrupted_after_scoring_reuses_committed_round(tmp_path, monkeypatch):
    args, hits, pool = inputs(tmp_path)
    original = SamplePreparation.finish

    def interrupt(*a):
        raise RuntimeError("interrupted after scored rounds")

    monkeypatch.setattr(SamplePreparation, "finish", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_sample(args, "s", hits, 42, lambda: pool.copy(), additional, score)
    monkeypatch.setattr(SamplePreparation, "finish", original)
    result = prepare_sample(args, "s", hits, 42, no_call, no_call, no_call)
    validate_matched_training_data(result)


def test_pipeline_error_drains_inflight_checkpoint_and_resume_reuses_it(tmp_path, monkeypatch):
    import threading
    from mhcflurry.processing_preparation import sample_preparation_steps, ScoredValues
    from mhcflurry.scoring_pipeline import scoring_pipeline
    args, hits, pool = inputs(tmp_path, "first")
    writing = threading.Event()
    release_write = threading.Event()
    original = SamplePreparation.save_round

    def save(self, frame, record, source=None):
        if self.sample == "first":
            writing.set()
            assert release_write.wait(3)
        return original(self, frame, record, source)

    monkeypatch.setattr(SamplePreparation, "save_round", save)

    def second_pool():
        assert writing.wait(3)
        return pool.assign(sample_id="second")

    first = sample_preparation_steps(args, "first", hits, 42, lambda: pool.copy(), no_call)
    second = sample_preparation_steps(args, "second", hits.assign(sample_id="second"), 43, second_pool, no_call)

    def predict(key, peptides):
        if key == "second":
            release_write.set()
            raise RuntimeError("scoring interrupted")
        return ScoredValues(numpy.repeat(50.0, len(peptides)), .01)

    with pytest.raises(RuntimeError, match="scoring interrupted"):
        list(scoring_pipeline([("first", first), ("second", second)], predict, 2, 2))
    assert len(list((tmp_path / "train.csv.matching").glob("*.complete.json"))) == 1
    result = prepare_sample(args, "first", hits, 42, no_call, no_call, no_call)
    validate_matched_training_data(result)


@pytest.mark.parametrize("corrupt", ["pool", "result", "hits"])
def test_resume_rejects_changed_saved_data(tmp_path, corrupt):
    args, hits, pool = inputs(tmp_path)
    prepare_sample(args, "s", hits, 42, lambda: pool.copy(), additional, score)
    directory = tmp_path / "train.csv.matching"
    if corrupt == "pool":
        next(directory.glob("*.round-00.csv.bz2")).write_bytes(b"corrupt")
    elif corrupt == "result":
        next(directory.glob("*.matched.csv.bz2")).write_bytes(b"corrupt")
    else:
        hits.loc[0, "n_flank"] = "YY"
    with pytest.raises(ValueError, match="Corrupt|different observed"):
        prepare_sample(args, "s", hits, 42, no_call, no_call, no_call)


def test_import_legacy_pool_avoids_scoring_and_preserves_source(tmp_path):
    args, hits, pool = inputs(tmp_path / "new")
    prior = tmp_path / "old"
    prior.mkdir()
    args.resume_matching_dir = str(prior)
    name = hashlib.sha256(b"s").hexdigest()[:24]
    source = prior / (name + ".candidate_pool.csv.bz2")
    pool.assign(affinity_prediction=50).to_csv(source, index=False)
    before = source.read_bytes()
    result = prepare_sample(args, "s", hits, 42, no_call, no_call, no_call)
    validate_matched_training_data(result)
    assert source.read_bytes() == before
    record = json.loads(next((tmp_path / "new/train.csv.matching").glob("*.round-00.json")).read_text())
    assert record["origin_sha256"] == hashlib.sha256(before).hexdigest()
    assert record["origin_had_round_checksums"] is False


def test_import_expanded_pool_keeps_all_scored_rounds(tmp_path):
    old, hits, pool = inputs(tmp_path / "old")
    first = prepare_sample(old, "s", hits, 42, lambda: pool.copy(), additional, score)
    new, _, _ = inputs(tmp_path / "new")
    new.resume_matching_dir = old.out + ".matching"
    second = prepare_sample(new, "s", hits, 42, no_call, no_call, no_call)
    pandas.testing.assert_frame_equal(first, second, check_dtype=False)
    assert len(list((tmp_path / "new/train.csv.matching").glob("*.round-*.json"))) == 2


@pytest.mark.parametrize("lengths", [[8], [8, 9, 11], [2, 3]])
@pytest.mark.parametrize("seed", [1, 42])
def test_numeric_sampler_preserves_exact_population(lengths, seed):
    sequences = {"p1": "ACDEFGHIKLMNPQRSTVWY", "p2": "ACDEXGHIKLMNPQRSTVWY", "short": "AC"}
    reference = make_peptide_frame_for_accessions(sequences, sequences, lengths, 3)
    excluded = {reference.peptide.iloc[0]}
    reference = reference.loc[~reference.peptide.isin(excluded)]
    numpy.random.seed(seed)
    sampled = sample_peptide_frame_for_accessions(sequences, sequences, lengths, 3,
                                                 excluded, len(reference))
    order = ["protein_accession", "start_position", "peptide"]
    pandas.testing.assert_frame_equal(reference.sort_values(order).reset_index(drop=True),
                                     sampled.sort_values(order).reset_index(drop=True))
    with pytest.raises(ValueError, match="larger sample"):
        sample_peptide_frame_for_accessions(sequences, sequences, lengths, 3,
                                            excluded, len(reference) + 1)
    assert len(sample_peptide_frame_for_accessions(sequences, sequences, lengths, 3,
                excluded, len(reference) + 1, allow_smaller=True)) == len(reference)


def test_numeric_sampler_is_seeded_and_does_not_call_string_iterator(monkeypatch):
    import mhcflurry.proteome_decoys as module
    monkeypatch.setattr(module, "iter_protein_peptide_records", no_call)
    numpy.random.seed(42)
    first = sample_peptide_frame_for_accessions(["p"], {"p": "ACDEFGHIKLMNPQRSTVWY" * 10}, n=10)
    numpy.random.seed(42)
    second = sample_peptide_frame_for_accessions(["p"], {"p": "ACDEFGHIKLMNPQRSTVWY" * 10}, n=10)
    pandas.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["protein_accession", "start_position", "peptide"]).any()


def test_matching_provenance_rejects_changed_inputs():
    identity = dict(policy="policy", decoys_per_hit=1, max_log10_affinity_distance=.25,
                    candidate_pool_multiplier=100, seed=42, affinity_reference={"sha256": "a"},
                    inputs={"/old/hits.csv": "h", "/old/proteins.csv": "p"})
    relocated = {**identity, "inputs": {"/new/hits.csv": "h", "/new/proteins.csv": "p"}}
    validate_preparation_provenance(identity, relocated)
    for change in ({"seed": 1}, {"max_log10_affinity_distance": .3},
                   {"affinity_reference": {"sha256": "b"}},
                   {"inputs": {"/new/hits.csv": "changed", "/new/proteins.csv": "p"}}):
        with pytest.raises(ValueError, match="Changed"):
            validate_preparation_provenance(identity, {**relocated, **change})


def test_initialize_resume_is_explicit_and_verifies_source(tmp_path, monkeypatch):
    import mhcflurry.processing_matching as module
    hits = tmp_path / "hits.csv"
    proteins = tmp_path / "proteins.csv"
    hits.write_text("hits")
    proteins.write_text("proteins")
    args = SimpleNamespace(out=str(tmp_path / "train.csv"), negative_policy="matched",
        max_affinity_distance=.25, decoys_per_hit=1, ppv_multiplier=100, random_seed=42,
        hits=str(hits), proteome_peptides=None, proteome_reference_csv=str(proteins),
        affinity_predictor="reference", resume=False, resume_matching_dir=None)
    monkeypatch.setattr(module, "affinity_reference_fingerprint", lambda p: {"sha256": "a"})
    initialize_matching_artifacts(args)
    with pytest.raises(ValueError, match="fresh"):
        initialize_matching_artifacts(args)
    args.resume = True
    initialize_matching_artifacts(args)
    hits.write_text("changed hits")
    with pytest.raises(ValueError, match="Changed"):
        initialize_matching_artifacts(args)

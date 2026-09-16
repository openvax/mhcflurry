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


def test_expansion_supplies_distinct_negatives_for_competing_hits(tmp_path):
    args, _, _ = inputs(tmp_path)
    hits = pandas.DataFrame(dict(sample_id="s", peptide=["A" * 8, "C" * 8],
        protein_accession="p", n_flank="NN", c_flank="CC", hit=1))
    pool = pandas.concat([hits, hits.iloc[:1].assign(peptide="D" * 8, hit=0)], ignore_index=True)
    draws = []

    def expand(length, excluded, seed):
        draws.append(length)
        assert {"A" * 8, "C" * 8, "D" * 8} <= excluded
        return hits.iloc[:1].assign(peptide="E" * 8, hit=0)

    result = prepare_sample(args, "s", hits, 42, lambda: pool, expand,
                            lambda peptides: numpy.full(len(peptides), 100.0))
    assert draws == [8]
    assert set(result.loc[result.hit.eq(0), "peptide"]) == {"D" * 8, "E" * 8}
    cached = prepare_sample(args, "s", hits, 42, no_call, no_call, no_call)
    pandas.testing.assert_frame_equal(result, cached, check_dtype=False)


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


def window_population(frame):
    """The eligible (accession, start, length) windows a peptide frame covers."""
    return {(row.protein_accession, int(row.start_position), len(row.peptide))
            for row in frame.itertuples()}


@pytest.mark.parametrize("lengths", [[8], [9], [8, 9], [8, 9, 10, 11]])
def test_both_candidate_helpers_match_the_reservoir_window_population(lengths):
    """Enumerate every eligible window for both samplers against the reservoir.

    The historical reservoir (``iter_protein_peptide_records``) drops only the
    C-terminal *minimum*-length-mer, so its eligible set depends on the minimum
    length of the requesting call. ``sample_peptide_frame_for_accessions`` is
    pinned against the reservoir for the same length set; ``ProcessingProteome``
    exposes a single-length API, so it is pinned against ``lengths=[length]``,
    which is how make_train_data.processing.py draws both its initial pool and
    its expansion rounds. Protein lengths cover L == the requested length (an
    empty population), L one longer, and beyond.
    """
    from mhcflurry.numeric_proteome import ProcessingProteome
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for size in range(min(lengths), min(lengths) + 5):
        sequences = {"p": alphabet[:size]}
        positions = window_population(sample_peptide_frame_for_accessions(
            sequences, sequences, lengths, 0, (), n=10 ** 6, allow_smaller=True))
        assert positions == window_population(make_peptide_frame_for_accessions(
            sequences, sequences, lengths, 0)), (size, lengths)
        index = ProcessingProteome(sequences)
        for length in lengths:
            per_length = window_population(index.sample(
                ["p"], length, 10 ** 6, [], numpy.random.RandomState(0)).to_frame(0))
            assert per_length == window_population(make_peptide_frame_for_accessions(
                sequences, sequences, [length], 0)), (size, length)


def test_multi_length_position_call_keeps_longer_terminal_windows():
    """Pin the one documented divergence between the two candidate helpers.

    A multi-length ``positions`` call keeps each longer length's C-terminal
    window, because the window the reservoir drops is the *minimum*-length one.
    The per-length numeric sampler drops it because for its own call the
    requested length is the minimum. Both therefore reproduce the reservoir for
    the call shape they are given, and production only ever draws one length
    per call, so no window is reachable by one path and not the other.
    """
    from mhcflurry.numeric_proteome import ProcessingProteome
    sequences = {"p": "ACDEFGHIKLMNPQRSTVWY"[:12]}
    multi = window_population(sample_peptide_frame_for_accessions(
        sequences, sequences, [8, 9], 0, (), n=10 ** 6, allow_smaller=True))
    index = ProcessingProteome(sequences)
    per_length = set()
    for length in (8, 9):
        per_length |= window_population(index.sample(
            ["p"], length, 10 ** 6, [], numpy.random.RandomState(0)).to_frame(0))
    assert multi - per_length == {("p", 3, 9)}
    assert per_length - multi == set()


def test_expansion_draw_sequence_is_pinned_for_a_fixed_seed(tmp_path):
    """Pin the exact fixed-seed expansion draws, their order, and the result.

    Guards the per-round exclusion set in ``sample_preparation_steps``: every
    length in a round must still see the whole committed pool, and the drawn
    windows and peptides must stay identical for a given seed.
    """
    from mhcflurry.numeric_proteome import (
        ProcessingProteome, ProteinWindows, NumericCandidatePool)
    setup = numpy.random.RandomState(20260913)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    sequences = {"p%d" % i: "".join(setup.choice(alphabet, size=160)) for i in range(4)}
    proteome = ProcessingProteome(sequences)
    accessions = ["p0", "p1", "p2", "p3"]
    hits = pandas.DataFrame({
        "sample_id": "s", "hit": 1, "protein_accession": "p0",
        "peptide": [sequences["p0"][10:18], sequences["p0"][30:39]],
        "n_flank": "NNNNN", "c_flank": "CCCCC"})
    args = SimpleNamespace(out=str(tmp_path / "train.csv"),
        matching_reference={"sha256": "a" * 64}, decoys_per_hit=1,
        max_affinity_distance=.25, max_expansion_rounds=4,
        expansion_candidates_per_length=12, resume_matching_dir=None)
    (tmp_path / "train.csv.matching").mkdir(parents=True)
    draws, requests = [], []

    def initial_pool():
        parts = [proteome.sample(accessions, length, 3, set(hits.peptide),
                                 numpy.random.RandomState(4242)) for length in (8, 9)]
        return NumericCandidatePool(hits, ProteinWindows.concatenate(parts), "s", 5)

    def additional_candidates(length, excluded, round_seed):
        # Each length must see the complete committed pool, not a subset of it.
        assert excluded == set(requests[0])
        windows = proteome.sample(accessions, length, 12, excluded,
                                  numpy.random.RandomState(int(round_seed) % (2 ** 32)))
        draws.append([[int(p), int(s), int(w)] for p, s, w
                      in zip(windows.proteins, windows.starts, windows.lengths)])
        return NumericCandidatePool(hits.iloc[:0], windows, "s", 5)

    def score(request):
        peptides = request.prediction_input("cpu").to_strings().tolist()
        requests.append(peptides)
        # Initial decoys land outside the caliper; expansion decoys match.
        return numpy.array([100.0 if p in set(hits.peptide)
                            else (10000.0 if len(requests) == 1 else 100.0)
                            for p in peptides])

    result = prepare_sample(args, "s", hits, 4242, initial_pool,
                            additional_candidates, score)
    assert draws == [
        [[2, 71, 8], [1, 47, 8], [0, 147, 8], [0, 9, 8], [2, 24, 8], [0, 143, 8],
         [2, 70, 8], [3, 100, 8], [3, 139, 8], [3, 111, 8], [3, 61, 8], [3, 38, 8]],
        [[1, 138, 9], [2, 134, 9], [3, 118, 9], [0, 97, 9], [1, 27, 9], [0, 125, 9],
         [1, 30, 9], [0, 76, 9], [2, 83, 9], [3, 71, 9], [2, 42, 9], [2, 44, 9]]]
    assert requests == [
        ["MEFSAPTH", "PYPWELVWL", "WDVSNLKD", "EHKELLGY", "LGYYDHKA", "TSVYSTYWL",
         "IDVENMREQ", "CLQRHDSPF"],
        ["VIPKCDFF", "QCPNVAHR", "CPFILRTM", "IMEFSAPT", "TKEPDERI", "CLVYCPFI",
         "AVIPKCDF", "TYSGPMVS", "CTYANYFV", "IGWMCQDI", "LTLQHLEE", "MIHSSWSV",
         "VHHREHVDH", "RPNKTKSPF", "ITQCYQDDR", "CGWDVSNLK", "GASGRFTSV",
         "MSTQQYFYS", "GRFTSVYST", "ANCFWDIHH", "HEERPRPNH", "SDEPCQREA",
         "DNCWIDAWW", "CWIDAWWVM"]]
    validate_matched_training_data(result)
    assert result.peptide.tolist() == [
        "MEFSAPTH", "CPFILRTM", "PYPWELVWL", "ANCFWDIHH"]


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

"""Scientific matching invariants and an independent feasibility oracle."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from mhcflurry.processing_matching import (
    IncompleteProcessingMatches, _random_unique_assignments, make_affinity_controlled_risk_sets,
    matched_training_data, validate_matched_training_data)


def test_dense_repair_preserves_released_assignments_and_rng_state():
    # Frozen outputs from 2c0736d61, including 28 augmenting-path repairs.
    # A faster traversal must not change any seeded scientific assignment.
    fixture = json.loads((Path(__file__).parent / "data" /
                          "processing_matching_seeded_replay.json").read_text())
    for case in fixture["cases"]:
        rng = np.random.default_rng(case["seed"])
        result = _random_unique_assignments(
            np.array(case["targets"]), np.array(case["proteins"], dtype=object),
            np.array(case["values"]), np.array(case["negative_proteins"], dtype=object),
            case["count"], case["caliper"], case["protein_caliper"], rng)
        np.testing.assert_array_equal(result, case["expected"])
        np.testing.assert_array_equal(rng.integers(0, 2**32, size=8), case["next_random"])


def pool(hits, negatives, proteins=None):
    values = np.r_[hits, negatives].astype(float)
    return pd.DataFrame(dict(
        sample_id="sample", peptide=[format(i, "08d") for i in range(len(values))],
        peptide_len=8, hit=[1] * len(hits) + [0] * len(negatives),
        protein_accession="p" if proteins is None else proteins,
        log10_affinity=values, affinity_prediction=10 ** values))


def test_competing_hits_never_reuse_a_negative():
    frame = pool([1, 1, 1], [1, 1.1, 1.2])
    for seed in range(10):
        result, diagnostics = make_affinity_controlled_risk_sets(frame, 1, random_seed=seed)
        assert result.loc[result.hit.eq(0), "peptide"].nunique() == 3
        assert diagnostics["replacement"] is False
        assert diagnostics["max_negative_reuse"] == 1


def test_random_draws_are_seeded_and_not_always_nearest():
    frame = pool([1], [1, 1.1, 1.2])
    choices = []
    for seed in range(60):
        first, _ = make_affinity_controlled_risk_sets(frame, 1, random_seed=seed)
        repeat, _ = make_affinity_controlled_risk_sets(frame, 1, random_seed=seed)
        pd.testing.assert_frame_equal(first, repeat)
        choices.append(first.loc[first.hit.eq(0), "source_row"].item())
    assert set(choices) == {1, 2, 3}


def test_augmenting_paths_repair_random_greedy_traps():
    # First hit can use either negative; second can use only the shared one.
    # Giving that negative to the first hit must be repaired, not reused.
    frame = pool([1, 1.3], [1.1, .8])
    for seed in range(30):
        result, _ = make_affinity_controlled_risk_sets(frame, 1, random_seed=seed)
        assert result.source_row.tolist() == [0, 3, 1, 2]


def test_duplicate_protein_mappings_do_not_create_extra_negatives():
    frame = pool([1, 1], [1, 1], proteins=["p", "q", "p", "q"])
    frame.loc[3, "peptide"] = frame.loc[2, "peptide"]
    with pytest.raises(IncompleteProcessingMatches, match="without replacement"):
        make_affinity_controlled_risk_sets(frame, 1)


def test_samples_are_independent_and_can_share_negative_sequences():
    frame = pool([1, 1], [1, 1.1, .9])
    combined = pd.concat([frame.assign(sample_id="a"), frame.assign(sample_id="b")], ignore_index=True)
    together, _ = make_affinity_controlled_risk_sets(combined, 1, random_seed=13)
    parts = []
    for sample, group in combined.groupby("sample_id"):
        result, _ = make_affinity_controlled_risk_sets(group.reset_index(drop=True), 1, random_seed=13)
        parts.append(result.drop(columns="source_row"))
    expected = pd.concat(parts, ignore_index=True)
    # risk IDs are local numbering; peptide assignments are sample-stable.
    pd.testing.assert_frame_equal(together.drop(columns=["source_row", "risk_set_id"]),
                                  expected.drop(columns="risk_set_id"))
    assert not together[together.hit.eq(0)].duplicated(["sample_id", "peptide"]).any()


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("count", [1, 2, 3])
def test_feasibility_agrees_with_independent_maximum_matching(seed, count):
    rng = np.random.default_rng(seed)
    hits = rng.integers(0, 9, size=8) / 8
    negatives = rng.integers(0, 9, size=20) / 8
    frame = pool(hits, negatives, proteins=rng.choice(["p", "q"], len(hits) + len(negatives)))
    # Separate oracle: explicitly construct the small bipartite graph.
    edges = np.abs(np.repeat(hits, count)[:, None] - negatives[None, :]) <= .25
    oracle = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
    if np.any(oracle < 0):
        with pytest.raises(IncompleteProcessingMatches):
            make_affinity_controlled_risk_sets(frame, count, random_seed=seed)
    else:
        result, _ = make_affinity_controlled_risk_sets(frame, count, random_seed=seed)
        assert result.hit.sum() == len(hits)
        assert result.loc[result.hit.eq(0), "peptide"].nunique() == len(hits) * count
        assert result.log10_affinity_distance.max() <= .25


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("dtype", [object, "str", "string"])
def test_missing_protein_is_not_a_preferred_shared_protein(missing, dtype):
    frame = pool([1], [1, 1.1])
    frame["protein_accession"] = pd.Series([missing, missing, "p"], dtype=dtype)
    result, _ = make_affinity_controlled_risk_sets(frame, 1)
    assert not result.loc[result.hit.eq(0), "same_protein_match"].any()


def test_saved_assignments_reject_reuse_across_pairs():
    frame = pool([1, 1], [1, 1.1])
    result, _ = matched_training_data(frame, {"sha256": "0" * 64})
    result.loc[3, "peptide"] = result.loc[1, "peptide"]
    with pytest.raises(ValueError, match="repeats a negative"):
        validate_matched_training_data(result)


def test_old_nearest_matching_policy_is_not_silently_relabelled():
    result, _ = matched_training_data(pool([1], [1]), {"sha256": "0" * 64})
    result["processing_matching_policy"] = "sample-length-affinity-v1"
    with pytest.raises(ValueError, match="matching metadata"):
        validate_matched_training_data(result)


def test_saved_seed_must_be_an_integer():
    result, _ = matched_training_data(pool([1], [1]), {"sha256": "0" * 64})
    result["matching_random_seed"] = "unrecorded"
    with pytest.raises(ValueError, match="random seed"):
        validate_matched_training_data(result)


def test_inclusive_caliper_boundary_and_no_match_outside():
    result, _ = make_affinity_controlled_risk_sets(pool([1], [.75]), 1)
    assert result.log10_affinity_distance.max() == .25
    with pytest.raises(IncompleteProcessingMatches):
        make_affinity_controlled_risk_sets(pool([1], [np.nextafter(.75, -np.inf)]), 1)


@pytest.mark.parametrize("seed", range(20))
def test_feasibility_near_float_boundaries(seed):
    rng = np.random.default_rng(seed)
    hits = rng.uniform(.1, 3, 8)
    negatives = np.r_[hits - .25, hits + .25, np.nextafter(hits - .25, -np.inf),
                      np.nextafter(hits + .25, np.inf)]
    edges = np.abs(hits[:, None] - negatives[None, :]) <= .25
    oracle = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
    if np.any(oracle < 0):
        with pytest.raises(IncompleteProcessingMatches):
            make_affinity_controlled_risk_sets(pool(hits, negatives), 1, random_seed=seed)
    else:
        result, _ = make_affinity_controlled_risk_sets(pool(hits, negatives), 1, random_seed=seed)
        assert result.log10_affinity_distance.max() <= .25

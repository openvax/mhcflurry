"""Independent scalar-oracle parity for grouped/vectorized risk sets."""

import numpy as np
import pandas as pd
import pytest

from mhcflurry.processing_matching import (
    IncompleteProcessingMatches, make_affinity_controlled_risk_sets,
    nearest_affinity_batch)


def scalar_nearest(pool, target, count, excluded=(), max_distance=None):
    """Frozen pre-vectorization algorithm, including its local tie window."""
    indices, values = pool
    if count <= 0 or not len(indices):
        return []
    position = int(np.searchsorted(values, target))
    radius = min(len(indices), max(count * 3, count + len(excluded)))
    start, end = max(0, position - radius), min(len(indices), position + radius)
    candidates, candidate_values = indices[start:end], values[start:end]
    selected = []
    for offset in np.argsort(np.abs(candidate_values - target), kind="stable"):
        index = int(candidates[offset])
        if max_distance is not None and abs(float(candidate_values[offset]) - target) > max_distance:
            continue
        if index not in excluded:
            selected.append(index)
            if len(selected) == count:
                break
    if len(selected) < count and len(candidates) < len(indices):
        for offset in np.argsort(np.abs(values - target), kind="stable"):
            index = int(indices[offset])
            if max_distance is not None and abs(float(values[offset]) - target) > max_distance:
                continue
            if index not in excluded and index not in selected:
                selected.append(index)
                if len(selected) == count:
                    break
    return selected


def scalar_assignments(frame, count, caliper, protein_caliper):
    """Return assignments, protein flags and failures from the old policy."""
    negatives = frame.loc[frame.hit == 0].drop_duplicates(["sample_id", "peptide"])
    pools = []
    for keys in (["sample_id", "peptide_len"], ["sample_id", "peptide_len", "protein_accession"]):
        mapping = {}
        for key, group in negatives.groupby(keys, sort=False, dropna=False):
            order = np.argsort(group.log10_affinity.to_numpy(), kind="stable")
            mapping[key] = (group.index.to_numpy()[order], group.log10_affinity.to_numpy()[order])
        pools.append(mapping)
    empty = (np.array([], dtype="int64"), np.array([]))
    rows, flags, failures = [], [], []
    for index, hit in frame.loc[frame.hit == 1].iterrows():
        selected = scalar_nearest(pools[1].get((hit.sample_id, hit.peptide_len, hit.protein_accession), empty),
            float(hit.log10_affinity), count, max_distance=min(caliper, protein_caliper)
            if protein_caliper is not None else caliper)
        same = len(selected)
        selected += scalar_nearest(pools[0].get((hit.sample_id, hit.peptide_len), empty),
            float(hit.log10_affinity), count - same, excluded=selected, max_distance=caliper)
        if len(selected) != count:
            failures.append(dict(sample_id=str(hit.sample_id), peptide=hit.peptide, available=len(selected),
                                 log10_affinity=float(hit.log10_affinity), peptide_length=int(hit.peptide_len)))
        rows.extend([index] + selected)
        flags.extend([True] + [True] * same + [False] * (len(selected) - same))
    return rows, flags, failures


@pytest.mark.parametrize("seed", range(12))
def test_batch_search_matches_scalar_for_ties_exclusions_and_calipers(seed):
    rng = np.random.RandomState(seed)
    for size in (0, 1, 2, 7, 80):
        values = np.sort(rng.randint(0, 10, size=size).astype(float) / 4)
        indices = rng.permutation(size)
        targets = np.r_[rng.uniform(-1, 4, 30), values, np.nextafter(values, np.inf)]
        exclusions = rng.randint(-1, max(1, size), size=(len(targets), 12))
        # Excluded identities are a set in the old contract.
        for row in exclusions:
            seen = set()
            for i, item in enumerate(row):
                if item in seen:
                    row[i] = -1
                seen.add(item)
        for count in (1, 3, 10):
            for caliper in (None, 0, .25):
                actual = nearest_affinity_batch((indices, values), targets, count, exclusions, caliper, chunk_size=7)
                expected = np.full(actual.shape, -1)
                for i, target in enumerate(targets):
                    found = scalar_nearest((indices, values), target, count,
                                           exclusions[i][exclusions[i] >= 0], caliper)
                    expected[i, :len(found)] = found
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("count,caliper,protein_caliper", [(1, .25, .25), (3, 5, .1), (10, 5, None)])
def test_grouped_matches_scalar_including_duplicate_peptides_and_missing_proteins(seed, count, caliper, protein_caliper):
    rng = np.random.RandomState(seed)
    n = 350
    frame = pd.DataFrame(dict(
        peptide=[format(i, "08d") for i in range(n)], peptide_len=8,
        sample_id=rng.choice(["s1", "s2"], n),
        protein_accession=rng.choice(["p1", "p2", "p3", None], n),
        log10_affinity=rng.randint(0, 12, n) / 4,
        hit=np.arange(n) < 25))
    frame["hit"] = frame.hit.astype(int)
    frame.loc[100:102, "peptide"] = frame.loc[99, "peptide"]
    expected, flags, failures = scalar_assignments(frame, count, caliper, protein_caliper)
    if failures:
        with pytest.raises(IncompleteProcessingMatches) as error:
            make_affinity_controlled_risk_sets(frame, count, protein_caliper, caliper)
        assert error.value.failures == failures
    else:
        actual, diagnostics = make_affinity_controlled_risk_sets(frame, count, protein_caliper, caliper)
        assert actual.source_row.tolist() == expected
        assert actual.same_protein_match.tolist() == flags
        assert diagnostics["fallback_decoys"] == sum(not f for f in flags)
        assert actual.risk_set_id.tolist() == np.repeat(np.arange(25), count + 1).tolist()


def test_dense_ties_keep_local_window_not_global_tie_order():
    values = np.r_[np.zeros(100), 2.0]
    actual = nearest_affinity_batch((np.arange(101), values), [1.0], 1)
    assert actual.tolist() == [[97]]  # Global stable nearest would pick row 0.


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_float_precision_and_caliper_nextafter_match_scalar(dtype):
    values = np.array([.5, .75, 1, 1.25, 1.5], dtype=dtype)
    targets = np.r_[values, np.nextafter(values, np.inf), np.nextafter(values, -np.inf), [-10.0, 100.0]]
    indices = np.arange(len(values))
    for caliper in (0, .25, 1, None):
        actual = nearest_affinity_batch((indices, values), targets, 3, max_distance=caliper)
        expected = np.full(actual.shape, -1)
        for i, target in enumerate(targets):
            rows = scalar_nearest((indices, values), float(target), 3, max_distance=caliper)
            expected[i, :len(rows)] = rows
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("dtype", [object, "str", "string"])
def test_missing_protein_tuple_semantics_are_preserved(missing, dtype):
    frame = pd.DataFrame(dict(peptide=["A" * 8, "C" * 8, "D" * 8], peptide_len=8,
                              sample_id="s", log10_affinity=[1, 1.1, 1], hit=[1, 0, 0]))
    frame["protein_accession"] = pd.Series([missing, missing, "p"], dtype=dtype)
    rows, flags, failures = scalar_assignments(frame, 1, .25, .25)
    assert not failures
    actual, _ = make_affinity_controlled_risk_sets(frame, 1)
    assert actual.source_row.tolist() == rows
    assert actual.same_protein_match.tolist() == flags

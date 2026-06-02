"""
Integration tests for allele-name canonicalization shared by prediction and
training ingestion.

These exercise the helpers in ``mhcflurry.common`` against the *real*
``allele_sequences`` key set (not synthetic keys), so they catch regressions in
the no-alias-first resolution that unit tests with hand-made key sets can miss.
"""

import pandas
import pytest

from mhcflurry.common import (
    normalize_allele_name,
    build_allele_alias_map,
    canonicalize_allele_to_keys,
    canonicalize_allele_series,
)
from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.downloads import get_path
from mhcflurry.pseudosequences import LEGACY_ALLELE_SEQUENCES_FILENAME

from mhcflurry.testing_utils import cleanup, startup


@pytest.fixture(autouse=True, scope="module")
def setup_module():
    startup()
    yield
    cleanup()


def _allele_sequence_keys():
    seqs = pandas.read_csv(
        get_path("allele_sequences", LEGACY_ALLELE_SEQUENCES_FILENAME),
        index_col=0)
    return list(seqs.index)


def test_canonicalize_allele_series_against_real_allele_sequences():
    keys = _allele_sequence_keys()
    key_set = set(keys)
    assert "HLA-A*02:01" in key_set, "test assumes A*02:01 is a known key"

    inputs = [
        "HLA-A*02:01",   # already canonical -> itself
        "HLA-A0201",     # alternative spelling -> canonical key
        "A*02:01",       # missing HLA- prefix -> canonical key
        "HLA-A2",        # serotype -> not an allele -> dropped
        "NONSENSE",      # junk -> dropped
    ]
    out = canonicalize_allele_series(inputs, keys, log_label="test alleles")

    assert out[0] == "HLA-A*02:01"
    assert out[1] == "HLA-A*02:01"
    assert out[2] == "HLA-A*02:01"
    assert out[3] is None, "serotype must not resolve to an allele key"
    assert out[4] is None, "junk must be dropped"
    # Every non-None result is guaranteed to be a real pseudosequence key.
    assert all(x in key_set for x in out if x is not None)


def test_canonicalize_allele_series_resolves_a_real_retired_alias():
    # Find a real retired/aliased name in the key set: a key whose alias-applied
    # normalization differs from the key itself. Then confirm a request for that
    # alias routes back to the original key (no-alias-first preserves the key's
    # own pseudosequence), rather than being lost.
    keys = _allele_sequence_keys()
    key_set = set(keys)
    alias_map = build_allele_alias_map(keys)
    if not alias_map:
        pytest.skip("no aliased alleles in this allele_sequences table")

    alias_name, original_key = next(iter(alias_map.items()))
    resolved = canonicalize_allele_to_keys(alias_name, key_set, alias_map)
    assert resolved == original_key
    assert resolved in key_set


def test_predictor_canonicalize_delegates_to_shared_helper():
    # Single source of truth: canonicalize_allele_name must delegate to
    # canonicalize_allele_to_keys using the predictor's own maps, so the two
    # never diverge.
    keys = _allele_sequence_keys()
    subset = {k: "X" for k in keys[:200]}
    predictor = Class1AffinityPredictor(allele_to_sequence=subset)
    for name in ["HLA-A*02:01", "HLA-A0201"] + list(keys[:3]):
        if normalize_allele_name(name, raise_on_error=False) is None:
            continue
        expected = canonicalize_allele_to_keys(
            name, predictor.allele_to_sequence, predictor.allele_to_canonical)
        assert predictor.canonicalize_allele_name(name) == expected


def test_allele_specific_normalization_merges_spellings():
    # The allele-specific ingestion path canonicalizes with plain normalization
    # (aliases applied) so different spellings of one allele collapse to a
    # single training allele instead of fragmenting into separate models.
    spellings = ["HLA-A*02:01", "HLA-A0201", "A*02:01", "HLA-A02:01"]
    canonical = {normalize_allele_name(s) for s in spellings}
    assert canonical == {"HLA-A*02:01"}, canonical
    # Unparseable names resolve to None (dropped at ingestion).
    assert normalize_allele_name("NONSENSE", raise_on_error=False) is None

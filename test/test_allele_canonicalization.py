# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    AlleleKeyResolver,
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
    resolved = AlleleKeyResolver(key_set, alias_map).resolve(alias_name)
    assert resolved == original_key
    assert resolved in key_set


def test_predictor_canonicalize_matches_resolver():
    # Single source of truth: the predictor canonicalizes via an AlleleKeyResolver
    # over its own maps, so canonicalize_allele_name matches a resolver built the
    # same way.
    keys = _allele_sequence_keys()
    subset = {k: "X" for k in keys[:200]}
    predictor = Class1AffinityPredictor(allele_to_sequence=subset)
    resolver = AlleleKeyResolver(
        predictor.allele_to_sequence, predictor.allele_to_canonical)
    for name in ["HLA-A*02:01", "HLA-A0201"] + list(keys[:3]):
        if normalize_allele_name(name, raise_on_error=False) is None:
            continue
        assert predictor.canonicalize_allele_name(name) == resolver.resolve(
            name, raise_on_error=True)


def test_predictor_reuses_cached_resolver():
    # The per-row calls in predict_to_dataframe must reuse one resolver, not
    # reconstruct it each time.
    predictor = Class1AffinityPredictor(
        allele_to_sequence={"HLA-A*02:01": "AAAA"})
    predictor.canonicalize_allele_name("HLA-A*02:01")
    first = predictor._cache["allele_key_resolver"]
    predictor.canonicalize_allele_name("HLA-A*02:01")
    assert predictor._cache["allele_key_resolver"] is first


def test_predictor_clear_cache_invalidates_cached_resolver():
    # clear_cache() must drop the cached resolver (it lives in self._cache), so
    # a resolver built before the allele maps change can't go stale.
    predictor = Class1AffinityPredictor(
        allele_to_sequence={"HLA-A*02:01": "AAAA"})
    predictor.canonicalize_allele_name("HLA-A*02:01")
    assert "allele_key_resolver" in predictor._cache
    predictor.clear_cache()
    assert "allele_key_resolver" not in predictor._cache
    # Still resolves afterward (a fresh resolver is built on demand).
    assert predictor.canonicalize_allele_name("HLA-A*02:01") == "HLA-A*02:01"


def test_canonicalize_allele_series_builds_reverse_map_lazily(monkeypatch):
    # The ~O(keys) reverse alias map must be built only when a name actually
    # needs it (a current-name request for a retired key), never for data that
    # resolves via the two cheap per-name paths.
    import mhcflurry.common as common
    calls = []
    real = common.build_allele_alias_map
    monkeypatch.setattr(
        common, "build_allele_alias_map",
        lambda keys: calls.append(1) or real(keys))

    keys = ["HLA-A*02:01", "HLA-B*44:01"]

    # Canonical / alternative-spelling / junk inputs: no reverse map needed.
    out = common.canonicalize_allele_series(
        ["HLA-A*02:01", "HLA-A0201", "NONSENSE"], keys)
    assert out == ["HLA-A*02:01", "HLA-A*02:01", None]
    assert calls == [], "reverse map built for already-resolvable input"

    # A retired key requested by its current name forces a single lazy build.
    out2 = common.canonicalize_allele_series(["HLA-B*44:02"], keys)
    assert out2 == ["HLA-B*44:01"]
    assert len(calls) == 1, "reverse map should build exactly once, on demand"


def test_allele_specific_normalization_merges_spellings():
    # The allele-specific ingestion path canonicalizes with plain normalization
    # (aliases applied) so different spellings of one allele collapse to a
    # single training allele instead of fragmenting into separate models.
    spellings = ["HLA-A*02:01", "HLA-A0201", "A*02:01", "HLA-A02:01"]
    canonical = {normalize_allele_name(s) for s in spellings}
    assert canonical == {"HLA-A*02:01"}, canonical
    # Unparseable names resolve to None (dropped at ingestion).
    assert normalize_allele_name("NONSENSE", raise_on_error=False) is None

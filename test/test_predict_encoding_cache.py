"""Inference-time encoding cache tests.

The encoding cache is shared between training and inference. With
``encoding_cache_dir=<dir>``, a single pass encodes the peptide list and
every network in the ensemble hits the prepopulated cache on its
``peptides_to_network_input`` call.

The load-bearing invariants these tests protect:

  1. ``encoding_cache_dir=None`` is bit-identical to pre-change
     predict() output (backward-compat).
  2. ``encoding_cache_dir=<dir>`` produces the SAME predictions as the
     uncached path — the cache is a pure perf-optimization.
  3. Networks actually hit the cache (no re-encoding) — detected by
     spying on ``EncodableSequences.variable_length_to_fixed_length_vector_encoding``.
  4. Edge cases: empty peptide list, predictor with zero networks,
     unrecognized peptide_encoding kwargs — all fall back to uncached
     path without crashing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from numpy.testing import assert_array_equal

from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.encoding_cache import _verify_cache_key_shape


# Test fixtures: small peptide set scored on one canonical allele.
TEST_PEPTIDES = [
    "SIINFEKL", "GILGFVFTL", "NLVPMVATV", "YLQPRTFLL", "KLVALGINAV",
    "FLRGRAYGL", "ELAGIGILTV", "RMFPNAPYL",
]
TEST_ALLELE = "HLA-A*02:01"


@pytest.fixture(scope="module")
def public_predictor():
    """Load the public mhcflurry pan-allele predictor once per module."""
    path = Path(get_path("models_class1_pan")) / "models.combined"
    if not path.exists():
        pytest.skip(f"public predictor not downloaded: {path}")
    return Class1AffinityPredictor.load(str(path))


def test_predict_with_cache_matches_uncached(public_predictor, tmp_path):
    """THE GATE: encoding_cache_dir=<dir> produces identical predictions.

    Any divergence here means the cache is a behavior change, not a
    pure optimization — unacceptable for an inference-side feature.
    """
    # Uncached path.
    preds_uncached = public_predictor.predict(
        peptides=TEST_PEPTIDES, allele=TEST_ALLELE
    )
    # Cached path.
    preds_cached = public_predictor.predict(
        peptides=TEST_PEPTIDES,
        allele=TEST_ALLELE,
        encoding_cache_dir=str(tmp_path / "encoding_cache"),
    )
    assert_array_equal(preds_uncached, preds_cached)


def test_predict_with_cache_warm_second_call_no_peptide_reencoding(
    public_predictor, tmp_path
):
    """Second call on same peptides must skip peptide re-encoding.

    Prime the cache with a first call, then spy on the low-level
    ``amino_acid.fixed_vectors_encoding`` (only called on true cache
    miss) and run a second call. Result scales with the ALLELE-encoding
    path (always 1 call per predict regardless of peptide cache), not
    with the number of networks in the ensemble.

    With N networks and NO peptide cache: N peptide encodes + 1 allele
    encode = N+1 calls on each predict. With the peptide cache: just
    the 1 allele encode, regardless of N.

    The assertion is "encode calls ≤ 2" — tight enough to catch a
    regression (would be ≥ N+1 = 11 on the 10-network public predictor)
    while not being so tight that an orthogonal allele-encoding call
    breaks it.
    """
    from mhcflurry import amino_acid

    # Prime the cache-key-shape self-test so it doesn't show up as
    # encode calls under the spy (it runs once per process).
    _verify_cache_key_shape()

    cache_dir = str(tmp_path / "encoding_cache")

    # First call builds the cache.
    public_predictor.predict(
        peptides=TEST_PEPTIDES,
        allele=TEST_ALLELE,
        encoding_cache_dir=cache_dir,
    )

    # Second call should skip peptide encoding; we spy on the actual
    # encode work (not the wrapper function which always runs regardless
    # of cache hit).
    calls = []
    original = amino_acid.fixed_vectors_encoding

    def spy(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    amino_acid.fixed_vectors_encoding = spy
    try:
        public_predictor.predict(
            peptides=TEST_PEPTIDES,
            allele=TEST_ALLELE,
            encoding_cache_dir=cache_dir,
        )
    finally:
        amino_acid.fixed_vectors_encoding = original

    n_networks = len(public_predictor.neural_networks)
    # Allele encoding always triggers 1 call; peptide cache should cut
    # the N peptide encodes down to 0. Budget = 2 (allele + slack).
    assert len(calls) <= 2, (
        f"warm-cache predict ran fixed_vectors_encoding {len(calls)} times "
        f"with {n_networks} networks in the ensemble. Expected ≤2 "
        f"(just the allele-encoding path). If ≥ {n_networks}+1, the "
        f"peptide cache isn't being hit. Check that "
        f"make_prepopulated_encodable_sequences is producing the right key "
        f"tuple + that the EncodableSequences instance isn't being "
        f"recreated somewhere in predict_to_dataframe."
    )


def test_predict_with_cache_handles_empty_peptide_list(public_predictor, tmp_path):
    """Empty peptide list shouldn't crash the cache build path."""
    result = public_predictor.predict(
        peptides=[],
        alleles=[],
        encoding_cache_dir=str(tmp_path / "encoding_cache"),
    )
    # predict() returns an array of predictions; empty input → empty output.
    assert len(result) == 0


def test_predict_cache_dir_none_is_default_path(public_predictor):
    """encoding_cache_dir=None is the pre-change default (backward-compat)."""
    # Verify explicit None and omitted arg produce the same thing.
    preds_default = public_predictor.predict(
        peptides=TEST_PEPTIDES, allele=TEST_ALLELE
    )
    preds_none = public_predictor.predict(
        peptides=TEST_PEPTIDES, allele=TEST_ALLELE, encoding_cache_dir=None
    )
    assert_array_equal(preds_default, preds_none)


def test_prepare_peptides_helper_fallback_on_empty_predictor(tmp_path):
    """Predictor with zero networks falls back to uncached path cleanly."""
    empty = Class1AffinityPredictor(class1_pan_allele_models=[])
    # Should NOT try to introspect hyperparameters from networks[0].
    result = empty._prepare_peptides_with_optional_cache(
        TEST_PEPTIDES, encoding_cache_dir=str(tmp_path / "cache"),
    )
    assert isinstance(result, EncodableSequences)
    assert list(result.sequences) == TEST_PEPTIDES


def test_prepare_peptides_helper_fallback_on_unknown_encoding_kwarg(
    public_predictor, tmp_path
):
    """Unknown peptide_encoding kwarg falls back to uncached, not crash.

    Guards against future upstream additions to peptide_encoding that
    EncodingParams doesn't know about yet.
    """
    # Temporarily patch the first network's hyperparameters to include
    # a key that EncodingParams doesn't accept.
    net0 = public_predictor.neural_networks[0]
    original_hp = dict(net0.hyperparameters)
    try:
        net0.hyperparameters["peptide_encoding"] = {
            **original_hp.get("peptide_encoding", {}),
            "brand_new_unknown_kwarg": 42,
        }
        # Should NOT crash — falls back to uncached path.
        result = public_predictor._prepare_peptides_with_optional_cache(
            TEST_PEPTIDES, encoding_cache_dir=str(tmp_path / "cache"),
        )
        assert isinstance(result, EncodableSequences)
        # Cache was not populated (fallback path).
        assert result.encoding_cache == {}
    finally:
        net0.hyperparameters.clear()
        net0.hyperparameters.update(original_hp)


def test_prepare_peptides_helper_accepts_encodablesequences_input(
    public_predictor, tmp_path
):
    """Passing an EncodableSequences (not a list) works through the cache path."""
    es = EncodableSequences(TEST_PEPTIDES)
    result = public_predictor._prepare_peptides_with_optional_cache(
        es, encoding_cache_dir=str(tmp_path / "cache"),
    )
    assert isinstance(result, EncodableSequences)
    # Cache entry should be populated since we gave a non-empty list.
    assert len(result.encoding_cache) == 1
    assert list(result.sequences) == TEST_PEPTIDES


def test_predict_cache_reuses_across_multiple_predictor_instances(
    public_predictor, tmp_path
):
    """Shared cache_dir between two predictor instances avoids re-encoding.

    This is the killer use case: scoring the same peptide set with
    multiple predictors (ours vs public comparison) shares the encoding
    cost across both runs.
    """
    from mhcflurry import amino_acid

    # Prime the cache-key-shape self-test so it doesn't count here.
    _verify_cache_key_shape()

    cache_dir = str(tmp_path / "shared_cache")
    # First predictor builds the cache.
    public_predictor.predict(
        peptides=TEST_PEPTIDES,
        allele=TEST_ALLELE,
        encoding_cache_dir=cache_dir,
    )

    # A NEW predictor (same models, different Python instance) pointed
    # at the same cache_dir should hit warm cache on first call.
    public_predictor_2 = Class1AffinityPredictor.load(
        str(Path(get_path("models_class1_pan")) / "models.combined")
    )

    calls = []
    original = amino_acid.fixed_vectors_encoding

    def spy(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    amino_acid.fixed_vectors_encoding = spy
    try:
        public_predictor_2.predict(
            peptides=TEST_PEPTIDES,
            allele=TEST_ALLELE,
            encoding_cache_dir=cache_dir,
        )
    finally:
        amino_acid.fixed_vectors_encoding = original

    # Same budget as the single-predictor warm-cache test: ≤2 allowed
    # for the allele-encoding path, but the N peptide encodes per
    # network (would be ≥ N+1 = 11 without the cache) must be 0.
    assert len(calls) <= 2, (
        f"cross-predictor cache sharing is broken — the second predictor "
        f"ran fixed_vectors_encoding {len(calls)} times. Expected ≤2 "
        f"(just the allele-encoding path). Likely the orchestrator-side "
        f"cache-key hash differs from the predict-side hash."
    )

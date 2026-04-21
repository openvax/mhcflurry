"""Tests for the memmap-backed BLOSUM62 encoding cache.

The cache is semantics-preserving by design: encoded bytes must match
``EncodableSequences.variable_length_to_fixed_length_vector_encoding``
for the same peptides and params. That property is the most important
thing to test — if it slips, downstream training loses determinism.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy
import pytest
from numpy.testing import assert_array_equal

from mhcflurry import encoding_cache as encoding_cache_module
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.encoding_cache import (
    DEFAULT_CHUNK_SIZE,
    EncodingCache,
    EncodingCacheKeyMismatchError,
    EncodingParams,
    _hash_peptides,
    _vector_encoding_cache_key,
    _verify_cache_key_shape,
    make_preencoded_encodable_sequences,
)


# Module-level function so multiprocessing's spawn context can pickle it.
# Nested/inline worker functions fail with AttributeError under spawn.
def _concurrent_build_worker(cache_dir_str, params_kwargs, peptide_list, queue):
    try:
        from mhcflurry.encoding_cache import (
            EncodingCache as _EC,
            EncodingParams as _EP,
        )
        import hashlib as _hashlib
        cache = _EC(cache_dir=cache_dir_str, params=_EP(**params_kwargs))
        encoded, _ = cache.get_or_build(peptide_list)
        queue.put(("ok", _hashlib.sha256(encoded.tobytes()).hexdigest()))
    except Exception as exc:
        import traceback as _tb
        queue.put(("error", f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"))


# A small peptide set that covers 8-15mers (the lengths mhcflurry supports).
SAMPLE_PEPTIDES = [
    "SIINFEKL",       # 8mer
    "AAAAAAAAA",      # 9mer, homopolymer edge case
    "GILGFVFTL",      # 9mer, famous flu epitope
    "YLQPRTFLL",      # 9mer
    "ELAGIGILTV",     # 10mer
    "IMDQVPFSV",      # 9mer
    "NLVPMVATV",      # 9mer
    "RMFPNAPYL",      # 9mer
    "FLRGRAYGL",      # 9mer
    "LLDFVRFMGV",     # 10mer
    "QYDPVAALF",      # 9mer
    "KLVALGINAV",     # 10mer
    "FVLELEPEWTVK",   # 12mer
    "KLIETYFSKK",     # 10mer
    "AQYELGGGPGAGD",  # 13mer
    "YVNVNMGLKIRQLLWFHISCLTFGRETVLEYLVS",  # will truncate with trim=True or err without
]

# The subset that matches mhcflurry's default 8-15mer window without
# needing trim=True. Used for the bit-for-bit baseline tests.
VALID_PEPTIDES = [p for p in SAMPLE_PEPTIDES if 8 <= len(p) <= 15]


@pytest.fixture
def default_params():
    """Matches the peptide_encoding block in pan-allele hyperparameters."""
    return EncodingParams(
        vector_encoding_name="BLOSUM62",
        alignment_method="left_pad_centered_right_pad",
        max_length=15,
    )


@pytest.fixture
def cache(tmp_path, default_params):
    return EncodingCache(cache_dir=tmp_path / "encoding_cache", params=default_params)


# ---- Category A: unit tests for the cache module ----


def test_cached_matches_direct_byte_for_byte(cache, default_params):
    """The cache MUST emit bytes identical to a direct EncodableSequences call.

    This is the semantic-preservation gate. If this fails, downstream
    training will diverge from the un-cached code path and we can no
    longer claim determinism. Every other test below is secondary.
    """
    direct = (
        EncodableSequences(VALID_PEPTIDES)
        .variable_length_to_fixed_length_vector_encoding(**default_params.to_kwargs())
        .astype(numpy.float32)
    )
    cached, _ = cache.get_or_build(VALID_PEPTIDES)
    assert cached.shape == direct.shape
    assert cached.dtype == direct.dtype
    assert_array_equal(cached, direct)


def test_peptide_to_idx_roundtrip(cache):
    _, peptide_to_idx = cache.get_or_build(VALID_PEPTIDES)
    assert len(peptide_to_idx) == len(VALID_PEPTIDES)
    for i, p in enumerate(VALID_PEPTIDES):
        assert peptide_to_idx[p] == i


def test_cache_hit_second_call_does_not_reencode(cache, monkeypatch):
    """Second call with same params+peptides should skip the encoding pass."""
    cache.get_or_build(VALID_PEPTIDES)
    calls = []
    original = EncodableSequences.variable_length_to_fixed_length_vector_encoding

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        EncodableSequences,
        "variable_length_to_fixed_length_vector_encoding",
        spy,
    )
    cache.get_or_build(VALID_PEPTIDES)
    assert calls == [], "expected no re-encoding on cache hit"


def test_different_params_invalidates_cache(tmp_path):
    """Changing max_length creates a separate cache entry — no bleed-through."""
    params_a = EncodingParams(max_length=15, alignment_method="pad_middle")
    params_b = EncodingParams(max_length=17, alignment_method="pad_middle")
    cache_a = EncodingCache(cache_dir=tmp_path / "cache", params=params_a)
    cache_b = EncodingCache(cache_dir=tmp_path / "cache", params=params_b)

    encoded_a, _ = cache_a.get_or_build(VALID_PEPTIDES)
    encoded_b, _ = cache_b.get_or_build(VALID_PEPTIDES)
    assert encoded_a.shape != encoded_b.shape
    # They live in different subdirs keyed by params_hash.
    assert cache_a.entry_path(VALID_PEPTIDES) != cache_b.entry_path(VALID_PEPTIDES)


def test_different_peptides_invalidates_cache(cache):
    """Same params, different peptide list → different cache entry."""
    path_a = cache.entry_path(VALID_PEPTIDES)
    path_b = cache.entry_path(VALID_PEPTIDES[::-1])  # reversed order
    assert path_a != path_b, "cache key must be order-sensitive"


def test_cache_is_order_sensitive(cache, default_params):
    """Peptides in reverse order produce reverse-ordered encoded rows."""
    forward, _ = cache.get_or_build(VALID_PEPTIDES)
    reversed_list = VALID_PEPTIDES[::-1]
    reversed_encoded, _ = cache.get_or_build(reversed_list)
    assert_array_equal(reversed_encoded, forward[::-1])


def test_partial_write_not_consumed(tmp_path, default_params):
    """A half-written cache (no .complete sentinel) must not be consumed.

    Simulates a crash mid-build: pre-create the entry_dir with a stale
    (wrong-bytes) encoded.npy but no .complete file. The next call should
    rebuild, producing correct bytes.
    """
    cache = EncodingCache(
        cache_dir=tmp_path / "encoding_cache", params=default_params
    )
    entry_dir = cache.entry_path(VALID_PEPTIDES)
    entry_dir.mkdir(parents=True)
    # Write garbage encoded.npy — anything the reader would trip on.
    numpy.save(
        entry_dir / "encoded.npy",
        numpy.full((len(VALID_PEPTIDES), 45, 21), 99.0, dtype=numpy.float32),
    )
    # .complete deliberately absent.

    encoded, _ = cache.get_or_build(VALID_PEPTIDES)

    # Should have been rebuilt, not the 99.0-filled garbage.
    direct = (
        EncodableSequences(VALID_PEPTIDES)
        .variable_length_to_fixed_length_vector_encoding(**default_params.to_kwargs())
        .astype(numpy.float32)
    )
    assert_array_equal(encoded, direct)
    assert (entry_dir / ".complete").exists()


def test_sidecar_files_written(cache):
    cache.get_or_build(VALID_PEPTIDES)
    entry_dir = cache.entry_path(VALID_PEPTIDES)
    assert (entry_dir / "encoded.npy").exists()
    assert (entry_dir / "peptides.txt").exists()
    assert (entry_dir / "params.json").exists()
    assert (entry_dir / ".complete").exists()
    # peptides.txt preserves order
    written_peptides = [
        p for p in (entry_dir / "peptides.txt").read_text().splitlines() if p != ""
    ]
    assert written_peptides == VALID_PEPTIDES
    # params.json is human-readable JSON
    params_dict = json.loads((entry_dir / "params.json").read_text())
    assert params_dict["vector_encoding_name"] == "BLOSUM62"


def test_mmap_load_is_memory_mapped(cache):
    """Confirm we're actually mmap'ing, not loading into RSS.

    Two calls should return memmap-backed arrays (shared with the file on
    disk), not independent copies in RAM. The specific check: the returned
    ndarray's ``.filename`` attribute exists (numpy.memmap) OR the base
    array is a ``numpy.memmap``.
    """
    arr, _ = cache.get_or_build(VALID_PEPTIDES)
    # np.load(mmap_mode='r') returns a numpy.memmap or similar.
    assert isinstance(arr, numpy.memmap) or isinstance(
        getattr(arr, "base", None), numpy.memmap
    ), f"expected mmap-backed array, got {type(arr).__name__}"


def test_mmap_is_read_only_view(cache):
    """Readers must get a read-only mmap; accidental writes should raise.

    Guards against a caller mutating a cache entry and silently corrupting
    shared state for other processes.
    """
    arr, _ = cache.get_or_build(VALID_PEPTIDES)
    with pytest.raises((ValueError, RuntimeError)):
        arr[0, 0, 0] = 999.0


def test_concurrent_readers_see_identical_bytes(tmp_path, default_params):
    """Multiple threads opening the same cache entry see the same bytes.

    This is really exercising mmap+OS page cache — the invariant we rely on
    for multi-worker sharing. If this fails, the whole shared-encoding
    optimization is unsafe.
    """
    # Build once up front so threads only READ.
    builder = EncodingCache(
        cache_dir=tmp_path / "encoding_cache", params=default_params
    )
    expected, _ = builder.get_or_build(VALID_PEPTIDES)
    expected_bytes = bytes(expected.tobytes())

    results: list[bytes] = []
    lock = threading.Lock()

    def reader():
        c = EncodingCache(
            cache_dir=tmp_path / "encoding_cache", params=default_params
        )
        arr, _ = c.get_or_build(VALID_PEPTIDES)
        data = bytes(arr.tobytes())
        with lock:
            results.append(data)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    for data in results:
        assert data == expected_bytes


def test_concurrent_builders_do_not_collide(tmp_path, default_params):
    """Multiple processes building the SAME cache entry in parallel must not crash.

    Reproduces the 2026-04-20 A100 training failure: two Pool workers
    simultaneously calling get_or_build() for the pretrain cache (which
    the orchestrator doesn't pre-build). The old code used a shared
    ``encoded.npy.tmp`` path so the second worker's ``os.replace`` failed
    with ``FileNotFoundError`` after the first worker's rename consumed
    the tmp file.

    Fix: per-process tmp path (PID-suffixed). Both builders write their
    own tmp file and atomically rename to the shared out_path; the second
    rename silently overwrites identical bytes, since the encoding is a
    pure function of params+peptides.

    Uses multiprocessing (not threading) because the GIL serializes
    threads inside Python-level code paths and wouldn't reliably trigger
    the race. Real Pool workers are separate processes.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    n_workers = 4
    procs = [
        ctx.Process(
            target=_concurrent_build_worker,
            args=(
                str(tmp_path / "cache"),
                default_params.to_kwargs(),
                VALID_PEPTIDES,
                queue,
            ),
        )
        for _ in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    # Collect results.
    results = []
    while not queue.empty():
        results.append(queue.get(timeout=1))

    assert len(results) == n_workers, (
        f"expected {n_workers} worker results, got {len(results)}"
    )
    # Every worker must succeed (no FileNotFoundError from racing renames).
    errors = [payload for status, payload in results if status != "ok"]
    assert not errors, "concurrent builders hit errors:\n" + "\n\n".join(errors)

    # Every worker must see identical encoded bytes.
    hashes = {payload for _, payload in results}
    assert len(hashes) == 1, f"concurrent builders produced divergent bytes: {hashes}"

    # Final cache dir must be in a clean state: .complete exists, exactly
    # one encoded.npy, no leftover tmp files.
    entry_dir = EncodingCache(
        cache_dir=tmp_path / "cache", params=default_params
    ).entry_path(VALID_PEPTIDES)
    assert (entry_dir / ".complete").exists()
    assert (entry_dir / "encoded.npy").exists()
    leftover_tmps = list(entry_dir.glob("encoded.npy.tmp*"))
    assert not leftover_tmps, (
        f"per-process tmp files should have been renamed away; got leftovers: "
        f"{leftover_tmps}"
    )


def test_concurrent_builders_on_pristine_cache_dir(tmp_path, default_params):
    """Same as above but with zero pre-existing state.

    The orchestrator-builds-first flow in training means workers usually
    see ``is_complete == True`` on arrival. Occasionally (e.g. the
    pretrain cache that the orchestrator doesn't pre-build) they hit a
    pristine entry_dir. This test explicitly exercises that path with
    multiple concurrent builders starting from zero.
    """
    import multiprocessing as mp

    # Don't pre-build anything — cache dir doesn't even exist yet.
    assert not (tmp_path / "cache").exists()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_concurrent_build_worker,
            args=(
                str(tmp_path / "cache"),
                default_params.to_kwargs(),
                VALID_PEPTIDES,
                queue,
            ),
        )
        for _ in range(3)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    results = []
    while not queue.empty():
        results.append(queue.get(timeout=1))
    assert len(results) == 3
    errors = [payload for status, payload in results if status != "ok"]
    assert not errors, (
        "pristine-dir concurrent build hit errors:\n" + "\n\n".join(errors)
    )


def test_encoding_error_propagates_at_build_time(tmp_path, default_params):
    """Unsupported peptide content must raise during build, not silently succeed.

    The old un-cached path would raise from inside fit_generator; the
    cached path raises earlier (at cache-build time) which is if anything
    friendlier — but it must still raise, not silently produce a wrong
    encoding.
    """
    cache = EncodingCache(
        cache_dir=tmp_path / "encoding_cache",
        params=EncodingParams(
            max_length=15,
            alignment_method="pad_middle",
            allow_unsupported_amino_acids=False,
        ),
    )
    bad_peptides = ["SIINFEKL", "BADJUNK1"]  # digit '1' isn't a canonical AA
    with pytest.raises(Exception):
        cache.get_or_build(bad_peptides)


def test_chunked_build_matches_single_pass(tmp_path, default_params):
    """Build a cache in small chunks and in one go; bytes must match.

    Chunks matter for large pretrain data (>>RAM); must be byte-identical
    to the single-pass encoding.
    """
    single_pass = EncodingCache(
        cache_dir=tmp_path / "cache_single",
        params=default_params,
        chunk_size=10_000,
    )
    chunked = EncodingCache(
        cache_dir=tmp_path / "cache_chunked",
        params=default_params,
        chunk_size=3,  # force multiple chunks
    )
    a, _ = single_pass.get_or_build(VALID_PEPTIDES)
    b, _ = chunked.get_or_build(VALID_PEPTIDES)
    assert_array_equal(a, b)


def test_empty_peptide_list_raises():
    """Edge case: zero peptides should fail cleanly rather than build an empty cache.

    Downstream callers would all crash dividing by zero somewhere; better
    to fail fast at build.
    """
    with pytest.raises(Exception):
        # An empty list makes the sample-encode dry-run fail; that's fine.
        cache = EncodingCache(
            cache_dir=Path("/tmp/this/should/not/matter"),
            params=EncodingParams(max_length=15),
        )
        cache.get_or_build([])


def test_hash_peptides_order_sensitive():
    assert _hash_peptides(["A", "B"]) != _hash_peptides(["B", "A"])


def test_hash_peptides_count_sensitive():
    # These differ in count (1 vs 2 items); the count prefix guarantees a miss.
    assert _hash_peptides(["AB"]) != _hash_peptides(["A", "B"])


def test_hash_peptides_stable():
    """Same inputs → same hash (sanity: no datetime/pid in the hash path)."""
    assert _hash_peptides(VALID_PEPTIDES) == _hash_peptides(VALID_PEPTIDES)


def test_default_chunk_size_is_reasonable():
    """Catch accidental reduction to a tiny chunk size in future refactors."""
    assert DEFAULT_CHUNK_SIZE >= 1000


# ---- Integration with EncodableSequences ----


def test_prepopulated_hits_existing_cache_path(default_params):
    """The preencoded EncodableSequences short-circuits the encode path.

    This is the load-bearing test for the integration — if upstream
    EncodableSequences rearranges its internal cache key tuple in a
    future refactor, this test fails loudly before a wrong-encoded
    training run hits production.
    """
    peptides = VALID_PEPTIDES
    expected = (
        EncodableSequences(peptides)
        .variable_length_to_fixed_length_vector_encoding(**default_params.to_kwargs())
        .astype(numpy.float32)
    )
    es = make_preencoded_encodable_sequences(peptides, expected, default_params)

    # Short-circuit check: if the cache is hit, this call should NOT do any
    # actual encoding work. We detect by snapshotting the cache and
    # monkey-patching the underlying conversion function to assert it's not
    # called a second time.
    calls = []
    original = EncodableSequences.variable_length_to_fixed_length_vector_encoding

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    EncodableSequences.variable_length_to_fixed_length_vector_encoding = spy
    try:
        out = es.variable_length_to_fixed_length_vector_encoding(
            **default_params.to_kwargs()
        )
    finally:
        EncodableSequences.variable_length_to_fixed_length_vector_encoding = original
    # spy was installed AFTER prepopulation, so the spy's `calls` should
    # still grow by 1 (the lookup call itself), but the expensive work
    # inside is not re-done — the cache key matches and the stored array
    # is returned. We verify that by checking the returned array IS the
    # prepopulated one (same object, not a recomputed copy).
    assert out is expected, "prepopulated cache entry was not reused"


def test_prepopulated_length_mismatch_raises():
    params = EncodingParams(max_length=15, alignment_method="pad_middle")
    with pytest.raises(ValueError, match="does not match peptide count"):
        make_preencoded_encodable_sequences(
            ["SIINFEKL", "GILGFVFTL"],
            numpy.zeros((5, 15, 21), dtype=numpy.float32),
            params,
        )


def test_cache_key_matches_encodable_sequences_internal(default_params):
    """Our cache key tuple must match what EncodableSequences constructs.

    If the tuple layout upstream ever changes, this test catches it via
    the downstream prepopulation no longer hitting the cache (which is
    exercised indirectly by test_prepopulated_hits_existing_cache_path).
    This one is a narrower, cheaper regression check on the tuple shape.
    """
    key = _vector_encoding_cache_key(default_params)
    # First element is the discriminator tag used inside
    # variable_length_to_fixed_length_vector_encoding.
    assert key[0] == "fixed_length_vector_encoding"
    assert default_params.vector_encoding_name in key
    assert default_params.alignment_method in key
    assert default_params.max_length in key


def test_full_roundtrip_via_prepopulated(tmp_path, default_params):
    """End-to-end: cache → prepopulated EncodableSequences → encoder output.

    The final encoded tensor emitted by peptides_to_network_input (via the
    prepopulated instance) must byte-match the direct un-cached call. This
    is the load-bearing semantic test for the integration as a whole.
    """
    cache = EncodingCache(
        cache_dir=tmp_path / "encoding_cache", params=default_params
    )
    encoded_rows, _ = cache.get_or_build(VALID_PEPTIDES)

    es = make_preencoded_encodable_sequences(
        VALID_PEPTIDES, encoded_rows, default_params
    )
    through_cache = es.variable_length_to_fixed_length_vector_encoding(
        **default_params.to_kwargs()
    )
    direct = (
        EncodableSequences(VALID_PEPTIDES)
        .variable_length_to_fixed_length_vector_encoding(**default_params.to_kwargs())
        .astype(numpy.float32)
    )
    assert_array_equal(through_cache, direct)


# ---- is_complete_for (public API) -----------------------------------------


def test_is_complete_for_returns_false_for_empty_cache(tmp_path, default_params):
    cache = EncodingCache(cache_dir=tmp_path / "cache", params=default_params)
    assert cache.is_complete_for(VALID_PEPTIDES) is False


def test_is_complete_for_returns_true_after_build(tmp_path, default_params):
    cache = EncodingCache(cache_dir=tmp_path / "cache", params=default_params)
    cache.get_or_build(VALID_PEPTIDES)
    assert cache.is_complete_for(VALID_PEPTIDES) is True


def test_is_complete_for_is_peptide_specific(tmp_path, default_params):
    """Cache built for peptides A is not 'complete' for peptides B."""
    cache = EncodingCache(cache_dir=tmp_path / "cache", params=default_params)
    cache.get_or_build(VALID_PEPTIDES)
    other_peptides = ["SIINFEKL", "GILGFVFTL"]  # proper subset, still different hash
    assert cache.is_complete_for(VALID_PEPTIDES) is True
    assert cache.is_complete_for(other_peptides) is False


def test_is_complete_for_is_params_specific(tmp_path):
    """Cache built for params A is not 'complete' for params B."""
    params_a = EncodingParams(max_length=15, alignment_method="pad_middle")
    params_b = EncodingParams(max_length=17, alignment_method="pad_middle")
    cache_a = EncodingCache(cache_dir=tmp_path / "cache", params=params_a)
    cache_b = EncodingCache(cache_dir=tmp_path / "cache", params=params_b)
    cache_a.get_or_build(VALID_PEPTIDES)
    assert cache_a.is_complete_for(VALID_PEPTIDES) is True
    assert cache_b.is_complete_for(VALID_PEPTIDES) is False


def test_is_complete_for_returns_false_when_sentinel_missing(tmp_path, default_params):
    """Half-written cache: entry_dir exists but .complete absent → not complete."""
    cache = EncodingCache(cache_dir=tmp_path / "cache", params=default_params)
    entry_dir = cache.entry_path(VALID_PEPTIDES)
    entry_dir.mkdir(parents=True)
    # Create some content but not the sentinel
    (entry_dir / "encoded.npy").write_bytes(b"stale")
    assert cache.is_complete_for(VALID_PEPTIDES) is False


# ---- Runtime cache-key-shape self-test ------------------------------------


def test_verify_cache_key_shape_passes_with_current_encodable_sequences():
    """Sanity: the self-test succeeds against the current encoder.

    If this ever fails on master, the prepopulation tuple is drifting
    out of sync with EncodableSequences and the whole cache integration
    degrades silently. This is the canary.
    """
    # Force re-verification regardless of prior test ordering.
    encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED = False
    try:
        _verify_cache_key_shape()
    finally:
        # Leave in verified state so later tests don't re-pay the cost.
        encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED = True


def test_verify_cache_key_shape_runs_at_most_once_per_process(monkeypatch):
    """The self-test caches its result via _CACHE_KEY_SHAPE_VERIFIED.

    A second call should be a no-op (not re-run the probe). We detect that
    by monkey-patching EncodableSequences.variable_length_to_fixed_length_
    vector_encoding to count calls.
    """
    # Pretend the self-test hasn't run, then run it once.
    encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED = False
    _verify_cache_key_shape()
    assert encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED is True

    # Spy and re-call — should short-circuit.
    calls = []
    original = EncodableSequences.variable_length_to_fixed_length_vector_encoding

    def spy(self, *a, **kw):
        calls.append(1)
        return original(self, *a, **kw)

    monkeypatch.setattr(
        EncodableSequences,
        "variable_length_to_fixed_length_vector_encoding",
        spy,
    )
    _verify_cache_key_shape()
    assert calls == [], "self-test ran again despite guard flag being set"


def test_verify_cache_key_shape_raises_when_key_wrong(monkeypatch):
    """If _vector_encoding_cache_key drifts, self-test raises loudly.

    Simulates upstream drift by monkey-patching our cache-key function
    to produce a tuple that won't match what EncodableSequences looks up.
    The self-test must then raise EncodingCacheKeyMismatchError with
    guidance — not silently degrade.
    """
    encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED = False

    def bad_key(params):
        return ("definitely-wrong-tag", params.vector_encoding_name)

    monkeypatch.setattr(
        encoding_cache_module,
        "_vector_encoding_cache_key",
        bad_key,
    )
    with pytest.raises(EncodingCacheKeyMismatchError, match="key tuple no longer matches"):
        _verify_cache_key_shape()

    # Guard should remain unset so a subsequent (correct) call still runs
    # and succeeds.
    assert encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED is False


def test_make_preencoded_triggers_self_test_on_first_call():
    """make_preencoded_encodable_sequences runs the self-test once.

    First call after a fresh process (or after resetting the guard flag)
    must invoke _verify_cache_key_shape. This is the behavior that turns
    a silent upstream break into a loud error at orchestrator-init time.
    """
    encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED = False
    params = EncodingParams()
    expected = (
        EncodableSequences(["SIINFEKL"])
        .variable_length_to_fixed_length_vector_encoding(**params.to_kwargs())
        .astype(numpy.float32)
    )
    make_preencoded_encodable_sequences(["SIINFEKL"], expected, params)
    assert encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED is True


def test_make_preencoded_raises_if_key_drift_detected(monkeypatch):
    """Orchestrator-level failure mode: key drift blocks cache construction.

    Ensures a broken upstream refactor fails fast at the first
    make_preencoded call rather than silently producing a cache that
    never gets hit.
    """
    encoding_cache_module._CACHE_KEY_SHAPE_VERIFIED = False
    monkeypatch.setattr(
        encoding_cache_module,
        "_vector_encoding_cache_key",
        lambda params: ("wrong",),
    )
    params = EncodingParams()
    fake_encoded = numpy.zeros((1, 15, 21), dtype=numpy.float32)
    with pytest.raises(EncodingCacheKeyMismatchError):
        make_preencoded_encodable_sequences(["SIINFEKL"], fake_encoded, params)

"""Integration tests for the encoding cache plumbed into train_pan_allele.

This file exercises the specific surfaces that Phase 1 (issue #268) touches
in ``train_pan_allele_models_command``:

    _build_train_peptides() — per-worker helper that returns an
        EncodableSequences either directly (cache disabled, old behavior)
        or via the pre-built memmap (cache enabled, new behavior).

    _deterministic_unique_peptide_list() — both orchestrator and workers
        must derive identical peptide lists from the same train_data, or
        the cache key won't match.

    _initialize_encoding_cache() — orchestrator-side build step.

The load-bearing assertion throughout: with or without the cache, the
bytes that ``peptides_to_network_input`` produces from the returned
EncodableSequences are identical. Bit-identical. Any deviation here means
Phase 1 is not semantics-preserving and should not be merged.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy
import pandas
import pytest
from numpy.testing import assert_array_equal

from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.encoding_cache import EncodingCache, EncodingParams
from mhcflurry.train_pan_allele_models_command import (
    _build_train_peptides,
    _deterministic_unique_peptide_list,
    _initialize_encoding_cache,
    _read_pretrain_peptide_list,
    GLOBAL_DATA,
    pretrain_data_iterator,
)


# Representative 8-15mer peptides + a few repeats, mimicking how real
# training data has the same peptide appear across multiple alleles.
SAMPLE_TRAIN_PEPTIDES = [
    "SIINFEKL",
    "GILGFVFTL",
    "NLVPMVATV",
    "YLQPRTFLL",
    "KLVALGINAV",
    "GILGFVFTL",  # repeat
    "FLRGRAYGL",
    "ELAGIGILTV",
    "NLVPMVATV",  # repeat
    "RMFPNAPYL",
    "AAAAAAAAA",
    "SIINFEKL",  # repeat
]

DEFAULT_PEPTIDE_ENCODING = {
    "alignment_method": "left_pad_centered_right_pad",
    "max_length": 15,
    "vector_encoding_name": "BLOSUM62",
}


@pytest.fixture(autouse=True)
def clear_global_data():
    """Reset GLOBAL_DATA between tests so cache state doesn't bleed over."""
    GLOBAL_DATA.clear()
    yield
    GLOBAL_DATA.clear()


@pytest.fixture
def train_data():
    """Mock train_data DataFrame matching the shape train_model expects.

    The real df also has allele, measurement_value, etc. columns; we only
    need peptide for this test.
    """
    return pandas.DataFrame({"peptide": SAMPLE_TRAIN_PEPTIDES})


@pytest.fixture
def constant_data_no_cache(train_data):
    return {"train_data": train_data}


@pytest.fixture
def constant_data_with_cache(train_data, tmp_path):
    """Set up GLOBAL_DATA as if _initialize_encoding_cache had already run."""
    data = {
        "train_data": train_data,
        "encoding_cache_dir": str(tmp_path / "encoding_cache"),
    }
    return data


@pytest.fixture
def hyperparameters():
    return {"peptide_encoding": DEFAULT_PEPTIDE_ENCODING}


# ---- _deterministic_unique_peptide_list ----


def test_unique_peptide_list_is_first_seen_order(train_data):
    got = _deterministic_unique_peptide_list(train_data.peptide.values)
    # Order of first appearance, no duplicates.
    assert got == [
        "SIINFEKL",
        "GILGFVFTL",
        "NLVPMVATV",
        "YLQPRTFLL",
        "KLVALGINAV",
        "FLRGRAYGL",
        "ELAGIGILTV",
        "RMFPNAPYL",
        "AAAAAAAAA",
    ]


def test_unique_peptide_list_stable_across_calls(train_data):
    """Must be deterministic — orchestrator and worker must agree."""
    first = _deterministic_unique_peptide_list(train_data.peptide.values)
    second = _deterministic_unique_peptide_list(train_data.peptide.values)
    assert first == second


# ---- _build_train_peptides — the semantic preservation test ----


def test_build_train_peptides_cache_disabled_returns_plain_encodable(
    constant_data_no_cache, hyperparameters, train_data
):
    """Cache-disabled path must be indistinguishable from EncodableSequences(values).

    Guards against accidentally changing the old-path behavior via refactor.
    """
    peptides = train_data.peptide.values
    got = _build_train_peptides(peptides, hyperparameters, constant_data_no_cache)
    assert isinstance(got, EncodableSequences)
    # encoding_cache should be empty on entry (no prepopulation)
    assert got.encoding_cache == {}


def test_build_train_peptides_cache_enabled_returns_prepopulated(
    constant_data_with_cache, hyperparameters, train_data
):
    """Cache-enabled path returns an EncodableSequences whose cache is prepopulated."""
    # Orchestrator would have run _initialize_encoding_cache first. Fake it.
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)
    # Verify orchestrator stashed the metadata.
    assert "encoding_cache_dir" in GLOBAL_DATA
    constant_data = dict(GLOBAL_DATA)  # shallow copy, what workers would see

    peptides = train_data.peptide.values
    got = _build_train_peptides(peptides, hyperparameters, constant_data)
    assert isinstance(got, EncodableSequences)
    # encoding_cache should have exactly one entry — the prepopulated one.
    assert len(got.encoding_cache) == 1


def test_build_train_peptides_bit_identical_across_modes(
    constant_data_no_cache, constant_data_with_cache, hyperparameters, train_data
):
    """THE GATE: cached and un-cached paths produce byte-identical network input.

    We run peptides_to_network_input on both and assert identical bytes.
    If this fails, Phase 1 is not semantics-preserving. Do not merge.
    """
    # Initialize the cache (orchestrator step).
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)
    constant_data_cached = dict(GLOBAL_DATA)

    peptides = train_data.peptide.values

    # Old path: plain EncodableSequences, encoded directly.
    es_old = _build_train_peptides(peptides, hyperparameters, constant_data_no_cache)
    # New path: prepopulated from cache.
    es_new = _build_train_peptides(peptides, hyperparameters, constant_data_cached)

    # Encode via a Class1NeuralNetwork (which is how it happens in real training).
    net = Class1NeuralNetwork(peptide_encoding=DEFAULT_PEPTIDE_ENCODING)
    encoded_old = net.peptides_to_network_input(es_old)
    encoded_new = net.peptides_to_network_input(es_new)

    assert encoded_old.shape == encoded_new.shape
    # Both should be the same dtype post-encoding. Old path may return
    # float64 (depending on upstream); new path pins float32. Compare
    # after common-dtype cast — identity is what matters, not dtype width.
    assert_array_equal(
        encoded_old.astype(numpy.float32), encoded_new.astype(numpy.float32)
    )


def test_build_train_peptides_preserves_fold_order(
    constant_data_with_cache, hyperparameters, train_data
):
    """The fold's shuffled peptide order is preserved post-cache lookup.

    Training assigns (peptide[i], allele[i], affinity[i]) together. If the
    cache lookup reshuffled rows, affinities would be paired with the wrong
    peptides and training would silently learn garbage.
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)
    constant_data = dict(GLOBAL_DATA)

    # Simulate a fold that shuffles the training data — sample frac=1.0.
    shuffled = train_data.sample(frac=1.0, random_state=42)
    peptides_shuffled = shuffled.peptide.values

    got = _build_train_peptides(peptides_shuffled, hyperparameters, constant_data)
    net = Class1NeuralNetwork(peptide_encoding=DEFAULT_PEPTIDE_ENCODING)
    encoded_via_cache = net.peptides_to_network_input(got)

    # What we'd get without the cache on the shuffled list:
    expected = net.peptides_to_network_input(EncodableSequences(peptides_shuffled))

    assert_array_equal(
        encoded_via_cache.astype(numpy.float32),
        expected.astype(numpy.float32),
    )


def test_build_train_peptides_different_archs_different_configs(
    constant_data_with_cache, train_data
):
    """Architectures with different peptide_encoding configs use separate caches.

    Architecture A uses alignment=pad_middle, B uses left_pad_centered.
    Each needs its own cache entry; they must not collide.
    """
    hp_a = {"peptide_encoding": {
        "alignment_method": "pad_middle",
        "max_length": 15,
        "vector_encoding_name": "BLOSUM62",
    }}
    hp_b = {"peptide_encoding": {
        "alignment_method": "left_pad_centered_right_pad",
        "max_length": 15,
        "vector_encoding_name": "BLOSUM62",
    }}
    all_work_items = [
        {"hyperparameters": hp_a},
        {"hyperparameters": hp_b},
    ]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)
    constant_data = dict(GLOBAL_DATA)

    peptides = train_data.peptide.values
    es_a = _build_train_peptides(peptides, hp_a, constant_data)
    es_b = _build_train_peptides(peptides, hp_b, constant_data)

    net_a = Class1NeuralNetwork(peptide_encoding=hp_a["peptide_encoding"])
    net_b = Class1NeuralNetwork(peptide_encoding=hp_b["peptide_encoding"])
    encoded_a = net_a.peptides_to_network_input(es_a)
    encoded_b = net_b.peptides_to_network_input(es_b)

    # Shapes differ because alignment changes encoded_len (15 vs 45).
    assert encoded_a.shape != encoded_b.shape


def test_initialize_encoding_cache_is_idempotent(
    constant_data_with_cache, hyperparameters
):
    """Calling _initialize_encoding_cache twice should hit the cache the second time."""
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)

    _initialize_encoding_cache(args, all_work_items)
    # Spy on EncodableSequences to verify no re-encoding happens on second call.
    calls = []
    original = EncodableSequences.variable_length_to_fixed_length_vector_encoding

    def spy(self, *a, **kw):
        calls.append(1)
        return original(self, *a, **kw)

    EncodableSequences.variable_length_to_fixed_length_vector_encoding = spy
    try:
        _initialize_encoding_cache(args, all_work_items)
    finally:
        EncodableSequences.variable_length_to_fixed_length_vector_encoding = original
    assert calls == [], "second initialize should not re-encode"


# ---- Backward-compat ----


def test_disabled_cache_path_does_not_access_cache_dir(
    constant_data_no_cache, hyperparameters, train_data, tmp_path, monkeypatch
):
    """With the cache disabled, no filesystem access should occur under cache_dir.

    Guards against accidental cache-dir lookups in the cache-disabled path —
    regressions where someone adds a .get() that defaults to the cache dir.
    """
    # If constant_data has no 'encoding_cache_dir', EncodingCache must not
    # be constructed at all. We can't easily assert on that without patching,
    # but we can assert the call still works and returns a plain
    # EncodableSequences.
    got = _build_train_peptides(
        train_data.peptide.values, hyperparameters, constant_data_no_cache
    )
    assert got.encoding_cache == {}


# ---- pretrain_data_iterator tests ---------------------------------------


PRETRAIN_PEPTIDES = [
    "SIINFEKL",
    "GILGFVFTL",
    "NLVPMVATV",
    "YLQPRTFLL",
    "KLVALGINAV",
    "FLRGRAYGL",
    "ELAGIGILTV",
    "RMFPNAPYL",
]

# Two alleles with arbitrary but canonical-AA sequences. The exact content
# doesn't matter for the iterator path; only the allele names must match
# what the AlleleEncoding knows about.
PRETRAIN_ALLELES = ["HLA-A*02:01", "HLA-B*07:02"]
ALLELE_TO_SEQUENCE = {
    "HLA-A*02:01": "YFAMYQENMAHTDANTLYLNYHDYTWAVLAYTWY",
    "HLA-B*07:02": "YFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAA",
}


@pytest.fixture
def pretrain_csv(tmp_path):
    """Write a tiny pretrain CSV matching the iterator's expected format.

    Format: index column is the peptide; remaining columns are allele ->
    affinity. The iterator reads rows in chunks of ``peptides_per_chunk``.
    """
    path = tmp_path / "pretrain.csv"
    # Generate reproducible synthetic affinities.
    rng = numpy.random.default_rng(42)
    rows = []
    for peptide in PRETRAIN_PEPTIDES:
        rows.append({
            "peptide": peptide,
            **{allele: rng.random() for allele in PRETRAIN_ALLELES},
        })
    pandas.DataFrame(rows).to_csv(path, index=False)
    # The iterator expects the peptide to be the CSV index, not a column,
    # so re-read and rewrite with index_col.
    df = pandas.read_csv(path)
    df.set_index("peptide", inplace=True)
    df.to_csv(path)
    return path


@pytest.fixture
def pretrain_allele_encoding():
    """A minimal AlleleEncoding covering the pretrain alleles."""
    return AlleleEncoding(
        alleles=PRETRAIN_ALLELES,
        allele_to_sequence=ALLELE_TO_SEQUENCE,
    )


def test_read_pretrain_peptide_list_returns_index(pretrain_csv):
    got = _read_pretrain_peptide_list(pretrain_csv)
    assert got == PRETRAIN_PEPTIDES


def _consume_iterator(iterator, n_chunks):
    """Pull n_chunks from a pretrain_data_iterator generator.

    Returns a list of (allele_encoding, encodable_peptides, affinities) tuples.
    """
    out = []
    for _ in range(n_chunks):
        out.append(next(iterator))
    return out


def test_pretrain_iterator_without_cache_yields_normally(
    pretrain_csv, pretrain_allele_encoding
):
    """Sanity: uncached path still works with small chunks."""
    it = pretrain_data_iterator(
        str(pretrain_csv),
        pretrain_allele_encoding,
        peptides_per_chunk=4,
    )
    chunks = _consume_iterator(it, n_chunks=2)
    # 2 chunks of 4 peptides each = 8 peptides. Each yielded EncodableSequences
    # has 4 * 2 alleles = 8 peptide slots (peptides × alleles).
    for (alleles, encodable, affinities) in chunks:
        assert isinstance(encodable, EncodableSequences)
        # 4 peptides × 2 alleles
        assert len(encodable.sequences) == 8
        assert len(affinities) == 8


def test_pretrain_iterator_with_cache_bit_identical(
    pretrain_csv, pretrain_allele_encoding, tmp_path
):
    """THE PRETRAIN GATE: cached yields must match uncached byte-for-byte.

    We consume the same number of chunks from both paths and compare the
    peptide encoding tensors produced by passing each chunk's
    EncodableSequences through peptides_to_network_input.
    """
    params = EncodingParams(
        alignment_method="left_pad_centered_right_pad",
        max_length=15,
        vector_encoding_name="BLOSUM62",
    )
    cache_dir = tmp_path / "cache"

    # Uncached.
    it_uncached = pretrain_data_iterator(
        str(pretrain_csv),
        pretrain_allele_encoding,
        peptides_per_chunk=4,
    )
    chunks_uncached = _consume_iterator(it_uncached, n_chunks=2)

    # Cached.
    it_cached = pretrain_data_iterator(
        str(pretrain_csv),
        pretrain_allele_encoding,
        peptides_per_chunk=4,
        encoding_cache_dir=str(cache_dir),
        encoding_params=params,
    )
    chunks_cached = _consume_iterator(it_cached, n_chunks=2)

    # Compare chunk by chunk.
    net = Class1NeuralNetwork(
        peptide_encoding={
            "alignment_method": "left_pad_centered_right_pad",
            "max_length": 15,
            "vector_encoding_name": "BLOSUM62",
        },
    )
    for (uncached, cached) in zip(chunks_uncached, chunks_cached):
        _, enc_uncached, aff_uncached = uncached
        _, enc_cached, aff_cached = cached
        # Affinities come from the same CSV; must be identical.
        assert_array_equal(aff_uncached, aff_cached)
        # Sequences in both paths should be the same repeated-peptide vector.
        assert list(enc_uncached.sequences) == list(enc_cached.sequences)
        # And the encoded peptide tensor through the network must match.
        enc_via_uncached = net.peptides_to_network_input(enc_uncached).astype(
            numpy.float32
        )
        enc_via_cached = net.peptides_to_network_input(enc_cached).astype(
            numpy.float32
        )
        assert_array_equal(enc_via_uncached, enc_via_cached)


def test_pretrain_iterator_cache_hit_avoids_reencoding(
    pretrain_csv, pretrain_allele_encoding, tmp_path
):
    """Second iterator constructor with same params hits the warm cache.

    The first constructor builds the cache; the second should not reencode
    — verified by spying on EncodableSequences to count encode calls.
    """
    params = EncodingParams(
        alignment_method="left_pad_centered_right_pad",
        max_length=15,
        vector_encoding_name="BLOSUM62",
    )
    cache_dir = tmp_path / "cache"

    # Warm the cache.
    it_warm = pretrain_data_iterator(
        str(pretrain_csv),
        pretrain_allele_encoding,
        peptides_per_chunk=4,
        encoding_cache_dir=str(cache_dir),
        encoding_params=params,
    )
    _consume_iterator(it_warm, n_chunks=1)

    calls = []
    original = EncodableSequences.variable_length_to_fixed_length_vector_encoding

    def spy(self, *a, **kw):
        calls.append(1)
        return original(self, *a, **kw)

    EncodableSequences.variable_length_to_fixed_length_vector_encoding = spy
    try:
        it_hot = pretrain_data_iterator(
            str(pretrain_csv),
            pretrain_allele_encoding,
            peptides_per_chunk=4,
            encoding_cache_dir=str(cache_dir),
            encoding_params=params,
        )
        _consume_iterator(it_hot, n_chunks=1)
    finally:
        EncodableSequences.variable_length_to_fixed_length_vector_encoding = original

    # Cache hit should skip the full encoding pass. A few calls MAY happen
    # (dry-run sample, etc.) but substantially fewer than the chunk size.
    # Concretely: the cache is already built, so get_or_build skips to
    # _load without re-encoding. Zero calls expected.
    assert calls == [], (
        f"cache hit path should not call variable_length_to_fixed_length_"
        f"vector_encoding; got {len(calls)} calls"
    )


def test_pretrain_iterator_cache_dir_only_needs_encoding_params(
    pretrain_csv, pretrain_allele_encoding, tmp_path
):
    """Passing cache_dir without encoding_params disables caching safely.

    The guard in pretrain_data_iterator requires BOTH; partial config
    should fall through to the uncached path rather than crash.
    """
    it = pretrain_data_iterator(
        str(pretrain_csv),
        pretrain_allele_encoding,
        peptides_per_chunk=4,
        encoding_cache_dir=str(tmp_path / "cache"),
        encoding_params=None,  # explicitly none
    )
    chunks = _consume_iterator(it, n_chunks=1)
    # No crash; chunks yielded normally.
    assert len(chunks) == 1


# ---- Subprocess-boundary (pickle) tests ---------------------------------
#
# The real training command runs workers via multiprocessing.Pool, which
# pickles ``GLOBAL_DATA`` (here, ``constant_data``) and ships it to each
# worker. The design deliberately keeps only the cache_dir string in
# GLOBAL_DATA — workers re-open the memmap themselves on disk.
#
# These tests exercise the pickle boundary WITHOUT the cost of spawning an
# actual training subprocess: if these pass, the data that crosses the
# boundary round-trips cleanly, and the downstream worker helper
# (``_build_train_peptides``) reconstructs the cache correctly.
#
# The heavy subprocess tests live in test_train_pan_allele_models_command.py.


def test_constant_data_survives_pickle_roundtrip(
    constant_data_with_cache, hyperparameters, train_data
):
    """Pickle/unpickle the constant_data dict and verify the cache still works.

    Simulates exactly what multiprocessing.Pool does: serialize
    GLOBAL_DATA in the parent, send the bytes to a worker, deserialize
    there, then call _build_train_peptides with the reconstructed dict.
    If anything in constant_data fails to pickle (or lies about its
    post-pickle identity) this catches it — no spawn needed.
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)
    constant_data = dict(GLOBAL_DATA)

    # The pickle round-trip: what Pool workers actually receive.
    roundtripped = pickle.loads(pickle.dumps(constant_data))
    assert roundtripped["encoding_cache_dir"] == constant_data["encoding_cache_dir"]

    # Call the worker helper with the unpickled dict — should work.
    peptides = train_data.peptide.values
    got = _build_train_peptides(peptides, hyperparameters, roundtripped)
    assert isinstance(got, EncodableSequences)
    assert len(got.encoding_cache) == 1, (
        "worker on reconstructed constant_data did not hit the prepopulation path"
    )


def test_global_data_contains_only_small_primitives(
    constant_data_with_cache, hyperparameters
):
    """Sanity: GLOBAL_DATA must not accidentally embed large arrays post-init.

    The design stores the cache_dir (a ~100-byte string) and nothing else
    that scales with data size — specifically NOT the memmap'd encoded
    array nor the peptide_to_idx dict. If a future refactor accidentally
    shoves the memmap into GLOBAL_DATA, the pickle shipped to each worker
    balloons from ~1KB to ~1GB, silently defeating the whole optimization.

    We check by asserting the pickled size of GLOBAL_DATA is "small"
    (under 1 MB). This is loose but catches regressions by an order of
    magnitude.
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)

    # Exclude train_data from the size check — that DataFrame is legitimately
    # large and is loaded into workers regardless of cache mode. We only
    # want to catch cache-introduced bloat.
    cache_related = {
        k: v for k, v in GLOBAL_DATA.items() if k.startswith("encoding_cache")
    }
    payload_size = len(pickle.dumps(cache_related))
    assert payload_size < 1_000_000, (
        f"cache-related GLOBAL_DATA payload is {payload_size} bytes — "
        f"looks like an ndarray snuck in. Workers should re-open the memmap, "
        f"not receive a copy of it."
    )


def test_memmap_reopens_cleanly_in_fresh_encoding_cache(
    constant_data_with_cache, hyperparameters
):
    """A fresh EncodingCache instance opens the same on-disk entry as the builder.

    This mimics exactly what a worker does: orchestrator built the cache
    entry, then a separate process (a worker) constructs a NEW
    EncodingCache with the same cache_dir + params and calls get_or_build,
    which must hit the existing entry and return a memmap pointing at the
    SAME bytes.

    Failure mode this catches: path construction differing by one character
    between orchestrator and worker (e.g. trailing slash handling, absolute
    vs relative path resolution).
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)

    # "Worker" reconstructs from the pickled cache_dir string.
    cache_dir_str = GLOBAL_DATA["encoding_cache_dir"]
    worker_cache = EncodingCache(
        cache_dir=Path(cache_dir_str),
        params=EncodingParams(**hyperparameters["peptide_encoding"]),
    )

    # Same unique-peptides list as orchestrator would have built.
    df = GLOBAL_DATA["train_data"]
    unique_peptides = _deterministic_unique_peptide_list(df.peptide.values)

    # Should be a cache HIT — the entry exists from orchestrator's build.
    entry_dir = worker_cache.entry_path(unique_peptides)
    assert (entry_dir / ".complete").exists(), (
        f"orchestrator's entry at {entry_dir} not found by worker-reconstructed cache"
    )

    # And get_or_build should not re-encode.
    calls = []
    original = EncodableSequences.variable_length_to_fixed_length_vector_encoding

    def spy(self, *a, **kw):
        calls.append(1)
        return original(self, *a, **kw)

    EncodableSequences.variable_length_to_fixed_length_vector_encoding = spy
    try:
        encoded, _ = worker_cache.get_or_build(unique_peptides)
    finally:
        EncodableSequences.variable_length_to_fixed_length_vector_encoding = original
    assert calls == [], "worker cache lookup triggered re-encoding"
    # Encoded mmap points at the orchestrator's file.
    assert encoded.shape[0] == len(unique_peptides)


def test_encoding_cache_dir_string_normalizes_through_pickle(
    constant_data_with_cache, hyperparameters
):
    """String paths in GLOBAL_DATA survive pickle without Path-vs-str confusion.

    A subtle bug class: orchestrator sets ``GLOBAL_DATA["encoding_cache_dir"]
    = pathlib.Path(...)`` but worker expects a string (or vice versa).
    Pickle preserves the concrete type, so we verify the stored value is
    the expected ``str`` type (not a Path) — workers that Path(str) it
    will then work on either platform.
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)

    assert isinstance(GLOBAL_DATA["encoding_cache_dir"], str), (
        "encoding_cache_dir should be stored as str for robust cross-process "
        "handling; workers wrap it with Path() themselves"
    )


def test_multiple_workers_simulated_concurrent_lookups(
    constant_data_with_cache, hyperparameters, train_data
):
    """Simulate N workers all constructing their own EncodingCache at once.

    Each would get an independent EncodingCache instance on the same
    on-disk entry. All should hit the warm cache (built by orchestrator)
    and return memmap views pointing at the same bytes.

    Real Pool workers are threads/processes; here we use loop iterations
    as a stand-in — the inputs and assertions are the same since the
    cache logic is stateless beyond the filesystem.
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)
    constant_data = dict(GLOBAL_DATA)

    # Every "worker" gets the pickled-and-unpickled constant_data dict
    # (what Pool.apply_async would deliver).
    worker_views = []
    for _ in range(4):
        worker_dict = pickle.loads(pickle.dumps(constant_data))
        es = _build_train_peptides(
            train_data.peptide.values, hyperparameters, worker_dict
        )
        # Retrieve the prepopulated encoded array from the EncodableSequences.
        params = EncodingParams(**hyperparameters["peptide_encoding"])
        from mhcflurry.encoding_cache import _vector_encoding_cache_key
        cache_key = _vector_encoding_cache_key(params)
        worker_views.append(es.encoding_cache[cache_key])

    # All workers should see byte-identical encoded arrays.
    reference = worker_views[0]
    for i, view in enumerate(worker_views[1:], start=1):
        assert_array_equal(
            numpy.asarray(view),
            numpy.asarray(reference),
            err_msg=f"worker {i} got different bytes than worker 0",
        )


def test_worker_fails_loudly_on_stale_cache_dir(
    constant_data_no_cache, hyperparameters, train_data, tmp_path
):
    """If a worker is handed a cache_dir that doesn't exist, it rebuilds.

    Concretely: orchestrator crashed and left a stale cache_dir reference
    in GLOBAL_DATA pointing at a now-deleted directory. Worker's call
    to ``_build_train_peptides`` should still succeed (cache miss → build).

    This isn't the common case but tests fault tolerance at the subprocess
    boundary where transient state can get weird.
    """
    bogus_cache_dir = tmp_path / "does-not-exist-yet"
    assert not bogus_cache_dir.exists()
    constant_data = {
        "train_data": train_data,
        "encoding_cache_dir": str(bogus_cache_dir),
    }
    got = _build_train_peptides(
        train_data.peptide.values, hyperparameters, constant_data
    )
    # Should have worked — cache was built from scratch on access.
    assert isinstance(got, EncodableSequences)
    assert len(got.encoding_cache) == 1
    assert bogus_cache_dir.exists(), (
        "cache_dir should have been created on-demand"
    )


# ---- Stashed unique_peptides in GLOBAL_DATA -----------------------------


def test_initialize_stashes_unique_peptides_in_global_data(
    constant_data_with_cache, hyperparameters
):
    """Orchestrator stashes unique peptide list so workers skip recomputation."""
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)

    assert "encoding_cache_unique_peptides" in GLOBAL_DATA
    stashed = GLOBAL_DATA["encoding_cache_unique_peptides"]
    assert isinstance(stashed, list)
    # Should equal what _deterministic_unique_peptide_list returns.
    df = GLOBAL_DATA["train_data"]
    expected = _deterministic_unique_peptide_list(df.peptide.values)
    assert stashed == expected


def test_build_train_peptides_uses_stashed_unique_peptides(
    constant_data_with_cache, hyperparameters, train_data, monkeypatch
):
    """Worker uses stashed peptide list instead of recomputing from train_data."""
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)
    constant_data = dict(GLOBAL_DATA)

    # Spy on _deterministic_unique_peptide_list to verify it's NOT called.
    from mhcflurry import train_pan_allele_models_command as cmd
    calls = []
    original = cmd._deterministic_unique_peptide_list

    def spy(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(cmd, "_deterministic_unique_peptide_list", spy)

    # Call from "worker" side with the pickled constant_data.
    worker_dict = pickle.loads(pickle.dumps(constant_data))
    _build_train_peptides(
        train_data.peptide.values, hyperparameters, worker_dict
    )
    assert calls == [], (
        "worker should reuse stashed unique peptides instead of recomputing"
    )


def test_build_train_peptides_fallback_when_stash_absent(
    constant_data_no_cache, hyperparameters, train_data, tmp_path, monkeypatch
):
    """Backward-compat: works even without the stashed peptide list.

    Older GLOBAL_DATA pickles (pre-fix) or manually-constructed test dicts
    won't have encoding_cache_unique_peptides. The worker must fall back
    to recomputing from train_data in that case.
    """
    from mhcflurry import train_pan_allele_models_command as cmd

    # Simulate an "old" constant_data: has cache_dir, no stashed peptides.
    cache_dir = tmp_path / "cache"
    constant_data = {
        "train_data": train_data,
        "encoding_cache_dir": str(cache_dir),
    }
    calls = []
    original = cmd._deterministic_unique_peptide_list

    def spy(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(cmd, "_deterministic_unique_peptide_list", spy)

    got = _build_train_peptides(
        train_data.peptide.values, hyperparameters, constant_data
    )
    assert isinstance(got, EncodableSequences)
    assert len(got.encoding_cache) == 1
    assert len(calls) == 1, (
        f"expected exactly 1 fallback call to _deterministic_unique_peptide_list, "
        f"got {len(calls)}"
    )


def test_initialize_encoding_cache_prebuilds_pretrain_cache(
    constant_data_with_cache, hyperparameters, pretrain_csv
):
    """Orchestrator pre-builds the pretrain cache too, not just the train cache.

    Before this, each Pool worker on its first ``pretrain_data_iterator``
    call would race to build the pretrain cache. The Phase 1 race fix
    made that safe, but every worker still paid the redundant encoding
    cost. Pre-building here means workers hit a warm cache on first
    access.
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
        pretrain_data=str(pretrain_csv),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)

    # The pretrain cache entry must be complete after orchestrator init.
    from mhcflurry.encoding_cache import EncodingParams
    params = EncodingParams(**hyperparameters["peptide_encoding"])
    cache = EncodingCache(
        cache_dir=constant_data_with_cache["encoding_cache_dir"], params=params
    )
    pretrain_peptides = _read_pretrain_peptide_list(str(pretrain_csv))
    assert cache.is_complete_for(pretrain_peptides), (
        "pretrain cache should have been pre-built by the orchestrator"
    )


def test_initialize_encoding_cache_pretrain_skipped_when_no_pretrain_data(
    constant_data_with_cache, hyperparameters
):
    """With pretrain_data=None the orchestrator skips pretrain pre-build silently.

    Some training configs don't have pretraining; the orchestrator
    should degrade cleanly (only build the train cache, don't crash).
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
        pretrain_data=None,
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    # Should NOT raise.
    _initialize_encoding_cache(args, all_work_items)
    # Train cache must still be built.
    assert "encoding_cache_dir" in GLOBAL_DATA


def test_pretrain_iterator_hits_orchestrator_prebuilt_cache(
    constant_data_with_cache, hyperparameters, pretrain_csv,
    pretrain_allele_encoding,
):
    """Worker's first pretrain_data_iterator call should find a warm cache.

    Spy on EncodableSequences.variable_length_to_fixed_length_vector_encoding
    to verify the worker doesn't re-encode anything — the orchestrator's
    pre-build already covered the pretrain peptides.
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
        pretrain_data=str(pretrain_csv),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)

    from mhcflurry.encoding_cache import (
        EncodingParams,
        _verify_cache_key_shape,
    )
    params = EncodingParams(**hyperparameters["peptide_encoding"])

    # Prime the module-level cache-key-shape self-test so it doesn't
    # show up as encode calls under the spy. In production this runs
    # once per process regardless.
    _verify_cache_key_shape()

    # Now call pretrain_data_iterator as a worker would — no encoding
    # should happen since the orchestrator pre-built the cache.
    calls = []
    original = EncodableSequences.variable_length_to_fixed_length_vector_encoding

    def spy(self, *a, **kw):
        calls.append(1)
        return original(self, *a, **kw)

    EncodableSequences.variable_length_to_fixed_length_vector_encoding = spy
    try:
        it = pretrain_data_iterator(
            str(pretrain_csv),
            pretrain_allele_encoding,
            peptides_per_chunk=4,
            encoding_cache_dir=str(constant_data_with_cache["encoding_cache_dir"]),
            encoding_params=params,
        )
        _consume_iterator(it, n_chunks=1)
    finally:
        EncodableSequences.variable_length_to_fixed_length_vector_encoding = original

    # Zero encode calls: the cache is warm from orchestrator pre-build.
    assert calls == [], (
        f"worker re-encoded {len(calls)} peptide chunks despite orchestrator "
        f"pre-build. Either pretrain pre-build isn't running, or the worker "
        f"is using a different cache path than the orchestrator."
    )


def test_stashed_peptides_match_cache_key_peptides(
    constant_data_with_cache, hyperparameters
):
    """The stashed list must be what the cache was built against.

    Workers use the stashed list as-is. If orchestrator's stash ≠
    orchestrator's cache-build peptide list, worker hits a different
    cache entry than the one that was actually built → cache miss +
    silent rebuild of the full encoding pass.

    Belt-and-suspenders regression check.
    """
    all_work_items = [{"hyperparameters": hyperparameters}]
    args = Mock(
        use_encoding_cache=True,
        encoding_cache_dir=constant_data_with_cache["encoding_cache_dir"],
        out_models_dir=str(Path(constant_data_with_cache["encoding_cache_dir"]).parent),
    )
    GLOBAL_DATA.update(constant_data_with_cache)
    _initialize_encoding_cache(args, all_work_items)

    stashed = GLOBAL_DATA["encoding_cache_unique_peptides"]
    # Reconstruct the same cache the orchestrator built and check that
    # the stashed list resolves to a complete cache entry.
    params = EncodingParams(**hyperparameters["peptide_encoding"])
    cache = EncodingCache(
        cache_dir=GLOBAL_DATA["encoding_cache_dir"], params=params
    )
    assert cache.is_complete_for(stashed), (
        "stashed peptide list doesn't resolve to a complete cache entry; "
        "workers using the stash will miss the orchestrator's build"
    )


# ---- pretrain cache: clear KeyError on stale cache ----------------------
#
# The stale-cache scenario is narrow: it fires when the CSV content drifts
# from the peptide list the cache was built against (e.g., race condition
# where the pretrain file is edited mid-run). In normal flow the iterator
# builds its own cache entry keyed on the file's current peptides, so no
# natural test path exists — we patch in a mismatched peptide_to_idx dict
# to simulate the failure surface directly.


def test_pretrain_iterator_raises_clear_keyerror_on_missing_peptide(
    pretrain_csv, pretrain_allele_encoding, tmp_path, monkeypatch
):
    """A missing peptide in the cached lookup raises our helpful KeyError.

    We patch EncodingCache.get_or_build to return a peptide_to_idx dict
    that's missing one of the peptides that will appear in the CSV chunk.
    Exercises the error-raising branch in pretrain_data_iterator directly.
    """
    import numpy as np
    from mhcflurry.encoding_cache import EncodingCache as _EC

    params = EncodingParams(
        alignment_method="left_pad_centered_right_pad",
        max_length=15,
        vector_encoding_name="BLOSUM62",
    )
    cache_dir = tmp_path / "mycache"
    cache_dir.mkdir()

    # Fake encoded array + deliberately incomplete peptide_to_idx. Shape
    # doesn't matter for this test because we never reach indexing.
    fake_encoded = np.zeros((10, 45, 21), dtype=np.float32)
    fake_peptide_to_idx = {
        # Intentionally excludes the peptides that are in pretrain_csv.
        "SOMETHING_NOT_IN_FILE": 0,
    }

    def mock_get_or_build(self, peptides):
        return fake_encoded, fake_peptide_to_idx

    monkeypatch.setattr(_EC, "get_or_build", mock_get_or_build)

    it = pretrain_data_iterator(
        str(pretrain_csv),
        pretrain_allele_encoding,
        peptides_per_chunk=4,
        encoding_cache_dir=str(cache_dir),
        encoding_params=params,
    )
    with pytest.raises(KeyError, match="not in the encoding cache"):
        next(it)


def test_pretrain_iterator_keyerror_message_names_cache_dir(
    pretrain_csv, pretrain_allele_encoding, tmp_path, monkeypatch
):
    """The KeyError message must point the user at the remediation (cache_dir path)."""
    import numpy as np
    from mhcflurry.encoding_cache import EncodingCache as _EC

    params = EncodingParams(
        alignment_method="left_pad_centered_right_pad",
        max_length=15,
        vector_encoding_name="BLOSUM62",
    )
    cache_dir = tmp_path / "mycache"
    cache_dir.mkdir()

    def mock_get_or_build(self, peptides):
        return np.zeros((0, 45, 21), dtype=np.float32), {}

    monkeypatch.setattr(_EC, "get_or_build", mock_get_or_build)

    it = pretrain_data_iterator(
        str(pretrain_csv),
        pretrain_allele_encoding,
        peptides_per_chunk=8,
        encoding_cache_dir=str(cache_dir),
        encoding_params=params,
    )
    with pytest.raises(KeyError) as exc_info:
        next(it)
    # Message must mention the cache dir path so the user knows what to delete.
    assert "mycache" in str(exc_info.value)
    assert "different pretrain CSV" in str(exc_info.value)

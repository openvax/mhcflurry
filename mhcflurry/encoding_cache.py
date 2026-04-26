"""Memmap-backed BLOSUM62 peptide encoding cache.

Training the pan-allele model re-encodes the same ~945K training peptides
from scratch for every architecture×fold×replicate combination, and every
epoch re-encodes the 256K pretrain peptides. The encoding is a pure
function of (peptide_string, encoding_params); there's no reason to redo
it 32+ times.

This module pre-encodes a peptide list once, writes a memmap-compatible
.npy file to disk, and hands out read-only mmap views. Multiple training
workers on the same box share one physical copy via the OS page cache.

Semantics
---------
The encoded bytes are identical to what
``EncodableSequences.variable_length_to_fixed_length_vector_encoding``
returns for the same peptides and params — this module does NOT define
a new encoding, it only caches the existing one. A regression test
(``test_cached_matches_direct_byte_for_byte``) asserts this.

Cache layout
------------
Cache directory (default ``$MHCFLURRY_OUT/encoding_cache/`` or user-provided)
contains one subdirectory per (params, peptides) pair::

    <cache_dir>/<params_hash>_<peptides_hash>/
        encoded.npy       # float32, shape (N, encoded_len, 21)
        peptides.txt      # one peptide per line, order matches encoded.npy rows
        params.json       # encoding params (human-readable)
        .complete         # sentinel: present iff build finished atomically

Reader invariant: only consume a cache entry if ``.complete`` exists.
The build pass writes encoded.npy as ``encoded.npy.tmp`` first, renames,
then touches ``.complete`` last; any crash before the sentinel leaves a
half-written cache that the next reader will rebuild instead of using.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
import numpy.lib.format

from .encodable_sequences import EncodableSequences


# Default chunk size for the encoding pass. Chunks amortize the overhead of
# constructing an ``EncodableSequences`` instance over many peptides and
# cap peak memory during build (one chunk's worth of encoded float32s).
DEFAULT_CHUNK_SIZE = 100_000


@dataclass(frozen=True)
class EncodingParams:
    """The tuple of knobs that determines encoded output bytes.

    Matches the kwargs accepted by
    ``EncodableSequences.variable_length_to_fixed_length_vector_encoding``.
    All fields participate in the cache key.
    """
    vector_encoding_name: str = "BLOSUM62"
    alignment_method: str = "pad_middle"
    left_edge: int = 4
    right_edge: int = 4
    max_length: int = 15
    trim: bool = False
    allow_unsupported_amino_acids: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "vector_encoding_name": self.vector_encoding_name,
            "alignment_method": self.alignment_method,
            "left_edge": self.left_edge,
            "right_edge": self.right_edge,
            "max_length": self.max_length,
            "trim": self.trim,
            "allow_unsupported_amino_acids": self.allow_unsupported_amino_acids,
        }

    def hash_key(self) -> str:
        payload = json.dumps(self.to_kwargs(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def _hash_peptides(peptides: list[str]) -> str:
    """Stream-hash the peptide list preserving order.

    Order matters: peptide at row i maps to training row i in the caller.
    Same peptides in different order is intentionally a cache miss.
    """
    h = hashlib.sha256()
    # Hash count first so ["A", "B"] and ["AB"] produce different hashes.
    h.update(str(len(peptides)).encode("utf-8"))
    h.update(b"\0")
    for peptide in peptides:
        h.update(peptide.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


@dataclass
class EncodingCache:
    """Memmap-backed cache of BLOSUM62-encoded peptides.

    Typical usage from the orchestrator process::

        cache = EncodingCache(cache_dir=Path("out/encoding_cache"),
                              params=EncodingParams(**hyperparams["peptide_encoding"]))
        encoded, peptide_to_idx = cache.get_or_build(train_data.peptide.values)
        # encoded is a read-only mmap'd array of shape (N, encoded_len, 21)
        # peptide_to_idx is a dict; use encoded[peptide_to_idx[pep]] to look up.

    Workers subsequently open the same cache_dir+params+peptides; they
    hit the existing cache and get mmap views with zero extra encoding cost.
    """

    cache_dir: Path
    params: EncodingParams = field(default_factory=EncodingParams)
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)

    # ---- public API ----

    def get_or_build(
        self, peptides: list[str] | numpy.ndarray
    ) -> tuple[numpy.ndarray, dict[str, int]]:
        """Return (encoded_memmap_array, peptide_to_row_idx).

        Builds the cache if not present. Safe to call from multiple processes;
        a per-entry build lock avoids redundant builders, and the
        atomic-rename + sentinel-file pattern makes concurrent readers never
        see a partial write.
        """
        entry_dir = self.ensure_built(peptides)
        return self._load(entry_dir)

    def ensure_built(self, peptides: list[str] | numpy.ndarray) -> Path:
        """Build the cache entry if needed and return its directory.

        Unlike ``get_or_build``, this intentionally does not load
        ``peptides.txt`` or construct the peptide-to-row dict. Orchestrator
        prebuild paths use it when they only need the encoded memmap to exist
        on disk; loading the index map for million-row peptide files can be a
        large, useless memory spike.
        """
        peptides = _as_peptide_list(peptides)
        entry_dir = self._entry_dir(peptides)
        if self._is_complete(entry_dir):
            return entry_dir

        entry_dir.mkdir(parents=True, exist_ok=True)
        lock_path = entry_dir / ".build.lock"
        held_lock = self._acquire_build_lock(lock_path, entry_dir / ".complete")
        if held_lock is None:
            return entry_dir
        try:
            if not self._is_complete(entry_dir):
                self._build(peptides, entry_dir)
        finally:
            self._release_build_lock(held_lock)
        return entry_dir

    def entry_path(self, peptides: list[str] | numpy.ndarray) -> Path:
        """Return the directory path for the cache entry for these peptides."""
        return self._entry_dir(_as_peptide_list(peptides))

    def is_complete_for(self, peptides: list[str] | numpy.ndarray) -> bool:
        """Return True iff this (params, peptides) entry is fully built on disk.

        Public callers (orchestrators, diagnostic tooling) use this to
        avoid calling ``get_or_build`` just to check cache state. Workers
        typically just call ``get_or_build`` directly — the atomic sentinel
        check happens inside.
        """
        return self._is_complete(self._entry_dir(_as_peptide_list(peptides)))

    # ---- internals ----

    def _entry_dir(self, peptides: list[str]) -> Path:
        params_hash = self.params.hash_key()
        peptides_hash = _hash_peptides(peptides)
        return self.cache_dir / f"{params_hash}_{peptides_hash}"

    @staticmethod
    def _is_complete(entry_dir: Path) -> bool:
        return (entry_dir / ".complete").exists()

    @staticmethod
    def _acquire_build_lock(
            lock_path: Path,
            complete_path: Path,
            stale_seconds: int = 6 * 60 * 60) -> Path | None:
        while True:
            if complete_path.exists():
                return None
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return lock_path
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > stale_seconds:
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(1.0)

    @staticmethod
    def _release_build_lock(lock_path: Path | None) -> None:
        if lock_path is None:
            return
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _build(self, peptides: list[str], entry_dir: Path) -> None:
        # Re-check the sentinel now that entry_dir exists: another process
        # may have finished building while we were on the is_complete path.
        # Catches the common race where two workers simultaneously decide
        # to build; the second wakes up to find the build is done.
        if self._is_complete(entry_dir):
            return

        # Dry-run one peptide to learn the encoded shape and dtype.
        sample_encoder = EncodableSequences([peptides[0]])
        sample_encoded = sample_encoder.variable_length_to_fixed_length_vector_encoding(
            **self.params.to_kwargs()
        )
        encoded_row_shape = sample_encoded.shape[1:]
        # Preserve the encoder's native dtype. For BLOSUM62 (our only
        # vector_encoding_name in practice) upstream returns int8 — the
        # substitution-matrix values fit tightly in [-4, +6]. Storing as
        # int8 is LOSSLESS for that encoding and cuts memmap size by 4×
        # vs float32. At training-step time the batch is cast int8→fp32
        # on the GPU, which is a free widening op and reduces H2D
        # bandwidth by the same 4× factor (cold cache slices are int8
        # from the memmap, not float32).
        #
        # If a future encoding returns float values, we preserve those
        # too — only the storage width shrinks for integer-valued
        # encoders.
        dtype = sample_encoded.dtype

        n = len(peptides)
        out_path = entry_dir / "encoded.npy"
        # Per-process tmp path. The build lock should leave only one builder,
        # but unique tmp names keep interrupted stale builders from colliding.
        tmp_path = entry_dir / f"encoded.npy.tmp.{os.getpid()}"
        # open_memmap writes a .npy header plus raw bytes that np.load(mmap_mode='r')
        # can consume.
        mm = numpy.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=dtype,
            shape=(n, *encoded_row_shape),
        )
        tmp_needs_cleanup = True
        try:
            try:
                for start in range(0, n, self.chunk_size):
                    chunk = peptides[start : start + self.chunk_size]
                    encoded = (
                        EncodableSequences(chunk)
                        .variable_length_to_fixed_length_vector_encoding(
                            **self.params.to_kwargs()
                        )
                        .astype(dtype, copy=False)
                    )
                    mm[start : start + len(chunk)] = encoded
                mm.flush()
            finally:
                # Release the memmap so the rename below isn't blocked on Windows
                # and so the OS can coalesce dirty pages. `del` is the conventional
                # way to close a np.memmap.
                del mm
            # Atomic move of our unique tmp file into place. POSIX ``os.replace``
            # is atomic — a concurrent reader either sees the old inode or the
            # new one, never a partial write. If another writer raced us to
            # produce out_path, our os.replace quietly overwrites their
            # (identical) bytes.
            os.replace(tmp_path, out_path)
            tmp_needs_cleanup = False
        finally:
            # If we raised mid-write (disk full, permission denied during
            # rename, interrupt), the per-pid tmp file is dead weight:
            # our PID will rarely reissue it (os.getpid() is stable per
            # process), and subsequent runs leak it permanently. Remove
            # it so failed builds don't accumulate garbage.
            if tmp_needs_cleanup:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

        # Write peptide list and params.
        (entry_dir / "peptides.txt").write_text("\n".join(peptides) + "\n")
        (entry_dir / "params.json").write_text(
            json.dumps(self.params.to_kwargs(), indent=2, sort_keys=True) + "\n"
        )

        # Sentinel last. Presence of `.complete` is the atomic signal that the
        # cache entry is fully written and safe to consume.
        (entry_dir / ".complete").touch()

    @staticmethod
    def _load(entry_dir: Path) -> tuple[numpy.ndarray, dict[str, int]]:
        encoded = numpy.load(entry_dir / "encoded.npy", mmap_mode="r")
        peptides_text = (entry_dir / "peptides.txt").read_text()
        # The trailing newline in peptides.txt produces an empty final token
        # after splitlines+split on newline; strip empties.
        peptide_list = [p for p in peptides_text.splitlines() if p != ""]
        # Build the lookup dict lazily-ish (caller often needs it).
        peptide_to_idx = {p: i for i, p in enumerate(peptide_list)}
        return encoded, peptide_to_idx


def _as_peptide_list(peptides: list[str] | numpy.ndarray) -> list[str]:
    if isinstance(peptides, numpy.ndarray):
        return peptides.tolist()
    return list(peptides)


# ---- Integration helper ---------------------------------------------------
#
# LOAD-BEARING COMPATIBILITY SURFACE:
#
# ``make_prepopulated_encodable_sequences`` writes into the per-instance
# ``EncodableSequences.encoding_cache`` dict at a key that must match the
# tuple that ``EncodableSequences.variable_length_to_fixed_length_
# vector_encoding`` builds internally (encodable_sequences.py around
# line 160). If upstream rearranges that tuple in a future refactor,
# prepopulation silently misses and the "cache hit" path degrades to a
# full re-encode + memmap read — silent perf regression, correct output.
#
# We protect against this two ways:
#
#  1. A module-level self-test (``_verify_cache_key_shape``, run once per
#     process on first use) that builds a prepopulated ``EncodableSequences``
#     and confirms the prepopulated tensor IS returned by a subsequent
#     ``variable_length_to_fixed_length_vector_encoding`` call. If it
#     isn't, we raise ``EncodingCacheKeyMismatchError`` loudly at first
#     use with a pointer to this module — no silent degradation.
#
#  2. A regression test (``test_prepopulated_hits_existing_cache_path``)
#     that asserts the same invariant at CI time.


class EncodingCacheKeyMismatchError(RuntimeError):
    """Raised when our cache-key tuple shape no longer matches EncodableSequences.

    Indicates a breaking upstream change: the internal cache key in
    ``EncodableSequences.variable_length_to_fixed_length_vector_encoding``
    has been refactored in a way that ``_vector_encoding_cache_key``
    didn't keep up with. Fix by reading the updated upstream code and
    mirroring the new tuple layout here.
    """


def _vector_encoding_cache_key(params: EncodingParams) -> tuple:
    """Construct the cache key tuple EncodableSequences uses internally.

    Must exactly match the tuple built inside
    ``EncodableSequences.variable_length_to_fixed_length_vector_encoding``.
    Protected by a module-level self-test — see the block comment above.
    """
    return (
        "fixed_length_vector_encoding",
        params.vector_encoding_name,
        params.alignment_method,
        params.left_edge,
        params.right_edge,
        params.max_length,
        params.trim,
        params.allow_unsupported_amino_acids,
    )


# Guard flag so ``_verify_cache_key_shape`` runs exactly once per process.
# Module-level mutable so ``reset`` helpers in tests can re-exercise it.
_CACHE_KEY_SHAPE_VERIFIED = False


def _verify_cache_key_shape() -> None:
    """Assert that our prepopulation key round-trips through EncodableSequences.

    Build a tiny EncodableSequences, prepopulate its instance cache using
    ``_vector_encoding_cache_key``, then call
    ``variable_length_to_fixed_length_vector_encoding`` and assert the
    returned array IS (object identity) the one we stored. If it isn't,
    the upstream key layout has drifted and we raise a clear error up
    front — no silent perf regression.

    Runs once per process (guarded by ``_CACHE_KEY_SHAPE_VERIFIED``).
    Fast: one 8-mer encoded with a 15-length pad. Amortized across all
    subsequent ``make_prepopulated_encodable_sequences`` calls.
    """
    global _CACHE_KEY_SHAPE_VERIFIED
    if _CACHE_KEY_SHAPE_VERIFIED:
        return

    # Use the default params for the self-test; the real check is the
    # tuple shape matching, which is param-value-independent.
    params = EncodingParams()
    # Minimal peptide set that works with default params
    # (max_length=15, pad_middle handles 8-15 without trim).
    probe_peptides = ["SIINFEKL"]
    probe_tensor = (
        EncodableSequences(probe_peptides)
        .variable_length_to_fixed_length_vector_encoding(**params.to_kwargs())
        .astype(numpy.float32)
    )

    sentinel = probe_tensor.copy() + 777.0  # distinguishable bytes
    es = EncodableSequences(probe_peptides)
    es.encoding_cache[_vector_encoding_cache_key(params)] = sentinel
    returned = es.variable_length_to_fixed_length_vector_encoding(
        **params.to_kwargs()
    )
    if returned is not sentinel:
        raise EncodingCacheKeyMismatchError(
            "The per-instance EncodableSequences.encoding_cache key tuple no "
            "longer matches what `_vector_encoding_cache_key` in "
            "mhcflurry/encoding_cache.py produces. Prepopulation silently "
            "missed. Update `_vector_encoding_cache_key` to match the key "
            "constructed inside "
            "`EncodableSequences.variable_length_to_fixed_length_vector_"
            "encoding` (see encodable_sequences.py)."
        )

    _CACHE_KEY_SHAPE_VERIFIED = True


def deterministic_unique_peptide_list(peptide_values) -> list[str]:
    """Return unique peptides in first-seen order.

    Stable: the orchestrator's list must match what workers compute, or
    the cache entry's content-addressed key won't match. ``pandas.Series
    .drop_duplicates()`` preserves first-seen order — use it consistently
    in every caller.
    """
    import pandas
    return list(pandas.Series(peptide_values).drop_duplicates())


def prebuild_encoding_caches(
    *,
    cache_dir: str | os.PathLike,
    peptides: list[str],
    encoding_configs: list[dict],
    label: str = "peptides",
    log=print,
) -> None:
    """Pre-encode ``peptides`` for each of ``encoding_configs``.

    Generic orchestrator-side helper: write one encoding cache entry per
    (config, peptides) pair under ``cache_dir``. Run this BEFORE forking
    training workers so each worker's first cache lookup is a hit
    instead of triggering N workers to race-build the same entry.

    The pan-allele orchestrator (``train_pan_allele_models_command``)
    wraps this with its own pretrain/folds/master-allele-encoding glue;
    other orchestrators (``train_processing_models_command``,
    ``train_allele_specific_models_command``) can call it directly when
    their datasets grow large enough to make encoding a bottleneck.

    Idempotent: cache hits short-circuit. Caller is responsible for
    de-duplicating ``encoding_configs`` (typically by hashing each
    config dict's items) — duplicates here only cost a no-op cache
    check, not a re-encode.
    """
    for cfg in encoding_configs:
        params = EncodingParams(**cfg)
        cache = EncodingCache(cache_dir=str(cache_dir), params=params)
        if cache.is_complete_for(peptides):
            log(
                f"Encoding cache hit ({label}): {cache.entry_path(peptides)} "
                f"({len(peptides)} peptides)"
            )
            continue
        log(
            f"Building encoding cache ({label}) for params "
            f"{params.to_kwargs()} ({len(peptides)} peptides) "
            f"at {cache.entry_path(peptides)}..."
        )
        t0 = time.time()
        cache.ensure_built(peptides)
        log(f"Encoding cache built ({label}) in {time.time() - t0:.1f}s.")


def make_prepopulated_encodable_sequences(
    peptides: list[str] | numpy.ndarray,
    encoded_rows: numpy.ndarray,
    params: EncodingParams,
) -> EncodableSequences:
    """Return an EncodableSequences whose per-instance encoding_cache is
    prepopulated with ``encoded_rows``.

    When downstream code (``Class1NeuralNetwork.peptides_to_network_input``)
    calls ``.variable_length_to_fixed_length_vector_encoding(**params)`` on
    the returned instance, it will hit the cache and return ``encoded_rows``
    directly without re-encoding — the whole point of the exercise.

    On first call in the process, runs a self-test
    (``_verify_cache_key_shape``) that raises
    ``EncodingCacheKeyMismatchError`` if the upstream cache-key layout has
    drifted. See the block comment above for rationale.

    ``encoded_rows`` must be shape-compatible with what the encoder would
    have produced (N_peptides, encoded_len, alphabet_size). Caller's
    responsibility. For safety the length is checked against len(peptides).
    """
    # Verify once — converts a silent upstream break into a loud error
    # on the first cache-integration call the orchestrator makes.
    _verify_cache_key_shape()

    peptides = _as_peptide_list(peptides)
    if len(peptides) != len(encoded_rows):
        raise ValueError(
            f"encoded_rows length {len(encoded_rows)} does not match "
            f"peptide count {len(peptides)}"
        )
    es = EncodableSequences(peptides)
    es.encoding_cache[_vector_encoding_cache_key(params)] = encoded_rows
    return es

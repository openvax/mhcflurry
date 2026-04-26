"""
Train Class1 pan-allele models.
"""
import argparse
import json
import os
from os.path import join
import signal
import sys
import time
import traceback
import random
import pprint
import hashlib
import pickle
import uuid
import resource
import multiprocessing
from functools import partial

import numpy
import pandas
import yaml
import tqdm  # progress bar

from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_neural_network import Class1NeuralNetwork
from .common import configure_logging, normalize_allele_name
from .local_parallelism import (
    add_local_parallelism_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped_kwargs)
from .cluster_parallelism import (
    add_cluster_parallelism_args,
    cluster_results_from_args)
from .allele_encoding import AlleleEncoding
from .encodable_sequences import EncodableSequences
from .encoding_cache import (
    EncodingCache,
    EncodingParams,
    make_prepopulated_encodable_sequences,
)
from .regression_target import from_ic50

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing us to share large data with the workers
# via shared memory.
GLOBAL_DATA = {}

# Note on parallelization:
# When running in parallel, avoid using the neural network backend in the main
# process. Model loading and inference should happen in worker processes.


def _pop_train_param(train_params, names, default, verbose=0):
    """Pop one training parameter, honoring backward-compatible aliases."""
    present = [name for name in names if name in train_params]
    if len(present) > 1:
        values = {name: train_params[name] for name in present}
        raise ValueError(
            "Conflicting train_data aliases specified for %s: %s"
            % (names[0], values)
        )
    if present:
        name = present[0]
        result = train_params.pop(name)
        if verbose:
            suffix = "" if name == names[0] else f" [alias for {names[0]}]"
            print("Train param", name, "=", result, suffix)
        return result
    if verbose:
        print("Train param", names[0], "=", default, "[default]")
    return default


def _log_process_telemetry(marker):
    """Emit lightweight per-process RSS / FD telemetry."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = (
            rss / (1024 * 1024)
            if sys.platform == "darwin"
            else rss / 1024
        )
        rss_field = f"{rss_mb:.1f}"
    except Exception:
        rss_field = "na"

    fd_field = "na"
    for path in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(path):
            try:
                fd_field = str(len(os.listdir(path)))
            except OSError:
                pass
            break

    print(
        f"PROCESS_TELEMETRY pid={os.getpid()} marker={marker} "
        f"rss_mb={rss_field} num_fds={fd_field}"
    )

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    help=(
        "Training data CSV. Expected columns: "
        "allele, peptide, measurement_value"))
parser.add_argument(
    "--pretrain-data",
    metavar="FILE.csv",
    help=(
        "Pre-training data CSV. Expected columns: "
        "allele, peptide, measurement_value"))
parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    required=True,
    help="Directory to write models and manifest")
parser.add_argument(
    "--hyperparameters",
    metavar="FILE.json",
    help="JSON or YAML of hyperparameters")
parser.add_argument(
    "--held-out-measurements-per-allele-fraction-and-max",
    type=float,
    metavar="X",
    nargs=2,
    default=[0.25, 100],
    help="Fraction of measurements per allele to hold out, and maximum number")
parser.add_argument(
    "--ignore-inequalities",
    action="store_true",
    default=False,
    help="Do not use affinity value inequalities even when present in data")
parser.add_argument(
    "--num-folds",
    type=int,
    default=4,
    metavar="N",
    help="Number of training folds.")
parser.add_argument(
    "--num-replicates",
    type=int,
    metavar="N",
    default=1,
    help="Number of replicates per (architecture, fold) pair to train.")
parser.add_argument(
    "--max-epochs",
    type=int,
    metavar="N",
    help="Max training epochs. If specified here it overrides any 'max_epochs' "
    "specified in the hyperparameters.")
parser.add_argument(
    "--allele-sequences",
    metavar="FILE.csv",
    help="Allele sequences file.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Verbosity. Default: %(default)s",
    default=0)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Launch python debugger on error")
parser.add_argument(
    "--continue-incomplete",
    action="store_true",
    default=False,
    help="Continue training models from an incomplete training run. If this is "
    "specified then the only required argument is --out-models-dir")
parser.add_argument(
    "--only-initialize",
    action="store_true",
    default=False,
    help="Do not actually train models. The initialized run can be continued "
    "later with --continue-incomplete.")
parser.add_argument(
    "--use-encoding-cache",
    action="store_true",
    default=False,
    help="Precompute BLOSUM62 peptide encodings once per training run and "
    "share the result across all worker processes via mmap. Preserves bit-"
    "identical semantics; the cache output matches EncodableSequences."
    "variable_length_to_fixed_length_vector_encoding exactly. Saves 30-50x "
    "per-epoch wall-time on the CPU-bound pan-allele training path. See "
    "mhcflurry/encoding_cache.py.")
parser.add_argument(
    "--encoding-cache-dir",
    metavar="DIR",
    default=None,
    help="Directory for the encoding cache. Default: <out-models-dir>/"
    "encoding_cache/. Only used when --use-encoding-cache is set.")
parser.add_argument(
    "--random-negative-shared-pool-dir",
    metavar="DIR",
    default=None,
    help="Directory under which the orchestrator pre-builds per-fold "
    "mmap-backed random-negative pools (see mhcflurry/shared_memory.py "
    "Layer 1). When set, training workers consume an OS-page-cache-"
    "shared encoded pool instead of regenerating + re-encoding their "
    "own each cycle. Requires the hyperparameters' "
    "``random_negative_pool_epochs`` to be > 1 (otherwise each epoch "
    "regenerates and there's nothing to share). The directory is "
    "populated by ``shared_memory.setup_shared_random_negative_pools`` "
    "and written before any training worker is forked, so workers fault "
    "in pages on first read. Default: None (each worker builds its own "
    "in-process pool, the legacy behavior).")

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)


def assign_folds(df, num_folds, held_out_fraction, held_out_max):
    """
    Split training data into multple test/train pairs, which we refer to as
    folds. Note that a given data point may be assigned to multiple test or
    train sets; these folds are NOT a non-overlapping partition as used in cross
    validation.

    A fold is defined by a boolean value for each data point, indicating whether
    it is included in the training data for that fold. If it's not in the
    training data, then it's in the test data.

    Folds are balanced in terms of allele content.

    Parameters
    ----------
    df : pandas.DataFrame
        training data
    num_folds : int
    held_out_fraction : float
        Fraction of data to hold out as test data in each fold
    held_out_max
        For a given allele, do not hold out more than held_out_max number of
        data points in any fold.

    Returns
    -------
    pandas.DataFrame
        index is same as df.index, columns are "fold_0", ... "fold_N" giving
        whether the data point is in the training data for the fold
    """
    result_df = pandas.DataFrame(index=df.index)

    # Per-allele invariants (medians, low/high peptides, held-out count)
    # are invariant across folds. The original loop recomputed them
    # num_folds times. Precompute once; the fold loop only does the
    # RNG-dependent sampling step, which is what bisects the
    # bit-identical behavior we need to preserve across this refactor.
    #
    # groupby("allele") without sort=False defaults to sort=True →
    # deterministic iteration order. We materialize the list so the
    # same order is reused across folds below.
    allele_groups = list(df.groupby("allele"))
    allele_precomputed = {}
    for (allele, sub_df) in allele_groups:
        medians = sub_df.groupby("peptide").measurement_value.median()
        median_of_medians = medians.median()
        allele_precomputed[allele] = {
            "medians": medians,
            "low_peptides": medians[medians < median_of_medians].index.values,
            "high_peptides": medians[medians >= median_of_medians].index.values,
            "held_out_count": int(
                min(len(medians) * held_out_fraction, held_out_max)),
            "sub_df_index": sub_df.index,
            "sub_df_peptide": sub_df.peptide,
        }

    # Fold loop: identical structure + identical sample() order as the
    # original (fold-outer, allele-inner), so pandas' global RNG is
    # advanced in the same sequence. Output is bit-identical to the
    # pre-refactor implementation.
    for fold in range(num_folds):
        result_df["fold_%d" % fold] = True
        for (allele, _sub_df) in allele_groups:
            cached = allele_precomputed[allele]
            medians = cached["medians"]
            low_peptides = cached["low_peptides"]
            high_peptides = cached["high_peptides"]
            held_out_count = cached["held_out_count"]

            held_out_peptides = set()
            if held_out_count == 0:
                pass
            elif held_out_count < 2:
                held_out_peptides = set(
                    medians.index.to_series().sample(n=held_out_count))
            else:
                held_out_low_count = min(
                    len(low_peptides),
                    int(held_out_count / 2))
                held_out_high_count = min(
                    len(high_peptides),
                    held_out_count - held_out_low_count)

                held_out_low = pandas.Series(low_peptides).sample(
                    n=held_out_low_count) if held_out_low_count else set()
                held_out_high = pandas.Series(high_peptides).sample(
                    n=held_out_high_count) if held_out_high_count else set()
                held_out_peptides = set(held_out_low).union(set(held_out_high))

            sub_df_peptide = cached["sub_df_peptide"]
            result_df.loc[
                cached["sub_df_index"][sub_df_peptide.isin(held_out_peptides)],
                "fold_%d" % fold
            ] = False

    print("Training points per fold")
    print(result_df.sum())

    print("Test points per fold")
    print((~result_df).sum())
    return result_df


def pretrain_data_iterator(
        filename,
        master_allele_encoding,
        peptides_per_chunk=1024,
        encoding_cache_dir=None,
        encoding_params=None,
        shard_rank=0,
        num_shards=1):
    """
    Step through a CSV file giving predictions for a large number of peptides
    (rows) and alleles (columns).

    Parameters
    ----------
    filename : string
    master_allele_encoding : AlleleEncoding
    peptides_per_chunk : int
    encoding_cache_dir : string, optional
        When set together with ``encoding_params``, the iterator pre-reads
        the CSV's peptide column, pre-encodes all peptides via
        ``EncodingCache``, and yields ``EncodableSequences`` instances whose
        per-instance ``encoding_cache`` is prepopulated with the memmap
        slice for each chunk's peptides. Downstream
        ``peptides_to_network_input`` calls hit the prepopulated cache and
        return the stored array without re-encoding. Bit-identical output
        to the uncached path (verified in
        ``test_pretrain_iterator_bit_identical`` in
        test_train_pan_allele_encoding_cache.py).
    encoding_params : EncodingParams, optional
        Required when ``encoding_cache_dir`` is set. Determines which
        entry in the cache to use / build.

    Returns
    -------
    Generator of (AlleleEncoding, EncodableSequences, float affinities) tuples

    """
    empty, usable_alleles, _ = _get_pretrain_allele_info(
        filename,
        master_allele_encoding,
        verbose=True,
    )

    allele_encoding_cache = {}

    # Optionally pre-encode the full CSV's peptides into a shared memmap.
    # Workers all hit the same mmap via the OS page cache; the physical
    # encoded bytes live on disk once regardless of how many training
    # workers ran. See mhcflurry/encoding_cache.py.
    pretrain_cache = None
    pretrain_peptide_to_idx = None
    pretrain_encoded_mmap = None
    if encoding_cache_dir is not None and encoding_params is not None:
        pretrain_peptides = _read_pretrain_peptide_list(filename)
        pretrain_cache = EncodingCache(
            cache_dir=encoding_cache_dir, params=encoding_params
        )
        t0 = time.time()
        pretrain_encoded_mmap, pretrain_peptide_to_idx = (
            pretrain_cache.get_or_build(pretrain_peptides)
        )
        print(
            f"Pretrain encoding cache ready ({len(pretrain_peptides)} peptides, "
            f"{time.time() - t0:.1f}s)."
        )

    while True:
        synthetic_iter = pandas.read_csv(
            filename, index_col=0, chunksize=peptides_per_chunk)
        for (k, df) in enumerate(synthetic_iter):
            if num_shards > 1 and (k % num_shards) != shard_rank:
                continue

            df.columns = empty.columns
            df = df[usable_alleles]
            chunk_len = len(df)
            if chunk_len == 0:
                continue
            if chunk_len not in allele_encoding_cache:
                allele_encoding_cache[chunk_len] = AlleleEncoding(
                    numpy.tile(usable_alleles, chunk_len),
                    borrow_from=master_allele_encoding,
                )
            allele_encoding = allele_encoding_cache[chunk_len]
            repeated_peptides = numpy.repeat(df.index.values, len(usable_alleles))

            if pretrain_encoded_mmap is None:
                encodable_peptides = EncodableSequences(repeated_peptides)
            else:
                # Index into the shared memmap and construct an
                # EncodableSequences with its cache prepopulated. Semantics
                # are identical to the fresh-encode path.
                try:
                    indices = numpy.fromiter(
                        (pretrain_peptide_to_idx[p] for p in df.index.values),
                        dtype=numpy.int64,
                        count=len(df.index),
                    )
                except KeyError as missing:
                    # Actionable error: the cache was built from a different
                    # file than the one we're now iterating. Point the user
                    # at the likely remediation (blow away the cache dir or
                    # pass a fresh --encoding-cache-dir).
                    raise KeyError(
                        f"Peptide {missing.args[0]!r} was not in the encoding "
                        f"cache built from {filename}. The cache at "
                        f"{encoding_cache_dir} likely corresponds to a "
                        f"different pretrain CSV. Delete the cache directory "
                        f"or pass a fresh --encoding-cache-dir and rerun."
                    ) from missing
                # Each unique peptide appears len(usable_alleles) times in
                # the repeated_peptides array; mirror that with a repeat on
                # the encoded rows.
                chunk_encoded = numpy.repeat(
                    pretrain_encoded_mmap[indices], len(usable_alleles), axis=0
                )
                encodable_peptides = make_prepopulated_encodable_sequences(
                    repeated_peptides, chunk_encoded, encoding_params
                )

            yield (allele_encoding, encodable_peptides, df.stack().values)


def _get_pretrain_allele_info(filename, master_allele_encoding, verbose):
    """Return (normalized-empty-df, usable_alleles, skipped_alleles)."""
    empty = pandas.read_csv(filename, index_col=0, nrows=0)
    empty.columns = empty.columns.map(normalize_allele_name)
    usable_alleles = [
        c for c in empty.columns
        if c in master_allele_encoding.allele_to_sequence
    ]
    skipped_alleles = [
        c for c in empty.columns
        if c not in master_allele_encoding.allele_to_sequence
    ]
    if verbose:
        print("Pretrain alleles available: ", *empty.columns.values)
        print("Using %d / %d alleles" % (len(usable_alleles), len(empty.columns)))
        print("Skipped alleles: ", skipped_alleles)
    return empty, usable_alleles, skipped_alleles


def _pretrain_batch_cache_dir(
    *,
    filename,
    usable_alleles,
    peptides_per_chunk,
    encoding_cache_dir,
    encoding_params,
):
    """Return the cache directory for pre-built pretrain chunks."""
    stat = os.stat(filename)
    source_token = json.dumps(
        {
            "path": os.path.abspath(filename),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "peptides_per_chunk": peptides_per_chunk,
            "usable_alleles": usable_alleles,
        },
        sort_keys=True,
    ).encode("utf-8")
    source_hash = hashlib.sha256(source_token).hexdigest()[:16]
    return join(
        encoding_cache_dir,
        "pretrain_batch_cache",
        f"{encoding_params.hash_key()}_{source_hash}",
    )


def _load_pretrain_batch_cache_manifest(cache_dir):
    manifest_path = join(cache_dir, "manifest.json")
    with open(manifest_path) as fd:
        return json.load(fd)


def _acquire_build_lock(lock_path, complete_path, stale_seconds=6 * 60 * 60):
    """Acquire a simple filesystem build lock or wait for completion."""
    while True:
        if os.path.exists(complete_path):
            return None
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_seconds:
                    os.unlink(lock_path)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(1.0)


def _release_build_lock(lock_path):
    if lock_path is None:
        return
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        pass


def _get_or_build_pretrain_batch_cache(
    *,
    filename,
    master_allele_encoding,
    peptides_per_chunk,
    encoding_cache_dir,
    encoding_params,
    verbose,
):
    """Build or load the reusable pretrain chunk cache manifest."""
    if encoding_cache_dir is None or encoding_params is None:
        raise ValueError("pretrain batch cache requires encoding cache + params")

    empty, usable_alleles, skipped_alleles = _get_pretrain_allele_info(
        filename,
        master_allele_encoding,
        verbose=verbose,
    )
    cache_dir = _pretrain_batch_cache_dir(
        filename=filename,
        usable_alleles=usable_alleles,
        peptides_per_chunk=peptides_per_chunk,
        encoding_cache_dir=encoding_cache_dir,
        encoding_params=encoding_params,
    )
    complete_path = join(cache_dir, ".complete")
    if os.path.exists(complete_path):
        manifest = _load_pretrain_batch_cache_manifest(cache_dir)
        manifest["cache_dir"] = cache_dir
        if verbose:
            print(
                f"Pretrain batch cache hit: {cache_dir} "
                f"({len(manifest['chunks'])} chunks)"
            )
        return manifest

    os.makedirs(cache_dir, exist_ok=True)
    lock_path = join(cache_dir, ".build.lock")
    held_lock = _acquire_build_lock(lock_path, complete_path)
    if held_lock is None:
        manifest = _load_pretrain_batch_cache_manifest(cache_dir)
        manifest["cache_dir"] = cache_dir
        return manifest
    if os.path.exists(complete_path):
        _release_build_lock(held_lock)
        manifest = _load_pretrain_batch_cache_manifest(cache_dir)
        manifest["cache_dir"] = cache_dir
        return manifest

    build_start = time.time()
    targets_tmp_path = join(cache_dir, f"targets.npy.tmp.{os.getpid()}")
    targets_path = join(cache_dir, "targets.npy")
    targets_mmap = None
    try:
        pretrain_peptides = _read_pretrain_peptide_list(filename)
        pretrain_cache = EncodingCache(
            cache_dir=encoding_cache_dir,
            params=encoding_params,
        )
        cache_entry_path = pretrain_cache.ensure_built(pretrain_peptides)

        total_target_rows = len(pretrain_peptides) * len(usable_alleles)
        targets_mmap = numpy.lib.format.open_memmap(
            targets_tmp_path,
            mode="w+",
            dtype=numpy.float32,
            shape=(total_target_rows,),
        )

        chunk_entries = []
        synthetic_iter = pandas.read_csv(
            filename,
            index_col=0,
            chunksize=peptides_per_chunk,
        )
        row_offset = 0
        target_offset = 0
        for chunk_num, df in enumerate(synthetic_iter):
            df.columns = empty.columns
            df = df[usable_alleles]
            chunk_len = len(df)
            chunk_start = row_offset
            row_offset += chunk_len
            if chunk_len == 0:
                continue
            targets = from_ic50(df.stack().values).astype(
                numpy.float32, copy=False
            )
            target_start = target_offset
            target_end = target_start + len(targets)
            targets_mmap[target_start:target_end] = targets
            target_offset = target_end
            chunk_entries.append(
                {
                    "chunk_num": chunk_num,
                    "chunk_len": chunk_len,
                    "num_rows": int(len(targets)),
                    "peptide_start": int(chunk_start),
                    "peptide_end": int(row_offset),
                    "target_start": int(target_start),
                    "target_end": int(target_end),
                }
            )
        if row_offset != len(pretrain_peptides):
            raise AssertionError(
                "Pretrain batch cache indexed %d rows but peptide list has %d rows"
                % (row_offset, len(pretrain_peptides))
            )
        if target_offset != total_target_rows:
            raise AssertionError(
                "Pretrain batch cache wrote %d target rows but expected %d"
                % (target_offset, total_target_rows)
            )
        targets_mmap.flush()
        del targets_mmap
        targets_mmap = None
        os.replace(targets_tmp_path, targets_path)

        manifest = {
            "version": 2,
            "source_path": os.path.abspath(filename),
            "source_size": os.path.getsize(filename),
            "source_mtime_ns": os.stat(filename).st_mtime_ns,
            "peptides_per_chunk": peptides_per_chunk,
            "usable_alleles": usable_alleles,
            "skipped_alleles": skipped_alleles,
            "chunks": chunk_entries,
            "targets_path": "targets.npy",
            "encoding_cache_entry_relpath": os.path.relpath(
                cache_entry_path,
                encoding_cache_dir,
            ),
        }
        manifest_path = join(cache_dir, "manifest.json")
        tmp_manifest_path = f"{manifest_path}.tmp.{os.getpid()}"
        with open(tmp_manifest_path, "w") as fd:
            json.dump(manifest, fd, indent=2, sort_keys=True)
        os.replace(tmp_manifest_path, manifest_path)
        with open(complete_path, "w"):
            pass
        if verbose:
            print(
                f"Pretrain batch cache built in {time.time() - build_start:.1f}s "
                f"({len(chunk_entries)} chunks) at {cache_dir}."
            )
        manifest["cache_dir"] = cache_dir
        return manifest
    finally:
        if targets_mmap is not None:
            del targets_mmap
        try:
            os.unlink(targets_tmp_path)
        except FileNotFoundError:
            pass
        _release_build_lock(held_lock)


def pretrain_network_input_iterator(
        filename,
        master_allele_encoding,
        peptide_encoding,
        peptides_per_chunk=1024,
        encoding_cache_dir=None,
        encoding_params=None,
        worker_id=0,
        num_workers=1,
        compact_peptide_repeats=False):
    """Yield pretrain batches as network-input ``(x_dict, y)`` tuples."""
    if encoding_cache_dir is not None and encoding_params is not None:
        manifest = _get_or_build_pretrain_batch_cache(
            filename=filename,
            master_allele_encoding=master_allele_encoding,
            peptides_per_chunk=peptides_per_chunk,
            encoding_cache_dir=encoding_cache_dir,
            encoding_params=encoding_params,
            verbose=(worker_id == 0),
        )
        encoding_entry = join(
            encoding_cache_dir,
            manifest["encoding_cache_entry_relpath"],
        )
        pretrain_encoded_mmap = numpy.load(
            join(encoding_entry, "encoded.npy"),
            mmap_mode="r",
        )
        allele_encoding_cache = {}
        allele_indices_cache = {}
        usable_alleles = manifest["usable_alleles"]
        targets_mmap = None
        if manifest.get("version") == 2:
            targets_mmap = numpy.load(
                join(manifest["cache_dir"], manifest["targets_path"]),
                mmap_mode="r",
            )
        for chunk_num, chunk in enumerate(manifest["chunks"]):
            if num_workers > 1 and (chunk_num % num_workers) != worker_id:
                continue
            chunk_len = chunk["chunk_len"]
            if chunk_len not in allele_encoding_cache:
                allele_encoding_cache[chunk_len] = AlleleEncoding(
                    numpy.tile(usable_alleles, chunk_len),
                    borrow_from=master_allele_encoding,
                )
                allele_indices_cache[chunk_len] = (
                    allele_encoding_cache[chunk_len].indices.values.copy()
                )
            if targets_mmap is None:
                # Backward compatibility for version-1 pretrain batch caches:
                # one compressed npz per chunk containing peptide_indices +
                # targets. New caches use one targets.npy memmap and contiguous
                # peptide row offsets to avoid thousands of small file opens in
                # the hot loop.
                chunk_path = join(manifest["cache_dir"], chunk["path"])
                with numpy.load(chunk_path, allow_pickle=False) as payload:
                    peptide_rows = numpy.array(
                        pretrain_encoded_mmap[payload["peptide_indices"]],
                        copy=True,
                    )
                    targets = payload["targets"]
            else:
                peptide_rows = numpy.array(
                    pretrain_encoded_mmap[
                        chunk["peptide_start"]:chunk["peptide_end"]
                    ],
                    copy=True,
                )
                targets = numpy.array(
                    targets_mmap[chunk["target_start"]:chunk["target_end"]],
                    copy=True,
                )
            if compact_peptide_repeats:
                x_peptide = peptide_rows
            else:
                x_peptide = numpy.repeat(
                    peptide_rows,
                    len(usable_alleles),
                    axis=0,
                )
            x_dict = {
                "peptide": x_peptide,
                "allele": allele_indices_cache[chunk_len],
            }
            if compact_peptide_repeats:
                # The training loop expands on-device. Avoid materializing a
                # fresh chunk_len × num_alleles peptide tensor on the CPU every
                # pretrain step.
                x_dict["peptide_repeat_count"] = len(usable_alleles)
            yield (x_dict, targets)
        return

    for allele_encoding, peptides, affinities in pretrain_data_iterator(
            filename=filename,
            master_allele_encoding=master_allele_encoding,
            peptides_per_chunk=peptides_per_chunk,
            encoding_cache_dir=encoding_cache_dir,
            encoding_params=encoding_params,
            shard_rank=worker_id,
            num_shards=num_workers):
        x_dict = {
            "peptide": peptides.variable_length_to_fixed_length_vector_encoding(
                **peptide_encoding
            ),
        }
        if allele_encoding is not None:
            x_dict["allele"] = allele_encoding.indices.values
        yield (x_dict, from_ic50(affinities))


def run(argv=sys.argv[1:]):
    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    if args.debug:
        try:
            return main(args)
        except Exception as e:
            print(e)
            import ipdb  # pylint: disable=import-error
            ipdb.set_trace()
            raise
    else:
        return main(args)


def main(args):
    print("Arguments:")
    print(args)

    args.out_models_dir = os.path.abspath(args.out_models_dir)
    configure_logging(verbose=args.verbosity > 1)

    if not args.continue_incomplete:
        initialize_training(args)

    if not args.only_initialize:
        train_models(args)


def initialize_training(args):
    required_arguments = [
        "data",
        "out_models_dir",
        "hyperparameters",
        "num_folds",
    ]
    for arg in required_arguments:
        if getattr(args, arg) is None:
            parser.error("Missing required arg: %s" % arg)

    # TIMING_MARKER lines let the sweep harness compute clean per-phase
    # wall times:
    #   start               — argparse + validation done; before any I/O
    #   data_loaded         — train_data CSV parsed + filtered + folded
    #   setup_done          — pool + encoding caches + all GLOBAL_DATA
    #   training_done       — all work items finished
    # Extract deltas by grepping stdout for ``TIMING_MARKER``.
    print(f"TIMING_MARKER start {time.time():.3f}")
    print("Initializing training.")
    hyperparameters_lst = yaml.safe_load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list)
    print("Loaded hyperparameters list:")
    pprint.pprint(hyperparameters_lst)

    allele_sequences = pandas.read_csv(
        args.allele_sequences, index_col=0).iloc[:,0]

    # pyarrow parser is 3–10× faster than the default C engine for
    # typical CSV. Still bzip2-bound on the I/O side but the parse phase
    # becomes negligible. Falls back to C engine if pyarrow isn't
    # installed (it's a pandas optional dep).
    try:
        df = pandas.read_csv(args.data, engine="pyarrow")
    except (ImportError, ValueError):
        df = pandas.read_csv(args.data)
    print("Loaded training data: %s" % (str(df.shape)))

    # Collapse the filter chain into a single boolean mask. The original
    # code chained 3 ``.loc[]`` calls which each materialize a full
    # DataFrame copy. Single-mask form scans each column once and allocates
    # one result buffer.
    peptide_len = df.peptide.str.len()
    mask = (
        (peptide_len >= 8)
        & (peptide_len <= 15)
        & df.measurement_value.notnull()
        & df.allele.isin(allele_sequences.index)
    )
    df = df.loc[mask]
    print("Filtered to valid (len, non-null, known-allele): %s" % (str(df.shape)))

    print("Data inequalities:")
    print(df.measurement_inequality.value_counts())

    if args.ignore_inequalities and "measurement_inequality" in df.columns:
        print("Dropping measurement_inequality column")
        del df["measurement_inequality"]
    # Allele names in data are assumed to be already normalized.
    print("Training data: %s" % (str(df.shape)))

    (held_out_fraction, held_out_max) = (
        args.held_out_measurements_per_allele_fraction_and_max)

    folds_df = assign_folds(
        df=df,
        num_folds=args.num_folds,
        held_out_fraction=held_out_fraction,
        held_out_max=held_out_max)
    print(f"TIMING_MARKER data_loaded {time.time():.3f}")

    allele_sequences_in_use = allele_sequences[
        allele_sequences.index.isin(df.allele)
    ]
    print("Will use %d / %d allele sequences" % (
        len(allele_sequences_in_use), len(allele_sequences)))

    # All alleles, not just those with training data.
    full_allele_encoding = AlleleEncoding(
        alleles=allele_sequences.index.values,
        allele_to_sequence=allele_sequences.to_dict()
    )

    # Only alleles with training data. For efficiency we perform model training
    # using only these alleles in the neural network embedding layer.
    allele_encoding = AlleleEncoding(
        alleles=allele_sequences_in_use.index.values,
        allele_to_sequence=allele_sequences_in_use.to_dict())

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    predictor = Class1AffinityPredictor(
        allele_to_sequence=allele_encoding.allele_to_sequence,
        metadata_dataframes={
            'train_data': pandas.merge(
                df,
                folds_df,
                left_index=True,
                right_index=True)
        })

    work_items = []
    for (h, hyperparameters) in enumerate(hyperparameters_lst):
        if 'n_models' in hyperparameters:
            raise ValueError("n_models is unsupported")

        if args.max_epochs:
            hyperparameters['max_epochs'] = args.max_epochs

        if hyperparameters.get("train_data", {}).get("pretrain", False):
            if not args.pretrain_data:
                raise ValueError("--pretrain-data is required")

        for fold in range(args.num_folds):
            for replicate in range(args.num_replicates):
                work_dict = {
                    'work_item_name': str(uuid.uuid4()),
                    'architecture_num': h,
                    'num_architectures': len(hyperparameters_lst),
                    'fold_num': fold,
                    'num_folds': args.num_folds,
                    'replicate_num': replicate,
                    'num_replicates': args.num_replicates,
                    'hyperparameters': hyperparameters,
                    'pretrain_data_filename': args.pretrain_data,
                }
                work_items.append(work_dict)

    training_init_info = {}
    training_init_info["train_data"] = df
    training_init_info["folds_df"] = folds_df
    training_init_info["allele_encoding"] = allele_encoding
    training_init_info["full_allele_encoding"] = full_allele_encoding
    training_init_info["work_items"] = work_items

    # Save empty predictor (for metadata)
    predictor.save(args.out_models_dir)

    # Write training_init_info.
    with open(join(args.out_models_dir, "training_init_info.pkl"), "wb") as fd:
        pickle.dump(training_init_info, fd, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done initializing training.")
    print(f"TIMING_MARKER setup_done {time.time():.3f}")


def _initialize_shared_random_negative_pools(args, all_work_items):
    """Pre-build per-fold mmap random-negative pools (Layer 1 SHM).

    Wraps ``shared_memory.setup_shared_random_negative_pools`` with the
    orchestrator-side glue: instantiates a sentinel
    ``Class1NeuralNetwork`` from the first work item's hyperparameters
    (so we can use its ``peptides_to_network_input`` as the encoder),
    derives a deterministic per-run seed, and parks the resulting
    ``(fold, cfg_key) -> pool_dir`` mapping in
    ``GLOBAL_DATA["random_negative_shared_pool_dirs"]``. Workers look
    their own pool up in ``train_model`` via
    ``shared_memory.lookup_pool_dir_for_work_item``.

    Validates that all work items share the same
    ``random_negative_pool_epochs`` and the same ``peptide_encoding``
    config — required for one encoder + one pool-epoch count to apply
    across the whole sweep.
    """
    from .class1_neural_network import Class1NeuralNetwork
    from .shared_memory import setup_shared_random_negative_pools

    # Validate uniform pool_epochs across the whole sweep. Mixed values
    # would need separate pools per (fold, pool_epochs); not supported
    # in this orchestrator hook (no caller has needed it).
    pool_epochs_seen = set()
    encoding_seen = {}
    for item in all_work_items:
        hp = item["hyperparameters"]
        pe = int(hp.get("random_negative_pool_epochs", 1) or 1)
        pool_epochs_seen.add(pe)
        cfg = tuple(sorted(hp.get("peptide_encoding", {}).items()))
        encoding_seen[cfg] = hp.get("peptide_encoding", {})
    if len(pool_epochs_seen) != 1:
        raise ValueError(
            "shared random-negative pool requires uniform "
            "random_negative_pool_epochs across all work items; saw %r"
            % (sorted(pool_epochs_seen),)
        )
    pool_epochs = pool_epochs_seen.pop()
    if pool_epochs <= 1:
        print(
            "shared random-negative pool: skipping orchestrator build "
            "(pool_epochs=%d means each epoch regenerates anyway). Set "
            "random_negative_pool_epochs > 1 in hyperparameters to use "
            "the shared mmap path." % pool_epochs
        )
        GLOBAL_DATA["random_negative_shared_pool_dirs"] = {}
        return
    if len(encoding_seen) != 1:
        raise ValueError(
            "shared random-negative pool requires uniform "
            "peptide_encoding across all work items; saw %d distinct "
            "configs" % len(encoding_seen)
        )

    # Sentinel network for its peptides_to_network_input. Instantiated
    # only for the encoder closure; never trained.
    sentinel_hp = all_work_items[0]["hyperparameters"]
    sentinel = Class1NeuralNetwork(**{
        k: v for k, v in sentinel_hp.items()
        if k in Class1NeuralNetwork.hyperparameter_defaults.defaults
    })
    encoder = sentinel.peptides_to_network_input

    # Per-run seed: deterministic across resumes of the same out_dir.
    # Combine a short hash of args.out_models_dir with a fixed salt.
    import hashlib
    seed = int(
        hashlib.sha256(
            ("mhcflurry-shared-pool::" + args.out_models_dir).encode("utf-8")
        ).hexdigest()[:8],
        16,
    )

    output_root = os.path.abspath(args.random_negative_shared_pool_dir)
    os.makedirs(output_root, exist_ok=True)
    print(
        "shared random-negative pool: building under %s "
        "(pool_epochs=%d, num_work_items=%d, seed=%d)" % (
            output_root, pool_epochs, len(all_work_items), seed,
        )
    )
    t0 = time.time()
    fold_pool_dirs = setup_shared_random_negative_pools(
        output_root_dir=output_root,
        work_items=all_work_items,
        train_data_df=GLOBAL_DATA["train_data"],
        folds_df=GLOBAL_DATA["folds_df"],
        peptide_encoder=encoder,
        pool_epochs=pool_epochs,
        seed=seed,
    )
    print(
        "shared random-negative pool: built %d (fold, cfg) pool(s) in "
        "%.1f sec" % (len(fold_pool_dirs), time.time() - t0)
    )
    GLOBAL_DATA["random_negative_shared_pool_dirs"] = fold_pool_dirs


def _initialize_encoding_cache(args, all_work_items):
    """Pre-build BLOSUM62 encoding caches used by training workers.

    Run once in the orchestrator. Distinct ``peptide_encoding`` configs
    across the hyperparameters sweep each get their own cache keyed by
    params hash. Workers subsequently call ``EncodingCache.get_or_build``
    with the same params + peptides and hit the already-built cache.

    Stores the cache_dir plus the orchestrator-computed unique-peptide
    list in ``GLOBAL_DATA``. The peptide list is small (a ~20 MB Python
    list of strings for a 1M-peptide training set) compared to the
    encoded tensor (~1+ GB), and stashing it means workers skip the
    `drop_duplicates` + `sha256` pass that would otherwise run 140+ times
    across a full sweep.

    Safe to call repeatedly; cache hits short-circuit.
    """
    cache_dir = args.encoding_cache_dir or join(
        args.out_models_dir, "encoding_cache"
    )
    # Group hyperparameters by their peptide_encoding config so we build one
    # cache per unique config. Most sweeps share the same config, in which
    # case this loop runs once.
    configs_seen = {}
    pretrain_batch_configs = {}
    for work_item in all_work_items:
        hyperparameters = work_item["hyperparameters"]
        cfg = hyperparameters.get("peptide_encoding", {})
        # dict isn't hashable; use a sorted-kv string key. The config dict
        # itself is small and we just need de-duplication.
        key = tuple(sorted(cfg.items()))
        configs_seen[key] = cfg
        train_params = dict(hyperparameters.get("train_data", {}))
        if train_params.get("pretrain", False):
            pretrain_chunk_size = _pop_train_param(
                train_params,
                names=(
                    "pretrain_peptides_per_step",
                    "pretrain_peptides_per_epoch",
                ),
                default=1024,
                verbose=0,
            )
            pretrain_batch_configs[(key, pretrain_chunk_size)] = {
                "peptide_encoding": cfg,
                "peptides_per_chunk": pretrain_chunk_size,
            }

    df = GLOBAL_DATA["train_data"]
    unique_peptides = _deterministic_unique_peptide_list(df.peptide.values)
    unique_peptide_to_idx = {
        peptide: i for i, peptide in enumerate(unique_peptides)
    }
    GLOBAL_DATA["train_peptide_encoding_cache_indices"] = pandas.Series(
        numpy.fromiter(
            (unique_peptide_to_idx[p] for p in df.peptide.values),
            dtype=numpy.int64,
            count=len(df),
        ),
        index=df.index,
    )
    del unique_peptide_to_idx

    for cfg in configs_seen.values():
        params = EncodingParams(**cfg)
        cache = EncodingCache(cache_dir=cache_dir, params=params)
        if cache.is_complete_for(unique_peptides):
            print(f"Encoding cache hit: {cache.entry_path(unique_peptides)} "
                  f"({len(unique_peptides)} peptides)")
            continue
        print(f"Building encoding cache for params "
              f"{params.to_kwargs()} ({len(unique_peptides)} peptides) "
              f"at {cache.entry_path(unique_peptides)}...")
        t0 = time.time()
        cache.ensure_built(unique_peptides)
        print(f"Encoding cache built in {time.time() - t0:.1f}s.")

    GLOBAL_DATA["encoding_cache_dir"] = str(cache_dir)
    # Per-arch params lookup: hyperparameters dict doesn't change pickle size
    # meaningfully, so just stash it. Workers use it to pick the right cache.
    GLOBAL_DATA["encoding_cache_configs"] = list(configs_seen.values())
    # Stash the orchestrator-built peptide list so workers don't re-run
    # drop_duplicates + sha256 over ~1M peptides on every fold.
    # The list pickles at roughly (8 bytes + avg peptide length) per entry
    # (~20 MB for 1M 12-mers) — fine for a single-box Pool. Revisit if this
    # ever ships to a distributed scheduler.
    GLOBAL_DATA["encoding_cache_unique_peptides"] = unique_peptides

    # Pre-build the pretrain cache too if a pretrain file is configured.
    # Without this, each worker racing to build the pretrain cache on
    # first use pays a redundant encoding pass — on an A100 subset run
    # that's ~15 min aggregate across 32 work items (the race fix in
    # EncodingCache._build makes this safe but not cheap). Pre-building
    # here amortizes the encoding to a single orchestrator-side pass. We
    # also pre-build the chunk manifest used by pretrain_network_input_iterator
    # so workers no longer reparse / stack the wide CSV every epoch.
    pretrain_data_path = getattr(args, "pretrain_data", None)
    if pretrain_data_path:
        pretrain_peptides = _read_pretrain_peptide_list(pretrain_data_path)
        for cfg in configs_seen.values():
            params = EncodingParams(**cfg)
            cache = EncodingCache(cache_dir=cache_dir, params=params)
            if cache.is_complete_for(pretrain_peptides):
                print(f"Pretrain encoding cache hit: "
                      f"{cache.entry_path(pretrain_peptides)} "
                      f"({len(pretrain_peptides)} peptides)")
                continue
            print(f"Building pretrain encoding cache for params "
                  f"{params.to_kwargs()} ({len(pretrain_peptides)} peptides) "
                  f"at {cache.entry_path(pretrain_peptides)}...")
            t0 = time.time()
            cache.ensure_built(pretrain_peptides)
            print(f"Pretrain encoding cache built in {time.time() - t0:.1f}s.")
        # Use the same AlleleEncoding the workers will use
        # (``allele_encoding``, restricted to alleles with training data),
        # NOT ``full_allele_encoding``. ``_get_pretrain_allele_info``
        # computes ``usable_alleles`` as the intersection of pretrain CSV
        # columns with ``master_allele_encoding.allele_to_sequence`` —
        # so the driver and each worker's DataLoader must pass the same
        # ``master_allele_encoding`` or they hash to different cache
        # dirs and the workers race to rebuild. Observed on the
        # 2026-04-23 8×A100 run: driver built a 97-allele cache that
        # every worker then discarded to build a parallel 96-allele
        # cache, thundering-herd writing ~30 GB across the 16-way race.
        master_allele_encoding = GLOBAL_DATA.get("allele_encoding")
        if pretrain_batch_configs and master_allele_encoding is None:
            raise KeyError(
                "allele_encoding is required to prebuild the pretrain "
                "batch cache"
            )
        for batch_cfg in pretrain_batch_configs.values():
            params = EncodingParams(**batch_cfg["peptide_encoding"])
            _get_or_build_pretrain_batch_cache(
                filename=pretrain_data_path,
                master_allele_encoding=master_allele_encoding,
                peptides_per_chunk=batch_cfg["peptides_per_chunk"],
                encoding_cache_dir=str(cache_dir),
                encoding_params=params,
                verbose=True,
            )


def _deterministic_unique_peptide_list(peptide_values):
    """Return unique peptides in first-seen order.

    Must be stable: orchestrator's list must match what workers compute, or
    the cache key (which hashes the list) won't match. pandas.Series
    .drop_duplicates() preserves first-seen order — use it consistently.
    """
    return list(pandas.Series(peptide_values).drop_duplicates())


def _read_pretrain_peptide_list(filename):
    """Read only the peptide column (index) from the pretrain CSV.

    The cache build pass needs the full peptide list up front. Reading
    just the index column avoids parsing the much-wider affinity matrix.
    For a 1M-row file this takes a few seconds vs minutes for a full read.
    """
    # Reading only usecols=[0] makes pandas parse just the peptide column.
    peptide_series = pandas.read_csv(filename, index_col=0, usecols=[0]).index
    return list(peptide_series)


def _local_pool_workers_inherit_global_data(worker_pool=None):
    """Return True when local Pool workers inherit GLOBAL_DATA by fork."""
    context = getattr(worker_pool, "_ctx", None)
    if context is not None:
        return context.get_start_method() == "fork"
    try:
        method = multiprocessing.get_start_method(allow_none=True)
        if method is None:
            method = multiprocessing.get_context().get_start_method()
        return method == "fork"
    except RuntimeError:
        return False


def train_models(args):
    global GLOBAL_DATA

    print("Beginning training.")
    predictor = Class1AffinityPredictor.load(
        args.out_models_dir, optimization_level=0)
    print("Loaded predictor with %d networks" % len(predictor.neural_networks))

    with open(join(args.out_models_dir, "training_init_info.pkl"), "rb") as fd:
        GLOBAL_DATA.update(pickle.load(fd))
    print("Loaded training init info.")

    all_work_items = GLOBAL_DATA["work_items"]

    # Optionally pre-build the shared BLOSUM62 encoding cache. Orchestrator
    # does the (single-threaded) encoding pass once; workers mmap the result.
    # See mhcflurry/encoding_cache.py and issue openvax/mhcflurry#268.
    if getattr(args, "use_encoding_cache", False):
        _initialize_encoding_cache(args, all_work_items)

    # Layer-1 SHM (per-run, mmap, read-only): pre-build per-fold random-
    # negative pools and stash the lookup table in GLOBAL_DATA. Each
    # work item's fit() will look up its own pool dir at task start.
    # See mhcflurry/shared_memory.py for the layered SHM design.
    if getattr(args, "random_negative_shared_pool_dir", None):
        _initialize_shared_random_negative_pools(args, all_work_items)
    complete_work_item_names = [
        network.fit_info[-1]["training_info"]["work_item_name"] for network in
        predictor.neural_networks
    ]
    work_items = [
        item for item in all_work_items
        if item["work_item_name"] not in complete_work_item_names
    ]
    print("Found %d work items, of which %d are incomplete and will run now." % (
        len(all_work_items), len(work_items)))

    serial_run = not args.cluster_parallelism and args.num_jobs == 0

    # The estimated time to completion is more accurate if we randomize
    # the order of the work.
    random.shuffle(work_items)
    for (work_item_num, item) in enumerate(work_items):
        item['work_item_num'] = work_item_num
        item['num_work_items'] = len(work_items)
        item['progress_print_interval'] = 60.0 if not serial_run else 5.0
        item['predictor'] = predictor if serial_run else None
        item['save_to'] = args.out_models_dir if serial_run else None
        item['verbose'] = args.verbosity
        if args.pretrain_data:
            item['pretrain_data_filename'] = args.pretrain_data

    start = time.time()

    worker_pool = None
    if serial_run:
        # Run in serial. Every worker is passed the same predictor,
        # which it adds models to, so no merging is required. It also saves
        # as it goes so no saving is required at the end.
        print("Processing %d work items in serial." % len(work_items))
        for _ in tqdm.trange(len(work_items)):
            item = work_items.pop(0)  # want to keep freeing up memory
            work_predictor = train_model(**item)
            assert work_predictor is predictor
        assert not work_items
        results_generator = None
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        results_generator = cluster_results_from_args(
            args,
            work_function=train_model,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="save_predictor")
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None

        print("Processing %d work items in parallel." % len(work_items))
        assert not serial_run

        if _local_pool_workers_inherit_global_data(worker_pool):
            print("Local Pool uses fork; workers inherit GLOBAL_DATA without "
                  "per-task pickle payloads.")
        else:
            print("Local Pool does not use fork; attaching GLOBAL_DATA to "
                  "each work item for worker delivery.")
            for item in work_items:
                item['constant_data'] = GLOBAL_DATA

        results_generator = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, train_model),
            work_items,
            chunksize=1)

    if results_generator:
        for new_predictor in tqdm.tqdm(results_generator, total=len(work_items)):
            save_start = time.time()
            (new_model_name,) = predictor.merge_in_place([new_predictor])
            predictor.save(
                args.out_models_dir,
                model_names_to_write=[new_model_name],
                write_metadata=False)
            print(
                "Saved predictor (%d models total) with 1 new models"
                "in %0.2f sec to %s" % (
                    len(predictor.neural_networks),
                    time.time() - save_start,
                    args.out_models_dir))

    # We want the final predictor to support all alleles with sequences, not
    # just those we actually used for model training.
    predictor.allele_to_sequence = (
        GLOBAL_DATA['full_allele_encoding'].allele_to_sequence)
    predictor.clear_cache()
    predictor.save(args.out_models_dir)
    print("Done.")

    print("*" * 30)
    training_time = time.time() - start
    print("Trained affinity predictor with %d networks in %0.2f min." % (
        len(predictor.neural_networks), training_time / 60.0))
    print("*" * 30)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Predictor written to: %s" % args.out_models_dir)
    print(f"TIMING_MARKER training_done {time.time():.3f}")


def _build_train_peptides(
        peptide_values, hyperparameters, constant_data, peptide_index=None):
    """Construct an EncodableSequences for the fold's training peptides.

    When the encoding cache is enabled (orchestrator set
    ``encoding_cache_dir`` in constant_data), we look up each peptide in
    the pre-built memmap and return an EncodableSequences with its
    ``encoding_cache`` prepopulated — so the subsequent
    ``peptides_to_network_input`` call inside ``fit()`` is a memmap slice
    instead of a fresh BLOSUM62 pass.

    When disabled (default), falls through to the old behavior — a plain
    ``EncodableSequences(peptides)`` whose first
    ``variable_length_to_fixed_length_vector_encoding`` call does the
    encoding work inline. Bit-identical to pre-change behavior.
    """
    cache_dir = constant_data.get("encoding_cache_dir")
    if cache_dir is None:
        return EncodableSequences(peptide_values)

    cfg = hyperparameters.get("peptide_encoding", {})
    params = EncodingParams(**cfg)
    cache = EncodingCache(cache_dir=cache_dir, params=params)

    # Prefer the orchestrator-stashed unique-peptide list so every worker
    # skips the drop_duplicates + sha256 pass over the ~1M-peptide training
    # set. Fall back to recomputation if the key is absent (older pickle,
    # manually-constructed constant_data in tests).
    unique_peptides = constant_data.get("encoding_cache_unique_peptides")
    if unique_peptides is None:
        df = constant_data["train_data"]
        unique_peptides = _deterministic_unique_peptide_list(df.peptide.values)
    cache_entry_path = cache.ensure_built(unique_peptides)
    encoded_all = numpy.load(
        join(str(cache_entry_path), "encoded.npy"),
        mmap_mode="r",
    )

    row_index_map = constant_data.get("train_peptide_encoding_cache_indices")
    if row_index_map is not None and peptide_index is not None:
        fold_indices = row_index_map.loc[peptide_index].to_numpy(
            dtype=numpy.int64,
            copy=False,
        )
    else:
        # Compatibility fallback for old constant_data dicts and direct tests.
        # This constructs the large peptide->row dict, so production local
        # workers pass peptide_index and use the orchestrator-built row map.
        _encoded_all, peptide_to_idx = cache.get_or_build(unique_peptides)
        encoded_all = _encoded_all
        fold_indices = numpy.fromiter(
            (peptide_to_idx[p] for p in peptide_values),
            dtype=numpy.int64,
            count=len(peptide_values),
        )

    # Lookup each fold-peptide's row in the memmap. Fancy indexing produces
    # a contiguous in-memory copy sized (len(fold), enc_len, alphabet); that's
    # the same materialized tensor the old path would have produced on first
    # encode call.
    fold_encoded = encoded_all[fold_indices]
    return make_prepopulated_encodable_sequences(peptide_values, fold_encoded, params)


def train_model(
        work_item_name,
        work_item_num,
        num_work_items,
        architecture_num,
        num_architectures,
        fold_num,
        num_folds,
        replicate_num,
        num_replicates,
        hyperparameters,
        pretrain_data_filename,
        verbose,
        progress_print_interval,
        predictor,
        save_to,
        constant_data=GLOBAL_DATA):

    df = constant_data["train_data"]
    folds_df = constant_data["folds_df"]
    allele_encoding = constant_data["allele_encoding"]

    if predictor is None:
        predictor = Class1AffinityPredictor(
            allele_to_sequence=allele_encoding.allele_to_sequence)

    numpy.testing.assert_equal(len(df), len(folds_df))

    train_data = df.loc[
        folds_df["fold_%d" % fold_num]
    ].sample(frac=1.0)

    train_peptides = _build_train_peptides(
        train_data.peptide.values,
        hyperparameters,
        constant_data,
        peptide_index=train_data.index,
    )
    train_alleles = AlleleEncoding(
        train_data.allele.values, borrow_from=allele_encoding)

    progress_preamble = (
        "[task %2d / %2d]: "
        "[%2d / %2d folds] "
        "[%2d / %2d architectures] "
        "[%4d / %4d replicates] " % (
            work_item_num + 1,
            num_work_items,
            fold_num + 1,
            num_folds,
            architecture_num + 1,
            num_architectures,
            replicate_num + 1,
            num_replicates))

    print("%s [pid %d]. Hyperparameters:" % (progress_preamble, os.getpid()))
    pprint.pprint(hyperparameters)

    # GPU memory telemetry at task boundaries. With max-tasks-per-worker>1
    # (the post-Phase-4 default of 1000), a single worker process handles
    # many arch×fold tasks sequentially. Logging allocated/reserved VRAM
    # plus the max-allocated high-water-mark at task start and end gives
    # us a trajectory to detect leaks: if the high-water mark or the
    # start-of-task "residual" allocation creeps upward across tasks
    # within a worker, there's a leak somewhere (PyTorch allocator fails
    # to release, cached tensors pinned, etc). Flat across 140 tasks is
    # the "no leak" signal.
    def _log_gpu_memory(marker):
        import torch
        if not torch.cuda.is_available():
            return
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"GPU_MEMORY_TELEMETRY pid={os.getpid()} "
            f"task={work_item_num + 1}/{num_work_items} marker={marker} "
            f"allocated_gb={alloc_gb:.3f} reserved_gb={reserved_gb:.3f} "
            f"max_allocated_gb={max_alloc_gb:.3f}"
        )

    _log_gpu_memory("START")
    _log_process_telemetry("START")

    train_params = dict(hyperparameters.get("train_data", {}))

    def get_train_param(param, default):
        return _pop_train_param(
            train_params,
            names=(param,),
            default=default,
            verbose=verbose,
        )


    def progress_callback():
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 10**9
            print("Current used GPU memory: ", mem, "gb")

    if get_train_param("pretrain", False):
        pretrain_patience = get_train_param("pretrain_patience", 10)
        pretrain_min_delta = get_train_param("pretrain_min_delta", 0.0)
        pretrain_steps_per_epoch = get_train_param(
            "pretrain_steps_per_epoch", 10)
        pretrain_max_epochs = get_train_param("pretrain_max_epochs", 1000)
        pretrain_min_epochs = get_train_param("pretrain_min_epochs", 0)
        pretrain_peptides_per_step = _pop_train_param(
            train_params,
            names=(
                "pretrain_peptides_per_step",
                "pretrain_peptides_per_epoch",
            ),
            default=1024,
            verbose=verbose,
        )
        max_val_loss = get_train_param("pretrain_max_val_loss", None)

        if verbose:
            print("Unused train params", train_params)

        attempt = 0
        while True:
            attempt += 1
            print("Pre-training attempt %d" % attempt)
            if attempt > 10:
                print("Too many pre-training attempts! Stopping pretraining.")
                break

            model = Class1NeuralNetwork(**hyperparameters)
            assert model.network() is None
            encoding_params = (
                EncodingParams(**hyperparameters["peptide_encoding"])
                if constant_data.get("encoding_cache_dir")
                else None
            )
            make_pretrain_generator = partial(
                pretrain_network_input_iterator,
                filename=pretrain_data_filename,
                master_allele_encoding=allele_encoding,
                peptide_encoding=hyperparameters["peptide_encoding"],
                peptides_per_chunk=pretrain_peptides_per_step,
                encoding_cache_dir=constant_data.get("encoding_cache_dir"),
                encoding_params=encoding_params,
                compact_peptide_repeats=True,
            )

            model.fit_generator(
                generator=make_pretrain_generator(),
                generator_factory=make_pretrain_generator,
                generator_batches_are_encoded=True,
                validation_peptide_encoding=train_peptides,
                validation_affinities=train_data.measurement_value.values,
                validation_allele_encoding=train_alleles,
                validation_inequalities=train_data.measurement_inequality.values,
                patience=pretrain_patience,
                min_delta=pretrain_min_delta,
                steps_per_epoch=pretrain_steps_per_epoch,
                epochs=pretrain_max_epochs,
                min_epochs=pretrain_min_epochs,
                verbose=verbose,
                progress_callback=progress_callback,
                progress_preamble=progress_preamble + "PRETRAIN",
                progress_print_interval=progress_print_interval,
            )
            model.fit_info[-1].setdefault(
                "training_info", {}).update({
                    "pretrain_attempt": attempt,
                    "phase": "pretrain",
                })
            if not max_val_loss:
                break
            final_val_loss = model.fit_info[-1]["val_loss"][-1]
            if final_val_loss >= max_val_loss:
                print("Val loss %f >= max val loss %f. Pre-training again." % (
                    final_val_loss, max_val_loss))
            else:
                print("Val loss %f < max val loss %f. Done pre-training." % (
                    final_val_loss, max_val_loss))
                break

        # Use a smaller learning rate for training on real data
        learning_rate = model.fit_info[-1]["learning_rate"]
        model.hyperparameters['learning_rate'] = learning_rate / 10
    else:
        model = Class1NeuralNetwork(**hyperparameters)

    # Derive a per-work-item random-negative seed ONLY when the pool
    # feature is actually in use. fit() has its own bypass for
    # pool_epochs=1 (Codex review on #270), but defense in depth: if
    # we don't pass a seed in the first place, a future refactor of
    # fit() that drops the bypass still can't silently flip training
    # to deterministic-per-work-item.
    #
    # hash() is randomized per-process via PYTHONHASHSEED, so use a
    # stable mix: SHA1 of the identity tuple truncated to 63 bits
    # (numpy.random.SeedSequence accepts arbitrary non-negative ints).
    pool_epochs = int(hyperparameters.get("random_negative_pool_epochs", 1) or 1)
    random_negative_seed = None
    if pool_epochs > 1:
        random_negative_seed = int(
            hashlib.sha1(
                ("|".join([
                    str(architecture_num),
                    str(fold_num),
                    str(replicate_num),
                    work_item_name or "",
                ])).encode()
            ).hexdigest()[:16],
            16,
        )

    # Layer-1 SHM lookup: if the orchestrator pre-built a per-fold
    # random-negative pool for this work item's (fold, random-negative
    # config), pass the dir through so fit() uses from_shared_mmap.
    # Otherwise the kwarg defaults to None and fit() falls back to the
    # in-process pool. See mhcflurry/shared_memory.py.
    fold_pool_dirs = constant_data.get("random_negative_shared_pool_dirs") or {}
    if fold_pool_dirs:
        from .shared_memory import lookup_pool_dir_for_work_item
        pool_dir = lookup_pool_dir_for_work_item(
            fold_pool_dirs,
            {"fold_num": fold_num, "hyperparameters": hyperparameters},
        )
    else:
        pool_dir = None
    # Cross-worker permutation diversity: distinct seed per work item so
    # workers reading the same pool see distinct orderings.
    pool_permutation_seed = (
        random_negative_seed if pool_dir is not None else None
    )

    model.fit(
        peptides=train_peptides,
        affinities=train_data.measurement_value.values,
        allele_encoding=train_alleles,
        inequalities=(
            train_data.measurement_inequality.values
            if "measurement_inequality" in train_data.columns else None),
        progress_preamble=progress_preamble,
        progress_callback=progress_callback,
        progress_print_interval=progress_print_interval,
        random_negative_seed=random_negative_seed,
        random_negative_shared_pool_dir=pool_dir,
        random_negative_permutation_seed=pool_permutation_seed,
        verbose=verbose)

    # Save model-specific training info
    train_peptide_hash = hashlib.sha1()
    for peptide in sorted(train_data.peptide.values):
        train_peptide_hash.update(peptide.encode())

    model.fit_info[-1].setdefault("training_info", {}).update({
        "phase": "finetune",
        "fold_num": fold_num,
        "num_folds": num_folds,
        "replicate_num": replicate_num,
        "num_replicates": num_replicates,
        "architecture_num": architecture_num,
        "num_architectures": num_architectures,
        "train_peptide_hash": train_peptide_hash.hexdigest(),
        "work_item_name": work_item_name,
    })

    numpy.testing.assert_equal(
        predictor.manifest_df.shape[0], len(predictor.class1_pan_allele_models))
    predictor.add_pan_allele_model(model, models_dir_for_save=save_to)
    numpy.testing.assert_equal(
        predictor.manifest_df.shape[0], len(predictor.class1_pan_allele_models))
    predictor.clear_cache()

    # Delete the network to release memory
    model.clear_allele_representations()
    model.update_network_description()  # save weights and config
    model._network = None  # release network to free memory

    # Release any cached allocator blocks we can, then log residual state.
    # empty_cache() doesn't free live tensors — just returns unused
    # cached blocks to the driver. If the END telemetry shows growing
    # "allocated_gb" across tasks despite this call, a reference is
    # being retained somewhere.
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_gpu_memory("END")
    _log_process_telemetry("END")

    return predictor


if __name__ == '__main__':
    run()

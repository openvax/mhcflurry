"""
Train Class1 pan-allele models.
"""
import argparse
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
    make_preencoded_encodable_sequences,
)

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

    for fold in range(num_folds):
        result_df["fold_%d" % fold] = True
        for (allele, sub_df) in df.groupby("allele"):
            medians = sub_df.groupby("peptide").measurement_value.median()

            low_peptides = medians[medians < medians.median()].index.values
            high_peptides = medians[medians >= medians.median()].index.values

            held_out_count = int(
                min(len(medians) * held_out_fraction, held_out_max))

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

            result_df.loc[
                sub_df.index[sub_df.peptide.isin(held_out_peptides)],
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
        encoding_params=None):
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
    empty = pandas.read_csv(filename, index_col=0, nrows=0)
    empty.columns = empty.columns.map(normalize_allele_name)
    print("Pretrain alleles available: ", *empty.columns.values)
    usable_alleles = [
        c for c in empty.columns
        if c in master_allele_encoding.allele_to_sequence
    ]
    print("Using %d / %d alleles" % (len(usable_alleles), len(empty.columns)))
    print("Skipped alleles: ", [
        c for c in empty.columns
        if c not in master_allele_encoding.allele_to_sequence
    ])

    allele_encoding = AlleleEncoding(
        numpy.tile(usable_alleles, peptides_per_chunk),
        borrow_from=master_allele_encoding)

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
            if len(df) != peptides_per_chunk:
                continue

            df.columns = empty.columns
            df = df[usable_alleles]
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
                encodable_peptides = make_preencoded_encodable_sequences(
                    repeated_peptides, chunk_encoded, encoding_params
                )

            yield (allele_encoding, encodable_peptides, df.stack().values)


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

    print("Initializing training.")
    hyperparameters_lst = yaml.safe_load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list)
    print("Loaded hyperparameters list:")
    pprint.pprint(hyperparameters_lst)

    allele_sequences = pandas.read_csv(
        args.allele_sequences, index_col=0).iloc[:,0]

    df = pandas.read_csv(args.data)
    print("Loaded training data: %s" % (str(df.shape)))
    df = df.loc[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    df = df.loc[~df.measurement_value.isnull()]
    print("Dropped NaNs: %s" % (str(df.shape)))

    df = df.loc[df.allele.isin(allele_sequences.index)]
    print("Subselected to alleles with sequences: %s" % (str(df.shape)))

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
    for work_item in all_work_items:
        cfg = work_item["hyperparameters"].get("peptide_encoding", {})
        # dict isn't hashable; use a sorted-kv string key. The config dict
        # itself is small and we just need de-duplication.
        key = tuple(sorted(cfg.items()))
        configs_seen[key] = cfg

    df = GLOBAL_DATA["train_data"]
    unique_peptides = _deterministic_unique_peptide_list(df.peptide.values)

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
        cache.get_or_build(unique_peptides)
        print(f"Encoding cache built in {time.time() - t0:.1f}s.")

    GLOBAL_DATA["encoding_cache_dir"] = str(cache_dir)
    # Per-arch params lookup: hyperparameters dict doesn't change pickle size
    # meaningfully, so just stash it. Workers use it to pick the right cache.
    GLOBAL_DATA["encoding_cache_configs"] = list(configs_seen.values())
    # Stash the orchestrator-built peptide list + its index map so workers
    # don't re-run drop_duplicates + sha256 over ~1M peptides on every fold.
    # The list pickles at roughly (8 bytes + avg peptide length) per entry
    # (~20 MB for 1M 12-mers) — fine for a single-box Pool. Revisit if this
    # ever ships to a distributed scheduler.
    GLOBAL_DATA["encoding_cache_unique_peptides"] = unique_peptides


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


def _build_train_peptides(peptide_values, hyperparameters, constant_data):
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
    encoded_all, peptide_to_idx = cache.get_or_build(unique_peptides)

    # Lookup each fold-peptide's row in the memmap. Fancy indexing produces
    # a contiguous in-memory copy sized (len(fold), enc_len, alphabet); that's
    # the same materialized tensor the old path would have produced on first
    # encode call.
    fold_indices = numpy.fromiter(
        (peptide_to_idx[p] for p in peptide_values),
        dtype=numpy.int64,
        count=len(peptide_values),
    )
    fold_encoded = encoded_all[fold_indices]
    return make_preencoded_encodable_sequences(peptide_values, fold_encoded, params)


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
        train_data.peptide.values, hyperparameters, constant_data
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

    train_params = dict(hyperparameters.get("train_data", {}))

    def get_train_param(param, default):
        if param in train_params:
            result = train_params.pop(param)
            if verbose:
                print("Train param", param, "=", result)
        else:
            result = default
            if verbose:
                print("Train param", param, "=", result, "[default]")
        return result


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
        pretrain_peptides_per_step = get_train_param(
            "pretrain_peptides_per_step", 1024)
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
            generator = pretrain_data_iterator(
                pretrain_data_filename,
                allele_encoding,
                peptides_per_chunk=pretrain_peptides_per_step,
                encoding_cache_dir=constant_data.get("encoding_cache_dir"),
                encoding_params=(
                    EncodingParams(**hyperparameters["peptide_encoding"])
                    if constant_data.get("encoding_cache_dir")
                    else None
                ),
            )

            model.fit_generator(
                generator,
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
                "training_info", {})["pretrain_attempt"] = attempt
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
        verbose=verbose)

    # Save model-specific training info
    train_peptide_hash = hashlib.sha1()
    for peptide in sorted(train_data.peptide.values):
        train_peptide_hash.update(peptide.encode())

    model.fit_info[-1].setdefault("training_info", {}).update({
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
    return predictor


if __name__ == '__main__':
    run()

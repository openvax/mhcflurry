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
import resource
from functools import partial

import numpy
import pandas
import yaml
import tqdm  # progress bar

from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_neural_network import (
    Class1NeuralNetwork,
    _peptide_uses_torch_encoding,
)
from .common import configure_logging, normalize_allele_name, write_generate_sh
from .local_parallelism import (
    add_local_parallelism_args,
    apply_dataloader_num_workers_to_work_items,
    apply_random_negative_pool_epochs_to_work_items,
    attach_constant_data_to_work_items_if_needed,
    call_wrapped_kwargs,
    resolve_local_parallelism_args,
    run_single_worker_torch_compile_warmup,
    worker_pool_with_gpu_assignments_from_args,
)
from .cluster_parallelism import (
    add_cluster_parallelism_args,
    cluster_results_from_args)
from .allele_encoding import AlleleEncoding
from .encodable_sequences import EncodableSequences
from .regression_target import from_ic50

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing local workers to read the same
# copy-on-write pages instead of receiving a pickled copy.
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
            encodable_peptides = EncodableSequences(repeated_peptides)
            yield (allele_encoding, encodable_peptides, df.stack().values)


_LOGGED_PRETRAIN_ALLELE_INFO = set()


def _get_pretrain_allele_info(filename, master_allele_encoding, verbose):
    """Return (normalized-empty-df, usable_alleles, skipped_alleles).

    Verbose output is dedup'd per (filename, skipped_alleles) tuple at
    module scope: this function gets called repeatedly by the per-fold
    pretrain loaders and was emitting 10× redundant 'Pretrain alleles'
    + 'Skipped alleles' blocks per training run that drowned the log.
    """
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
        log_key = (filename, tuple(skipped_alleles))
        if log_key not in _LOGGED_PRETRAIN_ALLELE_INFO:
            _LOGGED_PRETRAIN_ALLELE_INFO.add(log_key)
            print("Pretrain alleles available: ", *empty.columns.values)
            print("Using %d / %d alleles" % (
                len(usable_alleles), len(empty.columns)
            ))
            print("Skipped alleles: ", skipped_alleles)
    return empty, usable_alleles, skipped_alleles


def pretrain_network_input_iterator(
        filename,
        master_allele_encoding,
        peptide_encoding,
        peptides_per_chunk=1024,
        worker_id=0,
        num_workers=1,
        compact_peptide_repeats=False,
        peptide_amino_acid_encoding_torch=True):
    """Yield pretrain batches as network-input ``(x_dict, y)`` tuples."""
    empty, usable_alleles, _ = _get_pretrain_allele_info(
        filename,
        master_allele_encoding,
        verbose=(worker_id == 0),
    )
    allele_encoding_cache = {}
    allele_indices_cache = {}
    categorical_kwargs = {
        k: v for k, v in peptide_encoding.items()
        if k != "vector_encoding_name"
    }
    use_torch_peptide_encoding = _peptide_uses_torch_encoding({
        "peptide_encoding": peptide_encoding,
        "peptide_amino_acid_encoding_torch": peptide_amino_acid_encoding_torch,
    })

    synthetic_iter = pandas.read_csv(
        filename, index_col=0, chunksize=peptides_per_chunk)
    for chunk_num, df in enumerate(synthetic_iter):
        if num_workers > 1 and (chunk_num % num_workers) != worker_id:
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
            allele_indices_cache[chunk_len] = (
                allele_encoding_cache[chunk_len].indices.values.copy()
            )

        if compact_peptide_repeats:
            peptide_values = df.index.values
        else:
            peptide_values = numpy.repeat(df.index.values, len(usable_alleles))
        peptides = EncodableSequences(peptide_values)
        if use_torch_peptide_encoding:
            peptide_rows = (
                peptides.variable_length_to_fixed_length_categorical(
                    **categorical_kwargs
                )
                .astype("int8", copy=False)
            )
        else:
            peptide_rows = peptides.variable_length_to_fixed_length_vector_encoding(
                **peptide_encoding
            )
        x_dict = {
            "peptide": peptide_rows,
            "allele": allele_indices_cache[chunk_len],
        }
        if compact_peptide_repeats:
            x_dict["peptide_repeat_count"] = len(usable_alleles)
        yield (x_dict, from_ic50(df.stack().values))


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
    resolve_local_parallelism_args(
        args,
        cap_auto_num_jobs=not getattr(args, "cluster_parallelism", False),
    )

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
    #   setup_done          — pool + all GLOBAL_DATA
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

    # Strip any pre-existing fold_* columns from ``df`` before joining the
    # freshly-computed ``folds_df``. Public 2.2.0 train_data.csv.bz2 ships
    # with fold_0..3 already attached; without this, ``pandas.merge`` adds
    # ``_x``/``_y`` suffixes to disambiguate, which then breaks
    # ``select_pan_allele_models_command``'s ``int(col.split("_")[-1])``
    # parser. Re-folding is the intended behavior — fold assignment is a
    # function of (data, num_folds, held_out_*), so any prior fold cols
    # are stale.
    df_no_folds = df.drop(
        columns=[c for c in df.columns if c.startswith("fold_")],
        errors="ignore",
    )
    predictor = Class1AffinityPredictor(
        allele_to_sequence=allele_encoding.allele_to_sequence,
        metadata_dataframes={
            'train_data': pandas.merge(
                df_no_folds,
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

    # Apply the resolved --dataloader-num-workers (auto or pinned) to every
    # work item's hyperparameters. The resolver in
    # resolve_local_parallelism_args computed an int from box capacity;
    # injecting it here means the saved component-model configs reflect
    # the value the orchestrator actually chose for this run.
    if getattr(args, "dataloader_num_workers", None) is not None:
        apply_dataloader_num_workers_to_work_items(
            all_work_items, int(args.dataloader_num_workers)
        )

    # Inject the resolved (orchestrator-chosen) random_negative_pool_epochs
    # into every work item's hyperparameters. fit() reads the int from
    # ``self.hyperparameters`` when constructing its RandomNegativesPool.
    if getattr(args, "random_negative_pool_epochs", None) is not None:
        apply_random_negative_pool_epochs_to_work_items(
            all_work_items, int(args.random_negative_pool_epochs)
        )

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
        run_single_worker_torch_compile_warmup(
            args,
            work_items,
            train_model,
            constant_data=GLOBAL_DATA,
        )

        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None

        print("Processing %d work items in parallel." % len(work_items))
        assert not serial_run

        attach_constant_data_to_work_items_if_needed(
            work_items, GLOBAL_DATA, worker_pool
        )

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
    write_generate_sh(args.out_models_dir)
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


def _build_train_peptides(peptide_values):
    """Construct an EncodableSequences for the fold's training peptides."""
    return EncodableSequences(peptide_values)


def _run_compile_warmup(hyperparameters, fold_num, constant_data):
    """One forward+backward through a freshly-built network for compile-cache priming.

    Used by ``run_single_worker_torch_compile_warmup`` to populate the
    torch.compile on-disk cache once per unique architecture before the
    production worker pool launches. Trains for one epoch on
    ``minibatch_size`` rows with pretrain/validation/early-stop disabled,
    discards the resulting model, and returns. The compile cache write
    is the only durable side effect.
    """
    df = constant_data["train_data"]
    folds_df = constant_data["folds_df"]
    allele_encoding = constant_data["allele_encoding"]

    minibatch = max(int(hyperparameters.get("minibatch_size", 128) or 128), 1)
    fold_mask = folds_df["fold_%d" % fold_num]
    train_subset = df.loc[fold_mask].head(minibatch)
    if len(train_subset) == 0:
        train_subset = df.head(minibatch)

    train_peptides = _build_train_peptides(train_subset.peptide.values)
    train_alleles = AlleleEncoding(
        train_subset.allele.values, borrow_from=allele_encoding)

    # Subselect to keys this network actually accepts so warmup is
    # robust to upstream configs carrying parameters from a different
    # network or removed-but-still-present keys.
    hp = Class1NeuralNetwork.hyperparameter_defaults.subselect(
        dict(hyperparameters))
    hp["max_epochs"] = 1
    hp["validation_split"] = 0.0
    hp["early_stopping"] = False
    hp["data_dependent_initialization_method"] = None
    train_data_overrides = dict(hp.get("train_data") or {})
    train_data_overrides["pretrain"] = False
    hp["train_data"] = train_data_overrides

    print(
        "compile_warmup_only: layer_sizes=%s topology=%s minibatch=%d "
        "rows=%d" % (
            hp.get("layer_sizes"),
            hp.get("topology"),
            minibatch,
            len(train_subset),
        )
    )
    started = time.time()
    model = Class1NeuralNetwork(**hp)
    model.fit(
        peptides=train_peptides,
        affinities=train_subset.measurement_value.values,
        allele_encoding=train_alleles,
        inequalities=(
            train_subset.measurement_inequality.values
            if "measurement_inequality" in train_subset.columns else None
        ),
        verbose=0,
    )
    print("compile_warmup_only: completed in %.1f sec" % (time.time() - started))


def _random_negative_seed_for_work_item(
        architecture_num, fold_num, replicate_num, work_item_name):
    """Stable per-fit seed used only when random-negative pooling is enabled."""
    identity = (
        str(architecture_num),
        str(fold_num),
        str(replicate_num),
        str(work_item_name),
    )
    return int(hashlib.sha1("|".join(identity).encode()).hexdigest()[:16], 16)


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
        compile_warmup_only=False,
        constant_data=GLOBAL_DATA):

    if compile_warmup_only:
        _run_compile_warmup(hyperparameters, fold_num, constant_data)
        return None

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

    train_peptides = _build_train_peptides(train_data.peptide.values)
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

    # GPU memory telemetry at task boundaries. With max-tasks-per-worker>1,
    # a single worker process handles
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
            make_pretrain_generator = partial(
                pretrain_network_input_iterator,
                filename=pretrain_data_filename,
                master_allele_encoding=allele_encoding,
                peptide_encoding=hyperparameters["peptide_encoding"],
                peptides_per_chunk=pretrain_peptides_per_step,
                compact_peptide_repeats=True,
                peptide_amino_acid_encoding_torch=(
                    model.uses_peptide_torch_encoding()
                ),
            )

            model.fit_streaming_batches(
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

    # Derive a per-work-item random-negative seed only when pooled negatives
    # are in use. fit() bypasses ``random_negative_seed`` when pool_epochs == 1,
    # but keeping None here protects the default fresh-per-epoch random stream.
    pool_epochs = int(hyperparameters.get("random_negative_pool_epochs", 1) or 1)
    random_negative_seed = None
    if pool_epochs > 1:
        random_negative_seed = _random_negative_seed_for_work_item(
            architecture_num=architecture_num,
            fold_num=fold_num,
            replicate_num=replicate_num,
            work_item_name=work_item_name,
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

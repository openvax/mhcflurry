"""
Train Class1 pan-allele models.
"""
import argparse
import os
import signal
import sys
import time
import traceback
import random
import pprint
import hashlib
import pickle
import subprocess
from functools import partial

import numpy
import pandas
import yaml
from mhcnames import normalize_allele_name
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_neural_network import Class1NeuralNetwork
from .common import configure_logging
from .local_parallelism import (
    add_local_parallelism_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped_kwargs)
from .cluster_parallelism import (
    add_cluster_parallelism_args,
    cluster_results_from_args)
from .allele_encoding import AlleleEncoding
from .encodable_sequences import EncodableSequences


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing us to share large data with the workers
# via shared memory.
GLOBAL_DATA = {}

# Note on parallelization:
# It seems essential currently (tensorflow==1.4.1) that no processes are forked
# after tensorflow has been used at all, which includes merely importing
# keras.backend. So we must make sure not to use tensorflow in the main process
# if we are running in parallel.

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    required=True,
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
    required=True,
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
    "--ensemble-size",
    type=int,
    metavar="N",
    required=True,
    help="Ensemble size, i.e. how many models to retain the final predictor. "
    "In the current implementation, this is also the number of training folds.")
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
    "--save-interval",
    type=float,
    metavar="N",
    default=60,
    help="Write models to disk every N seconds. Only affects parallel runs; "
    "serial runs write each model to disk as it is trained.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
    default=0)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Launch python debugger on error")

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)

def assign_folds(df, num_folds, held_out_fraction, held_out_max):
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
        peptides_per_chunk=1024):
    empty = pandas.read_csv(filename, index_col=0, nrows=0)
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

    synthetic_iter = pandas.read_csv(
        filename, index_col=0, chunksize=peptides_per_chunk)
    for (k, df) in enumerate(synthetic_iter):
        if len(df) != peptides_per_chunk:
            continue

        df = df[usable_alleles]
        encodable_peptides = EncodableSequences(
            numpy.repeat(
                df.index.values,
                len(usable_alleles)))

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
            import ipdb ; ipdb.set_trace()
            raise
    else:
        return main(args)


def main(args):
    global GLOBAL_DATA

    print("Arguments:")
    print(args)

    args.out_models_dir = os.path.abspath(args.out_models_dir)

    configure_logging(verbose=args.verbosity > 1)

    hyperparameters_lst = yaml.load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list)
    print("Loaded hyperparameters list:")
    pprint.pprint(hyperparameters_lst)

    allele_sequences = pandas.read_csv(
        args.allele_sequences, index_col=0).sequence

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
        num_folds=args.ensemble_size,
        held_out_fraction=held_out_fraction,
        held_out_max=held_out_max)

    allele_sequences_in_use = allele_sequences[
        allele_sequences.index.isin(df.allele)
    ]
    print("Will use %d / %d allele sequences" % (
        len(allele_sequences_in_use), len(allele_sequences)))

    allele_encoding = AlleleEncoding(
        alleles=allele_sequences_in_use.index.values,
        allele_to_sequence=allele_sequences_in_use.to_dict())

    GLOBAL_DATA["train_data"] = df
    GLOBAL_DATA["folds_df"] = folds_df
    GLOBAL_DATA["allele_encoding"] = allele_encoding
    GLOBAL_DATA["args"] = args

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
    serial_run = args.num_jobs == 0

    work_items = []
    for (h, hyperparameters) in enumerate(hyperparameters_lst):
        if 'n_models' in hyperparameters:
            raise ValueError("n_models is unsupported")

        if args.max_epochs:
            hyperparameters['max_epochs'] = args.max_epochs

        if hyperparameters.get("train_data", {}).get("pretrain", False):
            if not args.pretrain_data:
                raise ValueError("--pretrain-data is required")

        for fold in range(args.ensemble_size):
            for replicate in range(args.num_replicates):
                work_dict = {
                    'architecture_num': h,
                    'num_architectures': len(hyperparameters_lst),
                    'fold_num': fold,
                    'num_folds': args.ensemble_size,
                    'replicate_num': replicate,
                    'num_replicates': args.num_replicates,
                    'hyperparameters': hyperparameters,
                    'pretrain_data_filename': args.pretrain_data,
                    'verbose': args.verbosity,
                    'progress_print_interval': 60.0 if not serial_run else 5.0,
                    'predictor': predictor if serial_run else None,
                    'save_to': args.out_models_dir if serial_run else None,
                }
                work_items.append(work_dict)

    start = time.time()

    # The estimated time to completion is more accurate if we randomize
    # the order of the work.
    random.shuffle(work_items)
    for (work_item_num, item) in enumerate(work_items):
        item['work_item_num'] = work_item_num
        item['num_work_items'] = len(work_items)

    if args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        results_generator = cluster_results_from_args(
            args,
            work_function=train_model,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="save_predictor")
        worker_pool = None
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)

        if worker_pool:
            print("Processing %d work items in parallel." % len(work_items))
            assert not serial_run

            results_generator = worker_pool.imap_unordered(
                partial(call_wrapped_kwargs, train_model),
                work_items,
                chunksize=1)
        else:
            # Run in serial. In this case, every worker is passed the same predictor,
            # which it adds models to, so no merging is required. It also saves
            # as it goes so no saving is required at the end.
            print("Processing %d work items in serial." % len(work_items))
            assert serial_run
            for _ in tqdm.trange(len(work_items)):
                item = work_items.pop(0)  # want to keep freeing up memory
                work_predictor = train_model(**item)
                assert work_predictor is predictor
            assert not work_items
            results_generator = None

    if results_generator:
        unsaved_predictors = []
        last_save_time = time.time()
        for new_predictor in tqdm.tqdm(results_generator, total=len(work_items)):
            unsaved_predictors.append(new_predictor)

            if time.time() > last_save_time + args.save_interval:
                # Save current predictor.
                save_start = time.time()
                new_model_names = predictor.merge_in_place(unsaved_predictors)
                predictor.save(
                    args.out_models_dir,
                    model_names_to_write=new_model_names,
                    write_metadata=False)
                print(
                    "Saved predictor (%d models total) including %d new models "
                    "in %0.2f sec to %s" % (
                        len(predictor.neural_networks),
                        len(new_model_names),
                        time.time() - save_start,
                        args.out_models_dir))
                unsaved_predictors = []
                last_save_time = time.time()

        predictor.merge_in_place(unsaved_predictors)

    print("Saving final predictor to: %s" % args.out_models_dir)
    predictor.save(args.out_models_dir)  # write all models just to be sure
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


def train_model(
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
    args = constant_data["args"]

    if predictor is None:
        predictor = Class1AffinityPredictor(
            allele_to_sequence=allele_encoding.allele_to_sequence)

    numpy.testing.assert_equal(len(df), len(folds_df))

    train_data = df.loc[
        folds_df["fold_%d" % fold_num]
    ].sample(frac=1.0)

    train_peptides = EncodableSequences(train_data.peptide.values)
    train_alleles = AlleleEncoding(
        train_data.allele.values, borrow_from=allele_encoding)

    model = Class1NeuralNetwork(**hyperparameters)

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

    assert model.network() is None
    if hyperparameters.get("train_data", {}).get("pretrain", False):
        generator = pretrain_data_iterator(pretrain_data_filename, allele_encoding)
        pretrain_patience = hyperparameters["train_data"].get(
            "pretrain_patience", 10)
        pretrain_min_delta = hyperparameters["train_data"].get(
            "pretrain_min_delta", 0.0)
        pretrain_steps_per_epoch = hyperparameters["train_data"].get(
            "pretrain_steps_per_epoch", 10)
        pretrain_max_epochs = hyperparameters["train_data"].get(
            "pretrain_max_epochs", 1000)

        max_val_loss =  hyperparameters["train_data"].get("pretrain_max_val_loss")

        attempt = 0
        while True:
            attempt += 1
            print("Pre-training attempt %d" % attempt)
            if attempt > 10:
                print("Too many pre-training attempts! Stopping pretraining.")
                break
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
                verbose=verbose,
            )
            if not max_val_loss:
                break
            if model.fit_info[-1]["val_loss"] >= max_val_loss:
                print("Val loss %f >= max val loss %f. Pre-training again." % (
                    model.fit_info[-1]["val_loss"], max_val_loss))
            else:
                print("Val loss %f < max val loss %f. Done pre-training." % (
                    model.fit_info[-1]["val_loss"], max_val_loss))
                break

        # Use a smaller learning rate for training on real data
        learning_rate = model.fit_info[-1]["learning_rate"]
        model.hyperparameters['learning_rate'] = learning_rate / 10

    model.fit(
        peptides=train_peptides,
        affinities=train_data.measurement_value.values,
        allele_encoding=train_alleles,
        inequalities=(
            train_data.measurement_inequality.values
            if "measurement_inequality" in train_data.columns else None),
        progress_preamble=progress_preamble,
        progress_print_interval=progress_print_interval,
        verbose=verbose)

    # Save model-specific training info
    train_peptide_hash = hashlib.sha1()
    for peptide in sorted(train_data.peptide.values):
        train_peptide_hash.update(peptide.encode())
    model.fit_info[-1]["training_info"] = {
        "fold_num": fold_num,
        "num_folds": num_folds,
        "replicate_num": replicate_num,
        "num_replicates": num_replicates,
        "architecture_num": architecture_num,
        "num_architectures": num_architectures,
        "train_peptide_hash": train_peptide_hash.hexdigest(),
    }

    numpy.testing.assert_equal(
        predictor.manifest_df.shape[0], len(predictor.class1_pan_allele_models))
    predictor.add_pan_allele_model(model, models_dir_for_save=save_to)
    numpy.testing.assert_equal(
        predictor.manifest_df.shape[0], len(predictor.class1_pan_allele_models))
    predictor.clear_cache()

    # Delete the network to release memory
    model.update_network_description()  # save weights and config
    model._network = None  # release tensorflow network
    return predictor


if __name__ == '__main__':
    run()


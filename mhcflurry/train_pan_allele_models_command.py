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
from functools import partial

import numpy
import pandas
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from mhcnames import normalize_allele_name
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_neural_network import Class1NeuralNetwork
from .common import configure_logging, set_keras_backend
from .parallelism import (
    add_worker_pool_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped_kwargs)
from .hyperparameters import HyperparameterDefaults
from .allele_encoding import AlleleEncoding
from .encodable_sequences import EncodableSequences
from .regression_target import to_ic50, from_ic50


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

add_worker_pool_args(parser)


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
            'train_data': df,
            'training_folds': folds_df,
        })
    serial_run = args.num_jobs == 1

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
                    'progress_print_interval': None if not serial_run else 5.0,
                    'predictor': predictor if serial_run else None,
                    'save_to': args.out_models_dir if serial_run else None,
                }
                work_items.append(work_dict)

    start = time.time()

    worker_pool = worker_pool_with_gpu_assignments_from_args(args)

    if worker_pool:
        print("Processing %d work items in parallel." % len(work_items))

        # The estimated time to completion is more accurate if we randomize
        # the order of the work.
        random.shuffle(work_items)

        results_generator = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, train_model),
            work_items,
            chunksize=1)

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

    else:
        # Run in serial. In this case, every worker is passed the same predictor,
        # which it adds models to, so no merging is required. It also saves
        # as it goes so no saving is required at the end.
        print("Processing %d work items in serial." % len(work_items))
        for _ in tqdm.trange(len(work_items)):
            item = work_items.pop(0)  # want to keep freeing up memory
            work_predictor = train_model(**item)
            assert work_predictor is predictor
        assert not work_items

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
        save_to):

    df = GLOBAL_DATA["train_data"]
    folds_df = GLOBAL_DATA["folds_df"]
    allele_encoding = GLOBAL_DATA["allele_encoding"]
    args = GLOBAL_DATA["args"]

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
    train_target = from_ic50(train_data.measurement_value.values)

    model = Class1NeuralNetwork(**hyperparameters)

    progress_preamble = (
        "[%2d / %2d folds] "
        "[%2d / %2d architectures] "
        "[%4d / %4d replicates] " % (
            fold_num + 1,
            num_folds,
            architecture_num + 1,
            num_architectures,
            replicate_num + 1,
            num_replicates))

    if hyperparameters.get("train_data", {}).get("pretrain", False):
        iterator = pretrain_data_iterator(pretrain_data_filename, allele_encoding)
        original_hyperparameters = dict(model.hyperparameters)
        model.hyperparameters['minibatch_size'] = int(len(next(iterator)[-1]) / 100)
        model.hyperparameters['max_epochs'] = 1
        model.hyperparameters['validation_split'] = 0.0
        model.hyperparameters['random_negative_rate'] = 0.0
        model.hyperparameters['random_negative_constant'] = 0
        pretrain_patience = hyperparameters["train_data"]["pretrain_patience"]
        scores = []
        best_score = float('inf')
        best_score_epoch = 0
        for (epoch, (alleles, peptides, affinities)) in enumerate(iterator):
            # Fit one epoch.
            start = time.time()
            model.fit(
                peptides=peptides,
                affinities=affinities,
                allele_encoding=alleles)

            fit_time = time.time() - start
            start = time.time()
            predictions = model.predict(
                train_peptides,
                allele_encoding=train_alleles)
            assert len(predictions) == len(train_data)

            print("Prediction histogram:")
            print(
                pandas.Series(
                    dict([k, v] for (v, k) in zip(*numpy.histogram(predictions)))))

            for (inequality, func) in [(">", numpy.minimum), ("<", numpy.maximum)]:
                mask = train_data.measurement_inequality == inequality
                predictions[mask.values] = func(
                    predictions[mask.values],
                    train_data.loc[mask].measurement_value.values)
            score_mse = numpy.mean((from_ic50(predictions) - train_target)**2)
            score_time = time.time() - start
            print(
                progress_preamble,
                "PRETRAIN epoch %d [%d values, %0.2f sec]. "
                "MSE [%0.2f sec.]: %10f" % (
                    epoch, len(affinities), fit_time, score_time, score_mse))
            scores.append(score_mse)

            if score_mse < best_score:
                print("New best score_mse", score_mse)
                best_score = score_mse
                best_score_epoch = epoch

            if epoch - best_score_epoch > pretrain_patience:
                print("Stopping pretraining")
                break

        model.hyperparameters = original_hyperparameters
        if model.hyperparameters['learning_rate']:
            model.hyperparameters['learning_rate'] /= 10
        else:
            model.hyperparameters['learning_rate'] = 0.0001

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

    numpy.testing.assert_equal(
        predictor.manifest_df.shape[0], len(predictor.class1_pan_allele_models))
    predictor.add_pan_allele_model(model, models_dir_for_save=save_to)
    numpy.testing.assert_equal(
        predictor.manifest_df.shape[0], len(predictor.class1_pan_allele_models))
    predictor.clear_cache()

    return predictor


if __name__ == '__main__':
    run()


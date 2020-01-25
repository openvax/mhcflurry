"""
Train Class1 processing models.
"""
from __future__ import print_function
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
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_processing_predictor import Class1ProcessingPredictor
from .class1_processing_neural_network import Class1ProcessingNeuralNetwork
from .common import configure_logging
from .local_parallelism import (
    add_local_parallelism_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped_kwargs)
from .cluster_parallelism import (
    add_cluster_parallelism_args,
    cluster_results_from_args)

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
    help="Training data CSV. Expected columns: peptide, n_flank, c_flank, hit")
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
    "--held-out-samples",
    type=int,
    metavar="N",
    default=10,
    help="Number of experiments to hold out per fold")
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
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
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


def assign_folds(df, num_folds, held_out_samples):
    """
    Split training data into mulitple test/train pairs, which we refer to as
    folds. Note that a given data point may be assigned to multiple test or
    train sets; these folds are NOT a non-overlapping partition as used in cross
    validation.

    A fold is defined by a boolean value for each data point, indicating whether
    it is included in the training data for that fold. If it's not in the
    training data, then it's in the test data.

    Parameters
    ----------
    df : pandas.DataFrame
        training data
    num_folds : int
    held_out_samples : int

    Returns
    -------
    pandas.DataFrame
        index is same as df.index, columns are "fold_0", ... "fold_N" giving
        whether the data point is in the training data for the fold
    """
    result_df = pandas.DataFrame(index=df.index)
    sample_names = pandas.Series(df.sample_id.unique())

    for fold in range(num_folds):
        samples_to_exclude = sample_names.sample(n=held_out_samples)
        result_df["fold_%d" % fold] = ~df.sample_id.isin(samples_to_exclude)
        print("Fold", fold, "holding out samples", *samples_to_exclude)

    print("Training points per fold")
    print(result_df.sum())

    print("Test points per fold")
    print((~result_df).sum())
    return result_df


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
    hyperparameters_lst = yaml.load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list)
    print("Loaded hyperparameters list:")

    if len(hyperparameters_lst) > 7:
        pprint.pprint(hyperparameters_lst[:3])
        print("...")
        pprint.pprint(hyperparameters_lst[-3:])
    else:
        pprint.pprint(hyperparameters_lst)
    print("Length of hyperparameters list: %d" % (len(hyperparameters_lst)))

    df = pandas.read_csv(args.data)
    print("Loaded training data: %s" % (str(df.shape)))
    df = df.loc[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))
    folds_df = assign_folds(
        df=df,
        num_folds=args.num_folds,
        held_out_samples=args.held_out_samples)

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    predictor = Class1ProcessingPredictor(
        models=[],
        metadata_dataframes={
            'train_data': pandas.merge(
                df,
                folds_df,
                left_index=True,
                right_index=True)
        })

    work_items = []
    for (h, hyperparameters) in enumerate(hyperparameters_lst):
        if args.max_epochs:
            hyperparameters['max_epochs'] = args.max_epochs

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
                }
                work_items.append(work_dict)

    training_init_info = {}
    training_init_info["train_data"] = df
    training_init_info["folds_df"] = folds_df
    training_init_info["work_items"] = work_items

    # Save empty predictor (for metadata)
    predictor.save(args.out_models_dir)

    # Write training_init_info.
    with open(join(args.out_models_dir, "training_init_info.pkl"), "wb") as fd:
        pickle.dump(training_init_info, fd, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done initializing training.")


def train_models(args):
    global GLOBAL_DATA

    print("Beginning training.")
    predictor = Class1ProcessingPredictor.load(args.out_models_dir)
    print("Loaded predictor with %d networks" % len(predictor.models))

    with open(join(args.out_models_dir, "training_init_info.pkl"), "rb") as fd:
        GLOBAL_DATA.update(pickle.load(fd))
    print("Loaded training init info.")

    all_work_items = GLOBAL_DATA["work_items"]
    complete_work_item_names = [
        network.fit_info[-1]["training_info"]["work_item_name"]
        for network in predictor.models
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
            pprint.pprint(predictor.models[-1].fit_info[-1]['training_info'])
        assert not work_items
        results_generator = None
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        results_generator = cluster_results_from_args(
            args,
            work_function=train_model,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="pickle")
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None

        print("Processing %d work items in parallel." % len(work_items))
        assert not serial_run

        results_generator = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, train_model),
            work_items,
            chunksize=1)

    if results_generator:
        for new_predictor in tqdm.tqdm(results_generator, total=len(work_items)):
            save_start = time.time()
            (model,) = new_predictor.models
            pprint.pprint(model.fit_info[-1]['training_info'])
            (new_model_name,) = predictor.add_models(new_predictor.models)
            predictor.save(
                args.out_models_dir,
                model_names_to_write=[new_model_name],
                write_metadata=False)
            print(
                "Saved predictor (%d models total) with 1 new models"
                "in %0.2f sec to %s" % (
                    len(predictor.models),
                    time.time() - save_start,
                    args.out_models_dir))

    predictor.save(args.out_models_dir)
    print("Done saving.")

    print("*" * 30)
    training_time = time.time() - start
    print("Trained affinity predictor with %d networks in %0.2f min." % (
        len(predictor.models), training_time / 60.0))
    print("*" * 30)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Predictor written to: %s" % args.out_models_dir)


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
        verbose,
        progress_print_interval,
        predictor,
        save_to,
        constant_data=GLOBAL_DATA):

    from sklearn.metrics import roc_auc_score
    from mhcflurry.flanking_encoding import FlankingEncoding

    df = constant_data["train_data"]
    folds_df = constant_data["folds_df"]

    if predictor is None:
        predictor = Class1ProcessingPredictor(models=[])

    numpy.testing.assert_equal(len(df), len(folds_df))

    train_data = df.loc[
        folds_df["fold_%d" % fold_num]
    ].sample(frac=1.0).copy()

    test_data = df.loc[~folds_df["fold_%d" % fold_num]].copy()

    print("Training on %d points (%d points held-out)." % (
        len(train_data), len(test_data)))

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

    model = Class1ProcessingNeuralNetwork(**hyperparameters)
    model.fit(
        sequences=FlankingEncoding(
            peptides=train_data.peptide.values,
            n_flanks=train_data.n_flank.values,
            c_flanks=train_data.c_flank.values),
        targets=train_data.hit.values,
        progress_preamble=progress_preamble,
        progress_print_interval=progress_print_interval,
        verbose=verbose)

    # Save model-specific training info
    train_peptide_hash = hashlib.sha1()
    for peptide in sorted(train_data.peptide.values):
        train_peptide_hash.update(peptide.encode())

    # Compute AUC on held-out data just so it can be logged.
    for some_df in [train_data, test_data]:
        some_df["prediction"] = model.predict(
            peptides=some_df.peptide.values,
            n_flanks=some_df.n_flank.values,
            c_flanks=some_df.c_flank.values)
    train_auc = roc_auc_score(
        train_data.hit.values, train_data.prediction.values)
    test_auc = roc_auc_score(test_data.hit.values, test_data.prediction.values)
    print("Train AUC", train_auc)
    print("Test AUC", test_auc)

    model.fit_info[-1].setdefault("training_info", {}).update({
        "fold_num": fold_num,
        "num_folds": num_folds,
        "replicate_num": replicate_num,
        "num_replicates": num_replicates,
        "architecture_num": architecture_num,
        "num_architectures": num_architectures,
        "train_peptide_hash": train_peptide_hash.hexdigest(),
        "work_item_name": work_item_name,
        "train_auc": train_auc,
        "test_auc": test_auc,
    })

    numpy.testing.assert_equal(
        predictor.manifest_df.shape[0], len(predictor.models))
    predictor.add_models([model])
    if save_to:
        predictor.save(save_to)
        print("Wrote", save_to)
    numpy.testing.assert_equal(
        predictor.manifest_df.shape[0], len(predictor.models))

    # Delete the network to release memory
    model._network = None  # release tensorflow network
    return predictor


if __name__ == '__main__':
    run()

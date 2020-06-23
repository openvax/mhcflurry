"""
Model select class1 pan-allele models.

APPROACH: For each training fold, we select at least min and at most max models
(where min and max are set by the --{min/max}-models-per-fold argument) using a
step-up (forward) selection procedure. The final ensemble is the union of all
selected models across all folds.
"""
import argparse
import os
import signal
import sys
import time
import traceback
import hashlib
from pprint import pprint

import numpy
import pandas

import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_affinity_predictor import Class1AffinityPredictor
from .encodable_sequences import EncodableSequences
from .allele_encoding import AlleleEncoding
from .common import configure_logging
from .local_parallelism import (
    worker_pool_with_gpu_assignments_from_args,
    add_local_parallelism_args)
from .cluster_parallelism import (
    add_cluster_parallelism_args,
    cluster_results_from_args)
from .regression_target import from_ic50


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing us to share large data with the workers
# via shared memory.
GLOBAL_DATA = {}


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    required=False,
    help=(
        "Model selection data CSV. Expected columns: "
        "allele, peptide, measurement_value"))
parser.add_argument(
    "--models-dir",
    metavar="DIR",
    required=True,
    help="Directory to read models")
parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    required=True,
    help="Directory to write selected models")
parser.add_argument(
    "--min-models-per-fold",
    type=int,
    default=2,
    metavar="N",
    help="Min number of models to select per fold")
parser.add_argument(
    "--max-models-per-fold",
    type=int,
    default=1000,
    metavar="N",
    help="Max number of models to select per fold")
parser.add_argument(
    "--mass-spec-regex",
    metavar="REGEX",
    default="mass[- ]spec",
    help="Regular expression for mass-spec data. Runs on measurement_source col."
    "Default: %(default)s.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
    default=0)

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)


def mse(
        predictions,
        actual,
        inequalities=None,
        affinities_are_already_01_transformed=False):
    """
    Mean squared error of predictions vs. actual

    Parameters
    ----------
    predictions : list of float
    actual : list of float
    inequalities : list of string (">", "<", or "=")
    affinities_are_already_01_transformed : boolean
        Predictions and actual are taken to be nanomolar affinities if
        affinities_are_already_01_transformed is False, otherwise 0-1 values.

    Returns
    -------
    float
    """
    if not affinities_are_already_01_transformed:
        predictions = from_ic50(predictions)
        actual = from_ic50(actual)

    deviations = (
        numpy.array(predictions, copy=False) - numpy.array(actual, copy=False))

    if inequalities is not None:
        # Must reverse meaning of inequality since we are working with
        # transformed 0-1 values, which are anti-correlated with the ic50s.
        # The measurement_inequality column is given in terms of ic50s.
        inequalities = numpy.array(inequalities, copy=False)
        deviations[
            ((inequalities == "<") & (deviations > 0)) | (
             (inequalities == ">") & (deviations < 0))
        ] = 0.0

    return (deviations ** 2).mean()


def run(argv=sys.argv[1:]):
    global GLOBAL_DATA

    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    args.out_models_dir = os.path.abspath(args.out_models_dir)

    configure_logging(verbose=args.verbosity > 1)

    df = pandas.read_csv(args.data)
    print("Loaded data: %s" % (str(df.shape)))

    input_predictor = Class1AffinityPredictor.load(
        args.models_dir, optimization_level=0)
    print("Loaded: %s" % input_predictor)

    alleles = input_predictor.supported_alleles
    (min_peptide_length, max_peptide_length) = (
        input_predictor.supported_peptide_lengths)

    metadata_dfs = {}

    fold_cols = [c for c in df if c.startswith("fold_")]
    num_folds = len(fold_cols)
    if num_folds <= 1:
        raise ValueError("Too few folds: ", num_folds)

    df = df.loc[
        (df.peptide.str.len() >= min_peptide_length) &
        (df.peptide.str.len() <= max_peptide_length)
    ]
    print("Subselected to %d-%dmers: %s" % (
        min_peptide_length, max_peptide_length, str(df.shape)))

    print("Num folds: ", num_folds, "fraction included:")
    print(df[fold_cols].mean())

    # Allele names in data are assumed to be already normalized.
    df = df.loc[df.allele.isin(alleles)]
    print("Subselected to supported alleles: %s" % str(df.shape))

    metadata_dfs["model_selection_data"] = df

    df["mass_spec"] = df.measurement_source.str.contains(
        args.mass_spec_regex)

    def make_train_peptide_hash(sub_df):
        train_peptide_hash = hashlib.sha1()
        for peptide in sorted(sub_df.peptide.values):
            train_peptide_hash.update(peptide.encode())
        return train_peptide_hash.hexdigest()

    folds_to_predictors = dict(
        (int(col.split("_")[-1]), (
            [],
            make_train_peptide_hash(df.loc[df[col] == 1])))
        for col in fold_cols)
    print(folds_to_predictors)
    for model in input_predictor.class1_pan_allele_models:
        training_info = model.fit_info[-1]['training_info']
        fold_num = training_info['fold_num']
        assert num_folds == training_info['num_folds']
        (lst, hash) = folds_to_predictors[fold_num]
        train_peptide_hash = training_info['train_peptide_hash']
        numpy.testing.assert_equal(hash, train_peptide_hash)
        lst.append(model)

    work_items = []
    for (fold_num, (models, _)) in folds_to_predictors.items():
        work_items.append({
            'fold_num': fold_num,
            'models': models,
            'min_models': args.min_models_per_fold,
            'max_models': args.max_models_per_fold,
        })

    GLOBAL_DATA["data"] = df
    GLOBAL_DATA["input_predictor"] = input_predictor

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    result_predictor = Class1AffinityPredictor(
        allele_to_sequence=input_predictor.allele_to_sequence,
        metadata_dataframes=metadata_dfs)

    serial_run = not args.cluster_parallelism and args.num_jobs == 0
    worker_pool = None
    start = time.time()
    if serial_run:
        # Serial run
        print("Running in serial.")
        results = (model_select(**item) for item in work_items)
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=model_select,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="pickle")
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None

        print("Processing %d work items in parallel." % len(work_items))
        assert not serial_run

        for item in work_items:
            item['constant_data'] = GLOBAL_DATA

        # Parallel run
        results = worker_pool.imap_unordered(
            do_model_select_task,
            work_items,
            chunksize=1)

    models_by_fold = {}
    summary_dfs = []
    for result in tqdm.tqdm(results, total=len(work_items)):
        pprint(result)
        fold_num = result['fold_num']
        (all_models_for_fold, _) = folds_to_predictors[fold_num]
        models = result['selected_models']
        summary_df = result['summary'].copy()
        summary_df.index = summary_df.index.map(
            lambda idx: all_models_for_fold[idx])
        summary_dfs.append(summary_df)

        print("Selected %d models for fold %d: %s" % (
            len(models), fold_num, result['selected_indices']))
        models_by_fold[fold_num] = models
        for model in models:
            result_predictor.add_pan_allele_model(model)

    summary_df = pandas.concat(summary_dfs, ignore_index=False)
    summary_df["model_config"] = summary_df.index.map(lambda m: m.get_config())
    result_predictor.metadata_dataframes["model_selection_summary"] = (
        summary_df.reset_index(drop=True))

    result_predictor.save(args.out_models_dir)

    model_selection_time = time.time() - start

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Model selection time %0.2f min." % (model_selection_time / 60.0))
    print("Predictor [%d models] written to: %s" % (
        len(result_predictor.neural_networks),
        args.out_models_dir))


def do_model_select_task(item, constant_data=GLOBAL_DATA):
    if 'constant_data' in item:
        constant_data = item.pop('constant_data')
    return model_select(constant_data=constant_data, **item)


def model_select(
        fold_num, models, min_models, max_models, constant_data=GLOBAL_DATA):
    """
    Model select for a fold.

    Parameters
    ----------
    fold_num : int
    models : list of Class1NeuralNetwork
    min_models : int
    max_models : int
    constant_data : dict

    Returns
    -------
    dict with keys 'fold_num', 'selected_indices', 'summary'
    """
    full_data = constant_data["data"]
    input_predictor = constant_data["input_predictor"]
    df = full_data.loc[
        full_data["fold_%d" % fold_num] == 0
    ]

    peptides = EncodableSequences.create(df.peptide.values)
    alleles = AlleleEncoding(
        df.allele.values,
        borrow_from=input_predictor.master_allele_encoding)

    predictions_df = df.copy()
    for (i, model) in enumerate(models):
        predictions_df[i] = from_ic50(model.predict(peptides, alleles))

    actual = from_ic50(predictions_df.measurement_value)

    selected = []
    selected_score = 0
    remaining_models = set(numpy.arange(len(models)))
    individual_model_scores = {}
    while remaining_models and len(selected) < max_models:
        best_model = None
        best_model_score = 0
        for i in remaining_models:
            possible_ensemble = list(selected) + [i]
            predictions = predictions_df[possible_ensemble].mean(1)
            mse_score = 1 - mse(
                predictions,
                actual,
                inequalities=(
                    predictions_df.measurement_inequality
                    if 'measurement_inequality' in predictions_df.columns
                    else None),
                affinities_are_already_01_transformed=True)
            if mse_score >= best_model_score:
                best_model = i
                best_model_score = mse_score
            if not selected:
                # First iteration. Store individual model scores.
                individual_model_scores[i] = mse_score
        if len(selected) < min_models or best_model_score > selected_score:
            selected_score = best_model_score
            remaining_models.remove(best_model)
            selected.append(best_model)
        else:
            break

    assert selected
    selected_models = [models[i] for i in selected]
    for model in selected_models:
        model.clear_allele_representations()

    summary_df = pandas.Series(individual_model_scores)[
        numpy.arange(len(models))
    ].to_frame()
    summary_df.columns = ['mse_score']

    return {
        'fold_num': fold_num,
        'selected_indices': selected,
        'selected_models': selected_models,
        'summary': summary_df,  # indexed by model index
    }


if __name__ == '__main__':
    run()

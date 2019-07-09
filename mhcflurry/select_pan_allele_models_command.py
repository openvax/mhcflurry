"""
Model select class1 single allele models.
"""
import argparse
import os
import signal
import sys
import time
import traceback
import random
import hashlib
from pprint import pprint

import numpy
import pandas
from scipy.stats import kendalltau, percentileofscore, pearsonr

import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_affinity_predictor import Class1AffinityPredictor
from .encodable_sequences import EncodableSequences
from .allele_encoding import AlleleEncoding
from .common import configure_logging, random_peptides
from .parallelism import worker_pool_with_gpu_assignments_from_args, add_worker_pool_args
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
    "--folds",
    metavar="FILE.csv",
    required=False,
    help=(""))
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

add_worker_pool_args(parser)


def mse(
        predictions,
        actual,
        inequalities=None,
        affinities_are_already_01_transformed=False):
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

    input_predictor = Class1AffinityPredictor.load(args.models_dir)
    print("Loaded: %s" % input_predictor)

    alleles = input_predictor.supported_alleles
    (min_peptide_length, max_peptide_length) = (
        input_predictor.supported_peptide_lengths)

    metadata_dfs = {}
    df = pandas.read_csv(args.data)
    print("Loaded data: %s" % (str(df.shape)))

    if args.folds:
        folds_df = pandas.read_csv(args.folds)
        matches = all([
            len(folds_df) == len(df),
            (folds_df.peptide == df.peptide).all(),
            (folds_df.allele == df.allele).all(),
        ])
        if not matches:
            raise ValueError("Training data and fold data do not match")
        fold_cols = [c for c in folds_df if c.startswith("fold_")]
        for col in fold_cols:
            df[col] = folds_df[col]

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
    df = df.loc[df.allele.isin(alleles)].dropna()
    print("Subselected to supported alleles: %s" % str(df.shape))

    print("Selected %d alleles: %s" % (len(alleles), ' '.join(alleles)))

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
        #numpy.testing.assert_equal(hash, train_peptide_hash)  #enable later
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

    worker_pool = worker_pool_with_gpu_assignments_from_args(args)

    start = time.time()

    if worker_pool is None:
        # Serial run
        print("Running in serial.")
        results = (do_model_select_task(item) for item in work_items)
    else:
        # Parallel run
        random.shuffle(alleles)
        results = worker_pool.imap_unordered(
            do_model_select_task,
            work_items,
            chunksize=1)

    models_by_fold = {}
    for result in tqdm.tqdm(results, total=len(work_items)):
        pprint(result)
        fold_num = result['fold_num']
        models = [
            folds_to_predictors[fold_num][0][i]
            for i in result['selected_indices']
        ]
        print("Selected %d models for fold %d: %s" % (
            len(models), fold_num, result['selected_indices']))
        models_by_fold[fold_num] = models
        for model in models:
            result_predictor.add_pan_allele_model(model)

    result_predictor.save(args.out_models_dir)

    model_selection_time = time.time() - start

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Model selection time %0.2f min." % (model_selection_time / 60.0))
    print("Predictor written to: %s" % args.out_models_dir)


def do_model_select_task(item):
    return model_select(**item)


def model_select(fold_num, models, min_models, max_models):
    global GLOBAL_DATA
    full_data = GLOBAL_DATA["data"]
    input_predictor = GLOBAL_DATA["input_predictor"]
    df = full_data.loc[
        full_data["fold_%d" % fold_num] == 0
    ]

    peptides = EncodableSequences.create(df.peptide.values)
    alleles = AlleleEncoding(
        df.allele.values,
        borrow_from=input_predictor.get_master_allele_encoding())

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
    return {
        'fold_num': fold_num,
        'selected_indices': selected,
        'individual_model_scores': pandas.Series(
            individual_model_scores)[numpy.arange(len(models))],
    }


if __name__ == '__main__':
    run()

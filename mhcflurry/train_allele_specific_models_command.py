"""
Train Class1 single allele models.
"""
import argparse
import os
import signal
import sys
import time
import traceback
from multiprocessing import Pool
from functools import partial
from pprint import pprint

import pandas
import yaml
from mhcnames import normalize_allele_name
import tqdm  # progress bar

from .class1_affinity_predictor import Class1AffinityPredictor
from .common import configure_logging, set_keras_backend


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing us to share large data with the workers
# efficiently.
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
    "--allele",
    default=None,
    nargs="+",
    help="Alleles to train models for. If not specified, all alleles with "
    "enough measurements will be used.")
parser.add_argument(
    "--min-measurements-per-allele",
    type=int,
    metavar="N",
    default=50,
    help="Train models for alleles with >=N measurements.")
parser.add_argument(
    "--only-quantitative",
    action="store_true",
    default=False,
    help="Use only quantitative training data")
parser.add_argument(
    "--ignore-inequalities",
    action="store_true",
    default=False,
    help="Do not use affinity value inequalities even when present in data")
parser.add_argument(
    "--percent-rank-calibration-num-peptides-per-length",
    type=int,
    metavar="N",
    default=int(1e5),
    help="Number of peptides per length to use to calibrate percent ranks. "
    "Set to 0 to disable percent rank calibration. The resulting models will "
    "not support percent ranks. Default: %(default)s.")
parser.add_argument(
    "--n-models",
    type=int,
    metavar="N",
    help="Ensemble size, i.e. how many models to train for each architecture. "
    "If specified here it overrides any 'n_models' specified in the "
    "hyperparameters.")
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
    "--num-jobs",
    default=[1],
    type=int,
    metavar="N",
    nargs="+",
    help="Number of processes to parallelize training and percent rank "
    "calibration over, respectively. Experimental. If only one value is specified "
    "then the same number of jobs is used for both phases."
    "Set to 1 for serial run. Set to 0 to use number of cores. Default: %(default)s.")
parser.add_argument(
    "--backend",
    choices=("tensorflow-gpu", "tensorflow-cpu"),
    help="Keras backend. If not specified will use system default.")


def run(argv=sys.argv[1:]):
    global GLOBAL_DATA

    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    if args.backend:
        set_keras_backend(args.backend)

    configure_logging(verbose=args.verbosity > 1)

    hyperparameters_lst = yaml.load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list)
    print("Loaded hyperparameters list: %s" % str(hyperparameters_lst))

    df = pandas.read_csv(args.data)
    print("Loaded training data: %s" % (str(df.shape)))

    df = df.ix[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    if args.only_quantitative:
        df = df.loc[
            df.measurement_type == "quantitative"
        ]
        print("Subselected to quantitative: %s" % (str(df.shape)))

    if args.ignore_inequalities and "measurement_inequality" in df.columns:
        print("Dropping measurement_inequality column")
        del df["measurement_inequality"]

    allele_counts = df.allele.value_counts()

    if args.allele:
        alleles = [normalize_allele_name(a) for a in args.allele]
    else:
        alleles = list(allele_counts.ix[
            allele_counts > args.min_measurements_per_allele
        ].index)

    # Allele names in data are assumed to be already normalized.
    df = df.loc[df.allele.isin(alleles)].dropna()

    print("Selected %d alleles: %s" % (len(alleles), ' '.join(alleles)))
    print("Training data: %s" % (str(df.shape)))

    GLOBAL_DATA["train_data"] = df

    predictor = Class1AffinityPredictor()
    if args.num_jobs[0] == 1:
        # Serial run
        print("Running in serial.")
        worker_pool = None
    else:
        worker_pool = Pool(
            processes=(
                args.num_jobs[0]
                if args.num_jobs[0] else None))
        print("Using worker pool: %s" % str(worker_pool))

    if args.out_models_dir and not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    start = time.time()
    work_items = []
    for (h, hyperparameters) in enumerate(hyperparameters_lst):
        n_models = None
        if 'n_models' in hyperparameters:
            n_models = hyperparameters.pop("n_models")
        if args.n_models:
            n_models = args.n_models
        if not n_models:
            raise ValueError("Specify --ensemble-size or n_models hyperparameter")

        if args.max_epochs:
            hyperparameters['max_epochs'] = args.max_epochs

        for (i, allele) in enumerate(df.allele.unique()):
            for model_group in range(n_models):
                work_dict = {
                    'model_group': model_group,
                    'n_models': n_models,
                    'allele_num': i,
                    'n_alleles': len(alleles),
                    'hyperparameter_set_num': h,
                    'num_hyperparameter_sets': len(hyperparameters_lst),
                    'allele': allele,
                    'data': None,  # subselect from GLOBAL_DATA["train_data"]
                    'hyperparameters': hyperparameters,
                    'verbose': args.verbosity,
                    'predictor': predictor if not worker_pool else None,
                    'save_to': args.out_models_dir if not worker_pool else None,
                }
                work_items.append(work_dict)

    if worker_pool:
        print("Processing %d work items in parallel." % len(work_items))

        # We sort here so the predictors are in order of hyperparameter set num.
        # This is convenient so that the neural networks get merged for each
        # allele in the same order.
        predictors = [
            predictor for (_, predictor)
            in sorted(
                tqdm.tqdm(
                    worker_pool.imap_unordered(
                        train_model_entrypoint, work_items, chunksize=1),
                    total=len(work_items)),
                key=lambda pair: pair[0])
        ]

        print("Merging %d predictors fit in parallel." % (len(predictors)))
        predictor = Class1AffinityPredictor.merge([predictor] + predictors)
        print("Saving merged predictor to: %s" % args.out_models_dir)
        predictor.save(args.out_models_dir)
    else:
        # Run in serial. In this case, every worker is passed the same predictor,
        # which it adds models to, so no merging is required. It also saves
        # as it goes so no saving is required at the end.
        for _ in tqdm.trange(len(work_items)):
            item = work_items.pop(0)  # want to keep freeing up memory
            (_, work_predictor) = train_model_entrypoint(item)
            assert work_predictor is predictor
        assert not work_items

    print("*" * 30)
    training_time = time.time() - start
    print("Trained affinity predictor with %d networks in %0.2f min." % (
        len(predictor.neural_networks), training_time / 60.0))
    print("*" * 30)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()
        worker_pool = None

    start = time.time()
    if args.percent_rank_calibration_num_peptides_per_length > 0:
        alleles = list(predictor.supported_alleles)

        print("Performing percent rank calibration. Encoding peptides.")
        encoded_peptides = predictor.calibrate_percentile_ranks(
            alleles=[],  # don't actually do any calibration, just return peptides
            num_peptides_per_length=args.percent_rank_calibration_num_peptides_per_length)

        # Now we encode the peptides for each neural network, so the encoding
        # becomes cached.
        for network in predictor.neural_networks:
            network.peptides_to_network_input(encoded_peptides)
        assert encoded_peptides.encoding_cache  # must have cached the encoding
        print("Finished encoding peptides for percent ranks in %0.2f sec." % (
            time.time() - start))
        print("Calibrating percent rank calibration for %d alleles." % len(alleles))

        if args.num_jobs[-1] == 1:
            # Serial run
            print("Running in serial.")
            worker_pool = None
            results = (
                calibrate_percentile_ranks(
                    allele=allele,
                    predictor=predictor,
                    peptides=encoded_peptides)
                for allele in alleles)
        else:
            # Parallel run
            # Store peptides in global variable so they are in shared memory
            # after fork, instead of needing to be pickled.
            GLOBAL_DATA["calibration_peptides"] = encoded_peptides
            worker_pool = Pool(
                processes=(
                    args.num_jobs[-1]
                    if args.num_jobs[-1] else None))
            print("Using worker pool: %s" % str(worker_pool))
            results = worker_pool.imap_unordered(
                partial(
                    calibrate_percentile_ranks,
                    predictor=predictor),
                alleles,
                chunksize=1)

        for result in tqdm.tqdm(results, total=len(alleles)):
            predictor.allele_to_percent_rank_transform.update(result)

        print("Done calibrating %d additional alleles." % len(alleles))
        predictor.save(args.out_models_dir, model_names_to_write=[])

    percent_rank_calibration_time = time.time() - start

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Train time: %0.2f min. Percent rank calibration time: %0.2f min." % (
        training_time / 60.0, percent_rank_calibration_time / 60.0))
    print("Predictor written to: %s" % args.out_models_dir)


def train_model_entrypoint(item):
    return train_model(**item)


def train_model(
        model_group,
        n_models,
        allele_num,
        n_alleles,
        hyperparameter_set_num,
        num_hyperparameter_sets,
        allele,
        data,
        hyperparameters,
        verbose,
        predictor,
        save_to):

    if predictor is None:
        predictor = Class1AffinityPredictor()

    if data is None:
        full_data = GLOBAL_DATA["train_data"]
        data = full_data.loc[full_data.allele == allele]

    progress_preamble = (
        "[%2d / %2d hyperparameters] "
        "[%4d / %4d alleles] "
        "[%2d / %2d replicates]: %s " % (
            hyperparameter_set_num + 1,
            num_hyperparameter_sets,
            allele_num + 1,
            n_alleles,
            model_group + 1,
            n_models,
            allele))

    train_data = data.sample(frac=1.0)
    (model,) = predictor.fit_allele_specific_predictors(
        n_models=1,
        architecture_hyperparameters_list=[hyperparameters],
        allele=allele,
        peptides=train_data.peptide.values,
        affinities=train_data.measurement_value.values,
        inequalities=(
            train_data.measurement_inequality.values
            if "measurement_inequality" in train_data.columns else None),
        models_dir_for_save=save_to,
        progress_preamble=progress_preamble,
        verbose=verbose)

    if allele_num == 0 and model_group == 0:
        # For the first model for the first allele, print the architecture.
        print("*** HYPERPARAMETER SET %d***" %
              (hyperparameter_set_num + 1))
        pprint(hyperparameters)
        print("*** ARCHITECTURE FOR HYPERPARAMETER SET %d***" %
              (hyperparameter_set_num + 1))
        model.network(borrow=True).summary()

    return (hyperparameter_set_num, predictor)


def calibrate_percentile_ranks(allele, predictor, peptides=None):
    """
    Private helper function.
    """
    global GLOBAL_DATA
    if peptides is None:
        peptides = GLOBAL_DATA["calibration_peptides"]
    predictor.calibrate_percentile_ranks(
        peptides=peptides,
        alleles=[allele])
    return {
        allele: predictor.allele_to_percent_rank_transform[allele],
    }


if __name__ == '__main__':
    run()

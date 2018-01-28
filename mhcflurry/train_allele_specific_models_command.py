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

import pandas
import yaml
from mhcnames import normalize_allele_name
import tqdm  # progress bar

from .class1_affinity_predictor import Class1AffinityPredictor
from .common import configure_logging, set_keras_backend

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
    "--parallelization-num-jobs",
    default=1,
    type=int,
    metavar="N",
    help="Parallelization jobs. Experimental. "
    "Set to 1 for serial run. Set to 0 to use number of cores. "
    "Default: %(default)s.")
parser.add_argument(
    "--backend",
    choices=("tensorflow-gpu", "tensorflow-cpu"),
    help="Keras backend. If not specified will use system default.")

def run(argv=sys.argv[1:]):
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

    predictor = Class1AffinityPredictor()
    if args.parallelization_num_jobs == 1:
        # Serial run
        worker_pool = None
    else:
        worker_pool = Pool(
            processes=(
                args.parallelization_num_jobs
                if args.parallelization_num_jobs else None))
        print("Using worker pool: %s" % str(worker_pool))

    if args.out_models_dir and not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    start = time.time()
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

        work_items = []
        total_data_to_train_on = 0
        for (i, (allele, sub_df)) in enumerate(df.groupby("allele")):
            total_data_to_train_on += len(sub_df) * n_models
            for model_group in range(n_models):
                work_dict = {
                    'model_group': model_group,
                    'n_models': n_models,
                    'allele_num': i,
                    'n_alleles': len(alleles),
                    'hyperparameter_set_num': h,
                    'num_hyperparameter_sets': len(hyperparameters_lst),
                    'allele': allele,
                    'data': sub_df,
                    'hyperparameters': hyperparameters,
                    'verbose': args.verbosity,
                    'predictor': predictor if not worker_pool else None,
                    'save_to': args.out_models_dir if not worker_pool else None,
                }
                work_items.append(work_dict)

        if worker_pool:
            print("Processing %d work items in parallel." % len(work_items))

            predictors = list(
                tqdm.tqdm(
                    worker_pool.imap_unordered(
                        work_entrypoint, work_items, chunksize=1),
                    ascii=True,
                    total=len(work_items)))

            print("Merging %d predictors fit in parallel." % (len(predictors)))
            predictor = Class1AffinityPredictor.merge([predictor] + predictors)
            print("Saving merged predictor to: %s" % args.out_models_dir)
            predictor.save(args.out_models_dir)
        else:
            # Run in serial. In this case, every worker is passed the same predictor,
            # which it adds models to, so no merging is required. It also saves
            # as it goes so no saving is required at the end.
            start = time.time()
            for _ in tqdm.trange(len(work_items)):
                item = work_items.pop(0)  # want to keep freeing up memory
                work_predictor = work_entrypoint(item)
                assert work_predictor is predictor
            assert not work_items

    print("*" * 30)
    training_time = time.time() - start
    print("Trained affinity predictor with %d networks in %0.2f min." % (
        len(predictor.neural_networks), training_time / 60.0))
    print("*" * 30)

    if args.percent_rank_calibration_num_peptides_per_length > 0:
        print("Performing percent rank calibration.")
        start = time.time()
        predictor.calibrate_percentile_ranks(
            num_peptides_per_length=args.percent_rank_calibration_num_peptides_per_length,
            worker_pool=worker_pool)
        percent_rank_calibration_time = time.time() - start
        print("Finished calibrating percent ranks in %0.2f sec." % (
            percent_rank_calibration_time))
        predictor.save(args.out_models_dir, model_names_to_write=[])

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Train time: %0.2f min. Percent rank calibration time: %0.2f min." % (
        training_time / 60.0, percent_rank_calibration_time / 60.0))
    print("Predictor written to: %s" % args.out_models_dir)


def work_entrypoint(item):
    return process_work(**item)


def process_work(
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
        print("*** ARCHITECTURE FOR HYPERPARAMETER SET %d***" %
              (hyperparameter_set_num + 1))
        model.network(borrow=True).summary()

    return predictor


if __name__ == '__main__':
    run()

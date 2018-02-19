"""
Train Class1 single allele models.
"""
import argparse
import os
import signal
import sys
import time
import traceback
import random
from functools import partial

import numpy
import pandas
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from mhcnames import normalize_allele_name
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_affinity_predictor import Class1AffinityPredictor
from .common import configure_logging, set_keras_backend
from .parallelism import (
    make_worker_pool, cpu_count, call_wrapped, call_wrapped_kwargs)
from .hyperparameters import HyperparameterDefaults
from .allele_encoding import AlleleEncoding


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
    "--allele-sequences",
    metavar="FILE.csv",
    help="Allele sequences file. Used for computing allele similarity matrix.")
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
    choices=("tensorflow-gpu", "tensorflow-cpu", "tensorflow-default"),
    help="Keras backend. If not specified will use system default.")
parser.add_argument(
    "--gpus",
    type=int,
    metavar="N",
    help="Number of GPUs to attempt to parallelize across. Requires running "
    "in parallel.")
parser.add_argument(
    "--max-workers-per-gpu",
    type=int,
    metavar="N",
    default=1000,
    help="Maximum number of workers to assign to a GPU. Additional tasks will "
    "run on CPU.")
parser.add_argument(
    "--save-interval",
    type=float,
    metavar="N",
    default=60,
    help="Write models to disk every N seconds. Only affects parallel runs; "
    "serial runs write each model to disk as it is trained.")
parser.add_argument(
    "--max-tasks-per-worker",
    type=int,
    metavar="N",
    default=None,
    help="Restart workers after N tasks. Workaround for tensorflow memory "
    "leaks. Requires Python >=3.2.")


TRAIN_DATA_HYPERPARAMETER_DEFAULTS = HyperparameterDefaults(
    subset="all",
    pretrain_min_points=None,
)


def run(argv=sys.argv[1:]):
    global GLOBAL_DATA

    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    args.out_models_dir = os.path.abspath(args.out_models_dir)

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

    if args.ignore_inequalities and "measurement_inequality" in df.columns:
        print("Dropping measurement_inequality column")
        del df["measurement_inequality"]

    # Allele counts are in terms of quantitative data only.
    allele_counts = (
        df.loc[df.measurement_type == "quantitative"].allele.value_counts())

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
    GLOBAL_DATA["args"] = args

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    predictor = Class1AffinityPredictor()
    serial_run = args.num_jobs[0] == 1

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

        hyperparameters['train_data'] = TRAIN_DATA_HYPERPARAMETER_DEFAULTS.with_defaults(
            hyperparameters.get('train_data', {})
        )

        if hyperparameters['train_data']['pretrain_min_points'] and (
                'allele_similarity_matrix' not in GLOBAL_DATA):
            print("Generating allele similarity matrix.")
            if not args.allele_sequences:
                parser.error(
                    "Allele sequences required when using pretrain_min_points")
            allele_sequences = pandas.read_csv(
                args.allele_sequences,
                index_col="allele")
            print("Read %d allele sequences" % len(allele_sequences))
            allele_sequences = allele_sequences.loc[
                allele_sequences.index.isin(df.allele.unique())
            ]
            blosum_encoding = (
                AlleleEncoding(
                    allele_sequences.index.values,
                    allele_sequences.pseudosequence.to_dict())
                .fixed_length_vector_encoded_sequences("BLOSUM62"))
            allele_similarity_matrix = pandas.DataFrame(
                cosine_similarity(
                    blosum_encoding.reshape((len(allele_sequences), -1))),
                index=allele_sequences.index.values,
                columns=allele_sequences.index.values)
            GLOBAL_DATA['allele_similarity_matrix'] = allele_similarity_matrix
            print("Computed allele similarity matrix")
            print(allele_similarity_matrix)

        for (i, allele) in enumerate(df.allele.unique()):
            for model_num in range(n_models):
                work_dict = {
                    'n_models': 1,
                    'allele_num': i,
                    'n_alleles': len(alleles),
                    'hyperparameter_set_num': h,
                    'num_hyperparameter_sets': len(hyperparameters_lst),
                    'allele': allele,
                    'hyperparameters': hyperparameters,
                    'verbose': args.verbosity,
                    'progress_print_interval': None if not serial_run else 5.0,
                    'predictor': predictor if serial_run else None,
                    'save_to': args.out_models_dir if serial_run else None,
                }
                work_items.append(work_dict)

    start = time.time()
    if serial_run:
        # Serial run.
        print("Running in serial.")
        worker_pool = None
        if args.backend:
            set_keras_backend(args.backend)
    else:
        # Parallel run.
        num_workers = args.num_jobs[0] if args.num_jobs[0] else cpu_count()
        worker_init_kwargs = None
        if args.gpus:
            print("Attempting to round-robin assign each worker a GPU.")
            if args.backend != "tensorflow-default":
                print("Forcing keras backend to be tensorflow-default")
                args.backend = "tensorflow-default"

            gpu_assignments_remaining = dict((
                (gpu, args.max_workers_per_gpu) for gpu in range(args.gpus)
            ))
            worker_init_kwargs = []
            for worker_num in range(num_workers):
                if gpu_assignments_remaining:
                    # Use a GPU
                    gpu_num = sorted(
                        gpu_assignments_remaining,
                        key=lambda key: gpu_assignments_remaining[key])[0]
                    gpu_assignments_remaining[gpu_num] -= 1
                    if not gpu_assignments_remaining[gpu_num]:
                        del gpu_assignments_remaining[gpu_num]
                    gpu_assignment = [gpu_num]
                else:
                    # Use CPU
                    gpu_assignment = []

                worker_init_kwargs.append({
                    'gpu_device_nums': gpu_assignment,
                    'keras_backend': args.backend
                })
                print("Worker %d assigned GPUs: %s" % (
                    worker_num, gpu_assignment))

        worker_pool = make_worker_pool(
            processes=num_workers,
            initializer=worker_init,
            initializer_kwargs_per_process=worker_init_kwargs,
            max_tasks_per_worker=args.max_tasks_per_worker)

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
                    args.out_models_dir, model_names_to_write=new_model_names)
                print(
                    "Saved predictor (%d models total) including %d new models "
                    "in %0.2f sec to %s" % (
                        len(predictor.neural_networks),
                        len(new_model_names),
                        time.time() - save_start,
                        args.out_models_dir))
                unsaved_predictors = []
                last_save_time = time.time()

        print("Saving final predictor to: %s" % args.out_models_dir)
        predictor.merge_in_place(unsaved_predictors)
        predictor.save(args.out_models_dir)  # write all models just to be sure
        print("Done.")

    else:
        # Run in serial. In this case, every worker is passed the same predictor,
        # which it adds models to, so no merging is required. It also saves
        # as it goes so no saving is required at the end.
        for _ in tqdm.trange(len(work_items)):
            item = work_items.pop(0)  # want to keep freeing up memory
            work_predictor = train_model(**item)
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

            worker_pool = make_worker_pool(
                processes=(
                    args.num_jobs[-1]
                    if args.num_jobs[-1] else None),
                max_tasks_per_worker=args.max_tasks_per_worker)

            results = worker_pool.imap_unordered(
                partial(
                    partial(call_wrapped, calibrate_percentile_ranks),
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


def alleles_by_similarity(allele):
    global GLOBAL_DATA
    allele_similarity = GLOBAL_DATA['allele_similarity_matrix']
    if allele not in allele_similarity.columns:
        # Use random alleles
        print("No similar alleles for: %s" % allele)
        return [allele] + list(allele_similarity.columns.to_series().sample(frac=1.0))
    return (
        allele_similarity[allele] + (
        allele_similarity.index == allele)  # force that we return specified allele first
    ).sort_values(ascending=False).index.tolist()


def train_model(
        n_models,
        allele_num,
        n_alleles,
        hyperparameter_set_num,
        num_hyperparameter_sets,
        allele,
        hyperparameters,
        verbose,
        progress_print_interval,
        predictor,
        save_to):

    if predictor is None:
        predictor = Class1AffinityPredictor()

    pretrain_min_points = hyperparameters['train_data']['pretrain_min_points']

    data = GLOBAL_DATA["train_data"]

    subset = hyperparameters.get("train_data", {}).get("subset", "all")
    if subset == "quantitative":
        data = data.loc[
            data.measurement_type == "quantitative"
        ]
    elif subset == "all":
        pass
    else:
        raise ValueError("Unsupported subset: %s" % subset)

    data_size_by_allele = data.allele.value_counts()

    if pretrain_min_points:
        similar_alleles = alleles_by_similarity(allele)
        alleles = []
        while not alleles or data_size_by_allele.loc[alleles].sum() < pretrain_min_points:
            alleles.append(similar_alleles.pop(0))
        print(alleles)
        data = data.loc[data.allele.isin(alleles)]
        assert len(data) >= pretrain_min_points, (len(data), pretrain_min_points)
        train_rounds = (data.allele == allele).astype(int).values
    else:
        train_rounds = None
        data = data.loc[data.allele == allele]

    progress_preamble = (
        "[%2d / %2d hyperparameters] "
        "[%4d / %4d alleles] %s " % (
            hyperparameter_set_num + 1,
            num_hyperparameter_sets,
            allele_num + 1,
            n_alleles,
            allele))

    train_data = data.sample(frac=1.0)
    predictor.fit_allele_specific_predictors(
        n_models=n_models,
        architecture_hyperparameters_list=[hyperparameters],
        allele=allele,
        peptides=train_data.peptide.values,
        affinities=train_data.measurement_value.values,
        inequalities=(
            train_data.measurement_inequality.values
            if "measurement_inequality" in train_data.columns else None),
        train_rounds=train_rounds,
        models_dir_for_save=save_to,
        progress_preamble=progress_preamble,
        progress_print_interval=progress_print_interval,
        verbose=verbose)

    return predictor


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


def worker_init(keras_backend=None, gpu_device_nums=None):
    # Each worker needs distinct random numbers
    numpy.random.seed()
    random.seed()
    if keras_backend or gpu_device_nums:
        print("WORKER pid=%d assigned GPU devices: %s" % (
            os.getpid(), gpu_device_nums))
        set_keras_backend(
            keras_backend, gpu_device_nums=gpu_device_nums)

if __name__ == '__main__':
    run()

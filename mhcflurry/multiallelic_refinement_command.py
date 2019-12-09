"""
Refine pan-allele predictors using multiallelic mass spec.
"""
import argparse
import os
import signal
import sys
import time
import traceback
import hashlib
import yaml
import pickle

import numpy
import pandas

import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_presentation_neural_network import Class1PresentationNeuralNetwork
from .class1_presentation_predictor import Class1PresentationPredictor
from .allele_encoding import MultipleAlleleEncoding
from .common import configure_logging
from .local_parallelism import (
    worker_pool_with_gpu_assignments_from_args,
    add_local_parallelism_args)
from .cluster_parallelism import (
    add_cluster_parallelism_args,
    cluster_results_from_args)


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing us to share large data with the workers
# via shared memory.
GLOBAL_DATA = {}


parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "--multiallelic-data",
    metavar="FILE.csv",
    required=False,
    help="Multiallelic mass spec data.")
parser.add_argument(
    "--monoallelic-data",
    metavar="FILE.csv",
    required=False,
    help="Affinity meaurements and monoallelic mass spec data.")
parser.add_argument(
    "--models-dir",
    metavar="DIR",
    required=True,
    help="Directory to read models")
parser.add_argument(
    "--hyperparameters",
    metavar="FILE.json",
    help="presentation predictor hyperparameters")
parser.add_argument(
    "--out-affinity-predictor-dir",
    metavar="DIR",
    required=True,
    help="Directory to write refined models")
parser.add_argument(
    "--out-presentation-predictor-dir",
    metavar="DIR",
    required=True,
    help="Directory to write preentation predictor")
parser.add_argument(
    "--max-models",
    type=int,
    default=None)
parser.add_argument(
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
    default=0)

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)


def run(argv=sys.argv[1:]):
    global GLOBAL_DATA

    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    hyperparameters = yaml.load(open(args.hyperparameters))

    args.out_affinity_predictor_dir = os.path.abspath(
        args.out_affinity_predictor_dir)
    args.out_presentation_predictor_dir = os.path.abspath(
        args.out_presentation_predictor_dir)

    configure_logging(verbose=args.verbosity > 1)

    multiallelic_df = pandas.read_csv(args.multiallelic_data)
    print("Loaded multiallelic data: %s" % (str(multiallelic_df.shape)))

    monoallelic_df = pandas.read_csv(args.monoallelic_data)
    print("Loaded monoallelic data: %s" % (str(monoallelic_df.shape)))

    input_predictor = Class1AffinityPredictor.load(
        args.models_dir, optimization_level=0, max_models=args.max_models)
    print("Loaded: %s" % input_predictor)

    sample_table = multiallelic_df.drop_duplicates(
        "sample_id").set_index("sample_id").loc[
        multiallelic_df.sample_id.unique()
    ]
    grouped = multiallelic_df.groupby("sample_id").nunique()
    for col in sample_table.columns:
        if (grouped[col] > 1).any():
            del sample_table[col]
    sample_table["alleles"] = sample_table.hla.str.split()

    fold_cols = [c for c in monoallelic_df if c.startswith("fold_")]
    num_folds = len(fold_cols)
    if num_folds <= 1:
        raise ValueError("Too few folds: ", num_folds)

    def make_train_peptide_hash(sub_df):
        train_peptide_hash = hashlib.sha1()
        for peptide in sorted(sub_df.peptide.values):
            train_peptide_hash.update(peptide.encode())
        return train_peptide_hash.hexdigest()

    work_items = []
    for model in input_predictor.class1_pan_allele_models:
        training_info = model.fit_info[-1]['training_info']
        fold_num = training_info['fold_num']
        assert num_folds == training_info['num_folds']
        fold_col = "fold_%d" % fold_num
        observed_hash = make_train_peptide_hash(
            monoallelic_df.loc[monoallelic_df[fold_col] == 1])
        saved_hash = training_info['train_peptide_hash']
        #numpy.testing.assert_equal(observed_hash, saved_hash)
        work_items.append({
            "work_item_num": len(work_items),
            "affinity_model": model,
            "fold_num": fold_num,
        })

    work_items_dict = dict((item['work_item_num'], item) for item in work_items)

    GLOBAL_DATA["monoallelic_data"] = monoallelic_df
    GLOBAL_DATA["multiallelic_data"] = multiallelic_df
    GLOBAL_DATA["multiallelic_sample_table"] = sample_table
    GLOBAL_DATA["hyperparameters"] = hyperparameters
    GLOBAL_DATA["allele_to_sequence"] = input_predictor.allele_to_sequence

    out_dirs = [
        args.out_affinity_predictor_dir,
        args.out_presentation_predictor_dir
    ]

    for out in out_dirs:
        if not os.path.exists(out):
            print("Attempting to create directory:", out)
            os.mkdir(out)
            print("Done.")

    metadata_dfs = {
        "monoallelic_train_data": monoallelic_df,
        "multiallelic_train_data": multiallelic_df,
    }

    affinity_models = []
    presentation_models = []

    serial_run = not args.cluster_parallelism and args.num_jobs == 0
    worker_pool = None
    start = time.time()
    if serial_run:
        # Serial run
        print("Running in serial.")
        results = (refine_model(**item) for item in work_items)
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=refine_model,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="pickle")
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None

        print("Processing %d work items in parallel." % len(work_items))
        assert not serial_run

        # Parallel run
        results = worker_pool.imap_unordered(
            do_refine_model_task,
            work_items,
            chunksize=1)

    for result in tqdm.tqdm(results, total=len(work_items)):
        work_item_num = result['work_item_num']
        work_item = work_items_dict[work_item_num]
        affinity_model = work_item['affinity_model']
        presentation_model = pickle.loads(result['presentation_model'])
        presentation_model.copy_weights_to_affinity_model(affinity_model)
        affinity_models.append(affinity_model)
        presentation_models.append(presentation_model)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Done model fitting. Writing predictors.")

    result_affinity_predictor = Class1AffinityPredictor(
        class1_pan_allele_models=affinity_models,
        allele_to_sequence=input_predictor.allele_to_sequence)
    result_affinity_predictor.save(args.out_affinity_predictor_dir)
    print("Wrote", args.out_affinity_predictor_dir)

    result_presentation_predictor = Class1PresentationPredictor(
        models=presentation_models,
        allele_to_sequence=input_predictor.allele_to_sequence,
        metadata_dataframes=metadata_dfs)
    result_presentation_predictor.save(args.out_presentation_predictor_dir)
    print("Wrote", args.out_presentation_predictor_dir)


def do_refine_model_task(item, constant_data=GLOBAL_DATA):
    return refine_model(constant_data=constant_data, **item)


def refine_model(
        work_item_num, affinity_model, fold_num, constant_data=GLOBAL_DATA):
    """
    Refine a model.
    """
    monoallelic_df = constant_data["monoallelic_data"]
    multiallelic_df = constant_data["multiallelic_data"]
    sample_table = constant_data["multiallelic_sample_table"]
    hyperparameters = constant_data["hyperparameters"]
    allele_to_sequence = constant_data["allele_to_sequence"]

    fold_col = "fold_%d" % fold_num

    multiallelic_train_df = multiallelic_df[
        ["sample_id", "peptide", "hit"]
    ].rename(columns={"hit": "label"}).copy()
    multiallelic_train_df["is_affinity"] = False
    multiallelic_train_df["validation_weight"] = 1.0

    monoallelic_train_df = monoallelic_df[
        ["peptide", "allele", "measurement_inequality", "measurement_value"]
    ].copy()
    monoallelic_train_df["label"] = monoallelic_train_df["measurement_value"]
    del monoallelic_train_df["measurement_value"]
    monoallelic_train_df["is_affinity"] = True

    # We force all validation affinities to be from the validation set used
    # originally to train the predictor. To ensure proportional sampling between
    # the affinity and multiallelic mass spec data, we set their weight to
    # as follows.
    monoallelic_train_df["validation_weight"] = (
        (monoallelic_df[fold_col] == 0).astype(float) * (
            (monoallelic_df[fold_col] == 1).sum() /
            (monoallelic_df[fold_col] == 0).sum()))

    combined_train_df = pandas.concat(
        [multiallelic_train_df, monoallelic_train_df],
        ignore_index=True,
        sort=False)

    allele_encoding = MultipleAlleleEncoding(
        experiment_names=multiallelic_train_df.sample_id.values,
        experiment_to_allele_list=sample_table.alleles.to_dict(),
        allele_to_sequence=allele_to_sequence,
    )
    allele_encoding.append_alleles(monoallelic_train_df.allele.values)
    allele_encoding = allele_encoding.compact()

    presentation_model = Class1PresentationNeuralNetwork(**hyperparameters)
    presentation_model.load_from_class1_neural_network(affinity_model)
    presentation_model.fit(
        peptides=combined_train_df.peptide.values,
        labels=combined_train_df.label.values,
        allele_encoding=allele_encoding,
        affinities_mask=combined_train_df.is_affinity.values,
        inequalities=combined_train_df.measurement_inequality.values,
        validation_weights=combined_train_df.validation_weight.values)

    return {
        'work_item_num': work_item_num,

        # We pickle it here so it always gets pickled, even when running in one
        # process. This prevents tensorflow errors when using thread-level
        # parallelism.
        'presentation_model': pickle.dumps(
            presentation_model, pickle.HIGHEST_PROTOCOL),
    }


if __name__ == '__main__':
    run()

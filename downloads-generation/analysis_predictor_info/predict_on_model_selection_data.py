"""
Evaluate affinity predictor on its held-out model selection data, using only
the individual models that were not trained on each particular data point.
"""
import sys
import argparse
import os
import numpy
import time
import collections
from functools import partial

import pandas
import tqdm

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from mhcflurry.common import configure_logging
from mhcflurry.local_parallelism import (
    add_local_parallelism_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped_kwargs)
from mhcflurry.cluster_parallelism import (
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
    "predictor",
    metavar="DIR",
    help="Class 1 affinity predictor to use")
parser.add_argument(
    "--data",
    metavar="CSV",
    help="Model selection data. If not specified will guess based on affinity "
    "predictor path")
parser.add_argument(
    "--out",
    metavar="CSV",
    required=True,
    help="File to write with predictions")

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)


def do_predict(predictor, key, sub_df, constant_data=None):
    import tqdm
    tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

    prediction = predictor.predict(sub_df.peptide, sub_df.allele, throw=False)
    return {
        "key": key,
        "index": sub_df.index,
        "prediction": prediction,
    }


def run():
    import mhcflurry

    args = parser.parse_args(sys.argv[1:])

    configure_logging()

    serial_run = not args.cluster_parallelism and args.num_jobs == 0

    if not args.data:
        args.data = os.path.join(args.predictor, 'model_selection_data.csv.bz2')
        print("Defaulting data to: ", args.data)

    data_df = pandas.read_csv(args.data)
    print("Read %d rows:" % len(data_df))
    print(data_df)

    fold_cols = [col for col in data_df.columns if col.startswith("fold_")]
    print("Fold cols", fold_cols)
    assert len(fold_cols) > 1

    eval_df = data_df.loc[
        data_df[fold_cols].sum(1) < len(fold_cols)
    ].copy()

    print("Reduced to data held-out at least once: ", len(eval_df))

    predictor = mhcflurry.Class1AffinityPredictor.load(
        args.predictor, optimization_level=0)
    print("Loaded predictor", predictor)

    fold_to_ensemble = collections.defaultdict(list)
    for n in predictor.neural_networks:
        fold = n.fit_info[-1]['training_info']['fold_num']
        fold_to_ensemble[fold].append(n)
    print("Constructed fold_to_ensemble", fold_to_ensemble)

    eval_df["ensemble_key"] = (
        (~eval_df[fold_cols]).astype(str) + "_"
    ).sum(1).str.strip("_")
    print("Established ensemble keys:")
    print(eval_df.ensemble_key.value_counts())

    def predictor_for_ensemble_key(key_string):
        indicators = [eval(s) for s in key_string.split("_")]
        ensemble = []
        for fold, indicator in enumerate(indicators):
            if indicator:
                ensemble.extend(fold_to_ensemble[fold])
        pred = mhcflurry.Class1AffinityPredictor(
            class1_pan_allele_models=ensemble,
            allele_to_sequence=predictor.allele_to_sequence)
        return pred

    tasks = []
    for (key, sub_df) in eval_df.groupby("ensemble_key"):
        print(key)
        pred = predictor_for_ensemble_key(key)
        assert len(pred.neural_networks) > 0
        eval_df.loc[
            sub_df.index,
            "ensemble_size"
        ] = len(pred.neural_networks)
        tasks.append({
            "key": key,
            "predictor": pred,
            "sub_df": sub_df[["peptide", "allele"]].copy()
        })

    worker_pool = None
    start = time.time()

    if serial_run:
        # Serial run
        print("Running in serial.")
        results = (
            do_predict(**task) for task in tasks)
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=do_predict,
            work_items=tasks,
            constant_data=GLOBAL_DATA,
            input_serialization_method="dill",
            result_serialization_method="pickle",
            clear_constant_data=False)
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None
        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, do_predict),
            tasks,
            chunksize=1)

    print("Reading results")

    for worker_result in tqdm.tqdm(results, total=len(tasks)):
        print("Received worker result:", worker_result['key'])
        print(worker_result)

        eval_df.loc[
            worker_result['index'],
            "prediction"
        ] = worker_result["prediction"]

    print("Received all results in %0.2f sec" % (time.time() - start))

    eval_df.to_csv(args.out, index=False)
    print("Wrote: ", args.out)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()


if __name__ == '__main__':
    run()

"""
Train Class1 presentation models.
"""
import argparse
import os
import signal
import sys
import time
import traceback
from functools import partial

import numpy
import pandas
import tqdm  # progress bar

from .class1_processing_predictor import Class1ProcessingPredictor
from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_presentation_predictor import Class1PresentationPredictor
from .common import configure_logging, write_generate_sh
from .local_parallelism import (
    add_local_parallelism_args,
    attach_constant_data_to_work_items_if_needed,
    call_wrapped_kwargs,
    resolve_local_parallelism_args,
    worker_pool_with_gpu_assignments_from_args,
)

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

GLOBAL_DATA = {}
PRESENTATION_FEATURE_WORKER_GB = 12.0

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
    "--affinity-predictor",
    metavar="DIR",
    required=True,
    help="Affinity predictor models dir")
parser.add_argument(
    "--processing-predictor-with-flanks",
    metavar="DIR",
    required=True,
    help="Processing predictor with flanks")
parser.add_argument(
    "--processing-predictor-without-flanks",
    metavar="DIR",
    required=True,
    help="Processing predictor without flanks")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Default: %(default)s",
    default=1)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Launch python debugger on error")
parser.add_argument(
    "--hla-column",
    default="hla",
    help="Column in data giving space-separated MHC I alleles")
parser.add_argument(
    "--target-column",
    default="hit",
    help="Column in data giving hit (1) vs decoy (0)")
parser.add_argument(
    "--feature-chunk-size",
    type=int,
    default=250000,
    metavar="N",
    help=(
        "Rows per parallel presentation feature-prediction task. Larger "
        "chunks reduce scheduling overhead but increase worker memory. "
        "Default: %(default)s"))

add_local_parallelism_args(parser)

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
    resolve_local_parallelism_args(
        args,
        per_worker_gb=PRESENTATION_FEATURE_WORKER_GB,
    )

    df = pandas.read_csv(
        args.data,
        dtype={"sample_id": str},
        low_memory=False,
    )
    print("Loaded training data: %s" % (str(df.shape)))
    df = df.loc[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    df["experiment_id"] = df[args.hla_column]
    experiment_to_alleles = dict((
        key, key.split()) for key in df.experiment_id.unique())
    if args.num_jobs:
        df = df.sort_values("experiment_id", kind="stable").reset_index(drop=True)

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    affinity_predictor = Class1AffinityPredictor.load(
        args.affinity_predictor,
        optimization_level=0)
    processing_predictor_with_flanks = Class1ProcessingPredictor.load(
        args.processing_predictor_with_flanks)
    processing_predictor_without_flanks = Class1ProcessingPredictor.load(
        args.processing_predictor_without_flanks)

    print("Loaded affinity predictor", affinity_predictor)
    print(
        "Loaded processing_predictor_with_flanks",
        processing_predictor_with_flanks)
    print("Loaded processing_predictor_without_flanks",
        processing_predictor_without_flanks)

    predictor = Class1PresentationPredictor(
        affinity_predictor=affinity_predictor,
        processing_predictor_with_flanks=processing_predictor_with_flanks,
        processing_predictor_without_flanks=processing_predictor_without_flanks)

    # We want to predict using an optimized Class1AffinityPredictor but
    # save the presentation models using an un-optimized Class1AffinityPredictor,
    # since the optimized (merged) network is only needed at inference time.
    print("Before fit: saving affinity and processing predictors.")
    predictor.save(
        args.out_models_dir,
        write_affinity_predictor = True,
        write_processing_predictor = True,
        write_weights = False,
        write_percent_ranks = False,
        write_info = False,
        write_metdata = False)
    print("Done writing: ", args.out_models_dir)

    print("Optimizing affinity predictor.")
    optimized = affinity_predictor.optimize()
    print("Optimization performed: ", optimized)

    print("Fitting.")
    start = time.time()
    if args.num_jobs:
        scores = predict_features_parallel(
            args=args,
            predictor=predictor,
            df=df,
            experiment_to_alleles=experiment_to_alleles,
        )
        predictor.fit_from_scores(
            targets=df[args.target_column].values,
            affinities=scores["affinity"],
            processing_scores_by_model=scores["processing_scores_by_model"],
            verbose=args.verbosity)
    else:
        predictor.fit(
            targets=df[args.target_column].values,
            peptides=df.peptide.values,
            alleles=experiment_to_alleles,
            sample_names=df.experiment_id,
            n_flanks=df.n_flank.values,
            c_flanks=df.c_flank.values,
            verbose=args.verbosity)
    print("Done fitting in", time.time() - start, "seconds")

    print("Saving weights and metadata.")
    predictor.save(
        args.out_models_dir,
        write_affinity_predictor = False,
        write_processing_predictor = False,
        write_weights = True,
        write_percent_ranks = True,
        write_info = True,
        write_metdata = True)
    write_generate_sh(args.out_models_dir)
    print("Wrote", args.out_models_dir)


def predict_features_parallel(args, predictor, df, experiment_to_alleles):
    """
    Predict BA/AP features in local worker processes.

    Presentation training fits only logistic-regression weights, but feature
    generation is expensive: it runs the affinity predictor and both
    processing predictors over tens of millions of rows. Split by sample and
    row chunk so each worker can use its assigned GPU independently.
    """
    global GLOBAL_DATA

    work_items = make_feature_work_items(df, args.feature_chunk_size)
    print(
        "Predicting presentation features in parallel: %d rows, %d chunks, "
        "num_jobs=%d, gpus=%d, max_workers_per_gpu=%s" % (
            len(df),
            len(work_items),
            args.num_jobs,
            args.gpus,
            args.max_workers_per_gpu,
        )
    )

    include_without_flanks = (
        predictor.processing_predictor_without_flanks is not None
    )
    include_with_flanks = (
        predictor.processing_predictor_with_flanks is not None
        and "n_flank" in df
        and "c_flank" in df
    )
    for item in work_items:
        item["include_without_flanks"] = include_without_flanks
        item["include_with_flanks"] = include_with_flanks

    GLOBAL_DATA.clear()
    GLOBAL_DATA.update({
        "predictor": predictor,
        "data": df,
        "experiment_to_alleles": experiment_to_alleles,
    })

    worker_pool = None
    try:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None
        attach_constant_data_to_work_items_if_needed(
            work_items, GLOBAL_DATA, worker_pool
        )

        affinity = numpy.empty(len(df), dtype="float64")
        processing_scores_by_model = {}
        if include_without_flanks:
            processing_scores_by_model["without_flanks"] = numpy.empty(
                len(df), dtype="float64")
        if include_with_flanks:
            processing_scores_by_model["with_flanks"] = numpy.empty(
                len(df), dtype="float64")

        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, predict_feature_chunk),
            work_items,
            chunksize=1,
        )
        for result in tqdm.tqdm(results, total=len(work_items)):
            start = result["start"]
            end = result["end"]
            affinity[start:end] = result["affinity"]
            for (model_name, values) in result["processing_scores"].items():
                processing_scores_by_model[model_name][start:end] = values

        return {
            "affinity": affinity,
            "processing_scores_by_model": processing_scores_by_model,
        }
    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()


def make_feature_work_items(df, chunk_size):
    chunk_size = max(int(chunk_size or 0), 1)
    work_items = []
    for (sample_name, sub_df) in df.groupby("experiment_id", sort=False):
        if len(sub_df) == 0:
            continue
        first = int(sub_df.index[0])
        last = int(sub_df.index[-1])
        if last - first + 1 != len(sub_df):
            raise ValueError(
                "Presentation feature chunks require experiment_id-contiguous "
                "data. Sort by experiment_id before chunking.")
        for start in range(first, last + 1, chunk_size):
            end = min(start + chunk_size, last + 1)
            work_items.append({
                "chunk_num": len(work_items),
                "start": start,
                "end": end,
                "sample_name": sample_name,
            })
    return work_items


def predict_feature_chunk(
        chunk_num,
        start,
        end,
        sample_name,
        include_without_flanks,
        include_with_flanks,
        constant_data=GLOBAL_DATA):
    predictor = constant_data["predictor"]
    df = constant_data["data"].iloc[start:end]
    experiment_to_alleles = constant_data["experiment_to_alleles"]

    print(
        "Presentation feature chunk %d: rows [%d, %d), sample=%s" % (
            chunk_num, start, end, sample_name)
    )

    affinity_df = predictor.predict_affinity(
        peptides=df.peptide.values,
        alleles={sample_name: experiment_to_alleles[sample_name]},
        sample_names=numpy.repeat(sample_name, len(df)),
        include_affinity_percentile=False,
        verbose=0,
        model_kwargs={"batch_size": "auto"},
    )
    result = {
        "chunk_num": chunk_num,
        "start": start,
        "end": end,
        "affinity": affinity_df.affinity.values.astype("float64", copy=False),
        "processing_scores": {},
    }

    if include_without_flanks:
        result["processing_scores"]["without_flanks"] = (
            predictor.predict_processing(
                peptides=df.peptide.values,
                n_flanks=None,
                c_flanks=None,
                verbose=0,
                batch_size="auto",
            ).astype("float64", copy=False)
        )
    if include_with_flanks:
        result["processing_scores"]["with_flanks"] = (
            predictor.predict_processing(
                peptides=df.peptide.values,
                n_flanks=df.n_flank.values,
                c_flanks=df.c_flank.values,
                verbose=0,
                batch_size="auto",
            ).astype("float64", copy=False)
        )
    return result


if __name__ == '__main__':
    run()

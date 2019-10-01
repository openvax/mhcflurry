"""
"""
import argparse
import os
import signal
import sys
import time
import traceback
import collections
import math
from functools import partial

import numpy
import pandas

from mhcnames import normalize_allele_name
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.common import configure_logging, random_peptides, amino_acid_distribution
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
    "input_peptides",
    metavar="CSV",
    help="CSV file with 'peptide' column")
parser.add_argument(
    "--models-dir",
    metavar="DIR",
    required=True,
    help="Directory to read MHCflurry models")
parser.add_argument(
    "--allele",
    default=None,
    required=True,
    nargs="+",
    help="Alleles to predict")
parser.add_argument(
    "--chunk-size",
    type=int,
    default=100000,
    help="Num peptides per job. Default: %(default)s")
parser.add_argument(
    "--batch-size",
    type=int,
    default=4096,
    help="Keras batch size for predictions. Default: %(default)s")
parser.add_argument(
    "--reuse-results",
    metavar="DIR",
    help="Reuse results from DIR")
parser.add_argument(
    "--out",
    metavar="DIR",
    help="Write results to DIR")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
    default=0)
parser.add_argument(
    "--max-peptides",
    type=int,
    help="Max peptides to process. For debugging.",
    default=None)


add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)


def run(argv=sys.argv[1:]):
    global GLOBAL_DATA

    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    args.models_dir = os.path.abspath(args.models_dir)

    configure_logging(verbose=args.verbosity > 1)

    serial_run = not args.cluster_parallelism and args.num_jobs == 0

    # It's important that we don't trigger a Keras import here since that breaks
    # local parallelism (tensorflow backend). So we set optimization_level=0.
    predictor = Class1AffinityPredictor.load(
        args.models_dir,
        optimization_level=0,
    )

    alleles = [normalize_allele_name(a) for a in args.allele]
    alleles = sorted(set(alleles))

    peptides = pandas.read_csv(
        args.input_peptides, nrows=args.max_peptides).peptide.drop_duplicates()
    print("Filtering to valid peptides. Starting at: ", len(peptides))
    peptides = peptides[peptides.str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")]
    print("Filtered to: ", len(peptides))
    peptides = peptides.unique()
    num_peptides = len(peptides)

    print("Predictions for %d alleles x %d peptides." % (
        len(alleles), num_peptides))

    if not os.path.exists(args.out):
        print("Creating", args.out)
        os.mkdir(args.out)

    # Write peptide and allele lists to out dir.
    out_peptides = os.path.abspath(os.path.join(args.out, "peptides.csv"))
    pandas.DataFrame({"peptide": peptides}).to_csv(out_peptides, index=False)
    print("Wrote: ", out_peptides)
    allele_to_file_path = dict(
        (allele, "%s.npz" % (allele.replace("*", ""))) for allele in alleles)
    out_alleles = os.path.abspath(os.path.join(args.out, "alleles.csv"))
    pandas.DataFrame({
        'allele': alleles,
        'path': [allele_to_file_path[allele] for allele in alleles],
    }).to_csv(out_alleles, index=False)
    print("Wrote: ", out_alleles)

    num_chunks = int(math.ceil(len(peptides) / args.chunk_size))
    print("Splitting peptides into %d chunks" % num_chunks)
    peptide_chunks = numpy.array_split(peptides, num_chunks)

    GLOBAL_DATA["predictor"] = predictor
    GLOBAL_DATA["args"] = {
        'verbose': args.verbosity > 0,
        'model_kwargs': {
            'batch_size': args.batch_size,
        }
    }

    work_items = []
    for (chunk_index, chunk_peptides) in enumerate(peptide_chunks):
        work_item = {
            'alleles': alleles,
            'chunk_index': chunk_index,
            'peptides': chunk_peptides,
        }
        work_items.append(work_item)
    print("Work items: ", len(work_items))

    worker_pool = None
    start = time.time()
    if serial_run:
        # Serial run
        print("Running in serial.")
        results = (
            do_predictions(**item) for item in work_items)
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=do_predictions,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="dill",
            clear_constant_data=True)
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None
        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, do_predictions),
            work_items,
            chunksize=1)

    allele_to_chunk_index_to_predictions = {}
    for allele in alleles:
        allele_to_chunk_index_to_predictions[allele] = {}

    for (chunk_index, allele_to_predictions) in tqdm.tqdm(
            results, total=len(work_items)):
        for (allele, predictions) in allele_to_predictions.items():
            chunk_index_to_predictions = allele_to_chunk_index_to_predictions[
                allele
            ]
            assert chunk_index not in chunk_index_to_predictions
            chunk_index_to_predictions[chunk_index] = predictions

            if len(allele_to_chunk_index_to_predictions[allele]) == num_chunks:
                chunk_predictions = sorted(chunk_index_to_predictions.items())
                assert [i for (i, _) in chunk_predictions] == list(
                    range(num_chunks))
                predictions = numpy.concatenate([
                    predictions for (_, predictions) in chunk_predictions
                ])
                assert len(predictions) == num_peptides
                out_path = os.path.join(
                    args.out, allele.replace("*", "")) + ".npz"
                out_path = os.path.abspath(out_path)
                numpy.savez(out_path, predictions)
                print("Wrote:", out_path)

                del allele_to_chunk_index_to_predictions[allele]

    assert not allele_to_chunk_index_to_predictions, (
        "Not all results written: ", allele_to_chunk_index_to_predictions)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    prediction_time = time.time() - start
    print("Done generating predictions in %0.2f min." % (
        prediction_time / 60.0))


def do_predictions(chunk_index, peptides, alleles, constant_data=GLOBAL_DATA):
    return predict_for_allele(
        chunk_index,
        peptides,
        alleles,
        predictor=constant_data['predictor'],
        **constant_data["args"])


def predict_for_allele(
        chunk_index,
        peptides,
        alleles,
        predictor,
        verbose=False,
        model_kwargs={}):
    predictor.optimize(warn=False)  # since we loaded with optimization_level=0
    start = time.time()
    results = {}
    peptides = EncodableSequences.create(peptides)
    for allele in alleles:
        results[allele] = predictor.predict(
            peptides=peptides,
            allele=allele,
            throw=False,
            model_kwargs=model_kwargs).astype('float32')
    if verbose:
        print("Done predicting in", time.time() - start, "sec")
    return (chunk_index, results)


if __name__ == '__main__':
    run()

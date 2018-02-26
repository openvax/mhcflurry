"""
Calibrate percentile ranks for models. Runs in-place.
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
from .common import configure_logging
from .parallelism import (
    add_worker_pool_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped)


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing us to share large data with the workers
# via shared memory.
GLOBAL_DATA = {}

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--models-dir",
    metavar="DIR",
    required=True,
    help="Directory to read and write models")
parser.add_argument(
    "--allele",
    default=None,
    nargs="+",
    help="Alleles to train models for. If not specified, all alleles with "
    "enough measurements will be used.")
parser.add_argument(
    "--num-peptides-per-length",
    type=int,
    metavar="N",
    default=int(1e5),
    help="Number of peptides per length to use to calibrate percent ranks. "
    "Default: %(default)s.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
    default=0)

add_worker_pool_args(parser)

def run(argv=sys.argv[1:]):
    global GLOBAL_DATA

    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    args.models_dir = os.path.abspath(args.models_dir)

    configure_logging(verbose=args.verbosity > 1)

    predictor = Class1AffinityPredictor.load(args.models_dir)

    if args.allele:
        alleles = [normalize_allele_name(a) for a in args.allele]
    else:
        alleles = predictor.supported_alleles

    start = time.time()

    print("Performing percent rank calibration. Encoding peptides.")
    encoded_peptides = predictor.calibrate_percentile_ranks(
        alleles=[],  # don't actually do any calibration, just return peptides
        num_peptides_per_length=args.num_peptides_per_length)

    # Now we encode the peptides for each neural network, so the encoding
    # becomes cached.
    for network in predictor.neural_networks:
        network.peptides_to_network_input(encoded_peptides)
    assert encoded_peptides.encoding_cache  # must have cached the encoding
    print("Finished encoding peptides for percent ranks in %0.2f sec." % (
        time.time() - start))
    print("Calibrating percent rank calibration for %d alleles." % len(alleles))

    # Store peptides in global variable so they are in shared memory
    # after fork, instead of needing to be pickled (when doing a parallel run).
    GLOBAL_DATA["calibration_peptides"] = encoded_peptides

    worker_pool = worker_pool_with_gpu_assignments_from_args(args)

    if worker_pool is None:
        # Serial run
        print("Running in serial.")
        results = (
            calibrate_percentile_ranks(
                allele=allele,
                predictor=predictor,
                peptides=encoded_peptides)
            for allele in alleles)
    else:
        # Parallel run
        results = worker_pool.imap_unordered(
            partial(
                partial(call_wrapped, calibrate_percentile_ranks),
                predictor=predictor),
            alleles,
            chunksize=1)

    for result in tqdm.tqdm(results, total=len(alleles)):
        predictor.allele_to_percent_rank_transform.update(result)
    print("Done calibrating %d additional alleles." % len(alleles))
    predictor.save(args.models_dir, model_names_to_write=[])

    percent_rank_calibration_time = time.time() - start

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Percent rank calibration time: %0.2f min." % (
       percent_rank_calibration_time / 60.0))
    print("Predictor written to: %s" % args.models_dir)


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

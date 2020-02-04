"""
Calibrate percentile ranks for models. Runs in-place.
"""
import argparse
import os
import signal
import sys
import time
import traceback
import collections
from functools import partial

import pandas

from mhcnames import normalize_allele_name
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_affinity_predictor import Class1AffinityPredictor
from .encodable_sequences import EncodableSequences
from .common import configure_logging, random_peptides, amino_acid_distribution
from .local_parallelism import (
    add_local_parallelism_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped_kwargs)
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
    "--models-dir",
    metavar="DIR",
    required=True,
    help="Directory to read and write models")
parser.add_argument(
    "--allele",
    default=None,
    nargs="+",
    help="Alleles to calibrate percentile ranks for. If not specified all "
    "alleles are used")
parser.add_argument(
    "--match-amino-acid-distribution-data",
    help="Sample random peptides from the amino acid distribution of the "
    "peptides listed in the supplied CSV file, which must have a 'peptide' "
    "column. If not specified a uniform distribution is used.")
parser.add_argument(
    "--num-peptides-per-length",
    type=int,
    metavar="N",
    default=int(1e5),
    help="Number of peptides per length to use to calibrate percent ranks. "
    "Default: %(default)s.")
parser.add_argument(
    "--motif-summary",
    default=False,
    action="store_true",
    help="Calculate motifs and length preferences for each allele")
parser.add_argument(
    "--summary-top-peptide-fraction",
    default=[0.0001, 0.001, 0.01, 0.1, 1.0],
    nargs="+",
    type=float,
    metavar="X",
    help="The top X fraction of predictions (i.e. tightest binders) to use to "
    "generate motifs and length preferences. Default: %(default)s")
parser.add_argument(
    "--length-range",
    default=(8, 15),
    type=int,
    nargs=2,
    help="Min and max peptide length to calibrate, inclusive. "
    "Default: %(default)s")
parser.add_argument(
    "--prediction-batch-size",
    type=int,
    default=4096,
    help="Keras batch size for predictions")
parser.add_argument(
    "--alleles-per-work-chunk",
    type=int,
    metavar="N",
    default=1,
    help="Number of alleles per work chunk. Default: %(default)s.")
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

    args.models_dir = os.path.abspath(args.models_dir)

    configure_logging(verbose=args.verbosity > 1)

    # It's important that we don't trigger a Keras import here since that breaks
    # local parallelism (tensorflow backend). So we set optimization_level=0.
    predictor = Class1AffinityPredictor.load(
        args.models_dir,
        optimization_level=0,
    )

    if args.allele:
        alleles = [normalize_allele_name(a) for a in args.allele]
    else:
        alleles = predictor.supported_alleles

    allele_set = set(alleles)

    if predictor.allele_to_sequence:
        # Remove alleles that have the same sequence.
        new_allele_set = set()
        sequence_to_allele = collections.defaultdict(set)
        for allele in list(allele_set):
            sequence_to_allele[predictor.allele_to_sequence[allele]].add(allele)
        for equivalent_alleles in sequence_to_allele.values():
            equivalent_alleles = sorted(equivalent_alleles)
            keep = equivalent_alleles.pop(0)
            new_allele_set.add(keep)
        print(
            "Sequence comparison reduced num alleles from",
            len(allele_set),
            "to",
            len(new_allele_set))
        allele_set = new_allele_set

    alleles = sorted(allele_set)

    distribution = None
    if args.match_amino_acid_distribution_data:
        distribution_peptides = pandas.read_csv(
            args.match_amino_acid_distribution_data).peptide.unique()
        distribution = amino_acid_distribution(distribution_peptides)
        print("Using amino acid distribution:")
        print(distribution)

    start = time.time()

    print("Percent rank calibration for %d alleles. Generating peptides." % (
        len(alleles)))
    peptides = []
    lengths = range(args.length_range[0], args.length_range[1] + 1)
    for length in lengths:
        peptides.extend(
            random_peptides(
                args.num_peptides_per_length, length, distribution=distribution))
    print("Done generating peptides in %0.2f sec." % (time.time() - start))
    print("Encoding %d peptides." % len(peptides))
    start = time.time()

    encoded_peptides = EncodableSequences.create(peptides)
    del peptides

    # Now we encode the peptides for each neural network, so the encoding
    # becomes cached.
    for network in predictor.neural_networks:
        network.peptides_to_network_input(encoded_peptides)
    assert encoded_peptides.encoding_cache  # must have cached the encoding
    print("Finished encoding peptides in %0.2f sec." % (time.time() - start))

    # Store peptides in global variable so they are in shared memory
    # after fork, instead of needing to be pickled (when doing a parallel run).
    GLOBAL_DATA["calibration_peptides"] = encoded_peptides
    GLOBAL_DATA["predictor"] = predictor
    GLOBAL_DATA["args"] = {
        'motif_summary': args.motif_summary,
        'summary_top_peptide_fractions': args.summary_top_peptide_fraction,
        'verbose': args.verbosity > 0,
        'model_kwargs': {
            'batch_size': args.prediction_batch_size,
        }
    }
    del encoded_peptides

    serial_run = not args.cluster_parallelism and args.num_jobs == 0
    worker_pool = None
    start = time.time()

    work_items = []
    for allele in alleles:
        if not work_items or len(
                work_items[-1]['alleles']) >= args.alleles_per_work_chunk:
            work_items.append({"alleles": []})
        work_items[-1]['alleles'].append(allele)

    if serial_run:
        # Serial run
        print("Running in serial.")
        results = (
            do_calibrate_percentile_ranks(**item) for item in work_items)
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=do_calibrate_percentile_ranks,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="pickle",
            clear_constant_data=True)
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None
        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, do_calibrate_percentile_ranks),
            work_items,
            chunksize=1)

    summary_results_lists = collections.defaultdict(list)
    for work_item in tqdm.tqdm(results, total=len(work_items)):
        for (transforms, summary_results) in work_item:
            predictor.allele_to_percent_rank_transform.update(transforms)
            if summary_results is not None:
                for (item, value) in summary_results.items():
                    summary_results_lists[item].append(value)
    print("Done calibrating %d alleles." % len(alleles))
    if summary_results_lists:
        for (name, lst) in summary_results_lists.items():
            df = pandas.concat(lst, ignore_index=True)
            predictor.metadata_dataframes[name] = df
            print("Including summary result: %s" % name)
            print(df)

    predictor.save(args.models_dir, model_names_to_write=[])

    percent_rank_calibration_time = time.time() - start

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Percent rank calibration time: %0.2f min." % (
        percent_rank_calibration_time / 60.0))
    print("Predictor written to: %s" % args.models_dir)


def do_calibrate_percentile_ranks(alleles, constant_data=GLOBAL_DATA):
    result_list = []
    for (i, allele) in enumerate(alleles):
        print("Processing allele", i + 1, "of", len(alleles))
        result_item = calibrate_percentile_ranks(
            allele,
            constant_data['predictor'],
            peptides=constant_data['calibration_peptides'],
            **constant_data["args"])
        result_list.append(result_item)
    return result_list


def calibrate_percentile_ranks(
        allele,
        predictor,
        peptides=None,
        motif_summary=False,
        summary_top_peptide_fractions=[0.001],
        verbose=False,
        model_kwargs={}):
    if verbose:
        print("Calibrating", allele)
    predictor.optimize()  # since we loaded with optimization_level=0
    start = time.time()
    summary_results = predictor.calibrate_percentile_ranks(
        peptides=peptides,
        alleles=[allele],
        motif_summary=motif_summary,
        summary_top_peptide_fractions=summary_top_peptide_fractions,
        verbose=verbose,
        model_kwargs=model_kwargs)
    if verbose:
        print("Done calibrating", allele, "in", time.time() - start, "sec")
    transforms = {
        allele: predictor.allele_to_percent_rank_transform[allele],
    }
    return (transforms, summary_results)


if __name__ == '__main__':
    run()

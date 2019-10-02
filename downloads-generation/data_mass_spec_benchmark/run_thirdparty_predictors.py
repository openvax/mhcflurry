"""
"""
import argparse
import os
import signal
import sys
import time
import traceback
import math
from functools import partial

import numpy
import pandas

from mhcnames import normalize_allele_name
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
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
    "input_peptides",
    metavar="CSV",
    help="CSV file with 'peptide' column")
parser.add_argument(
    "--predictor",
    required=True,
    choices=("netmhcpan4", "netmhcpan4-el"))
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
    "--out",
    metavar="DIR",
    help="Write results to DIR")
parser.add_argument(
    "--max-peptides",
    type=int,
    help="Max peptides to process. For debugging.",
    default=None)
parser.add_argument(
    "--reuse-predictions",
    metavar="DIR",
    help="Take predictions from indicated DIR instead of re-running them")

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)


def load_results(dirname, result_df=None, col_names=None):
    peptides = pandas.read_csv(
        os.path.join(dirname, "peptides.csv")).peptide.values
    manifest_df = pandas.read_csv(os.path.join(dirname, "alleles.csv"))
    if col_names:
        manifest_df = manifest_df.loc[manifest_df.col.isin(col_names)]

    if result_df is None:
        result_df = pandas.DataFrame(
            index=peptides, columns=manifest_df.col.values, dtype="float32")
        result_df[:] = numpy.nan

    for _, row in manifest_df.iterrows():
        with open(os.path.join(dirname, row.path), "rb") as fd:
            result_df.loc[
                peptides, row.col
            ] = numpy.load(fd)['arr_0']

    return result_df


def run(argv=sys.argv[1:]):
    global GLOBAL_DATA

    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    configure_logging()

    serial_run = not args.cluster_parallelism and args.num_jobs == 0

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


    GLOBAL_DATA["predictor"] = args.predictor

    # Write peptide and allele lists to out dir.
    out_peptides = os.path.abspath(os.path.join(args.out, "peptides.csv"))
    pandas.DataFrame({"peptide": peptides}).to_csv(out_peptides, index=False)
    print("Wrote: ", out_peptides)

    manifest_df = []
    for allele in alleles:
        for col in ["affinity", "percentile_rank", "elution_score"]:
            manifest_df.append((allele, col))
    manifest_df = pandas.DataFrame(
        manifest_df, columns=["allele", "kind"])
    manifest_df["col"] = (
            manifest_df.allele + " " + manifest_df.kind)
    manifest_df["path"] = manifest_df.col.map(
        lambda s: s.replace("*", "").replace(" ", ".")) + ".npz"
    out_manifest = os.path.abspath(os.path.join(args.out, "alleles.csv"))
    manifest_df.to_csv(out_manifest, index=False)
    col_to_filename = manifest_df.set_index("col").path.map(
        lambda s: os.path.abspath(os.path.join(args.out, s)))
    print("Wrote: ", out_manifest)

    result_df = pandas.DataFrame(
        index=peptides, columns=manifest_df.columns.values, dtype="float32")
    result_df[:] = numpy.nan

    if args.reuse_predictions:
        raise NotImplementedError()
    else:
        # Same number of chunks for all alleles
        num_chunks = int(math.ceil(len(peptides) / args.chunk_size))
        print("Splitting peptides into %d chunks" % num_chunks)
        peptide_chunks = numpy.array_split(peptides, num_chunks)

        work_items = []
        for (chunk_index, chunk_peptides) in enumerate(peptide_chunks):
            work_item = {
                'alleles': alleles,
                'peptides': chunk_peptides,
            }
            work_items.append(work_item)
    print("Work items: ", len(work_items))

    for (i, work_item) in enumerate(work_items):
        work_item["work_item_num"] = i

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
            input_serialization_method="dill",
            result_serialization_method="pickle",
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

    for (work_item_num, col_to_predictions) in tqdm.tqdm(
            results, total=len(work_items)):
        for (col, predictions) in col_to_predictions.items():
            result_df.loc[
                work_items[work_item_num]['peptides'],
                col
            ] = predictions
            out_path = os.path.join(
                args.out, col_to_filename[col])
            numpy.savez(out_path, result_df[col].values)
            print(
                "Wrote [%f%% null]:" % (
                    result_df[col].isnull().mean() * 100.0),
                out_path)

    print("Overall null rate (should be 0): %f" % (
        100.0 * result_df.isnull().values.flatten().mean()))

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    prediction_time = time.time() - start
    print("Done generating predictions in %0.2f min." % (
        prediction_time / 60.0))


def do_predictions(work_item_num, peptides, alleles, constant_data=None):
    # This may run on the cluster in a way that misses all top level imports,
    # so we have to re-import everything here.
    import time
    import numpy
    import numpy.testing
    import mhctools

    if constant_data is None:
        constant_data = GLOBAL_DATA

    predictor_name = constant_data['predictor']
    if predictor_name == "netmhcpan4":
        predictor = mhctools.NetMHCpan4(
            alleles=alleles, program_name="netMHCpan-4.0")
    else:
        raise ValueError("Unsupported", predictor_name)

    start = time.time()
    df = predictor.predict_peptides_dataframe(peptides)
    print("Generated predictions for %d peptides x %d alleles in %0.2f sec." % (
        len(peptides), len(alleles), (time.time() - start)))

    results = {}
    for (allele, sub_df) in df.groupby("allele"):
        for col in ["affinity", "percentile_rank", "elution_score"]:
            results["%s %s" % (allele, col)] = sub_df[col].values.astype(
                'float32')
    return (work_item_num, results)


if __name__ == '__main__':
    run()

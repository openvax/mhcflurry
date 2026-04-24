"""
Make training data by selecting decoys, etc.
"""
import sys
import argparse
import os
import numpy
import time
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
    "--hits",
    metavar="CSV",
    required=True,
    help="Multiallelic mass spec")
parser.add_argument(
    "--affinity-predictor",
    required=True,
    metavar="CSV",
    help="Class 1 affinity predictor to use (exclusive with --predictions)")
parser.add_argument(
    "--proteome-peptides",
    metavar="CSV",
    required=True,
    help="Proteome peptides")
parser.add_argument(
    "--hit-multiplier-to-take",
    type=float,
    default=1,
    help="")
parser.add_argument(
    "--ppv-multiplier",
    type=int,
    metavar="N",
    default=1000,
    help="Take top 1/N predictions.")
parser.add_argument(
    "--exclude-contig",
    help="Exclude entries annotated to the given contig")
parser.add_argument(
    "--out",
    metavar="CSV",
    required=True,
    help="File to write")
parser.add_argument(
    "--alleles",
    nargs="+",
    help="Include only the specified alleles")

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)


def do_process_samples(samples, constant_data=None):
    import mhcflurry
    import pandas
    import tqdm
    tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

    columns_to_keep = [
        "hit_id",
        "protein_accession",
        "n_flank",
        "c_flank",
        "peptide",
        "sample_id",
        "affinity_prediction",
        "hit",
    ]

    if constant_data is None:
        constant_data = GLOBAL_DATA

    args = constant_data['args']
    lengths = constant_data['lengths']
    all_peptides_by_length = constant_data['all_peptides_by_length']
    sample_table = constant_data['sample_table']

    hit_df = constant_data['hit_df']
    hit_df = hit_df.loc[
        hit_df.sample_id.isin(samples)
    ]

    affinity_predictor = mhcflurry.Class1AffinityPredictor.load(
        args.affinity_predictor)
    print("Loaded", affinity_predictor)

    result_df = []
    for sample_id, sub_hit_df in tqdm.tqdm(
            hit_df.groupby("sample_id"), total=hit_df.sample_id.nunique()):

        sub_hit_df = sub_hit_df.copy()
        sub_hit_df["hit"] = 1

        decoys_df = []
        for length in lengths:
            universe = all_peptides_by_length[length]
            decoys_df.append(
                universe.loc[
                    (~universe.peptide.isin(sub_hit_df.peptide.unique())) &
                    (universe.protein_accession.isin(sub_hit_df.protein_accession.unique()))
                ].sample(
                    n=int(len(sub_hit_df) * args.ppv_multiplier / len(lengths)))[[
                        "protein_accession", "peptide", "n_flank", "c_flank"
                ]].drop_duplicates("peptide"))

        merged_df = pandas.concat(
            [sub_hit_df] + decoys_df, ignore_index=True, sort=False)

        prediction_col = "%s affinity" % sample_table.loc[sample_id].hla
        predictions_df = pandas.DataFrame(
            index=merged_df.peptide.unique(),
            columns=[prediction_col])

        predictions_df[prediction_col] = affinity_predictor.predict(
            predictions_df.index,
            allele=sample_table.loc[sample_id].hla)

        merged_df["affinity_prediction"] = merged_df.peptide.map(
            predictions_df[prediction_col])
        merged_df = merged_df.sort_values("affinity_prediction", ascending=True)

        num_to_take = int(len(sub_hit_df) * args.hit_multiplier_to_take)
        selected_df = merged_df.head(num_to_take)[
                columns_to_keep
        ].sample(frac=1.0).copy()
        selected_df["hit"] = selected_df["hit"].fillna(0)
        selected_df["sample_id"] = sample_id
        result_df.append(selected_df)

        print(
            "Processed sample",
            sample_id,
            "with hit and decoys:\n",
            selected_df.hit.value_counts())

    result_df = pandas.concat(result_df, ignore_index=True, sort=False)
    return result_df


def run():
    import mhcflurry

    args = parser.parse_args(sys.argv[1:])

    configure_logging()

    serial_run = not args.cluster_parallelism and args.num_jobs == 0

    hit_df = pandas.read_csv(args.hits)
    numpy.testing.assert_equal(hit_df.hit_id.nunique(), len(hit_df))
    hit_df = hit_df.loc[
        (hit_df.mhc_class == "I") &
        (hit_df.peptide.str.len() <= 11) &
        (hit_df.peptide.str.len() >= 8) &
        (~hit_df.protein_ensembl.isnull()) &
        (hit_df.peptide.str.match("^[%s]+$" % "".join(
            mhcflurry.amino_acid.COMMON_AMINO_ACIDS)))
    ]
    print("Loaded hits from %d samples" % hit_df.sample_id.nunique())
    hit_df = hit_df.loc[hit_df.format == "MONOALLELIC"].copy()
    print("Subselected to %d monoallelic samples" % hit_df.sample_id.nunique())
    hit_df["allele"] = hit_df.hla

    hit_df = hit_df.loc[hit_df.allele.str.match("^HLA-[ABC]")]
    print("Subselected to %d HLA-A/B/C samples" % hit_df.sample_id.nunique())

    if args.exclude_contig:
        new_hit_df = hit_df.loc[
            hit_df.protein_primary_ensembl_contig.astype(str) !=
            args.exclude_contig
        ]
        print(
            "Excluding contig",
            args.exclude_contig,
            "reduced dataset from",
            len(hit_df),
            "to",
            len(new_hit_df))
        hit_df = new_hit_df.copy()
    if args.alleles:
        filter_alleles = set(args.alleles)
        new_hit_df = hit_df.loc[
            hit_df.allele.isin(filter_alleles)
        ]
        print(
            "Selecting alleles",
            args.alleles,
            "reduced dataset from",
            len(hit_df),
            "to",
            len(new_hit_df))
        hit_df = new_hit_df.copy()

    sample_table = hit_df.drop_duplicates("sample_id").set_index("sample_id")
    grouped = hit_df.groupby("sample_id").nunique()
    for col in sample_table.columns:
        if (grouped[col] > 1).any():
            del sample_table[col]
    sample_table["total_hits"] = hit_df.groupby("sample_id").peptide.nunique()

    print("Loading proteome peptides")
    all_peptides_df = pandas.read_csv(args.proteome_peptides)
    print("Loaded: ", all_peptides_df.shape)

    all_peptides_df = all_peptides_df.loc[
        all_peptides_df.protein_accession.isin(hit_df.protein_accession.unique()) &
        all_peptides_df.peptide.str.match("^[%s]+$" % "".join(
            mhcflurry.amino_acid.COMMON_AMINO_ACIDS))
    ].copy()
    all_peptides_df["length"] = all_peptides_df.peptide.str.len()
    print("Subselected proteome peptides by accession: ", all_peptides_df.shape)

    all_peptides_by_length = dict(iter(all_peptides_df.groupby("length")))

    print("Selecting decoys.")

    GLOBAL_DATA['args'] = args
    GLOBAL_DATA['lengths'] = [8, 9, 10, 11]
    GLOBAL_DATA['all_peptides_by_length'] = all_peptides_by_length
    GLOBAL_DATA['sample_table'] = sample_table
    GLOBAL_DATA['hit_df'] = hit_df

    worker_pool = None
    start = time.time()

    tasks = [
        {"samples": [sample]} for sample in hit_df.sample_id.unique()
    ]

    if serial_run:
        # Serial run
        print("Running in serial.")
        results = [do_process_samples(hit_df.sample_id.unique())]
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=do_process_samples,
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
            partial(call_wrapped_kwargs, do_process_samples),
            tasks,
            chunksize=1)

    print("Reading results")

    result_df = []
    for worker_result in tqdm.tqdm(results, total=len(tasks)):
        for sample_id, selected_df in worker_result.groupby("sample_id"):
            print(
                "Received result for sample",
                sample_id,
                "with hit and decoys:\n",
                selected_df.hit.value_counts())
        result_df.append(worker_result)

    print("Received all results in %0.2f sec" % (time.time() - start))

    result_df = pandas.concat(result_df, ignore_index=True, sort=False)
    result_df["hla"] = result_df.sample_id.map(sample_table.hla)

    print(result_df)
    print("Counts:")
    print(result_df.groupby(["sample_id", "hit"]).peptide.nunique())

    print("Hit counts:")
    print(
        result_df.loc[
            result_df.hit == 1
        ].groupby("sample_id").hit.count().sort_values())

    print("Hit rates:")
    print(result_df.groupby("sample_id").hit.mean().sort_values())

    result_df.to_csv(args.out, index=False)
    print("Wrote: ", args.out)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()


if __name__ == '__main__':
    run()

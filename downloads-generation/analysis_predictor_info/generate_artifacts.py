"""
Generate images for MHC binding motifs.

Note: a shared filesystem is assumed even when running on an HPC cluster.
The --out directory should be on an NFS filesystem and available to the workers.
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
from mhcflurry.downloads import get_path
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
    "--affinity-predictor",
    metavar="DIR",
    help="Pan-allele class I affinity predictor")
parser.add_argument(
    "--frequency-matrices",
    metavar="CSV",
    help="Frequency matrices")
parser.add_argument(
    "--length-distributions",
    metavar="CSV",
    help="Length distributions")
parser.add_argument(
    "--train-data",
    metavar="CSV",
    help="Training data")
parser.add_argument(
    "--alleles",
    nargs="+",
    help="Alleles to process. If not specified all alleles are used")
parser.add_argument(
    "--max-alleles",
    type=int,
    help="Max number of allelels to process. For debugging.")
parser.add_argument(
    "--chunk-size",
    type=int,
    default=100,
    help="Number of alleles per job")
parser.add_argument(
    "--logo-lengths",
    type=int,
    nargs="+",
    default=[8, 9, 10, 11],
    help="Peptide lengths for motif logos")
parser.add_argument(
    "--length-distribution-lengths",
    nargs="+",
    default=[8, 9, 10, 11, 12, 13, 14, 15],
    type=int,
    help="Peptide lengths for length distribution plots",
)
parser.add_argument(
    "--logo-cutoff",
    default=0.01,
    type=float,
    help="Fraction of top to use for motifs",
)
parser.add_argument(
    "--length-cutoff",
    default=0.01,
    type=float,
    help="Fraction of top to use for length distribution",
)
parser.add_argument(
    "--out",
    metavar="DIR",
    required=True,
    help="Directory to write results to")

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)


def run():
    from mhcflurry.amino_acid import COMMON_AMINO_ACIDS

    args = parser.parse_args(sys.argv[1:])

    configure_logging()

    serial_run = not args.cluster_parallelism and args.num_jobs == 0

    if not args.affinity_predictor:
        args.affinity_predictor = get_path(
            "models_class1_pan", "models.combined")
        print("Using downloaded affinity predictor: ", args.affinity_predictor)

    if not args.frequency_matrices:
        args.frequency_matrices = os.path.join(
            args.affinity_predictor, "frequency_matrices.csv.bz2")

    if not args.length_distributions:
        args.length_distributions = os.path.join(args.affinity_predictor,
            "length_distributions.csv.bz2")

    if not args.train_data:
        args.train_data = os.path.join(args.affinity_predictor,
            "train_data.csv.bz2")

    frequency_matrices_df = pandas.read_csv(args.frequency_matrices)
    length_distributions = pandas.read_csv(args.length_distributions)
    train_data = pandas.read_csv(args.train_data)

    alleles = args.alleles
    if alleles:
        print("Using specified alleles, ", *alleles)
    else:
        alleles = frequency_matrices_df.allele.unique()

    if args.max_alleles:
        alleles = alleles[:args.max_alleles]

    print("Using %d alleles" % len(alleles), alleles)

    amino_acids = sorted(COMMON_AMINO_ACIDS)

    distribution = frequency_matrices_df.loc[
        (frequency_matrices_df.cutoff_fraction == 1.0), amino_acids
    ].mean(0)

    normalized_frequency_matrices = frequency_matrices_df.copy()
    normalized_frequency_matrices.loc[:, amino_acids] = (
            normalized_frequency_matrices[amino_acids] / distribution)

    GLOBAL_DATA["args"] = args
    GLOBAL_DATA["normalized_frequency_matrices"] = normalized_frequency_matrices
    GLOBAL_DATA["length_distributions"] = length_distributions
    GLOBAL_DATA["train_data"] = train_data

    artifacts_out = os.path.join(args.out, "artifacts")

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    if not os.path.exists(artifacts_out):
        os.mkdir(artifacts_out)

    tasks = [
        {
            "task_num": i,
            "allele": allele,
            "out_dir": artifacts_out,
        }
        for (i, allele) in enumerate(alleles)
    ]

    jobs = []
    for task in tasks:
        if not jobs or len(jobs[-1]['tasks']) >= args.chunk_size:
            jobs.append({'tasks': []})
        jobs[-1]['tasks'].append(task)

    print("Generated %d tasks, packed into %d jobs" % (len(tasks), len(jobs)))

    worker_pool = None
    start = time.time()

    if serial_run:
        # Serial run
        print("Running in serial.")
        results = (
            do_job(**job) for job in jobs)
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=do_job,
            work_items=jobs,
            constant_data=GLOBAL_DATA,
            input_serialization_method="dill",
            result_serialization_method="pickle",
            clear_constant_data=False)
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(args)
        print("Worker pool", worker_pool)
        assert worker_pool is not None

        for task in tasks:
            task['constant_data'] = GLOBAL_DATA

        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, do_job),
            jobs,
            chunksize=1)

    print("Reading results")

    task_results = {}

    for job_result in tqdm.tqdm(results, total=len(jobs)):
        for task_result in job_result:
            task_results[task_result['task_num']] = task_result

    print("Received all results in %0.2f sec" % (time.time() - start))

    artifacts_df = pandas.DataFrame(task_results).T.set_index("task_num")

    normalized_frequency_matrices_out = os.path.join(
        args.out, "normalized_frequency_matrices.csv")
    normalized_frequency_matrices.to_csv(
        normalized_frequency_matrices_out, index=False)
    print("Wrote: ", normalized_frequency_matrices_out)

    length_distributions_out = os.path.join(args.out,
        "length_distributions.csv")
    length_distributions.to_csv(length_distributions_out,
        index=False)
    print("Wrote: ", length_distributions_out)

    artifacts_summary_out = os.path.join(args.out, "artifacts.csv")
    artifacts_df.to_csv(artifacts_summary_out)
    print("Wrote: ", artifacts_summary_out)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()


def do_job(tasks, constant_data=GLOBAL_DATA):
    # Nested functions are so that the do_job function can be pickled for
    # running on an HPC cluster.
    GLOBAL_DATA = constant_data

    def do_task(task_num, allele, out_dir, constant_data=GLOBAL_DATA):
        args = constant_data['args']
        normalized_frequency_matrices = constant_data[
            'normalized_frequency_matrices'
        ]
        length_distributions = constant_data[
            'length_distributions'
        ]
        train_data = constant_data[
            'train_data'
        ]

        logo_filename = write_logo(
            normalized_frequency_matrices,
            allele=allele,
            lengths=args.logo_lengths,
            cutoff=args.logo_cutoff,
            models_label="standard",
            out_dir=out_dir,
        )

        length_distribution_filename = write_length_distribution(
            length_distributions,
            allele=allele,
            lengths=args.length_distribution_lengths,
            cutoff=args.length_cutoff,
            models_label="standard",
            out_dir=out_dir)

        (train_data_filename, num_train_points) = write_train_data(
            train_data,
            allele=allele,
            models_label="standard",
            out_dir=out_dir)

        return {
            'task_num': task_num,
            'allele': allele,
            'logo_filename': logo_filename,
            'length_distribution_filename': length_distribution_filename,
            'train_data_filename': train_data_filename,
            'num_train_points': num_train_points,
        }


    def write_logo(
            normalized_frequency_matrices,
            allele,
            lengths,
            cutoff,
            models_label,
            out_dir):

        import seaborn
        from matplotlib import pyplot
        import logomaker
        import os
        from mhcflurry.amino_acid import COMMON_AMINO_ACIDS

        amino_acids = sorted(COMMON_AMINO_ACIDS)

        fig = pyplot.figure(figsize=(8,10))

        for (i, length) in enumerate(lengths):
            ax = pyplot.subplot(len(lengths), 1, i + 1)
            matrix = normalized_frequency_matrices.loc[
                (normalized_frequency_matrices.allele == allele) &
                (normalized_frequency_matrices.length == length) &
                (normalized_frequency_matrices.cutoff_fraction == cutoff)
            ].set_index("position")[amino_acids]
            if matrix.shape[0] == 0:
                return None

            matrix = (matrix.T / matrix.sum(1)).T  # row normalize

            ss_logo = logomaker.Logo(
                matrix,
                color_scheme="NajafabadiEtAl2017",
                font_name="Arial",
                width=.8,
                vpad=.05,
                fade_probabilities=True,
                stack_order='small_on_top',
                ax=ax,
            )
            pyplot.title(
                "%s %d-mer" % (allele, length), y=0.85)
            pyplot.xticks(matrix.index.values)
            seaborn.despine()

        pyplot.tight_layout()
        name = "%s.motifs.%s.png" % (
            allele.replace("*", "-").replace(":", "-"), models_label)
        filename = os.path.abspath(os.path.join(out_dir, name))
        pyplot.savefig(filename)
        print("Wrote: ", filename)
        fig.clear()
        pyplot.close(fig)
        return name


    def write_length_distribution(
            length_distributions_df, allele, lengths, cutoff, models_label, out_dir):

        from matplotlib import pyplot
        import seaborn
        import os

        length_distribution = length_distributions_df.loc[
            (length_distributions_df.allele == allele) &
            (length_distributions_df.cutoff_fraction == cutoff)
        ]
        if length_distribution.shape[0] == 0:
            return None

        length_distribution = length_distribution.set_index(
            "length").reindex(lengths).fillna(0.0).reset_index()

        length_distribution.plot(
            x="length", y="fraction", kind="bar", figsize=(5, 3))
        fig = pyplot.gcf()
        pyplot.title("%s" % allele, fontsize=10)
        pyplot.xlabel("Peptide length", fontsize=10)
        pyplot.xticks(rotation=0)
        pyplot.ylim(ymin=0, ymax=1.0)
        pyplot.ylabel("Fraction of top %0.1f%%" % (cutoff * 100.0), fontsize=10)
        pyplot.gca().get_legend().remove()
        pyplot.tight_layout()

        seaborn.despine()

        name = "%s.lengths.%s.png" % (
            allele.replace("*", "-").replace(":", "-"), models_label)

        filename = os.path.abspath(os.path.join(out_dir, name))
        pyplot.savefig(filename)
        print("Wrote: ", filename)
        fig.clear()
        pyplot.close(fig)
        return name

    def write_train_data(train_data, allele, models_label, out_dir):
        import os
        sub_train = train_data.loc[
            train_data.allele == allele
        ]

        name = None
        if sub_train.shape[0] > 0:
            name = "%s.train_data.%s.csv" % (
                allele.replace("*", "-").replace(":", "-"), models_label)
            filename = os.path.abspath(os.path.join(out_dir, name))
            sub_train.to_csv(filename, index=False)
            print("Wrote: ", filename)
        return (name, len(sub_train))

    return [do_task(constant_data=constant_data, **task) for task in tasks]

if __name__ == '__main__':
    run()

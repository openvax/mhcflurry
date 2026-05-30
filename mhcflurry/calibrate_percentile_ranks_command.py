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
import numpy


import tqdm  # progress bar

from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_presentation_predictor import Class1PresentationPredictor
from .common import (
    add_random_seed_arg,
    amino_acid_distribution,
    configure_logging,
    configure_pytorch,
    configure_random_seed,
    filter_canonicalizable_alleles,
    normalize_allele_name,
    random_peptides,
    write_generate_sh,
)
from .encodable_sequences import EncodableSequences
from .local_parallelism import (
    attach_constant_data_to_work_items_if_needed,
    add_local_parallelism_args,
    chunk_ranges_for_local_parallelism,
    num_workers_per_gpu_from_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped_kwargs)
from .workload_planning import (
    WORKLOAD_AFFINITY_CALIBRATION,
    WORKLOAD_PRESENTATION_CALIBRATION,
    path_size_bytes,
)
from .cluster_parallelism import (
    add_cluster_parallelism_args,
    cluster_results_from_args)

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing local workers to read the same
# copy-on-write pages instead of receiving a pickled copy.
GLOBAL_DATA = {}

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "--predictor-kind",
    choices=("class1_affinity", "class1_presentation"),
    default="class1_affinity",
    help="Type of predictor to calibrate")
parser.add_argument(
    "--models-dir",
    metavar="DIR",
    required=True,
    help="Directory to read and write models")
parser.add_argument(
    "--allele", "--alleles",
    dest="allele",
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
    "--alleles-file",
    default=None,
    help="Use alleles in supplied CSV file, which must have an 'allele' column.")
parser.add_argument(
    "--list-percent-rank-status",
    default=False,
    action="store_true",
    help="For class1 affinity predictors, print a CSV indicating which requested "
    "alleles already have percentile-rank calibration and exit without "
    "generating calibration peptides.")
parser.add_argument(
    "--only-missing",
    dest="only_missing_percent_ranks",
    default=False,
    action="store_true",
    help="For class1 affinity predictors, calibrate only requested alleles that "
    "do not already have direct or sequence-equivalent percentile-rank "
    "calibration.")
parser.add_argument(
    "--num-peptides-per-length",
    type=int,
    metavar="N",
    default=int(1e5),
    help="Number of peptides per length to use to calibrate percent ranks. "
    "Default: %(default)s.")
parser.add_argument(
    "--num-genotypes",
    type=int,
    metavar="N",
    default=25,
    help="Used when calibrrating a presentation predictor. Number of genotypes"
    "to sample")
parser.add_argument(
    "--alleles-per-genotype",
    type=int,
    metavar="N",
    default=6,
    help="Used when calibrating a presentation predictor. Number of alleles "
    "per genotype. Use 1 to calibrate for single alleles. Default: %(default)s")
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


def _batch_size_arg(value):
    """Accept either an int or the literal string 'auto' for --prediction-batch-size.

    ``auto`` (the default) delegates sizing to
    ``mhcflurry.class1_neural_network.compute_prediction_batch_size``,
    which picks a per-GPU-memory batch, reserving the VRAM partition
    across co-resident workers.
    """
    if isinstance(value, str) and value.strip().lower() in ("auto", ""):
        return "auto"
    return int(value)


parser.add_argument(
    "--prediction-batch-size",
    type=_batch_size_arg,
    default="auto",
    help="Batch size for predictions. Pass an int to pin, or 'auto' "
         "(default) to size per GPU free memory / workers-per-GPU — see "
         "mhcflurry.class1_neural_network.compute_prediction_batch_size.")
parser.add_argument(
    "--alleles-per-work-chunk",
    type=int,
    metavar="N",
    default=1,
    help="Number of alleles per work chunk. Default: %(default)s.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Verbosity. Default: %(default)s",
    default=0)
parser.add_argument(
    "--gpu-batched",
    default=False,
    action="store_true",
    help="[class1 affinity predictors only] Use the GPU-hoisted "
         "calibration fast path: precompute peptide-side activations "
         "per network and batch "
         "--gpu-allele-batch-size alleles into a single forward through "
         "the merge + main dense path. Same output as the default path "
         "(bit-identical on CUDA, ~1e-6 log-IC50 drift on MPS due to "
         "missing fp64 support), typically 5-30x faster on CUDA for the "
         "full pan-allele universe. Ignored when running a presentation "
         "predictor or serial/cluster mode.")
parser.add_argument(
    "--gpu-allele-batch-size",
    type=_batch_size_arg,
    default="auto",
    help="Alleles per GPU forward when --gpu-batched. Pass an int to "
         "pin; 'auto' (default) partitions the VRAM budget with "
         "--max-workers-per-gpu. Larger values trade off more VRAM for "
         "fewer kernel launches.")
parser.add_argument(
    "--gpu-peptide-batch-size",
    type=_batch_size_arg,
    default="auto",
    help="Peptide chunk size on device when --gpu-batched. Pass an int "
         "to pin; 'auto' (default) picks the peptide axis of the "
         "auto-sized budget. Reducing keeps peak VRAM down on smaller "
         "GPUs but adds kernel-launch overhead.")

add_local_parallelism_args(parser)
add_cluster_parallelism_args(parser)
add_random_seed_arg(parser)


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)

    if not args.list_percent_rank_status:
        # On sigusr1 print stack trace
        print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
        signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args.models_dir = os.path.abspath(args.models_dir)

    configure_logging(verbose=args.verbosity > 1)

    # Seed all randomness up front (random peptide universe + genotype
    # sampling below both run in this process). Inference in workers is
    # deterministic, so global seeding here makes calibration reproducible.
    configure_random_seed(args.random_seed, name="calibrate-percentile-ranks")

    # Resolve --max-workers-per-gpu='auto' to an int now, before any
    # downstream consumer reads it (model_kwargs below + pool creation).
    # The shared helper also caps auto-sized local Pools to GPU capacity and
    # hoists torch.compile worker-thread defaults before forking.
    #
    # Workload profiles keep calibration from inheriting training's worker
    # footprint. Presentation calibration in particular keeps the full
    # presentation predictor stack resident while generating a large
    # peptide-by-genotype score universe.
    from .local_parallelism import resolve_local_parallelism_args
    num_lengths = args.length_range[1] - args.length_range[0] + 1
    prediction_rows = int(args.num_peptides_per_length) * max(num_lengths, 1)
    if args.predictor_kind == "class1_presentation":
        prediction_rows *= max(int(args.num_genotypes), 1)
    resolve_local_parallelism_args(
        args,
        cap_auto_num_jobs=not getattr(args, "cluster_parallelism", False),
        workload_name=(
            WORKLOAD_PRESENTATION_CALIBRATION
            if args.predictor_kind == "class1_presentation"
            else WORKLOAD_AFFINITY_CALIBRATION
        ),
        workload_hints={
            "data_bytes": sum(
                value or 0 for value in (
                    path_size_bytes(args.match_amino_acid_distribution_data),
                    path_size_bytes(args.alleles_file),
                )
            ) or None,
            "num_work_items": (
                args.num_genotypes
                if args.predictor_kind == "class1_presentation"
                else None
            ),
            "prediction_rows": prediction_rows,
        },
    )

    # If we're going to run in-process (serial — no worker pool), the
    # forward kernels in this process must respect the requested
    # backend. Worker pools take care of this in worker_init via
    # configure_pytorch, but the parent never gets that path.
    if (
            getattr(args, "num_jobs", 0) == 0
            and not getattr(args, "cluster_parallelism", False)):
        configure_pytorch(backend=getattr(args, "backend", "auto") or "auto")

    if (
            args.only_missing_percent_ranks
            and args.predictor_kind != "class1_affinity"):
        raise ValueError(
            "--only-missing is only supported for class1_affinity predictors")

    if args.list_percent_rank_status:
        if args.predictor_kind != "class1_affinity":
            raise ValueError(
                "--list-percent-rank-status is only supported for "
                "class1_affinity predictors")
        return run_class1_affinity_percent_rank_status(args)

    aa_distribution = None
    if args.match_amino_acid_distribution_data:
        distribution_peptides = pandas.read_csv(
            args.match_amino_acid_distribution_data).peptide.unique()
        distribution_peptides = [
            x for x in distribution_peptides if 'X' not in x and 'B' not in x and 'U' not in x
        ]
        aa_distribution = amino_acid_distribution(distribution_peptides)
        print("Using amino acid distribution:")
        print(aa_distribution)

    start = time.time()
    peptides = []
    lengths = range(args.length_range[0], args.length_range[1] + 1)
    for length in lengths:
        peptides.extend(
            random_peptides(
                args.num_peptides_per_length,
                length,
                distribution=aa_distribution))
    print("Done generating peptides in %0.2f sec." % (time.time() - start))

    if args.predictor_kind == "class1_affinity":
        return run_class1_affinity_predictor(args, peptides)
    elif args.predictor_kind == "class1_presentation":
        return run_class1_presentation_predictor(args, peptides)
    else:
        raise ValueError("Unsupported kind %s" % args.predictor_kind)


def requested_calibration_alleles(args, predictor):
    """Return normalized alleles selected by CLI arguments."""
    if args.allele:
        # Explicit CLI alleles should fail loudly if they cannot be normalized.
        return [normalize_allele_name(a) for a in args.allele]
    if args.alleles_file:
        return filter_canonicalizable_alleles(
            pandas.read_csv(args.alleles_file).allele.unique()
        )
    return filter_canonicalizable_alleles(predictor.supported_alleles)


def missing_percent_rank_alleles(predictor, alleles):
    """Return alleles lacking direct or sequence-equivalent calibration."""
    return [
        allele for allele in alleles
        if predictor.percent_rank_calibrated_allele(allele) is None
    ]


def percent_rank_status_df(predictor, alleles):
    """Return per-allele percentile-rank calibration status."""
    rows = []
    supported_alleles = set(predictor.supported_alleles)
    for allele in alleles:
        normalized = predictor.canonicalize_allele_name(allele)
        source_allele = predictor.percent_rank_calibrated_allele(normalized)
        rows.append({
            "allele": allele,
            "normalized_allele": normalized,
            "supported": normalized in supported_alleles,
            "has_affinity_percent_rank": source_allele is not None,
            "affinity_percent_rank_source_allele": source_allele or "",
        })
    return pandas.DataFrame(rows)


def run_class1_affinity_percent_rank_status(args):
    """Print percent-rank status for requested affinity alleles and exit."""
    predictor = Class1AffinityPredictor.load(args.models_dir, optimization_level=0)
    alleles = requested_calibration_alleles(args, predictor)
    percent_rank_status_df(predictor, alleles).to_csv(sys.stdout, index=False)


def run_class1_presentation_predictor(args, peptides):
    predictor = Class1PresentationPredictor.load(args.models_dir)

    alleles = requested_calibration_alleles(args, predictor)

    print("Num alleles", len(alleles))

    genotypes = {}
    if args.alleles_per_genotype == 6:
        gene_to_alleles = collections.defaultdict(list)
        for a in alleles:
            for gene in ["A", "B", "C"]:
                if a.startswith("HLA-%s" % gene):
                    gene_to_alleles[gene].append(a)

        for _ in range(args.num_genotypes):
            genotype = []
            for gene in ["A", "A", "B", "B", "C", "C"]:
                genotype.append(numpy.random.choice(gene_to_alleles[gene]))
            genotypes[",".join(genotype)] = genotype
    elif args.alleles_per_genotype == 1:
        for _ in range(args.num_genotypes):
            genotype = [numpy.random.choice(alleles)]
            genotypes[",".join(genotype)] = genotype
    else:
        raise ValueError("Alleles per genotype must be 6 or 1")

    print("Sampled genotypes: ", list(genotypes))
    print("Num peptides: ", len(peptides))

    GLOBAL_DATA["presentation_models_dir"] = args.models_dir
    GLOBAL_DATA["presentation_peptides"] = peptides
    affinity_model_kwargs = {"batch_size": args.prediction_batch_size}
    if hasattr(args, "max_workers_per_gpu"):
        affinity_model_kwargs["num_workers_per_gpu"] = (
            num_workers_per_gpu_from_args(args)
        )
    GLOBAL_DATA["presentation_predict_kwargs"] = {
        "affinity_model_kwargs": affinity_model_kwargs,
        "processing_batch_size": args.prediction_batch_size,
    }
    GLOBAL_DATA.pop("_presentation_predictor", None)
    GLOBAL_DATA.pop("_presentation_predictor_models_dir", None)

    serial_run = not args.cluster_parallelism and args.num_jobs == 0
    if serial_run:
        GLOBAL_DATA["_presentation_predictor"] = predictor

    genotype_items = list(genotypes.items())
    work_items = []
    for (chunk_num, start, end) in chunk_ranges_for_local_parallelism(
            len(genotype_items), args.num_jobs):
        work_items.append({
            "chunk_num": chunk_num,
            "genotypes": dict(genotype_items[start:end]),
        })

    start = time.time()
    print("Generating predictions")
    worker_pool = None
    if serial_run:
        print("Running in serial.")
        results = (
            do_class1_presentation_percent_rank_scores(**item)
            for item in work_items)
    elif args.cluster_parallelism:
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=do_class1_presentation_percent_rank_scores,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="pickle",
            clear_constant_data=True)
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(
            args,
            # Workers pin to GPUs via CUDA_VISIBLE_DEVICES. Force spawn so
            # they don't inherit any PyTorch / CUDA state from the parent,
            # matching predict / predict-scan.
            start_method="spawn",
        )
        print("Worker pool", worker_pool)
        assert worker_pool is not None
        attach_constant_data_to_work_items_if_needed(
            work_items, GLOBAL_DATA, worker_pool
        )
        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs,
                    do_class1_presentation_percent_rank_scores),
            work_items,
            chunksize=1)

    try:
        score_chunks = [
            scores for scores in tqdm.tqdm(results, total=len(work_items))
        ]
        if worker_pool:
            worker_pool.close()
            worker_pool.join()
            worker_pool = None
    finally:
        # On any failure mid-iteration, tear the pool down rather than
        # leaking non-daemon workers.
        if worker_pool is not None:
            worker_pool.terminate()
            worker_pool.join()
    print("Finished in %0.2f sec." % (time.time() - start))
    scores = (
        numpy.concatenate(score_chunks, axis=0)
        if score_chunks
        else numpy.array([], dtype=float)
    )
    print("Generated %d presentation scores." % len(scores))

    print("Calibrating ranks")
    predictor.calibrate_percentile_ranks(scores)
    print("Done. Saving.")

    predictor.save(
        args.models_dir,
        write_affinity_predictor=False,
        write_processing_predictor=False,
        write_weights=False,
        write_percent_ranks=True,
        write_info=False,
        write_metdata=False)
    write_generate_sh(args.models_dir)
    print("Wrote predictor to: %s" % args.models_dir)


def _presentation_predictor_for_calibration(constant_data):
    cache_key = constant_data["presentation_models_dir"]
    predictor = GLOBAL_DATA.get("_presentation_predictor")
    if (
            predictor is None
            or GLOBAL_DATA.get("_presentation_predictor_models_dir") != cache_key):
        predictor = Class1PresentationPredictor.load(cache_key)
        GLOBAL_DATA["_presentation_predictor"] = predictor
        GLOBAL_DATA["_presentation_predictor_models_dir"] = cache_key
    return predictor


def do_class1_presentation_percent_rank_scores(
        genotypes, chunk_num=None, constant_data=GLOBAL_DATA):
    del chunk_num
    predictor = _presentation_predictor_for_calibration(constant_data)
    predictions_df = predictor.predict(
        peptides=constant_data["presentation_peptides"],
        alleles=genotypes,
        **constant_data["presentation_predict_kwargs"])
    return predictions_df.presentation_score.values


def run_class1_affinity_predictor(args, peptides):
    # Load with optimization_level=0 so we can optimize per-worker later.
    predictor = Class1AffinityPredictor.load(
        args.models_dir,
        optimization_level=0,
    )

    alleles = requested_calibration_alleles(args, predictor)

    if args.only_missing_percent_ranks:
        before_missing_filter = len(alleles)
        alleles = missing_percent_rank_alleles(predictor, alleles)
        print(
            "Missing-percent-rank filter reduced num alleles from",
            before_missing_filter,
            "to",
            len(alleles))
        if not alleles:
            print("No requested alleles are missing percentile-rank calibration.")
            return

    allele_set = set(alleles)

    if predictor.allele_to_sequence:
        unknown_alleles = [
            allele for allele in allele_set
            if allele not in predictor.allele_to_sequence
        ]
        if unknown_alleles:
            raise ValueError(
                "Cannot calibrate unsupported allele(s): %s" % (
                    " ".join(sorted(unknown_alleles))))

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

    print("Percent rank calibration for %d alleles. " % (len(alleles)))

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

    # Store peptides in GLOBAL_DATA so forked local workers inherit the same
    # copy-on-write pages instead of receiving a pickled copy.
    GLOBAL_DATA["calibration_peptides"] = encoded_peptides
    GLOBAL_DATA["predictor"] = predictor
    model_kwargs = {
        'batch_size': args.prediction_batch_size,
    }
    # Thread workers-per-GPU into the auto-size budget. The VRAM
    # partition across co-resident workers matters only when the
    # underlying predict path resolves ``"auto"`` — but it's cheap to
    # always pass it through.
    if hasattr(args, 'max_workers_per_gpu'):
        model_kwargs['num_workers_per_gpu'] = num_workers_per_gpu_from_args(args)
    num_workers_per_gpu = num_workers_per_gpu_from_args(args)
    GLOBAL_DATA["args"] = {
        'motif_summary': args.motif_summary,
        'summary_top_peptide_fractions': args.summary_top_peptide_fraction,
        'verbose': args.verbosity > 0,
        'model_kwargs': model_kwargs,
        'gpu_batched': getattr(args, 'gpu_batched', False),
        'gpu_allele_batch_size': getattr(args, 'gpu_allele_batch_size', 'auto'),
        'gpu_peptide_batch_size': getattr(args, 'gpu_peptide_batch_size', 'auto'),
        'num_workers_per_gpu': num_workers_per_gpu,
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
            do_class1_affinity_calibrate_percentile_ranks(**item) for item in work_items)
    elif args.cluster_parallelism:
        # Run using separate processes HPC cluster.
        print("Running on cluster.")
        results = cluster_results_from_args(
            args,
            work_function=do_class1_affinity_calibrate_percentile_ranks,
            work_items=work_items,
            constant_data=GLOBAL_DATA,
            result_serialization_method="pickle",
            clear_constant_data=True)
    else:
        worker_pool = worker_pool_with_gpu_assignments_from_args(
            args,
            # Same CUDA-fork-safety rationale as the presentation path above.
            start_method="spawn",
        )
        print("Worker pool", worker_pool)
        assert worker_pool is not None

        attach_constant_data_to_work_items_if_needed(
            work_items, GLOBAL_DATA, worker_pool
        )
        results = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, do_class1_affinity_calibrate_percentile_ranks),
            work_items,
            chunksize=1)

    try:
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
        write_generate_sh(args.models_dir)

        percent_rank_calibration_time = time.time() - start

        if worker_pool:
            worker_pool.close()
            worker_pool.join()
            worker_pool = None
    finally:
        # On any failure mid-iteration, tear the pool down rather than
        # leaking non-daemon workers.
        if worker_pool is not None:
            worker_pool.terminate()
            worker_pool.join()

    print("Percent rank calibration time: %0.2f min." % (
        percent_rank_calibration_time / 60.0))
    print("Predictor written to: %s" % args.models_dir)


def do_class1_affinity_calibrate_percentile_ranks(
        alleles, constant_data=GLOBAL_DATA):

    if 'predictor' not in constant_data:
        raise ValueError("No predictor provided: " + str(constant_data))

    args = dict(constant_data["args"])
    gpu_batched = args.pop('gpu_batched', False)
    gpu_allele_batch_size = args.pop('gpu_allele_batch_size', "auto")
    gpu_peptide_batch_size = args.pop('gpu_peptide_batch_size', "auto")
    num_workers_per_gpu = args.pop('num_workers_per_gpu', 1)

    if gpu_batched:
        # Single fast-path call over the whole chunk — see
        # Class1AffinityPredictor.calibrate_percentile_ranks_fast. The
        # worker's chunk size (--alleles-per-work-chunk) gates the
        # outer partition; within a chunk the allele sweep runs as
        # fewer larger GPU forwards. num_workers_per_gpu narrows the
        # VRAM partition the fast path claims so co-resident workers
        # don't race each other into OOM.
        return [
            class1_affinity_calibrate_percentile_ranks_fast(
                alleles=alleles,
                predictor=constant_data['predictor'],
                peptides=constant_data['calibration_peptides'],
                motif_summary=args['motif_summary'],
                summary_top_peptide_fractions=args['summary_top_peptide_fractions'],
                verbose=args['verbose'],
                gpu_allele_batch_size=gpu_allele_batch_size,
                gpu_peptide_batch_size=gpu_peptide_batch_size,
                num_workers_per_gpu=num_workers_per_gpu,
            )
        ]

    result_list = []
    for (i, allele) in enumerate(alleles):
        print("Processing allele", i + 1, "of", len(alleles))
        result_item = class1_affinity_calibrate_percentile_ranks(
            allele,
            constant_data['predictor'],
            peptides=constant_data['calibration_peptides'],
            **args)
        result_list.append(result_item)
    return result_list


def class1_affinity_calibrate_percentile_ranks_fast(
        alleles,
        predictor,
        peptides,
        motif_summary=False,
        summary_top_peptide_fractions=(0.001,),
        verbose=False,
        gpu_allele_batch_size="auto",
        gpu_peptide_batch_size="auto",
        num_workers_per_gpu=1):
    """Worker-side fast-path wrapper for the GPU-batched calibration path.

    Returns the same ``(transforms_dict, summary_results)`` tuple the
    per-allele wrapper produces so the surrounding result-aggregation
    code doesn't need to know which path ran.
    """
    predictor.optimize()
    start = time.time()
    summary_results = predictor.calibrate_percentile_ranks_fast(
        peptides=peptides,
        alleles=alleles,
        motif_summary=motif_summary,
        summary_top_peptide_fractions=tuple(summary_top_peptide_fractions),
        allele_batch_size=gpu_allele_batch_size,
        peptide_batch_size=gpu_peptide_batch_size,
        num_workers_per_gpu=num_workers_per_gpu,
        verbose=verbose,
    )
    if verbose:
        print(
            "Done calibrating %d alleles in %0.2f sec via fast path" % (
                len(alleles), time.time() - start,
            )
        )
    transforms = {
        allele: predictor.allele_to_percent_rank_transform[allele]
        for allele in alleles
    }
    # Motif summary comes back pre-concatenated when fast path is used;
    # mirror the legacy per-allele wrapper's return shape.
    return (transforms, summary_results if motif_summary else None)


def class1_affinity_calibrate_percentile_ranks(
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

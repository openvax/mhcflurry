"""
Train Class1 single allele models.
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
from sklearn.model_selection import StratifiedKFold
from .common import normalize_allele_name
import tqdm  # progress bar

from .class1_affinity_predictor import Class1AffinityPredictor
from .common import (
    add_random_seed_arg,
    configure_logging,
    configure_random_seed,
    derive_seed,
    write_generate_sh,
)
from .local_parallelism import (
    add_local_parallelism_args,
    apply_resolved_training_hyperparameters_to_work_items,
    attach_constant_data_to_work_items_if_needed,
    resolve_local_parallelism_args,
    worker_pool_with_gpu_assignments_from_args,
    call_wrapped_kwargs)
from .workload_planning import (
    WORKLOAD_AFFINITY_TRAINING,
    path_size_bytes,
)
from .hyperparameters import HyperparameterDefaults
from .allele_encoding import AlleleEncoding

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing local workers to read the same
# copy-on-write pages instead of receiving a pickled copy.
GLOBAL_DATA = {}

# Note on parallelization:
# When running in parallel, avoid using the neural network backend in the main
# process. Model loading and inference should happen in worker processes.

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    required=True,
    help=(
        "Training data CSV. Expected columns: "
        "allele, peptide, measurement_value"))
parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    required=True,
    help="Directory to write models and manifest")
parser.add_argument(
    "--hyperparameters",
    metavar="FILE.json",
    required=True,
    help="JSON or YAML of hyperparameters")
parser.add_argument(
    "--allele",
    default=None,
    nargs="+",
    help="Alleles to train models for. If not specified, all alleles with "
    "enough measurements will be used.")
parser.add_argument(
    "--min-measurements-per-allele",
    type=int,
    metavar="N",
    default=50,
    help="Train models for alleles with >=N measurements.")
parser.add_argument(
    "--held-out-fraction-reciprocal",
    type=int,
    metavar="N",
    default=None,
    help="Hold out 1/N fraction of data (for e.g. subsequent model selection. "
    "For example, specify 5 to hold out 20 percent of the data.")
parser.add_argument(
    "--held-out-fraction-seed",
    type=int,
    metavar="N",
    default=None,
    help="Seed for randomizing which measurements are held out. Only matters "
    "when --held-out-fraction is specified. When omitted, the held-out split "
    "is derived from --random-seed (so the whole run reproduces from one "
    "value). Pass this explicitly to control the split directly — e.g. to "
    "reproduce a pre-2.3.0 split — overriding --random-seed for the split.")
add_random_seed_arg(parser)
parser.add_argument(
    "--ignore-inequalities",
    action="store_true",
    default=False,
    help="Do not use affinity value inequalities even when present in data")
parser.add_argument(
    "--n-models",
    type=int,
    metavar="N",
    help="Ensemble size, i.e. how many models to train for each architecture. "
    "If specified here it overrides any 'n_models' specified in the "
    "hyperparameters.")
parser.add_argument(
    "--max-epochs",
    type=int,
    metavar="N",
    help="Max training epochs. If specified here it overrides any 'max_epochs' "
    "specified in the hyperparameters.")
parser.add_argument(
    "--allele-sequences",
    metavar="FILE.csv",
    help="Allele sequences file. Used for computing allele similarity matrix.")
parser.add_argument(
    "--save-interval",
    type=float,
    metavar="N",
    default=60,
    help="Write models to disk every N seconds. Only affects parallel runs; "
    "serial runs write each model to disk as it is trained.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Verbosity. Default: %(default)s",
    default=0)

add_local_parallelism_args(parser)

TRAIN_DATA_HYPERPARAMETER_DEFAULTS = HyperparameterDefaults(
    subset="all",
    pretrain_min_points=None,
)


def run(argv=sys.argv[1:]):
    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    args.out_models_dir = os.path.abspath(args.out_models_dir)

    configure_logging(verbose=args.verbosity > 1)

    # Resolve the master seed once for the whole run and seed this process's
    # global RNGs before any main-process randomness: the held-out split
    # (subselect_df_held_out), the per-allele ``.sample(frac=1.0)`` data
    # shuffles in train_model, the ``random.shuffle(work_items)`` dispatch
    # ordering, and alleles_by_similarity's ``.sample``. Per-fit weight init
    # and shuffles happen inside workers (which reseed from entropy), so
    # those are pinned separately via derive_seed below. --random-seed
    # defaults to 42 (reproducible out of the box); the resolved value is
    # logged either way.
    master_seed = configure_random_seed(
        args.random_seed, name="train-allele-specific")

    hyperparameters_lst = yaml.safe_load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list), hyperparameters_lst
    print("Loaded hyperparameters list: %s" % str(hyperparameters_lst))

    df = pandas.read_csv(args.data)
    print("Loaded training data: %s" % (str(df.shape)))

    df = df.loc[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    if args.ignore_inequalities and "measurement_inequality" in df.columns:
        print("Dropping measurement_inequality column")
        del df["measurement_inequality"]

    # Canonicalize allele names (merge aliased / retired / alternative spellings
    # to one canonical name) before counting and grouping, so the same allele
    # under different spellings trains as a single model rather than fragmenting
    # into several. Unparseable names are dropped. Allele-specific models have no
    # shared pseudosequence space, so plain normalization (mhcgnomes aliases
    # applied) is the canonical form here.
    allele_norm = {
        a: normalize_allele_name(a, raise_on_error=False)
        for a in df.allele.unique()
    }
    n_unparseable = sum(1 for v in allele_norm.values() if v is None)
    if n_unparseable:
        print("Dropping %d unparseable allele name(s) from training data"
              % n_unparseable)
    df = df.assign(allele=df.allele.map(allele_norm)).dropna(subset=["allele"])

    # Allele counts are in terms of quantitative data only.
    allele_counts = (
        df.loc[df.measurement_type == "quantitative"].allele.value_counts())

    if args.allele:
        alleles = [normalize_allele_name(a) for a in args.allele]
    else:
        alleles = list(allele_counts.loc[
            allele_counts > args.min_measurements_per_allele
        ].index)

    print("Selected %d/%d alleles: %s" % (len(alleles), df.allele.nunique(), ' '.join(alleles)))
    df = df.loc[df.allele.isin(alleles)].dropna()

    if args.held_out_fraction_reciprocal:
        # An explicit --held-out-fraction-seed controls the split directly
        # (legacy behavior, e.g. to reproduce a pre-2.3.0 split). Otherwise
        # --random-seed governs the split too, so the whole run reproduces
        # from one value; derive a stable sub-seed so it's decorrelated from
        # the per-fit seeds.
        held_out_seed = (
            args.held_out_fraction_seed
            if args.held_out_fraction_seed is not None
            else derive_seed(master_seed, "held_out_fraction"))
        df = subselect_df_held_out(
            df,
            recriprocal_held_out_fraction=args.held_out_fraction_reciprocal,
            seed=held_out_seed % (2 ** 32))

    print("Training data: %s" % (str(df.shape)))

    GLOBAL_DATA["train_data"] = df
    GLOBAL_DATA["args"] = args
    # Persist the resolved master seed so workers (which reseed from entropy
    # in worker_init) can derive a stable per-fit seed via derive_seed. The
    # main-process global seeding above doesn't reach worker fits.
    GLOBAL_DATA["seed"] = master_seed
    resolve_local_parallelism_args(
        args,
        workload_name=WORKLOAD_AFFINITY_TRAINING,
        workload_hints={
            "data_bytes": path_size_bytes(args.data),
            "num_rows": len(df),
        },
    )

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    predictor = Class1AffinityPredictor(
        metadata_dataframes={
            'train_data': df,
        })
    serial_run = args.num_jobs == 0

    work_items = []
    for (h, hyperparameters) in enumerate(hyperparameters_lst):
        n_models = None
        if 'n_models' in hyperparameters:
            n_models = hyperparameters.pop("n_models")
        if args.n_models:
            n_models = args.n_models
        if not n_models:
            raise ValueError(
                "Specify --ensemble-size or n_models hyperparameter")

        if args.max_epochs:
            hyperparameters['max_epochs'] = args.max_epochs

        hyperparameters['train_data'] = (
            TRAIN_DATA_HYPERPARAMETER_DEFAULTS.with_defaults(
                hyperparameters.get('train_data', {})))

        if hyperparameters['train_data']['pretrain_min_points'] and (
                'allele_similarity_matrix' not in GLOBAL_DATA):
            print("Generating allele similarity matrix.")
            if not args.allele_sequences:
                parser.error(
                    "Allele sequences required when using pretrain_min_points")
            allele_sequences = pandas.read_csv(
                args.allele_sequences,
                index_col="allele")
            print("Read %d allele sequences" % len(allele_sequences))
            allele_sequences = allele_sequences.loc[
                allele_sequences.index.isin(df.allele.unique())
            ]
            print("Allele sequences matching train data: %d" % len(allele_sequences))
            blosum_encoding = (
                AlleleEncoding(
                    allele_sequences.index.values,
                    allele_sequences.pseudosequence.to_dict())
                .fixed_length_vector_encoded_sequences("BLOSUM62"))
            allele_similarity_matrix = pandas.DataFrame(
                cosine_similarity(
                    blosum_encoding.reshape((len(allele_sequences), -1))),
                index=allele_sequences.index.values,
                columns=allele_sequences.index.values)
            GLOBAL_DATA['allele_similarity_matrix'] = allele_similarity_matrix
            print("Computed allele similarity matrix")
            print(allele_similarity_matrix)

        for (i, allele) in enumerate(df.allele.unique()):
            for model_num in range(n_models):
                work_dict = {
                    'n_models': 1,
                    'allele_num': i,
                    'n_alleles': len(alleles),
                    'model_num': model_num,
                    'hyperparameter_set_num': h,
                    'num_hyperparameter_sets': len(hyperparameters_lst),
                    'allele': allele,
                    'hyperparameters': hyperparameters,
                    'verbose': args.verbosity,
                    'progress_print_interval': None if not serial_run else 5.0,
                    'predictor': predictor if serial_run else None,
                    'save_to': args.out_models_dir if serial_run else None,
                }
                work_items.append(work_dict)

    apply_resolved_training_hyperparameters_to_work_items(work_items, args)

    start = time.time()

    worker_pool = worker_pool_with_gpu_assignments_from_args(args)
    if args.num_jobs != 0:
        print("Processing %d work items in parallel." % len(work_items))

        # The estimated time to completion is more accurate if we randomize
        # the order of the work.
        random.shuffle(work_items)

        # NOTE: torch.compile warmup is currently wired only for the
        # pan-allele and processing trainers, which expose a
        # ``compile_warmup_only=True`` short-circuit in their
        # ``train_model``. The allele-specific trainer goes through
        # ``fit_allele_specific_predictors`` and would need its own
        # warmup short-circuit; until that exists, the production pool
        # eats one ~30 s first-compile per architecture per process,
        # which is acceptable for this trainer's smaller sweeps.

        assert worker_pool is not None
        attach_constant_data_to_work_items_if_needed(
            work_items, GLOBAL_DATA, worker_pool
        )
        results_generator = worker_pool.imap_unordered(
            partial(call_wrapped_kwargs, train_model),
            work_items,
            chunksize=1)

        unsaved_predictors = []
        last_save_time = time.time()
        for new_predictor in tqdm.tqdm(results_generator, total=len(work_items)):
            unsaved_predictors.append(new_predictor)

            if time.time() > last_save_time + args.save_interval:
                # Save current predictor.
                save_start = time.time()
                new_model_names = predictor.merge_in_place(unsaved_predictors)
                predictor.save(
                    args.out_models_dir,
                    model_names_to_write=new_model_names,
                    write_metadata=False)
                print(
                    "Saved predictor (%d models total) including %d new models "
                    "in %0.2f sec to %s" % (
                        len(predictor.neural_networks),
                        len(new_model_names),
                        time.time() - save_start,
                        args.out_models_dir))
                unsaved_predictors = []
                last_save_time = time.time()

        predictor.merge_in_place(unsaved_predictors)

    else:
        assert worker_pool is None
        # Run in serial. In this case, every worker is passed the same predictor,
        # which it adds models to, so no merging is required. It also saves
        # as it goes so no saving is required at the end.
        for _ in tqdm.trange(len(work_items)):
            item = work_items.pop(0)  # want to keep freeing up memory
            work_predictor = train_model(**item)
            assert work_predictor is predictor
        assert not work_items

    print("Saving final predictor to: %s" % args.out_models_dir)
    predictor.save(args.out_models_dir)  # write all models just to be sure
    write_generate_sh(args.out_models_dir)
    print("Done.")

    print("*" * 30)
    training_time = time.time() - start
    print("Trained affinity predictor with %d networks in %0.2f min." % (
        len(predictor.neural_networks), training_time / 60.0))
    print("*" * 30)

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Predictor written to: %s" % args.out_models_dir)


def alleles_by_similarity(allele):
    allele_similarity = GLOBAL_DATA['allele_similarity_matrix']
    if allele not in allele_similarity.columns:
        # Use random alleles
        print("No similar alleles for: %s" % allele)
        return [allele] + list(
            allele_similarity.columns.to_series().sample(frac=1.0))
    return (
        allele_similarity[allele] + (
            allele_similarity.index == allele)  # force specified allele first
    ).sort_values(ascending=False).index.tolist()


def train_model(
        n_models,
        allele_num,
        n_alleles,
        model_num,
        hyperparameter_set_num,
        num_hyperparameter_sets,
        allele,
        hyperparameters,
        verbose,
        progress_print_interval,
        predictor,
        save_to,
        constant_data=GLOBAL_DATA):

    if predictor is None:
        predictor = Class1AffinityPredictor()

    # One stable per-fit seed controls every stochastic step of this work
    # item. Seed the worker's global RNG from it up front — workers reseed
    # from entropy in worker_init, so the main process's global seeding does
    # not reach here. This pins the randomness that runs before fit(): the
    # similar-allele selection in alleles_by_similarity (pretrain path) and
    # the data shuffle below. Derived from the run's master seed plus this
    # fit's identity coordinates, so the whole run is reproducible from
    # --random-seed regardless of dispatch order. When no master seed was
    # recorded (legacy run) training is left stochastic.
    master_seed = constant_data.get("seed")
    work_item_seed = (
        None if master_seed is None
        else derive_seed(
            master_seed, allele, hyperparameter_set_num, model_num))
    if work_item_seed is not None:
        numpy.random.seed(work_item_seed % (2 ** 32))

    pretrain_min_points = hyperparameters['train_data']['pretrain_min_points']

    data = constant_data["train_data"]

    subset = hyperparameters.get("train_data", {}).get("subset", "all")
    if subset == "quantitative":
        data = data.loc[
            data.measurement_type == "quantitative"
        ]
    elif subset == "all":
        pass
    else:
        raise ValueError("Unsupported subset: %s" % subset)

    data_size_by_allele = data.allele.value_counts()

    if pretrain_min_points:
        similar_alleles = alleles_by_similarity(allele)
        alleles = []
        while not alleles or data_size_by_allele.loc[alleles].sum() < pretrain_min_points:
            alleles.append(similar_alleles.pop(0))
        data = data.loc[data.allele.isin(alleles)]
        assert len(data) >= pretrain_min_points, (len(data), pretrain_min_points)
        train_rounds = (data.allele == allele).astype(int).values
    else:
        train_rounds = None
        data = data.loc[data.allele == allele]

    progress_preamble = (
        "[%2d / %2d hyperparameters] "
        "[%4d / %4d alleles] %s " % (
            hyperparameter_set_num + 1,
            num_hyperparameter_sets,
            allele_num + 1,
            n_alleles,
            allele))

    # Shuffle with an explicit random_state derived from the work-item seed,
    # so the data order is reproducible and decoupled from however many draws
    # alleles_by_similarity took from the global RNG above.
    train_data = data.sample(
        frac=1.0,
        random_state=(
            None if work_item_seed is None
            else work_item_seed % (2 ** 32)))
    predictor.fit_allele_specific_predictors(
        n_models=n_models,
        architecture_hyperparameters_list=[hyperparameters],
        allele=allele,
        peptides=train_data.peptide.values,
        affinities=train_data.measurement_value.values,
        inequalities=(
            train_data.measurement_inequality.values
            if "measurement_inequality" in train_data.columns else None),
        train_rounds=train_rounds,
        models_dir_for_save=save_to,
        progress_preamble=progress_preamble,
        progress_print_interval=progress_print_interval,
        seed=work_item_seed,
        verbose=verbose)

    return predictor


def subselect_df_held_out(df, recriprocal_held_out_fraction=10, seed=0):
    df = df.copy()
    df["allele_peptide"] = df.allele + "_" + df.peptide

    kf = StratifiedKFold(
        n_splits=recriprocal_held_out_fraction,
        shuffle=True,
        random_state=seed)

    # Stratify by both allele and binder vs. nonbinder.
    df["key"] = [
        "%s_%s" % (
            row.allele,
            "binder" if row.measurement_value <= 500 else "nonbinder")
        for (_, row) in df.iterrows()
    ]

    (train, test) = next(kf.split(df, df.key))
    selected_allele_peptides = df.iloc[train].allele_peptide.unique()
    result_df = df.loc[
        df.allele_peptide.isin(selected_allele_peptides)
    ]
    del result_df["allele_peptide"]
    return result_df

if __name__ == '__main__':
    run()

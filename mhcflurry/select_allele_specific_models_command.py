"""
Model select class1 single allele models.
"""
import argparse
import os
import signal
import sys
import time
import traceback
import random

import numpy
import pandas
from scipy.stats import kendalltau

from mhcnames import normalize_allele_name
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_affinity_predictor import Class1AffinityPredictor
from .encodable_sequences import EncodableSequences
from .common import configure_logging, random_peptides
from .parallelism import worker_pool_with_gpu_assignments_from_args, add_worker_pool_args
from .regression_target import from_ic50


# To avoid pickling large matrices to send to child processes when running in
# parallel, we use this global variable as a place to store data. Data that is
# stored here before creating the thread pool will be inherited to the child
# processes upon fork() call, allowing us to share large data with the workers
# via shared memory.
GLOBAL_DATA = {}


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    required=False,
    help=(
        "Model selection data CSV. Expected columns: "
        "allele, peptide, measurement_value"))
parser.add_argument(
    "--exclude-data",
    metavar="FILE.csv",
    required=False,
    help=(
        "Data to EXCLUDE from model selection. Useful to specify the original "
        "training data used"))
parser.add_argument(
    "--models-dir",
    metavar="DIR",
    required=True,
    help="Directory to read models")
parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    required=True,
    help="Directory to write selected models")
parser.add_argument(
    "--out-unselected-predictions",
    metavar="FILE.csv",
    help="Write predictions for validation data using unselected predictor to "
    "FILE.csv")
parser.add_argument(
    "--allele",
    default=None,
    nargs="+",
    help="Alleles to select models for. If not specified, all alleles with "
    "enough measurements will be used.")
parser.add_argument(
    "--combined-min-models",
    type=int,
    default=8,
    metavar="N",
    help="Min number of models to select per allele when using combined selector")
parser.add_argument(
    "--combined-max-models",
    type=int,
    default=1000,
    metavar="N",
    help="Max number of models to select per allele when using combined selector")
parser.add_argument(
    "--combined-weights",
    type=int,
    nargs=3,
    default=[1,1,1],
    help="Weights for combined predictor in order: mass-spec MSE consensus")
parser.add_argument(
    "--mass-spec-min-measurements",
    type=int,
    metavar="N",
    default=50,
    help="Min number of measurements required for an allele to use mass-spec model "
    "selection")
parser.add_argument(
    "--mass-spec-min-models",
    type=int,
    default=8,
    metavar="N",
    help="Min number of models to select per allele when using mass-spec selector")
parser.add_argument(
    "--mass-spec-max-models",
    type=int,
    default=1000,
    metavar="N",
    help="Max number of models to select per allele when using mass-spec selector")
parser.add_argument(
    "--mse-min-measurements",
    type=int,
    metavar="N",
    default=50,
    help="Min number of measurements required for an allele to use MSE model "
    "selection")
parser.add_argument(
    "--mse-min-models",
    type=int,
    default=8,
    metavar="N",
    help="Min number of models to select per allele when using MSE selector")
parser.add_argument(
    "--mse-max-models",
    type=int,
    default=1000,
    metavar="N",
    help="Max number of models to select per allele when using MSE selector")
parser.add_argument(
    "--scoring",
    nargs="+",
    choices=("combined-all", "mse", "mass-spec", "consensus"),
    default=["mse", "consensus"],
    help="Scoring procedures to use in order")
parser.add_argument(
    "--consensus-min-models",
    type=int,
    default=8,
    metavar="N",
    help="Min number of models to select per allele when using consensus selector")
parser.add_argument(
    "--consensus-max-models",
    type=int,
    default=1000,
    metavar="N",
    help="Max number of models to select per allele when using consensus selector")
parser.add_argument(
    "--consensus-num-peptides-per-length",
    type=int,
    default=10000,
    help="Num peptides per length to use for consensus scoring")
parser.add_argument(
    "--mass-spec-regex",
    metavar="REGEX",
    default="mass[- ]spec",
    help="Regular expression for mass-spec data. Runs on measurement_source col."
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

    args.out_models_dir = os.path.abspath(args.out_models_dir)

    configure_logging(verbose=args.verbosity > 1)

    input_predictor = Class1AffinityPredictor.load(args.models_dir)
    print("Loaded: %s" % input_predictor)

    if args.allele:
        alleles = [normalize_allele_name(a) for a in args.allele]
    else:
        alleles = input_predictor.supported_alleles

    metadata_dfs = {}
    if args.data:
        df = pandas.read_csv(args.data)
        print("Loaded data: %s" % (str(df.shape)))

        df = df.ix[
            (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
        ]
        print("Subselected to 8-15mers: %s" % (str(df.shape)))

        # Allele names in data are assumed to be already normalized.
        df = df.loc[df.allele.isin(alleles)].dropna()
        print("Selected %d alleles: %s" % (len(alleles), ' '.join(alleles)))

        if args.exclude_data:
            exclude_df = pandas.read_csv(args.exclude_data)
            metadata_dfs["model_selection_exclude"] = exclude_df
            print("Loaded exclude data: %s" % (str(df.shape)))

            df["_key"] = df.allele + "__" + df.peptide
            exclude_df["_key"] = exclude_df.allele + "__" + exclude_df.peptide
            df["_excluded"] = df._key.isin(exclude_df._key.unique())
            print("Excluding measurements per allele (counts): ")
            print(df.groupby("allele")._excluded.sum())

            print("Excluding measurements per allele (fractions): ")
            print(df.groupby("allele")._excluded.mean())

            df = df.loc[~df._excluded]
            del df["_excluded"]
            del df["_key"]
            print("Reduced data to: %s" % (str(df.shape)))

        metadata_dfs["model_selection_data"] = df

        df["mass_spec"] = df.measurement_source.str.contains(
            args.mass_spec_regex)
    else:
        df = None

    if args.out_unselected_predictions:
        df["unselected_prediction"] = input_predictor.predict(
            alleles=df.allele.values,
            peptides=df.peptide.values)
        df.to_csv(args.out_unselected_predictions)
        print("Wrote: %s" % args.out_unselected_predictions)

    selectors = {}
    selector_to_model_selection_kwargs = {}

    def make_selector(scoring):
        start = time.time()
        if scoring == "combined-all":
            model_selection_kwargs = {
                'min_models': args.combined_min_models,
                'max_models': args.combined_max_models,
            }
            selector = CombinedModelSelector([
                make_selector("mass-spec")[0],
                make_selector("mse")[0],
                make_selector("consensus")[0],
            ], weights=args.combined_weights)
        elif scoring == "mse":
            model_selection_kwargs = {
                'min_models': args.mse_min_models,
                'max_models': args.mse_max_models,
            }
            selector = MSEModelSelector(
                df=df.loc[~df.mass_spec],
                predictor=input_predictor,
                min_measurements=args.mse_min_measurements)
        elif scoring == "mass-spec":
            mass_spec_df = df.loc[df.mass_spec]
            model_selection_kwargs = {
                'min_models': args.mass_spec_min_models,
                'max_models': args.mass_spec_max_models,
            }
            selector = MassSpecModelSelector(
                df=mass_spec_df,
                predictor=input_predictor,
                min_measurements=args.mass_spec_min_measurements)
        elif scoring == "consensus":
            model_selection_kwargs = {
                'min_models': args.consensus_min_models,
                'max_models': args.consensus_max_models,
            }
            selector = ConsensusModelSelector(
                predictor=input_predictor,
                num_peptides_per_length=args.consensus_num_peptides_per_length)
        else:
            raise ValueError("Unsupported scoring method: %s" % scoring)
        print("Instantiated model selector %s in %0.2f sec." % (
            scoring, time.time() - start))
        return (selector, model_selection_kwargs)

    for scoring in args.scoring:
        (selector, model_selection_kwargs) = make_selector(scoring)

        selectors[scoring] = selector
        selector_to_model_selection_kwargs[scoring] = model_selection_kwargs

    print("Selectors for alleles:")
    allele_to_selector = {}
    allele_to_model_selection_kwargs = {}
    for allele in alleles:
        selector = None
        for possible_selector in args.scoring:
            if selectors[possible_selector].usable_for_allele(allele=allele):
                selector = selectors[possible_selector]
                print("%20s %s" % (allele, selector.plan_summary(allele)))
                break
        if selector is None:
            raise ValueError("No selectors usable for allele: %s" % allele)
        allele_to_selector[allele] = selector
        allele_to_model_selection_kwargs[allele] = (
            selector_to_model_selection_kwargs[possible_selector])

    GLOBAL_DATA["input_predictor"] = input_predictor
    GLOBAL_DATA["allele_to_selector"] = allele_to_selector
    GLOBAL_DATA["allele_to_model_selection_kwargs"] = allele_to_model_selection_kwargs

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    result_predictor = Class1AffinityPredictor(metadata_dataframes=metadata_dfs)

    worker_pool = worker_pool_with_gpu_assignments_from_args(args)

    start = time.time()

    if worker_pool is None:
        # Serial run
        print("Running in serial.")
        results = (
            model_select(allele) for allele in alleles)
    else:
        # Parallel run
        random.shuffle(alleles)
        results = worker_pool.imap_unordered(
            model_select,
            alleles,
            chunksize=1)

    model_selection_dfs = []
    for result in tqdm.tqdm(results, total=len(alleles)):
        model_selection_dfs.append(
            result.metadata_dataframes['model_selection'])
        result_predictor.merge_in_place([result])

    model_selection_df = pandas.concat(model_selection_dfs, ignore_index=True)
    model_selection_df["selector"] = model_selection_df.allele.map(
        allele_to_selector)
    result_predictor.metadata_dataframes["model_selection"] = model_selection_df

    print("Done model selecting for %d alleles." % len(alleles))
    result_predictor.save(args.out_models_dir)

    model_selection_time = time.time() - start

    if worker_pool:
        worker_pool.close()
        worker_pool.join()

    print("Model selection time %0.2f min." % (model_selection_time / 60.0))
    print("Predictor written to: %s" % args.out_models_dir)


def model_select(allele):
    global GLOBAL_DATA
    predictor = GLOBAL_DATA["input_predictor"]
    selector = GLOBAL_DATA["allele_to_selector"][allele]
    model_selection_kwargs = GLOBAL_DATA[
        "allele_to_model_selection_kwargs"
    ][allele]
    return predictor.model_select(
        score_function=selector.score_function(allele=allele),
        alleles=[allele],
        **model_selection_kwargs)


def cache_encoding(predictor, peptides):
    # Encode the peptides for each neural network, so the encoding
    # becomes cached.
    for network in predictor.neural_networks:
        network.peptides_to_network_input(peptides)


class ScoreFunction(object):
    def __init__(self, function, summary=None):
        self.function = function
        self.summary = summary if summary else "(n/a)"

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class CombinedModelSelector(object):
    def __init__(self, model_selectors, weights=None):
        if weights is None:
            weights = numpy.ones(shape=(len(model_selectors),))
        self.model_selectors = model_selectors
        self.selector_to_weight = dict(zip(self.model_selectors, weights))

    def usable_for_allele(self, allele):
        return any(
            selector.usable_for_allele(allele)
            for selector in self.model_selectors)

    def plan_summary(self, allele):
        return self.score_function(allele, dry_run=True).summary

    def score_function(self, allele, dry_run=False):
        selector_to_max_weighted_score = {}
        for selector in self.model_selectors:
            weight = self.selector_to_weight[selector]
            if selector.usable_for_allele(allele):
                max_weighted_score = selector.max_absolute_value(allele) * weight
            else:
                max_weighted_score = 0
            selector_to_max_weighted_score[selector] = max_weighted_score
        max_total_score = sum(selector_to_max_weighted_score.values())

        # Use only selectors that can contribute >1% to the total score
        selectors_to_use = [
            selector
            for selector in self.model_selectors
            if selector_to_max_weighted_score[selector] > max_total_score / 100.
        ]

        summary = ", ".join([
            "%s(|%.3f|)" % (
                selector.plan_summary(allele),
                selector_to_max_weighted_score[selector])
            for selector in selectors_to_use
        ])

        if dry_run:
            score = None
        else:
            score_functions_and_weights = [
                (selector.score_function(allele=allele),
                 self.selector_to_weight[selector])
                for selector in selectors_to_use
            ]

            def score(predictor):
                scores = numpy.array([
                    score_function(predictor) * weight
                    for (score_function, weight) in score_functions_and_weights
                ])
                return scores.sum()
        return ScoreFunction(score, summary=summary)


class ConsensusModelSelector(object):
    def __init__(
            self,
            predictor,
            num_peptides_per_length=10000,
            multiply_score_by_value=10.0):

        (min_length, max_length) = predictor.supported_peptide_lengths
        peptides = []
        for length in range(min_length, max_length + 1):
            peptides.extend(
                random_peptides(num_peptides_per_length, length=length))

        self.peptides = EncodableSequences.create(peptides)
        self.predictor = predictor
        self.multiply_score_by_value = multiply_score_by_value
        cache_encoding(self.predictor, self.peptides)

    def usable_for_allele(self, allele):
        return True

    def max_absolute_value(self, allele):
        return self.multiply_score_by_value

    def plan_summary(self, allele):
        return "consensus (%d points)" % len(self.peptides)

    def score_function(self, allele):
        full_ensemble_predictions = self.predictor.predict(
            allele=allele,
            peptides=self.peptides)

        def score(predictor):
            predictions = predictor.predict(
                allele=allele,
                peptides=self.peptides,
            )
            return (
                kendalltau(predictions, full_ensemble_predictions).correlation *
                self.multiply_score_by_value)

        return ScoreFunction(
            score, summary=self.plan_summary(allele))


class MSEModelSelector(object):
    def __init__(
            self,
            df,
            predictor,
            min_measurements=1,
            multiply_score_by_data_size=True):

        self.df = df
        self.predictor = predictor
        self.min_measurements = min_measurements
        self.multiply_score_by_data_size = multiply_score_by_data_size

    def usable_for_allele(self, allele):
        return (self.df.allele == allele).sum() >= self.min_measurements

    def max_absolute_value(self, allele):
        if self.multiply_score_by_data_size:
            return (self.df.allele == allele).sum()
        else:
            return 1.0

    def plan_summary(self, allele):
        return self.score_function(allele).summary

    def score_function(self, allele):
        sub_df = self.df.loc[self.df.allele == allele]
        peptides = EncodableSequences.create(sub_df.peptide.values)

        def score(predictor):
            predictions = predictor.predict(
                allele=allele,
                peptides=peptides,
            )
            deviations = from_ic50(predictions) - from_ic50(
                sub_df.measurement_value)

            if 'measurement_inequality' in sub_df.columns:
                # Must reverse meaning of inequality since we are working with
                # transformed 0-1 values, which are anti-correlated with the ic50s.
                # The measurement_inequality column is given in terms of ic50s.
                deviations.loc[
                    (
                    (sub_df.measurement_inequality == "<") & (deviations > 0)) |
                    ((sub_df.measurement_inequality == ">") & (deviations < 0))
                    ] = 0.0

            return  (1 - (deviations ** 2).mean()) * (
                len(sub_df) if self.multiply_score_by_data_size else 1)

        summary = "mse (%d points)" % (len(sub_df))
        return ScoreFunction(score, summary=summary)


class MassSpecModelSelector(object):
    def __init__(
            self,
            df,
            predictor,
            decoys_per_length=0,
            min_measurements=100,
            multiply_score_by_data_size=True):

        # Index is peptide, columns are alleles
        hit_matrix = df.groupby(
            ["peptide", "allele"]).measurement_value.count().unstack().fillna(
            0).astype(bool)

        if decoys_per_length:
            (min_length, max_length) = predictor.supported_peptide_lengths
            decoys = []
            for length in range(min_length, max_length + 1):
                decoys.extend(
                    random_peptides(decoys_per_length, length=length))

            decoy_matrix = pandas.DataFrame(
                index=decoys, columns=hit_matrix.columns, dtype=bool)
            decoy_matrix[:] = False
            full_matrix = pandas.concat([hit_matrix, decoy_matrix])
        else:
            full_matrix = hit_matrix

        full_matrix = full_matrix.sample(frac=1.0).astype(float)

        self.df = full_matrix
        self.predictor = predictor
        self.min_measurements = min_measurements
        self.multiply_score_by_data_size = multiply_score_by_data_size

        self.peptides = EncodableSequences.create(full_matrix.index.values)
        cache_encoding(self.predictor, self.peptides)

    @staticmethod
    def ppv(y_true, predictions):
        df = pandas.DataFrame({"prediction": predictions, "y_true": y_true})
        return df.sort_values("prediction", ascending=True)[
            : int(y_true.sum())
        ].y_true.mean()

    def usable_for_allele(self, allele):
        return allele in self.df.columns and (
            self.df[allele].sum() >= self.min_measurements)

    def max_absolute_value(self, allele):
        if self.multiply_score_by_data_size:
            return self.df[allele].sum()
        else:
            return 1.0

    def plan_summary(self, allele):
        return self.score_function(allele).summary

    def score_function(self, allele):
        total_hits = self.df[allele].sum()
        total_decoys = (self.df[allele] == 0).sum()
        multiplier = total_hits if self.multiply_score_by_data_size else 1
        def score(predictor):
            predictions = predictor.predict(
                allele=allele,
                peptides=self.peptides,
            )
            return self.ppv(self.df[allele], predictions) * multiplier

        summary = "mass-spec (%d hits / %d decoys)" % (total_hits, total_decoys)
        return ScoreFunction(score, summary=summary)


if __name__ == '__main__':
    run()

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
from functools import partial

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
    choices=("mse", "mass-spec", "consensus"),
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
    default=100000,
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
    for scoring in args.scoring:
        if scoring == "mse":
            selector = MSEModelSelector(
                df=df,
                predictor=input_predictor,
                min_measurements=args.mse_min_measurements,
                model_selection_kwargs={
                    'min_models': args.mse_min_models,
                    'max_models': args.mse_max_models,
                })
        elif scoring == "mass-spec":
            mass_spec_df = df.loc[df.mass_spec]
            selector = MassSpecModelSelector(
                df=mass_spec_df,
                predictor=input_predictor,
                min_measurements=args.mass_spec_min_measurements,
                model_selection_kwargs={
                    'min_models': args.mass_spec_min_models,
                    'max_models': args.mass_spec_max_models,
                })
        elif scoring == "consensus":
            selector = ConsensusModelSelector(
                predictor=input_predictor,
                num_peptides_per_length=args.consensus_num_peptides_per_length,
                model_selection_kwargs={
                    'min_models': args.consensus_min_models,
                    'max_models': args.consensus_max_models,
                })
        else:
            raise ValueError("Unsupported scoring method: %s" % scoring)
        selectors[scoring] = selector

    print("Selectors for alleles:")
    allele_to_selector = {}
    for allele in alleles:
        selector = None
        for possible_selector in args.scoring:
            if selectors[possible_selector].usable_for_allele(allele=allele):
                selector = selectors[possible_selector]
                print("%20s %s" % (allele, possible_selector))
                break
        if selector is None:
            raise ValueError("No selectors usable for allele: %s" % allele)
        allele_to_selector[allele] = selector

    GLOBAL_DATA["allele_to_selector"] = allele_to_selector

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
    selector = GLOBAL_DATA["allele_to_selector"][allele]
    return selector.select(allele)


class ConsensusModelSelector(object):
    def __init__(
            self,
            predictor,
            num_peptides_per_length=100000,
            model_selection_kwargs={}):

        (min_length, max_length) = predictor.supported_peptide_lengths
        peptides = []
        for length in range(min_length, max_length + 1):
            peptides.extend(
                random_peptides(num_peptides_per_length, length=length))

        self.peptides = EncodableSequences.create(peptides)
        self.predictor = predictor
        self.model_selection_kwargs = model_selection_kwargs

        # Encode the peptides for each neural network, so the encoding
        # becomes cached.
        for network in predictor.neural_networks:
            network.peptides_to_network_input(self.peptides)

    def usable_for_allele(self, allele):
        return True

    def score_function(self, allele, ensemble_predictions, predictor):
        predictions = predictor.predict(
            allele=allele,
            peptides=self.peptides,
        )
        return kendalltau(predictions, ensemble_predictions).correlation

    def select(self, allele):
        full_ensemble_predictions = self.predictor.predict(
            allele=allele,
            peptides=self.peptides)

        return self.predictor.model_select(
            score_function=partial(
                self.score_function, allele, full_ensemble_predictions),
            alleles=[allele],
            **self.model_selection_kwargs
        )


class MSEModelSelector(object):
    def __init__(
            self,
            df,
            predictor,
            model_selection_kwargs={},
            min_measurements=1):

        self.df = df
        self.predictor = predictor
        self.model_selection_kwargs = model_selection_kwargs
        self.min_measurements = min_measurements

    def usable_for_allele(self, allele):
        return (self.df.allele == allele).sum() >= self.min_measurements

    @staticmethod
    def score_function(allele, sub_df, peptides, predictor):
        predictions = predictor.predict(
            allele=allele,
            peptides=peptides,
        )
        deviations = from_ic50(predictions) - from_ic50(sub_df.measurement_value)

        if 'measurement_inequality' in sub_df.columns:
            # Must reverse meaning of inequality since we are working with
            # transformed 0-1 values, which are anti-correlated with the ic50s.
            # The measurement_inequality column is given in terms of ic50s.
            deviations.loc[
                ((sub_df.measurement_inequality == "<") & (deviations > 0)) |
                ((sub_df.measurement_inequality == ">") & (deviations < 0))
            ] = 0.0

        return -1 * (deviations**2).mean()

    def select(self, allele):
        sub_df = self.df.loc[self.df.allele == allele]
        peptides = EncodableSequences.create(sub_df.peptide.values)

        return self.predictor.model_select(
            score_function=partial(
                self.score_function, allele, sub_df, peptides),
            alleles=[allele],
            **self.model_selection_kwargs
        )


class MassSpecModelSelector(object):
    def __init__(
            self,
            df,
            predictor,
            decoys_per_length=5000,
            model_selection_kwargs={},
            min_measurements=100):

        (min_length, max_length) = predictor.supported_peptide_lengths
        decoys = []
        for length in range(min_length, max_length + 1):
            decoys.extend(
                random_peptides(decoys_per_length, length=length))

        # Index is peptide, columns are alleles
        hit_matrix = df.groupby(["peptide", "allele"]).measurement_value.count().unstack().fillna(0).astype(bool)

        decoy_matrix = pandas.DataFrame(
            index=decoys, columns=hit_matrix.columns, dtype=bool)
        decoy_matrix[:] = False
        full_matrix = pandas.concat([hit_matrix, decoy_matrix]).sample(frac=1.0)

        self.df = full_matrix
        self.predictor = predictor
        self.model_selection_kwargs = model_selection_kwargs
        self.min_measurements = min_measurements

        self.peptides = EncodableSequences.create(full_matrix.index.values)

        # Encode the peptides for each neural network, so the encoding
        # becomes cached.
        for network in predictor.neural_networks:
            network.peptides_to_network_input(self.peptides)

    @staticmethod
    def ppv(y_true, predictions):
        df = pandas.DataFrame({"prediction": predictions, "y_true": y_true})
        return df.sort_values("prediction", ascending=True)[
            : int(y_true.sum())
        ].y_true.mean()

    def usable_for_allele(self, allele):
        return allele in self.df.columns and (
            self.df[allele].sum() >= self.min_measurements)

    def score_function(self, allele, predictor):
        predictions = predictor.predict(
            allele=allele,
            peptides=self.peptides,
        )
        return self.ppv(self.df[allele], predictions)

    def select(self, allele):
        return self.predictor.model_select(
            score_function=partial(
                self.score_function, allele),
            alleles=[allele],
            **self.model_selection_kwargs
        )


if __name__ == '__main__':
    run()

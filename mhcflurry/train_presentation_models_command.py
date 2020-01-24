"""
Train Class1 presentation models.
"""
from __future__ import print_function
import argparse
import os
import signal
import sys
import time
import traceback

import pandas
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

from .class1_cleavage_predictor import Class1CleavagePredictor
from .class1_affinity_predictor import Class1AffinityPredictor
from .class1_presentation_predictor import Class1PresentationPredictor
from .common import configure_logging

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    help="Training data CSV. Expected columns: peptide, n_flank, c_flank, hit")
parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    required=True,
    help="Directory to write models and manifest")
parser.add_argument(
    "--affinity-predictor",
    metavar="DIR",
    required=True,
    help="Affinity predictor models dir")
parser.add_argument(
    "--cleavage-predictor-with-flanks",
    metavar="DIR",
    required=True,
    help="Cleavage predictor with flanking")
parser.add_argument(
    "--cleavage-predictor-without-flanks",
    metavar="DIR",
    required=True,
    help="Cleavage predictor without flanking")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Default: %(default)s",
    default=0)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Launch python debugger on error")
parser.add_argument(
    "--hla-column",
    default="hla",
    help="Column in data giving space-separated MHC I alleles")
parser.add_argument(
    "--target-column",
    default="hit",
    help="Column in data giving hit (1) vs decoy (0)")

def run(argv=sys.argv[1:]):
    # On sigusr1 print stack trace
    print("To show stack trace, run:\nkill -s USR1 %d" % os.getpid())
    signal.signal(signal.SIGUSR1, lambda sig, frame: traceback.print_stack())

    args = parser.parse_args(argv)

    if args.debug:
        try:
            return main(args)
        except Exception as e:
            print(e)
            import ipdb  # pylint: disable=import-error
            ipdb.set_trace()
            raise
    else:
        return main(args)


def main(args):
    print("Arguments:")
    print(args)

    args.out_models_dir = os.path.abspath(args.out_models_dir)
    configure_logging(verbose=args.verbosity > 1)

    df = pandas.read_csv(args.data)
    print("Loaded training data: %s" % (str(df.shape)))
    df = df.loc[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    df["experiment_id"] = df[args.hla_columns]
    experiment_to_alleles = dict((
        key, key.split()) for key in df.experiment_id.unique())

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    affinity_predictor = Class1AffinityPredictor.load(args.affinity_predictor)
    cleavage_predictor_with_flanks = Class1CleavagePredictor.load(
        args.cleavage_predictor_with_flanking)
    cleavage_predictor_without_flanks = Class1CleavagePredictor.load(
        args.cleavage_predictor_without_flanking)

    predictor = Class1PresentationPredictor(
        affinity_predictor=affinity_predictor,
        cleavage_predictor_with_flanks=cleavage_predictor_with_flanks,
        cleavage_predictor_without_flanks=cleavage_predictor_without_flanks)

    print("Fitting.")
    start = time.time()
    predictor.fit(
        targets=df[args.target_column].values,
        peptides=df.peptide.values,
        alleles=experiment_to_alleles,
        experiment_names=df.experiment_id,
        n_flanks=df.n_flank.values,
        c_flanks=df.c_flank.values,
        verbose=args.verbose)
    print("Done fitting in", time.time() - start, "seconds")

    print("Saving")
    predictor.save(args.out_models_dir)
    print("Wrote", args.out_models_dir)


if __name__ == '__main__':
    run()

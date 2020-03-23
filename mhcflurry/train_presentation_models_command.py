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

from .class1_processing_predictor import Class1ProcessingPredictor
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
    "--processing-predictor-with-flanks",
    metavar="DIR",
    required=True,
    help="Processing predictor with flanks")
parser.add_argument(
    "--processing-predictor-without-flanks",
    metavar="DIR",
    required=True,
    help="Processing predictor without flanks")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Default: %(default)s",
    default=1)
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

    df["experiment_id"] = df[args.hla_column]
    experiment_to_alleles = dict((
        key, key.split()) for key in df.experiment_id.unique())

    if not os.path.exists(args.out_models_dir):
        print("Attempting to create directory: %s" % args.out_models_dir)
        os.mkdir(args.out_models_dir)
        print("Done.")

    affinity_predictor = Class1AffinityPredictor.load(
        args.affinity_predictor,
        optimization_level=0)
    processing_predictor_with_flanks = Class1ProcessingPredictor.load(
        args.processing_predictor_with_flanks)
    processing_predictor_without_flanks = Class1ProcessingPredictor.load(
        args.processing_predictor_without_flanks)

    predictor = Class1PresentationPredictor(
        affinity_predictor=affinity_predictor,
        processing_predictor_with_flanks=processing_predictor_with_flanks,
        processing_predictor_without_flanks=processing_predictor_without_flanks)

    print("Fitting.")
    start = time.time()
    predictor.fit(
        targets=df[args.target_column].values,
        peptides=df.peptide.values,
        alleles=experiment_to_alleles,
        sample_names=df.experiment_id,
        n_flanks=df.n_flank.values,
        c_flanks=df.c_flank.values,
        verbose=args.verbosity)
    print("Done fitting in", time.time() - start, "seconds")

    print("Saving")
    predictor.save(args.out_models_dir)
    print("Wrote", args.out_models_dir)


if __name__ == '__main__':
    run()

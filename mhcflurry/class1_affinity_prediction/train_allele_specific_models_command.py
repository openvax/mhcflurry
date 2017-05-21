"""
Train single allele models

"""
import sys
import argparse
import json
import os
import pickle

import pandas

import mhcnames


from .class1_neural_network import Class1NeuralNetwork
from ..common import configure_logging

def normalize_allele_name(s):
    try:
        return mhcnames.normalize_allele_name(s)
    except Exception:
        return "UNKNOWN"


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    required=True,
    help="Training data")
parser.add_argument(
    "--out-models-dir",
    required=True,
    help="Directory to write models and manifest")
parser.add_argument(
    "--hyperparameters",
    required=True,
    help="JSON of hyperparameters")
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
    "--only-quantitative",
    action="store_true",
    help="Exclude qualitative measurements",
    default=False)
parser.add_argument(
    "--verbosity",
    type=int,
    help="Default: %(default)s",
    default=1)



def run():
    args = parser.parse_args(sys.argv[1:])

    configure_logging(verbose=args.verbosity > 1)

    hyperparameters_lst = json.load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list)
    print("Loaded hyperparameters list: %s" % str(hyperparameters_lst))

    df = pandas.read_csv(args.data)
    print("Loaded training data: %s" % (str(df.shape)))

    df = df.ix[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    allele_counts = df.allele.value_counts()

    if args.allele:
        alleles = args.allelle
        df = df.ix[df.allele.isin(alleles)]
    else:
        alleles = list(allele_counts.ix[
            allele_counts > args.min_measurements_per_allele
        ].index)

    print("Selected %d alleles: %s" % (len(alleles), ' '.join(alleles)))
    print("Training data: %s" % (str(df.shape)))

    manifest = pandas.DataFrame()
    manifest["name"] = []
    manifest["hyperparameters_index"] = []
    manifest["model_group"] = []
    manifest["allele"] = []
    manifest["hyperparameters"] = []
    manifest["history"] = []
    manifest["num_measurements"] = []
    manifest["fit_seconds"] = []

    manifest_path = os.path.join(args.out_models_dir, "manifest.csv")

    for (h, hyperparameters) in enumerate(hyperparameters_lst):
        n_models = hyperparameters.pop("n_models")
        for model_group in range(n_models):
            for (i, allele) in enumerate(alleles):
                print(
                    "[%2d / %2d hyperparameters] "
                    "[%2d / %2d replicates] "
                    "[%4d / %4d alleles]: %s" % (
                        h + 1,
                        len(hyperparameters_lst),
                        model_group + 1,
                        n_models,
                        i + 1,
                        len(alleles), allele))

                train_data = df.ix[df.allele == allele].dropna().sample(
                    frac=1.0)

                model = Class1NeuralNetwork(
                    verbose=args.verbosity,
                    **hyperparameters)

                model.fit(
                    train_data.peptide.values,
                    train_data.measurement_value.values)
                print("Fit in %0.2f sec" % model.fit_seconds)

                name = "%s-%d-%d" % (
                    allele.replace("*", "_"),
                    h,
                    model_group)

                row = pandas.Series({
                    "hyperparameters_index": h,
                    "model_group": model_group,
                    "allele": allele,
                    "hyperparameters": hyperparameters,
                    "history": model.fit_history,
                    "name": name,
                    "num_measurements": len(train_data),
                    "fit_seconds": model.fit_seconds,
                }).to_frame().T
                manifest = pandas.concat([manifest, row], ignore_index=True)
                print(manifest)

                manifest.to_csv(manifest_path, index=False)
                print("Wrote: %s" % manifest_path)

                model_path = os.path.join(
                    args.out_models_dir, "%s.pickle" % name)
                with open(model_path, 'wb') as fd:
                    pickle.dump(model, fd, protocol=2)
                print("Wrote: %s" % model_path)


if __name__ == '__main__':
    run()

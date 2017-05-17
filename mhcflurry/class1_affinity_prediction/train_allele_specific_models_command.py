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


from .class1_binding_predictor import Class1BindingPredictor
from ..common import random_peptides


def normalize_allele_name(s):
    try:
        return mhcnames.normalize_allele_name(s)
    except Exception:
        return "UNKNOWN"


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data-csv",
    help="Path to data csv")
parser.add_argument(
    "--iedb-data-csv",
    help="Path to IEDB mhc_ligand_full.csv")

parser.add_argument(
    "--out-models-dir",
    help="Directory to write models and manifest")
parser.add_argument(
    "--hyperparameters",
    required=True,
    help="JSON of hyperparameters")
parser.add_argument(
    "--allele",
    default=None,
    nargs="+",
    help="Alleles")
parser.add_argument(
    "--min-measurements-per-category",
    type=int,
    default=500,
    help="Alleles")
parser.add_argument(
    "--min-measurements-per-allele",
    type=int,
    default=50,
    help="Alleles")
parser.add_argument(
    "--random-negative-rate",
    type=float,
    default=0.0)
parser.add_argument(
    "--random-negative-fixed",
    type=int,
    default=0)
parser.add_argument(
    "--pretrain",
    action="store_true",
    default=False)
parser.add_argument(
    "--only-quantitative",
    action="store_true",
    default=False)
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Alleles")


def add_random_negative_peptides(
        df,
        rate=1,
        fixed=0,
        affinity=50000.0,
        weight=1.0,
        lengths=range(8, 16)):
    new_dfs = [df]
    measurement_sources = df.measurement_source.unique()
    (allele,) = df.allele.unique()
    length_counts = df.peptide.str.len().value_counts().to_dict()
    for length in lengths:
        count = length_counts.get(length, 0)
        desired = int((count * rate + fixed))
        print("Adding %d * %d + %d = %d random negative %d-mers" % (
            count, rate, fixed, desired, length))
        peptides = random_peptides(desired, length=length)

        for measurement_source in measurement_sources:
            new_df = pandas.DataFrame({
                "allele": allele,
                "peptide": peptides,
                "measurement_type": "affinity",
                "measurement_source": measurement_source,
                "measurement_value": affinity,
                "weight": weight,
            })
            new_dfs.append(new_df)
    result = pandas.concat(new_dfs, ignore_index=True)
    print("Final result shape: %s" % str(result.shape))
    return result


def load_data_csv(filename, alleles):
    df = pandas.read_csv(filename)
    if alleles:
        df = df.ix[df.allele.isin(alleles)]
    return df


upper_thresholds = {
    "Negative": 50000.0,
    "Positive": 100.0,
    "Positive-High": 50.0,
    "Positive-Intermediate": 500.0,
    "Positive-Low": 5000.0,
}


def load_iedb_data_csv(
        iedb_csv,
        alleles=None,
        min_measurements_per_category=100,
        min_measurements_per_allele=50,
        include_qualitative=True):
    iedb_df = pandas.read_csv(iedb_csv, skiprows=1)
    print("Loaded iedb data: %s" % str(iedb_df.shape))
    iedb_df["allele"] = iedb_df["Allele Name"].map(normalize_allele_name)
    print("Dropping un-parseable alleles: %s" % ", ".join(
        iedb_df.ix[iedb_df.allele == "UNKNOWN"]["Allele Name"].unique()))
    iedb_df = iedb_df.ix[iedb_df.allele != "UNKNOWN"]

    if not alleles:
        print("Taking all alleles with %d measurements" % (
            min_measurements_per_allele))
        allele_counts = iedb_df.allele.value_counts()
        alleles = list(allele_counts.ix[
            allele_counts > min_measurements_per_allele
        ].index)
    print("Selected alleles: %s" % ' '.join(alleles))

    iedb_df = iedb_df.ix[
        iedb_df.allele.isin(alleles)
    ]
    print("IEDB measurements per allele:\n%s" % iedb_df.allele.value_counts())

    quantitative = iedb_df.ix[iedb_df["Units"] == "nM"]
    print("Quantitative measurements: %d" % len(quantitative))

    qualitative = iedb_df.ix[iedb_df["Units"] != "nM"].copy()
    print("Qualitative measurements: %d" % len(qualitative))
    non_mass_spec_qualitative = qualitative.ix[
        (~qualitative["Method/Technique"].str.contains("mass spec"))
    ].copy()
    non_mass_spec_qualitative["Quantitative measurement"] = (
        non_mass_spec_qualitative["Qualitative Measure"].map(upper_thresholds))
    print("Qualitative measurements after dropping MS: %d" % (
        len(non_mass_spec_qualitative)))

    iedb_df = pandas.concat(
        (
            ([quantitative]) +
            ([non_mass_spec_qualitative] if include_qualitative else [])),
        ignore_index=True)

    print("IEDB measurements per allele:\n%s" % iedb_df.allele.value_counts())

    print("Subselecting to valid peptides. Starting with: %d" % len(iedb_df))
    iedb_df["Description"] = iedb_df.Description.str.strip()
    iedb_df = iedb_df.ix[
        iedb_df.Description.str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")
    ]
    print("Now: %d" % len(iedb_df))

    print("Subselecting to 8-to-15-mers")
    iedb_df = iedb_df.ix[
        (iedb_df["Description"].str.len() >= 8) &
        (iedb_df["Description"].str.len() <= 15)
    ].copy()
    print("IEDB measurements per allele:\n%s" % iedb_df.allele.value_counts())

    print("Annotating last author and category")
    iedb_df["last_author"] = iedb_df.Authors.map(
        lambda x: (
            x.split(";")[-1]
            .split(",")[-1]
            .split(" ")[-1]
            .strip()
            .replace("*", "")))
    iedb_df["category"] = (
        iedb_df["last_author"] + " - " + iedb_df["Method/Technique"])

    to_concat = []

    for allele in alleles:
        sub_df = iedb_df.ix[iedb_df.allele == allele]
        top_categories = sub_df.category.value_counts().ix[
            sub_df.category.value_counts() >
            min_measurements_per_category
        ]

        top_categories = top_categories.index

        train_data = pandas.DataFrame()
        train_data["peptide"] = sub_df.Description
        train_data["measurement_value"] = sub_df[
            "Quantitative measurement"
        ]
        train_data["original_measurement_source"] = (
            sub_df.category.values)

        train_data["allele"] = sub_df["allele"]
        train_data["measurement_type"] = "affinity"
        train_data["measurement_source"] = [
            s if s in (top_categories) else "other"
            for s in train_data.original_measurement_source
        ]
        train_data["weight"] = 1.0
        train_data = train_data.drop_duplicates().reset_index(
            drop=True)
        to_concat.append(train_data)

    return pandas.concat(to_concat, ignore_index=True)


def run():
    args = parser.parse_args(sys.argv[1:])

    hyperparameters_lst = json.load(open(args.hyperparameters))
    if not isinstance(hyperparameters_lst, list):
        hyperparameters_lst = [hyperparameters_lst]
    print("Loaded hyperparameters list: %s" % str(hyperparameters_lst))

    dfs = []
    if args.iedb_data_csv:
        iedb_df = load_iedb_data_csv(
            args.iedb_data_csv,
            alleles=args.allele,
            min_measurements_per_category=args.min_measurements_per_category,
            include_qualitative=not args.only_quantitative)
        dfs.append(iedb_df)
    if args.data_csv:
        extra_data_csv = load_data_csv(
            args.data_csv, alleles=args.allele)
        print("Loaded extra data csv: %s %s" % (
            args.data_csv, str(extra_data_csv.shape)))
        dfs.append(extra_data_csv)

    df = pandas.concat(dfs, ignore_index=True)
    print("Combined df: %s" % (str(df.shape)))
    allele_counts = df.allele.value_counts()
    alleles = list(allele_counts.ix[
        allele_counts > args.min_measurements_per_allele
    ].index)
    print("Selected alleles: %s" % ' '.join(alleles))

    df = df.ix[df.allele.isin(alleles)]

    print("Combined allele-selected df: %s" % (str(df.shape)))

    manifest = pandas.DataFrame()
    manifest["name"] = []
    manifest["hyperparameters_index"] = []
    manifest["model_group"] = []
    manifest["allele"] = []
    manifest["hyperparameters"] = []
    manifest["history"] = []
    manifest["num_measurements"] = []
    manifest["random_negative_rate"] = []
    manifest["random_negative_fixed"] = []
    manifest["sources"] = []
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

                train_data = df.ix[df.allele == allele]

                train_data_expanded = add_random_negative_peptides(
                    train_data,
                    rate=args.random_negative_rate,
                    fixed=args.random_negative_fixed)

                train_data_expanded = train_data_expanded.dropna().sample(
                    frac=1.0)

                print("Measurement sources:\n%s" % (
                    train_data_expanded.measurement_source.value_counts()))

                model = Class1BindingPredictor(
                    verbose=args.verbose,
                    **hyperparameters)

                model.fit(
                    train_data_expanded.peptide.values,
                    train_data_expanded.measurement_value.values,
                    output_assignments=(
                        train_data_expanded.measurement_source.values))
                print("Done fitting in %0.2f sec" % model.fit_seconds)

                name = "%s-%d-%d" % (
                    allele.replace("*", "_"),
                    h,
                    model_group)

                row = pandas.Series({
                    "hyperparameters_index": h,
                    "model_group": model_group,
                    "allele": allele,
                    "hyperparameters": hyperparameters,
                    "history": model.fit_history.history,
                    "name": name,
                    "num_measurements": len(train_data),
                    "random_negative_rate": args.random_negative_rate,
                    "random_negative_fixed": args.random_negative_fixed,
                    "sources": train_data_expanded.measurement_source.unique(),
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

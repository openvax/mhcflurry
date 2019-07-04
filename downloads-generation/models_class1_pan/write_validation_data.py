"""
Write and summarize model validation data, which is obtained by taking a full
dataset and removing the data used for training.

"""
import argparse
import sys
from os.path import abspath

import pandas
import numpy
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(usage = __doc__)

parser.add_argument(
    "--include",
    metavar="INPUT.csv",
    nargs="+",
    help="Input CSV to include")
parser.add_argument(
    "--exclude",
    metavar="INPUT.csv",
    nargs="+",
    help="Input CSV to exclude")
parser.add_argument(
    "--out-data",
    metavar="RESULT.csv",
    help="Output dadta CSV")
parser.add_argument(
    "--out-summary",
    metavar="RESULT.csv",
    help="Output summary CSV")
parser.add_argument(
    "--mass-spec-regex",
    metavar="REGEX",
    default="mass[- ]spec",
    help="Regular expression for mass-spec data. Runs on measurement_source col."
    "Default: %(default)s.")
parser.add_argument(
    "--only-alleles-present-in-exclude",
    action="store_true",
    default=False,
    help="Filter to only alleles that are present in files given by --exclude. "
    "Useful for filtering to only alleles supported by a predictor, where the "
    "training data for the predictor is given by --exclude.")


def run(argv):
    args = parser.parse_args(argv)

    dfs = []
    for input in args.include:
        df = pandas.read_csv(input)
        dfs.append(df)
    df = pandas.concat(dfs, ignore_index=True)
    print("Loaded data with shape: %s" % str(df.shape))
    del dfs

    df = df.ix[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    if args.exclude:
        exclude_dfs = []
        for exclude in args.exclude:
            exclude_df = pandas.read_csv(exclude)
            exclude_dfs.append(exclude_df)
        exclude_df = pandas.concat(exclude_dfs, ignore_index=True)
        del exclude_dfs

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

        if args.only_alleles_present_in_exclude:
            df = df.loc[df.allele.isin(exclude_df.allele.unique())]

    df["mass_spec"] = df.measurement_source.str.contains(args.mass_spec_regex)
    df.loc[df.mass_spec , "measurement_inequality"] = "mass_spec"

    if args.out_summary:
        summary_df = df.groupby(
            ["allele", "measurement_inequality"]
        )["measurement_value"].count().unstack().fillna(0).astype(int)
        summary_df["total"] = summary_df.sum(1)
        summary_df.to_csv(args.out_summary)
        print("Wrote: %s" % args.out_summary)

    if args.out_data:
        df.to_csv(args.out_data, index=False)
        print("Wrote: %s" % args.out_data)

if __name__ == '__main__':
    run(sys.argv[1:])

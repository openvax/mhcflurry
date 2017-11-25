"""
Split training data into CV folds.
"""
import argparse
import sys
from os.path import abspath

import pandas
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(usage = __doc__)

parser.add_argument(
    "input", metavar="INPUT.csv", help="Input CSV")

parser.add_argument(
    "--folds", metavar="N", type=int, default=5)

parser.add_argument(
    "--allele",
    nargs="+",
    help="Include only the specified allele(s)")

parser.add_argument(
    "--min-measurements-per-allele",
    type=int,
    metavar="N",
    help="Use only alleles with >=N measurements.")

parser.add_argument(
    "--subsample",
    type=int,
    metavar="N",
    help="Subsample to first N rows")

parser.add_argument(
    "--random-state",
    metavar="N",
    type=int,
    help="Specify an int for deterministic splitting")

parser.add_argument(
    "--output-pattern-train",
    default="./train.fold_{}.csv",
    help="Pattern to use to generate output filename. Default: %(default)s")

parser.add_argument(
    "--output-pattern-test",
    default="./test.fold_{}.csv",
    help="Pattern to use to generate output filename. Default: %(default)s")


def run(argv):
    args = parser.parse_args(argv)

    df = pandas.read_csv(args.input)
    print("Loaded data with shape: %s" % str(df.shape))

    df = df.ix[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    allele_counts = df.allele.value_counts()

    if args.allele:
        alleles = args.allele
    else:
        alleles = list(
            allele_counts.ix[
                           allele_counts > args.min_measurements_per_allele
        ].index)

    df = df.ix[df.allele.isin(alleles)]
    print("Potentially subselected by allele to: %s" % str(df.shape))

    print("Data has %d alleles: %s" % (
        df.allele.nunique(), " ".join(df.allele.unique())))

    df = df.groupby(["allele", "peptide"]).measurement_value.median().reset_index()
    print("Took median for each duplicate peptide/allele pair: %s" % str(df.shape))

    if args.subsample:
        df = df.head(args.subsample)
        print("Subsampled to: %s" % str(df.shape))

    kf = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.random_state)

    # Stratify by both allele and binder vs. nonbinder.
    df["key"] = [
        "%s_%s" % (
            row.allele,
            "binder" if row.measurement_value < 500 else "nonbinder")
        for (_, row) in df.iterrows()
    ]

    for i, (train, test) in enumerate(kf.split(df, df.key)):
        train_filename = args.output_pattern_train.format(i)
        test_filename = args.output_pattern_test.format(i)

        df.iloc[train].to_csv(train_filename, index=False)
        print("Wrote: %s" % abspath(train_filename))

        df.iloc[test].to_csv(test_filename, index=False)
        print("Wrote: %s" % abspath(test_filename))


if __name__ == '__main__':
    run(sys.argv[1:])


"""
Split a big csv by a particular column (sample id)
"""
import sys
import argparse

import pandas


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "data",
    metavar="CSV")
parser.add_argument(
    "--out",
    help="Out pattern (%s will be replaced by sample)",
    metavar="CSV")
parser.add_argument(
    "--out-samples",
    help="Out sample list",
    metavar="CSV")
parser.add_argument(
    "--split-column",
    help="Column to split by",
    default="sample_id")


def run():
    args = parser.parse_args(sys.argv[1:])
    df = pandas.read_csv(args.data)

    names = []
    for (i, (sample, sub_df)) in enumerate(df.groupby(args.split_column)):
        name = sample.replace(" ", "") + (".%d" % i)
        dest = args.out % name
        sub_df.to_csv(dest, index=False)
        print("Wrote", dest)
        names.append(name)

    if args.out_samples:
        pandas.Series(names).to_csv(args.out_samples, index=False, header=False)
        print("Wrote", args.out_samples)

if __name__ == '__main__':
    run()

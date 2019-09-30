"""
"""
import sys
import argparse
import os

import pandas
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "input",
    metavar="FILE.csv",
    help="CSV of annotated mass spec hits")
parser.add_argument(
    "--out",
    metavar="OUT.txt",
    help="Out file path")


def run():
    args = parser.parse_args(sys.argv[1:])

    df = pandas.read_csv(args.input)
    print("Read peptides", df.shape, *df.columns.tolist())

    df = df.loc[df.mhc_class == "I"]

    hla_sets = df.hla.unique()
    all_hla = set()
    for hla_set in hla_sets:
        all_hla.update(hla_set.split())

    all_hla = pandas.Series(sorted(all_hla))
    all_hla.to_csv(args.out, index=False, header=False)
    print("Wrote [%d alleles]: %s" % (len(all_hla), os.path.abspath(args.out)))


if __name__ == '__main__':
    run()

"""
"""
import sys
import argparse
import os
import gzip

import pandas
import shellinford
from Bio import SeqIO

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "input_paths",
    nargs="+",
    help="Fasta files to process")
parser.add_argument(
    "--out-csv",
    required=True,
    metavar="FILE.csv",
    help="CSV output")
parser.add_argument(
    "--out-index",
    required=True,
    metavar="FILE.fm",
    help="Index output")


def run():
    args = parser.parse_args(sys.argv[1:])

    fm = shellinford.FMIndex()
    df = []
    for f in args.input_paths:
        print("Processing", f)
        with gzip.open(f, "rt") as fd:
            records = SeqIO.parse(fd, format='fasta')
            for (i, record) in enumerate(records):
                seq = str(record.seq).upper()
                df.append((record.name, record.description, seq))
                fm.push_back("$" + seq + "$")  # include sentinels
    df = pandas.DataFrame(df, columns=["name", "description", "seq"])

    print("Done reading fastas")
    print(df)

    print("Building index")
    fm.build()
    fm.write(args.out_index)
    print("Wrote: ", os.path.abspath((args.out_index)))

    df.to_csv(args.out_csv, index=True)
    print("Wrote: ", os.path.abspath((args.out_csv)))


if __name__ == '__main__':
    run()

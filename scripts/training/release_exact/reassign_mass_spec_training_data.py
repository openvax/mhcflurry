"""
Reassign affinity values for mass spec data
"""
import sys
import os
import argparse

import pandas

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument("data", metavar="CSV", help="Training data")
parser.add_argument("--ms-only", action="store_true", default=False)
parser.add_argument("--drop-negative-ms", action="store_true", default=False)
parser.add_argument("--set-measurement-value", type=float)
parser.add_argument("--out-csv")

pandas.set_option('display.max_columns', 500)


def go(args):
    df = pandas.read_csv(args.data)
    print(df)

    if args.drop_negative_ms:
        bad = df.loc[
            (df.measurement_kind == "mass_spec") &
            (df.measurement_inequality != "<")
        ]
        print("Dropping ", len(bad))
        df = df.loc[~df.index.isin(bad.index)].copy()

    if args.ms_only:
        print("Filtering to MS only")
        df = df.loc[df.measurement_kind == "mass_spec"].copy()

    if args.set_measurement_value is not None:
        indexer = df.measurement_kind == "mass_spec"
        df.loc[
            indexer,
            "measurement_value"
        ] = args.set_measurement_value
        print("Reassigned:")
        print(df.loc[indexer])

    if args.out_csv:
        out_csv = os.path.abspath(args.out_csv)
        df.to_csv(out_csv, index=False)
        print("Wrote", out_csv)


if __name__ == "__main__":
    go(parser.parse_args(sys.argv[1:]))

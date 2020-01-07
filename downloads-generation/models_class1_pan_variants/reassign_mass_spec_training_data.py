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
parser.add_argument("--set-measurement-value", type=float)
parser.add_argument("--out-csv")

pandas.set_option('display.max_columns', 500)


def go(args):
    df = pandas.read_csv(args.data)
    print(df)

    bad = df.loc[
        (df.measurement_kind == "mass_spec") &
        (df.measurement_inequality != "<")
    ]
    assert len(bad) == 0, bad

    if args.ms_only:
        print("Filtering to MS only")
        df = df.loc[df.kind == "mass_spec"]

    if args.set_measurement_value:
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

"""
"""
import sys
import argparse
import os

import pandas

from mhcflurry.peptide_reference import annotate_peptide_references


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "peptides",
    metavar="FILE.csv",
    help="CSV of mass spec hits")
parser.add_argument(
    "reference_csv",
    metavar="FILE.csv",
    help="CSV of protein sequences")
parser.add_argument(
    "reference_index",
    metavar="FILE.fm",
    nargs="?",
    help="Legacy shellinford index over protein sequences. Retained for "
    "compatibility with existing GENERATE.sh calls; ignored.")
parser.add_argument(
    "--out",
    metavar="OUT.csv",
    help="Out file path")
parser.add_argument(
    "--flanking-length",
    metavar="N",
    type=int,
    default=15,
    help="Length of flanking sequence to include")
parser.add_argument(
    "--debug-max-rows",
    metavar="N",
    type=int,
    default=None,
    help="Max rows to process. Useful for debugging. If specified an ipdb "
    "debugging session is also opened at the end of the script")


def run(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    df = pandas.read_csv(args.peptides)
    df["hit_id"] = "hit." + df.index.map('{0:07d}'.format)
    df = df.set_index("hit_id")
    print("Read peptides", df.shape, *df.columns.tolist())

    reference_df = pandas.read_csv(args.reference_csv, index_col=0)
    print("Read proteins", reference_df.shape, *reference_df.columns.tolist())

    merged_df = annotate_peptide_references(
        df,
        reference_df,
        flanking_length=args.flanking_length,
        debug_max_rows=args.debug_max_rows)

    merged_df.to_csv(args.out, index=False)
    print("Wrote: %s" % os.path.abspath(args.out))

    if args.debug_max_rows:
        # Leave user in a debugger
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    run()

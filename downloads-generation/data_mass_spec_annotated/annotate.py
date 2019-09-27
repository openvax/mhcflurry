"""
"""
import sys
import argparse
import os
import time
import collections
from six.moves import StringIO

import pandas
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

import shellinford


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
    help="shellinford index over protein sequences")
parser.add_argument(
    "--out",
    metavar="OUT.csv",
    help="Out file path")


def run():
    args = parser.parse_args(sys.argv[1:])

    df = pandas.read_csv(args.peptides)
    df["hit_id"] = "hit." + df.index.map(str)
    df = df.set_index("hit_id")
    print("Read peptides", df.shape, *df.columns.tolist())

    reference_df = pandas.read_csv(args.reference_csv, index_col=0)
    reference_df = reference_df.set_index("accession")
    print("Read proteins", reference_df.shape, *reference_df.columns.tolist())

    fm = shellinford.FMIndex()
    fm.read(args.reference_index)
    print("Read proteins index")

    join_df = []
    for (hit_id, row) in tqdm.tqdm(df.iterrows(), total=len(df)):
        matches = fm.search(row.peptide)
        for match in matches:
            join_df.append((hit_id, match.doc_id, len(matches)))

    join_df = pandas.DataFrame(
        join_df,
        columns=["hit_id", "match_index", "num_proteins"],
    )

    join_df["protein_accession"] = join_df.match_index.map(
        reference_df.index.to_series().reset_index(drop=True))

    del join_df["match_index"]

    protein_cols = [
        c for c in reference_df.columns
        if c not in ["name", "description", "seq"]
    ]
    for col in protein_cols:
        join_df["protein_%s" % col] = join_df.protein_accession.map(
            reference_df[col])

    merged_df = pandas.merge(
        join_df,
        df,
        how="left",
        left_on="hit_id",
        right_index=True)

    merged_df.to_csv(args.out, index=False)
    print("Wrote: %s" % os.path.abspath(args.out))


if __name__ == '__main__':
    run()

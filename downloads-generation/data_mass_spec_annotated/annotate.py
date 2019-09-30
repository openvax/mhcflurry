"""
"""
import sys
import argparse
import os
import time
import collections
import re
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


def run():
    args = parser.parse_args(sys.argv[1:])

    df = pandas.read_csv(args.peptides)
    df["hit_id"] = "hit." + df.index.map('{0:07d}'.format)
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
            reference_row = reference_df.iloc[match.doc_id]
            starts = [
                m.start() for m in
                re.finditer(row.peptide, reference_row.seq)
            ]
            assert len(starts) > 0, (row.peptide, reference_row.seq)
            for start in starts:
                end_pos = start + len(row.peptide)
                n_flank = reference_row.seq[
                    max(start - args.flanking_length, 0) : start
                ].rjust(args.flanking_length, 'X')
                c_flank = reference_row.seq[
                    end_pos : (end_pos + args.flanking_length)
                ].ljust(args.flanking_length, 'X')
                join_df.append((
                    hit_id,
                    match.doc_id,
                    len(matches),
                    len(starts),
                    start,
                    start / len(reference_row.seq),
                    n_flank,
                    c_flank,
                ))

        if args.debug_max_rows and len(join_df) > args.debug_max_rows:
            break

    join_df = pandas.DataFrame(
        join_df,
        columns=[
            "hit_id",
            "match_index",
            "num_proteins",
            "num_occurrences_in_protein",
            "start_position",
            "start_fraction_in_protein",
            "n_flank",
            "c_flank",
        ]).drop_duplicates()

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

    if args.debug_max_rows:
        # Leave user in a debugger
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    run()

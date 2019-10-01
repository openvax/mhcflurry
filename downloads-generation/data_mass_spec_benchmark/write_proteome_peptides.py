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


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "input",
    metavar="FILE.csv",
    help="CSV of annotated mass spec hits")
parser.add_argument(
    "reference_csv",
    metavar="FILE.csv",
    help="CSV of protein sequences")
parser.add_argument(
    "--out",
    metavar="OUT.csv",
    help="Out file path")
parser.add_argument(
    "--chromosome",
    metavar="CHR",
    nargs="+",
    help="Use only proteins from the specified chromosome(s)")
parser.add_argument(
    "--debug-max-rows",
    metavar="N",
    type=int,
    default=None,
    help="Max rows to process. Useful for debugging. If specified an ipdb "
    "debugging session is also opened at the end of the script")
parser.add_argument(
    "--lengths",
    metavar="N",
    type=int,
    nargs="+",
    default=[8,9,10,11],
    help="Peptide lengths")


def run():
    args = parser.parse_args(sys.argv[1:])

    df_original = pandas.read_csv(args.input)
    df = df_original
    print("Read peptides", df.shape, *df.columns.tolist())

    reference_df = pandas.read_csv(args.reference_csv, index_col=0)
    reference_df = reference_df.set_index("accession")
    print("Read proteins", reference_df.shape, *reference_df.columns.tolist())

    print("Subselecting to MHC I hits. Before: ", len(df))
    df = df.loc[df.mhc_class == "I"]
    print("After: ", len(df))

    print("Subselecting to gene-associated hits. Before: ", len(df))
    df = df.loc[~df.protein_ensembl_primary.isnull()]
    print("After: ", len(df))

    if args.chromosome:
        print("Subselecting to chromosome(s): ", *args.chromosome)
        print("Before: ", len(df))
        df = df.loc[df.protein_primary_ensembl_contig.isin(args.chromosome)]
        print("After: ", len(df))

    (flanking_length,) = list(
        set(df.n_flank.str.len().unique()).union(
            set(df.n_flank.str.len().unique())))
    print("Flanking length", flanking_length)

    proteins = df.protein_accession.unique()

    if args.debug_max_rows:
        proteins = proteins[:args.debug_max_rows]

    print("Writing decoys for %d proteins" % len(proteins))

    reference_df = reference_df.loc[proteins]

    lengths = sorted(args.lengths)
    rows = []
    total = len(reference_df)
    for (accession, info) in tqdm.tqdm(reference_df.iterrows(), total=total):
        seq = info.seq
        for start in range(0, len(seq) - min(args.lengths)):
            for length in lengths:
                end_pos = start + length
                if end_pos > len(seq):
                    break
                n_flank = seq[
                    max(start - flanking_length, 0) : start
                ].rjust(flanking_length, 'X')
                c_flank = seq[
                    end_pos : (end_pos + flanking_length)
                ].ljust(flanking_length, 'X')
                peptide = seq[start : start + length]

                rows.append((
                    accession,
                    peptide,
                    n_flank,
                    c_flank,
                    start
                ))

    result_df = pandas.DataFrame(
        rows,
        columns=[
            "protein_accession",
            "peptide",
            "n_flank",
            "c_flank",
            "start_position",
        ])

    result_df.to_csv(args.out, index=False)
    print("Wrote: %s" % os.path.abspath(args.out))

    if args.debug_max_rows:
        # Leave user in a debugger
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    run()

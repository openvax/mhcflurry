"""
Annotate hits with expression (tpm), and roll up to just the highest-expressed
gene for each peptide.
"""
import sys
import argparse
import os


import pandas
import tqdm


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--hits",
    metavar="CSV",
    required=True,
    help="Multiallelic mass spec")
parser.add_argument(
    "--expression",
    metavar="CSV",
    required=True,
    help="Expression data")
parser.add_argument(
    "--out",
    metavar="CSV",
    required=True,
    help="File to write")


def run():
    args = parser.parse_args(sys.argv[1:])
    args.out = os.path.abspath(args.out)

    hit_df = pandas.read_csv(args.hits)
    hit_df = hit_df.loc[
        (~hit_df.protein_ensembl.isnull())
    ]
    print("Loaded hits from %d samples" % hit_df.sample_id.nunique())
    expression_df = pandas.read_csv(args.expression, index_col=0).fillna(0)

    # Add a column to hit_df giving expression value for that sample and that gene
    print("Annotating expression.")
    hit_df["tpm"] = [
        expression_df.reindex(
            row.protein_ensembl.split())[row.expression_dataset].sum()
        for _, row in tqdm.tqdm(
            hit_df.iterrows(), total=len(hit_df), ascii=True, maxinterval=10000)
    ]

    # Discard hits except those that have max expression for each hit_id
    print("Selecting max-expression transcripts for each hit.")
    max_gene_hit_df = hit_df.loc[
        hit_df.tpm == hit_df.hit_id.map(hit_df.groupby("hit_id").tpm.max())
    ].sample(frac=1.0).drop_duplicates("hit_id")

    max_gene_hit_df.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == '__main__':
    run()

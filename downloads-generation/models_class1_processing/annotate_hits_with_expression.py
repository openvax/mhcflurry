# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def annotate_tpm(hit_df, expression_df):
    """Return per-hit TPM sums for each row's expression dataset."""
    tpm = pandas.Series(0.0, index=hit_df.index)
    for expression_dataset, sub_df in hit_df.groupby(
            "expression_dataset", sort=False):
        expression_by_gene = expression_df[expression_dataset]
        genes = sub_df.protein_ensembl.str.split().explode()
        gene_tpm = genes.map(expression_by_gene).fillna(0.0)
        tpm.loc[sub_df.index] = gene_tpm.groupby(level=0).sum()
    return tpm


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
    hit_df["tpm"] = annotate_tpm(hit_df, expression_df)

    # Discard hits except those that have max expression for each hit_id
    print("Selecting max-expression transcripts for each hit.")
    max_gene_hit_df = hit_df.loc[
        hit_df.tpm == hit_df.hit_id.map(hit_df.groupby("hit_id").tpm.max())
    ].sample(frac=1.0).drop_duplicates("hit_id")

    max_gene_hit_df.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == '__main__':
    run()

"""
Make multiallelic training data by selecting decoys, etc.
"""
import sys
import argparse
import os
import json
import collections
from six.moves import StringIO

import pandas
import tqdm

import mhcnames


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
    "--decoys-per-hit",
    type=int,
    default=None,
    help="If not specified will use all possible decoys")
parser.add_argument(
    "--exclude-contig",
    help="Exclude entries annotated to the given contig")
parser.add_argument(
    "--out",
    metavar="CSV",
    required=True,
    help="File to write")
parser.add_argument(
    "--alleles",
    nargs="+",
    help="Include only the specified alleles")


def run():
    args = parser.parse_args(sys.argv[1:])
    hit_df = pandas.read_csv(args.hits)
    expression_df = pandas.read_csv(args.expression, index_col=0).fillna(0)
    hit_df["alleles"] = hit_df.hla.str.split()

    hit_df = hit_df.loc[
        (hit_df.mhc_class == "I") &
        (hit_df.peptide.str.len() <= 15) &
        (hit_df.peptide.str.len() >= 7) &
        (~hit_df.protein_ensembl.isnull())
    ]
    if args.exclude_contig:
        new_hit_df = hit_df.loc[
            hit_df.protein_primary_ensembl_contig.astype(str) !=
            args.exclude_contig
        ]
        print(
            "Excluding contig",
            args.exclude_contig,
            "reduced dataset from",
            len(hit_df),
            "to",
            len(new_hit_df))
        hit_df = new_hit_df.copy()
    if args.alleles:
        filter_alleles = set(args.alleles)
        new_hit_df = hit_df.loc[
            hit_df.alleles.map(
                lambda a: len(set(a).intersection(filter_alleles)) > 0)
        ]
        print(
            "Selecting alleles",
            args.alleles,
            "reduced dataset from",
            len(hit_df),
            "to",
            len(new_hit_df))
        hit_df = new_hit_df.copy()

    sample_table = hit_df.drop_duplicates("sample_id").set_index("sample_id")
    grouped = hit_df.groupby("sample_id").nunique()
    for col in sample_table.columns:
        if (grouped[col] > 1).any():
            del sample_table[col]
    sample_table["total_hits"] = hit_df.groupby("sample_id").peptide.nunique()

    experiment_to_decoys = {}
    for sample_id, row in sample_table.iterrows():
        decoys = sample_table.loc[
            sample_table.alleles.map(
                lambda a: not set(a).intersection(set(row.alleles)))
        ].index.tolist()
        experiment_to_decoys[sample_id] = decoys
    experiment_to_decoys = pandas.Series(experiment_to_decoys)
    sample_table["decoys"] = experiment_to_decoys

    print("Samples:")
    print(sample_table)

    # Add a column to hit_df giving expression value for that sample and that gene
    print("Annotating expression.")
    hit_df["tpm"] = [
        expression_df.reindex(
            row.protein_ensembl.split())[row.expression_dataset].sum()
        for _, row in tqdm.tqdm(
            hit_df.iterrows(), total=len(hit_df), ascii=True, maxinterval=10000)
    ]

    print("Selecting max-expression transcripts for each hit.")

    # Discard hits except those that have max expression for each hit_id
    max_gene_hit_df = hit_df.loc[
        hit_df.tpm == hit_df.hit_id.map(hit_df.groupby("hit_id").tpm.max())
    ].sample(frac=1.0).drop_duplicates("hit_id")

    # Columns are accession, peptide, sample_id, hit
    result_df = []

    columns_to_keep = [
        "hit_id",
        "protein_accession",
        "protein_primary_ensembl_contig",
        "peptide",
        "sample_id"
    ]

    print("Selecting decoys.")
    for sample_id, sub_df in tqdm.tqdm(
            max_gene_hit_df.groupby("sample_id"),
            total=max_gene_hit_df.sample_id.nunique()):
        result_df.append(sub_df[columns_to_keep].copy())
        result_df[-1]["hit"] = 1

        possible_decoys = hit_df.loc[
            (hit_df.sample_id.isin(sample_table.loc[sample_id].decoys)) &
            (~hit_df.peptide.isin(sub_df.peptide.unique())) &
            (hit_df.protein_accession.isin(sub_df.protein_accession.unique()))
        ].drop_duplicates("peptide")
        if not args.decoys_per_hit:
            selected_decoys = possible_decoys[columns_to_keep].copy()
        elif len(possible_decoys) < len(sub_df) * args.decoys_per_hit:
            print(
                "Insufficient decoys",
                len(possible_decoys),
                len(sub_df),
                args.decoys_per_hit)
            selected_decoys = possible_decoys[columns_to_keep].copy()
        else:
            selected_decoys = possible_decoys.sample(
                n=len(sub_df) * args.decoys_per_hit)[columns_to_keep].copy()
        result_df.append(selected_decoys)
        result_df[-1]["hit"] = 0
        result_df[-1]["sample_id"] = sample_id

    result_df = pandas.concat(result_df, ignore_index=True, sort=False)
    result_df["hla"] = result_df.sample_id.map(sample_table.hla)

    print(result_df)
    print("Counts:")
    print(result_df.groupby(["sample_id", "hit"]).peptide.nunique())

    print("Hit counts:")
    print(
        result_df.loc[
            result_df.hit == 1
        ].groupby("sample_id").hit.count().sort_values())

    print("Hit rates:")
    print(result_df.groupby("sample_id").hit.mean().sort_values())

    result_df.to_csv(args.out, index=False)
    print("Wrote: ", args.out)


if __name__ == '__main__':
    run()

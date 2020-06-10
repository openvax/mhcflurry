"""
From affinity predictor model selection data, add decoys so that AUCs can be
calculated per-allele.
"""
import sys
import argparse
import os
import numpy
import math
import collections

import pandas
import tqdm

import mhcflurry
from mhcflurry.downloads import get_path

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "data",
    metavar="CSV",
    help="Model selection data")
parser.add_argument(
    "--proteome-peptides",
    metavar="CSV",
    required=True,
    help="Proteome peptides")
parser.add_argument(
    "--protein-data",
    metavar="CSV",
    default=get_path("data_references", "uniprot_proteins.csv.bz2", test_exists=False),
    help="Proteome data. Default: %(default)s.")
parser.add_argument(
    "--out",
    metavar="CSV",
    required=True,
    help="File to write")


def run():
    args = parser.parse_args(sys.argv[1:])

    data_df = pandas.read_csv(args.data)
    print("Read", args.data, len(data_df))
    print(data_df)

    fold_cols = [col for col in data_df.columns if col.startswith("fold_")]
    print("Fold cols", fold_cols)
    assert len(fold_cols) > 1

    eval_df = data_df.loc[
        data_df[fold_cols].sum(1) < len(fold_cols)
    ].copy()

    eval_df["binder"] = (eval_df.measurement_inequality != '>') & (
        eval_df.measurement_value <= 500)

    print("Reduced to data held-out at least once: ", len(eval_df))
    print("Binder rate per allele:")
    print(eval_df.groupby("allele").binder.mean())

    decoy_universe = pandas.read_csv(args.protein_data, usecols=["seq"])
    decoy_universe = pandas.Series(decoy_universe.seq.unique())
    decoy_universe = decoy_universe.loc[
        decoy_universe.str.match("^[%s]+$" % "".join(
            mhcflurry.amino_acid.COMMON_AMINO_ACIDS)) & (
            decoy_universe.str.len() >= 50)
    ]
    print("Read decoy universe from", args.protein_data)
    print(decoy_universe)

    def make_decoys(num, length):
        return decoy_universe.sample(num, replace=True).map(
            lambda s: s[numpy.random.randint(0, len(s) - length):][:length]).values

    lengths = [8,9,10,11]

    pieces = []
    real_df = eval_df.loc[
        eval_df.peptide.str.len().isin(lengths)].copy()
    real_df["synthetic"] = False
    pieces.append(real_df)

    for length in lengths:
        decoys_df = real_df.loc[real_df.binder].copy()
        decoys_df.binder = False
        decoys_df.measurement_value = numpy.nan
        decoys_df.synthetic = True
        decoys_df["peptide"] = make_decoys(len(decoys_df), length)
        pieces.append(decoys_df)

    result_df = pandas.concat(pieces, ignore_index=True)

    print("Final binder rate per allele:")
    print(result_df.groupby("allele").binder.mean())

    result_df.to_csv(args.out, index=False)
    print("Wrote: ", args.out)


if __name__ == '__main__':
    run()

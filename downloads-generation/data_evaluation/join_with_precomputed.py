"""
Join benchmark with precomputed predictions.
"""
import sys
import argparse
import os
import numpy
import collections

import pandas
import tqdm

import mhcflurry
from mhcflurry.downloads import get_path

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "benchmark")
parser.add_argument(
    "predictors",
    nargs="+",
    choices=("netmhcpan4.ba", "netmhcpan4.el", "mixmhcpred"))
parser.add_argument(
    "--out",
    metavar="CSV",
    required=True,
    help="File to write")


def load_results(dirname, result_df=None, columns=None):
    peptides = pandas.read_csv(os.path.join(dirname, "peptides.csv")).peptide
    manifest_df = pandas.read_csv(os.path.join(dirname, "alleles.csv"))

    print("Loading results. Existing data has", len(peptides), "peptides and",
        len(manifest_df), "columns")

    if columns is None:
        columns = manifest_df.col.values

    if result_df is None:
        result_df = pandas.DataFrame(
            index=peptides,
            columns=columns,
            dtype="float32")
        result_df[:] = numpy.nan
        peptides_to_assign = peptides
        mask = None
    else:
        mask = (peptides.isin(result_df.index)).values
        peptides_to_assign = peptides[mask]

    manifest_df = manifest_df.loc[manifest_df.col.isin(result_df.columns)]

    print("Will load", len(peptides), "peptides and", len(manifest_df), "cols")

    for _, row in tqdm.tqdm(manifest_df.iterrows(), total=len(manifest_df)):
        with open(os.path.join(dirname, row.path), "rb") as fd:
            value = numpy.load(fd)['arr_0'].astype(numpy.float32)
            if mask is not None:
                value = value[mask]
            result_df.loc[peptides_to_assign, row.col] = value

    return result_df


def run():
    args = parser.parse_args(sys.argv[1:])
    df = pandas.read_csv(args.benchmark)

    df["alleles"] = df.hla.str.split()

    peptides = df.peptide.unique()
    alleles = set()
    for some in df.hla.unique():
        alleles.update(some.split())

    precomputed_dfs = {}

    if 'netmhcpan4.ba' in args.predictors:
        precomputed_dfs['netmhcpan4.ba'] = load_results(
            get_path("data_mass_spec_benchmark", "predictions/all.netmhcpan4.ba"),
            result_df=pandas.DataFrame(
                dtype=numpy.float32,
                index=peptides,
                columns=["%s affinity" % a for a in alleles])).rename(
            columns=lambda s: s.replace("affinity", "").strip())
        precomputed_dfs['netmhcpan4.ba'] *= -1

    if 'netmhcpan4.el' in args.predictors:
        precomputed_dfs['netmhcpan4.el'] = load_results(
            get_path("data_mass_spec_benchmark", "predictions/all.netmhcpan4.el"),
            result_df=pandas.DataFrame(
                dtype=numpy.float32,
                index=peptides,
                columns=["%s score" % a for a in alleles])).rename(
            columns=lambda s: s.replace("score", "").strip())

    if 'mixmhcpred' in args.predictors:
        precomputed_dfs['mixmhcpred'] = load_results(
            get_path("data_mass_spec_benchmark", "predictions/all.mixmhcpred"),
            result_df=pandas.DataFrame(
                dtype=numpy.float32,
                index=peptides,
                columns=["%s score" % a for a in alleles])).rename(
            columns=lambda s: s.replace("score", "").strip())

    skip_experiments = set()

    for hla_text, sub_df in tqdm.tqdm(df.groupby("hla"), total=df.hla.nunique()):
        hla = hla_text.split()
        for (name, precomputed_df) in precomputed_dfs.items():
            df.loc[sub_df.index, name] = numpy.nan
            prediction_df = pandas.DataFrame(index=sub_df.peptide, dtype=float)
            for allele in hla:
                if allele not in precomputed_df.columns or precomputed_df[allele].isnull().all():
                    print(sub_df.sample_id.unique(), hla)
                    skip_experiments.update(sub_df.sample_id.unique())
                prediction_df[allele] = precomputed_df.loc[
                    prediction_df.index, allele
                ]
            df.loc[sub_df.index, name] = prediction_df.max(1, skipna=False).values
            df.loc[sub_df.index, name + " allele"] = prediction_df.idxmax(1, skipna=False).values

    print("Skip experiments", skip_experiments)
    print("results")
    print(df)

    df.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == '__main__':
    run()

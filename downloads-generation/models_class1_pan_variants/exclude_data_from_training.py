"""
Extract allele/peptide pairs to exclude from training data.
"""
import sys
import os
import argparse

import pandas

import mhcnames


def normalize_allele_name(s):
    try:
        return mhcnames.normalize_allele_name(s)
    except Exception:
        return "UNKNOWN"



parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument("data", metavar="CSV", help="Training data")
parser.add_argument(
    "--remove-filename",
    action="append",
    default=[],
    metavar="NAME",
    help="Data to drop",
    required=True)
parser.add_argument(
    "--remove-kind",
    action="append",
    default=[],
    metavar="KIND",
    help="Format of data to drop. For published data, use the PMID.",
    choices=[
        "30377561"  # Koşaloğlu-Yalçın, ..., Peters. Oncoimmunology 2018 [PMID 30377561]
    ],
    required=True)
parser.add_argument("--out", metavar="CSV", help="Result data path")
parser.add_argument(
    "--out-removed", metavar="CSV", help="Write removed data to given path")


pandas.set_option('display.max_columns', 500)


LOADERS = {}


def load_30377561(filename):
    # Koşaloğlu-Yalçın, ..., Peters. Oncoimmunology 2018 [PMID 30377561]
    dfs = pandas.read_excel(filename, sheet_name=None)

    df1 = dfs['Supp Table 5 positive & random']

    result_df = []
    result_df.append(df1.rename(
        columns={
            "mt.pep": "peptide",
            "hla": "allele",
        })[["allele", "peptide"]])
    result_df.append(df1.rename(
        columns={
            "wt.pep": "peptide",
            "hla": "allele",
        })[["allele", "peptide"]])


    df2 = dfs["Supp Table 4 viral epitopes"]

    result_df.append(
        df2.rename(
            columns={
                "Epitope": "peptide", "Restriction": "allele",
        })[["allele", "peptide"]])

    result_df = pandas.concat(result_df, ignore_index=True)
    return result_df


LOADERS["30377561"] = load_30377561


def go(args):
    df = pandas.read_csv(args.data)
    print("Read training data of length %d: " % len(df))
    print(df)

    df["allele_peptide"] = df.allele + "~" + df.peptide

    if len(args.remove_kind) != len(args.remove_filename):
        parser.error(
            "Number of arguments mismatch: --remove-kind [%d] != "
            "--remove-filename [%d]" % (
                len(args.remove_kind),
                len(args.remove_filename)))

    removed = []

    for (i, (kind, path)) in enumerate(
            zip(args.remove_kind, args.remove_filename)):
        print(
            "Processing file %d / %d: %s %s" % (
                i + 1, len(args.remove_kind), kind, path))
        to_remove = LOADERS[kind](path)
        print("Remove data contains %d entries" % len(to_remove))

        to_remove["normalized_allele"] = to_remove.allele.map(
            normalize_allele_name)

        remove_allele_peptides = set(
            to_remove.normalized_allele + "~" + to_remove.peptide)

        remove_mask = df.allele_peptide.isin(remove_allele_peptides)
        print("Will remove %d entries." % remove_mask.sum())

        removed.append(df.loc[remove_mask].copy())
        df = df.loc[~remove_mask].copy()

        print("New training data size: %d" % len(df))

    print("Done processing.")

    removed_df = pandas.concat(removed)
    print("Removed %d entries in total:" % len(removed_df))
    print(removed_df)

    if args.out_removed:
        removed_df.to_csv(args.out_removed, index=False)
        print("Wrote: ", args.out_removed)

    if args.out:
        df.to_csv(args.out, index=False)
        print("Wrote: ", args.out)


if __name__ == "__main__":
    go(parser.parse_args(sys.argv[1:]))

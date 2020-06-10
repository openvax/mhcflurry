"""
Make training data by selecting decoys, etc.
"""
import sys
import argparse
import os
import numpy
import collections

import pandas
import tqdm

import mhcflurry

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--hits",
    metavar="CSV",
    required=True,
    help="Multiallelic mass spec")
parser.add_argument(
    "--proteome-peptides",
    metavar="CSV",
    required=True,
    help="Proteome peptides")
parser.add_argument(
    "--decoys-per-hit",
    type=float,
    metavar="N",
    default=99,
    help="Decoys per hit")
parser.add_argument(
    "--exclude-pmid",
    nargs="+",
    default=[],
    help="Exclude given PMID")
parser.add_argument(
    "--only-pmid",
    nargs="*",
    default=[],
    help="Include only the given PMID")
parser.add_argument(
    "--exclude-train-data",
    nargs="+",
    default=[],
    help="Remove hits and decoys included in the given training data")
parser.add_argument(
    "--only-format",
    choices=("MONOALLELIC", "MULTIALLELIC"),
    help="Include only data of the given format")
parser.add_argument(
    "--sample-fraction",
    type=float,
    help="Subsample data by specified fraction (e.g. 0.1)")
parser.add_argument(
    "--out",
    metavar="CSV",
    required=True,
    help="File to write")


def run():
    args = parser.parse_args(sys.argv[1:])
    hit_df = pandas.read_csv(args.hits)
    hit_df["pmid"] = hit_df["pmid"].astype(str)
    original_samples_pmids = hit_df.pmid.unique()
    numpy.testing.assert_equal(hit_df.hit_id.nunique(), len(hit_df))
    hit_df = hit_df.loc[
        (hit_df.mhc_class == "I") &
        (hit_df.peptide.str.len() <= 11) &
        (hit_df.peptide.str.len() >= 8) &
        (~hit_df.protein_ensembl.isnull()) &
        (hit_df.peptide.str.match("^[%s]+$" % "".join(
            mhcflurry.amino_acid.COMMON_AMINO_ACIDS)))
    ]
    hit_df['alleles'] = hit_df.hla.str.split().map(tuple)
    print("Loaded hits from %d samples" % hit_df.sample_id.nunique())
    if args.only_format:
        hit_df = hit_df.loc[hit_df.format == args.only_format].copy()
        print("Subselected to %d %s samples" % (
            hit_df.sample_id.nunique(), args.only_format))

    if args.only_pmid or args.exclude_pmid:
        assert not (args.only_pmid and args.exclude_pmid)

        pmids = list(args.only_pmid) + list(args.exclude_pmid)
        missing = [pmid for pmid in pmids if pmid not in original_samples_pmids]
        assert not missing, (missing, original_samples_pmids)

        mask = hit_df.pmid.isin(pmids)
        if args.exclude_pmid:
            mask = ~mask

        new_hit_df = hit_df.loc[mask]
        print(
            "Selecting by pmids",
            pmids,
            "reduced dataset from",
            len(hit_df),
            "to",
            len(new_hit_df))
        hit_df = new_hit_df.copy()
        print("Subselected by pmid to %d samples" % hit_df.sample_id.nunique())

    allele_to_excluded_peptides = collections.defaultdict(set)
    for train_dataset in args.exclude_train_data:
        if not train_dataset:
            continue
        print("Excluding hits from", train_dataset)
        train_df = pandas.read_csv(train_dataset)
        for (allele, peptides) in train_df.groupby("allele").peptide.unique().iteritems():
            allele_to_excluded_peptides[allele].update(peptides)
        train_counts = train_df.groupby(
            ["allele", "peptide"]).measurement_value.count().to_dict()
        hit_no_train = hit_df.loc[
            [
                not any([
                    train_counts.get((allele, row.peptide))
                    for allele in row.alleles
                ])
            for _, row in tqdm.tqdm(hit_df.iterrows(), total=len(hit_df))]
        ]
        print(
            "Excluding hits from",
            train_dataset,
            "reduced dataset from",
            len(hit_df),
            "to",
            len(hit_no_train))
        hit_df = hit_no_train

    sample_table = hit_df.drop_duplicates("sample_id").set_index("sample_id")
    grouped = hit_df.groupby("sample_id").nunique()
    for col in sample_table.columns:
        if (grouped[col] > 1).any():
            del sample_table[col]

    print("Loading proteome peptides")
    all_peptides_df = pandas.read_csv(args.proteome_peptides)
    print("Loaded: ", all_peptides_df.shape)

    all_peptides_df = all_peptides_df.loc[
        all_peptides_df.protein_accession.isin(hit_df.protein_accession.unique()) &
        all_peptides_df.peptide.str.match("^[%s]+$" % "".join(
            mhcflurry.amino_acid.COMMON_AMINO_ACIDS))
    ].copy()
    all_peptides_df["length"] = all_peptides_df.peptide.str.len()
    print("Subselected proteome peptides by accession: ", all_peptides_df.shape)

    all_peptides_by_length = dict(iter(all_peptides_df.groupby("length")))

    print("Selecting decoys.")

    lengths = [8, 9, 10, 11]
    result_df = []

    for sample_id, sub_df in tqdm.tqdm(
            hit_df.groupby("sample_id"), total=hit_df.sample_id.nunique()):
        result_df.append(
            sub_df[[
                "protein_accession",
                "peptide",
                "sample_id",
                "n_flank",
                "c_flank",
            ]].copy())
        result_df[-1]["hit"] = 1

        excluded_peptides = set()
        for allele in sample_table.loc[sample_id].alleles:
            excluded_peptides.update(allele_to_excluded_peptides[allele])
        print(
            sample_id,
            "will exclude",
            len(excluded_peptides),
            "peptides from decoy universe")

        for length in lengths:
            universe = all_peptides_by_length[length]
            possible_universe = universe.loc[
                (~universe.peptide.isin(sub_df.peptide.unique())) &
                (~universe.peptide.isin(excluded_peptides)) &
                (universe.protein_accession.isin(sub_df.protein_accession.unique()))
            ]
            selected_decoys = possible_universe.sample(
                n=int(len(sub_df) * args.decoys_per_hit / len(lengths)))

            result_df.append(selected_decoys[
                ["protein_accession", "peptide", "n_flank", "c_flank"]
            ].copy())
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

    if args.sample_fraction:
        print("Subsampling to ", args.sample_fraction)
        result_df = result_df.sample(frac=args.sample_fraction)
        print("Subsampled:")
        print(result_df)
        print("Hit rates:")
        print(result_df.groupby("sample_id").hit.mean().sort_values())

    result_df.to_csv(args.out, index=False)
    print("Wrote: ", args.out)


if __name__ == '__main__':
    run()

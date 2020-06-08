"""
Make training data by selecting decoys, etc.
"""
import sys
import argparse
import os
import numpy

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
    "--predictions",
    metavar="CSV",
    required=True,
    help="Predictions data")
parser.add_argument(
    "--affinity-predictor",
    metavar="CSV",
    help="Class 1 affinity predictor to use (exclusive with --predictions)")
parser.add_argument(
    "--proteome-peptides",
    metavar="CSV",
    required=True,
    help="Proteome peptides")
parser.add_argument(
    "--hit-multiplier-to-take",
    type=float,
    default=1,
    help="")
parser.add_argument(
    "--ppv-multiplier",
    type=int,
    metavar="N",
    default=1000,
    help="Take top 1/N predictions.")
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


def load_predictions(dirname, result_df=None, columns=None):
    peptides = pandas.read_csv(
        os.path.join(dirname, "peptides.csv")).peptide
    manifest_df = pandas.read_csv(os.path.join(dirname, "alleles.csv"))

    print(
        "Loading results. Existing data has",
        len(peptides),
        "peptides and",
        len(manifest_df),
        "columns")

    if columns is None:
        columns = manifest_df.col.values

    if result_df is None:
        result_df = pandas.DataFrame(
            index=peptides, columns=columns, dtype="float32")
        result_df[:] = numpy.nan
        peptides_to_assign = peptides
        mask = None
    else:
        mask = (peptides.isin(result_df.index)).values
        peptides_to_assign = peptides[mask]

    manifest_df = manifest_df.loc[manifest_df.col.isin(result_df.columns)]

    for _, row in manifest_df.iterrows():
        with open(os.path.join(dirname, row.path), "rb") as fd:
            value = numpy.load(fd)['arr_0']
            if mask is not None:
                value = value[mask]
            result_df.loc[peptides_to_assign, row.col] = value

    return result_df


def run():
    args = parser.parse_args(sys.argv[1:])
    hit_df = pandas.read_csv(args.hits)
    numpy.testing.assert_equal(hit_df.hit_id.nunique(), len(hit_df))
    hit_df = hit_df.loc[
        (hit_df.mhc_class == "I") &
        (hit_df.peptide.str.len() <= 11) &
        (hit_df.peptide.str.len() >= 8) &
        (~hit_df.protein_ensembl.isnull()) &
        (hit_df.peptide.str.match("^[%s]+$" % "".join(
            mhcflurry.amino_acid.COMMON_AMINO_ACIDS)))
    ]
    print("Loaded hits from %d samples" % hit_df.sample_id.nunique())
    hit_df = hit_df.loc[hit_df.format == "MONOALLELIC"].copy()
    print("Subselected to %d monoallelic samples" % hit_df.sample_id.nunique())
    hit_df["allele"] = hit_df.hla

    hit_df = hit_df.loc[hit_df.allele.str.match("^HLA-[ABC]")]
    print("Subselected to %d HLA-A/B/C samples" % hit_df.sample_id.nunique())

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
            hit_df.allele.isin(filter_alleles)
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

    columns_to_keep = [
        "hit_id",
        "protein_accession",
        "n_flank",
        "c_flank",
        "peptide",
        "sample_id",
        "affinity_prediction",
        "hit",
    ]

    affinity_predictor = None
    if args.affinity_predictor:
        affinity_predictor = mhcflurry.Class1AffinityPredictor.load(args.affinity_predictor)
        print("Loaded", affinity_predictor)
    if (args.predictions and affinity_predictor) or not (args.predictions or affinity_predictor):
        parser.error("Specify one of --affinity-predictor or --predictions")

    print("Selecting decoys.")

    lengths = [8, 9, 10, 11]
    result_df = []
    for sample_id, sub_hit_df in tqdm.tqdm(
            hit_df.groupby("sample_id"), total=hit_df.sample_id.nunique()):

        sub_hit_df = sub_hit_df.copy()
        sub_hit_df["hit"] = 1

        decoys_df = []
        for length in lengths:
            universe = all_peptides_by_length[length]
            decoys_df.append(
                universe.loc[
                    (~universe.peptide.isin(sub_hit_df.peptide.unique())) &
                    (universe.protein_accession.isin(sub_hit_df.protein_accession.unique()))
                ].sample(
                    n=int(len(sub_hit_df) * args.ppv_multiplier / len(lengths)))[[
                        "protein_accession", "peptide", "n_flank", "c_flank"
                ]].drop_duplicates("peptide"))

        merged_df = pandas.concat(
            [sub_hit_df] + decoys_df, ignore_index=True, sort=False)

        prediction_col = "%s affinity" % sample_table.loc[sample_id].hla
        predictions_df = pandas.DataFrame(
            index=merged_df.peptide.unique(),
            columns=[prediction_col])
        if affinity_predictor:
            predictions_df[prediction_col] = affinity_predictor.predict(
                predictions_df.index,
                allele=sample_table.loc[sample_id].hla)
        else:
            load_predictions(args.predictions, result_df=predictions_df)

        merged_df["affinity_prediction"] = merged_df.peptide.map(
            predictions_df[prediction_col])
        merged_df = merged_df.sort_values("affinity_prediction", ascending=True)

        num_to_take = int(len(sub_hit_df) * args.hit_multiplier_to_take)
        selected_df = merged_df.head(num_to_take)[
                columns_to_keep
        ].sample(frac=1.0).copy()
        selected_df["hit"] = selected_df["hit"].fillna(0)
        selected_df["sample_id"] = sample_id
        result_df.append(selected_df)

        print(
            "Processed sample",
            sample_id,
            "with hit and decoys:\n",
            selected_df.hit.value_counts())

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

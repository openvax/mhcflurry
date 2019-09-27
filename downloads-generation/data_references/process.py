"""
"""
import sys
import argparse
import os
import gzip

import pandas

import gtfparse
import shellinford
from Bio import SeqIO

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "input_paths",
    nargs="+",
    help="Fasta files to process")
parser.add_argument(
    "--out-csv",
    required=True,
    metavar="FILE.csv",
    help="CSV output")
parser.add_argument(
    "--out-index",
    required=True,
    metavar="FILE.fm",
    help="Index output")
parser.add_argument(
    "--id-mapping",
    required=True,
    metavar="FILE.idmapping.gz",
    help="Uniprot mapping file")
parser.add_argument(
    "--ensembl-gtf",
    required=True,
    metavar="FILE.gtf.gz",
    help="Ensembl GTF file")

def run():
    args = parser.parse_args(sys.argv[1:])

    fm = shellinford.FMIndex()
    df = []
    for f in args.input_paths:
        print("Processing", f)
        with gzip.open(f, "rt") as fd:
            records = SeqIO.parse(fd, format='fasta')
            for (i, record) in enumerate(records):
                seq = str(record.seq).upper()
                df.append((record.name, record.description, seq))
                fm.push_back("$" + seq + "$")  # include sentinels
    df = pandas.DataFrame(df, columns=["name", "description", "seq"])

    print("Done reading fastas")
    print(df)

    pieces = df.name.str.split("|")
    df["db"] = pieces.str.get(0)
    df["accession"] = pieces.str.get(1)
    df["entry"] = pieces.str.get(2)

    print("Annotating using mapping", args.id_mapping)
    mapping_df = pandas.read_csv(
        args.id_mapping, sep="\t", header=None)
    mapping_df.columns = ['accession', 'key', 'value']

    for item in ["Ensembl", "Ensembl_TRS", "Gene_Name"]:
        accession_to_values = mapping_df.loc[
            mapping_df.key == item
        ].groupby("accession").value.unique().map(" ".join)
        df[item.lower()] = df.accession.map(accession_to_values)

    print("Annotating using gtf", args.ensembl_gtf)
    gtf_df = gtfparse.read_gtf(args.ensembl_gtf)
    matching_ensembl_genes = set(gtf_df.gene_id.unique())
    ensembl_primary = []
    for ensembls in df.ensembl.fillna("").str.split():
        result = ""
        for item in ensembls:
            if item in matching_ensembl_genes:
                result = item
                break
        ensembl_primary.append(result)
    df["ensembl_primary"] = ensembl_primary
    print("Fraction of records with matching ensembl genes", (
            df.ensembl_primary != "").mean())

    gene_records = gtf_df.loc[gtf_df.feature == "gene"].set_index("gene_id")
    df["primary_ensembl_contig"] = df.ensembl_primary.map(gene_records.seqname)
    df["primary_ensembl_start"] = df.ensembl_primary.map(gene_records.start)
    df["primary_ensembl_end"] = df.ensembl_primary.map(gene_records.end)
    df["primary_ensembl_strand"] = df.ensembl_primary.map(gene_records.strand)

    print("Done annotating")
    print(df)

    df.to_csv(args.out_csv, index=True)
    print("Wrote: ", os.path.abspath((args.out_csv)))

    print("Building index")
    fm.build()
    fm.write(args.out_index)
    print("Wrote: ", os.path.abspath((args.out_index)))


if __name__ == '__main__':
    run()

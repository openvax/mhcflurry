"""
Filter and combine various peptide/MHC datasets to derive a composite training set,
optionally including eluted peptides identified by mass-spec.
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

parser.add_argument(
    "--data-kim2014",
    action="append",
    default=[],
    help="Path to Kim 2014-style affinity data")
parser.add_argument(
    "--data-iedb",
    action="append",
    default=[],
    help="Path to IEDB-style affinity data (e.g. mhc_ligand_full.csv)")
parser.add_argument(
    "--data-additional-ms",
    action="append",
    default=[],
    help="Path to additional monoallelic mass spec hits")
parser.add_argument(
    "--data-systemhc-atlas",
    action="append",
    default=[],
    help="Path to systemhc-atlas-style mass-spec data")

parser.add_argument(
    "--out-csv",
    required=True,
    help="Combined result file")
parser.add_argument(
    "--out-affinity-csv",
    required=False,
    help="Result file")
parser.add_argument(
    "--out-mass-spec-csv",
    required=False,
    help="Result file")

QUALITATIVE_TO_AFFINITY_AND_INEQUALITY = {
    "Negative": (5000.0, ">"),
    "Positive": (500.0, "<"),  # used for mass-spec hits
    "Positive-High": (100.0, "<"),
    "Positive-Intermediate": (1000.0, "<"),
    "Positive-Low": (5000.0, "<"),
}
QUALITATIVE_TO_AFFINITY = dict(
    (key, value[0]) for (key, value)
    in QUALITATIVE_TO_AFFINITY_AND_INEQUALITY.items())
QUALITATIVE_TO_INEQUALITY = dict(
    (key, value[1]) for (key, value)
    in QUALITATIVE_TO_AFFINITY_AND_INEQUALITY.items())


EXCLUDE_IEDB_ALLELES = [
    "HLA class I",
    "HLA class II",
]


def load_data_kim2014(filename):
    df = pandas.read_table(filename)
    print("Loaded kim2014 data: %s" % str(df.shape))
    df["measurement_source"] = "kim2014"
    df["measurement_kind"] = "affinity"
    df["measurement_value"] = df.meas
    df["measurement_type"] = (df.inequality == "=").map({
        True: "quantitative",
        False: "qualitative",
    })
    df["measurement_inequality"] = df.inequality
    df["original_allele"] = df.mhc
    df["peptide"] = df.sequence
    df["allele"] = df.mhc.map(normalize_allele_name)
    print("Dropping un-parseable alleles: %s" % ", ".join(
        df.loc[df.allele == "UNKNOWN"]["mhc"].unique()))
    df = df.loc[df.allele != "UNKNOWN"]

    print("Loaded kim2014 data: %s" % str(df.shape))
    return df


def load_data_systemhc_atlas(filename, min_probability=0.99):
    df = pandas.read_csv(filename)
    print("Loaded systemhc atlas data: %s" % str(df.shape))

    df["measurement_kind"] = "mass_spec"
    df["measurement_source"] = "systemhc-atlas"
    df["measurement_value"] = QUALITATIVE_TO_AFFINITY["Positive"]
    df["measurement_inequality"] = "<"
    df["measurement_type"] = "qualitative"
    df["original_allele"] = df.top_allele
    df["peptide"] = df.search_hit
    df["allele"] = df.top_allele.map(normalize_allele_name)

    print("Dropping un-parseable alleles: %s" % ", ".join(
        str(x) for x in df.loc[df.allele == "UNKNOWN"]["top_allele"].unique()))
    df = df.loc[df.allele != "UNKNOWN"]
    print("Systemhc atlas data now: %s" % str(df.shape))

    print("Dropping data points with probability < %f" % min_probability)
    df = df.loc[df.prob >= min_probability]
    print("Systemhc atlas data now: %s" % str(df.shape))

    print("Removing duplicates")
    df = df.drop_duplicates(["allele", "peptide"])
    print("Systemhc atlas data now: %s" % str(df.shape))

    return df


def load_data_iedb(iedb_csv, include_qualitative=True):
    iedb_df = pandas.read_csv(iedb_csv, skiprows=1, low_memory=False)
    print("Loaded iedb data: %s" % str(iedb_df.shape))

    print("Selecting only class I")
    iedb_df = iedb_df.loc[
        iedb_df["MHC allele class"].str.strip().str.upper() == "I"
    ]
    print("New shape: %s" % str(iedb_df.shape))

    print("Dropping known unusuable alleles")
    iedb_df = iedb_df.loc[
        ~iedb_df["Allele Name"].isin(EXCLUDE_IEDB_ALLELES)
    ]
    iedb_df = iedb_df.loc[
        (~iedb_df["Allele Name"].str.contains("mutant")) &
        (~iedb_df["Allele Name"].str.contains("CD1"))
    ]

    # Drop insufficiently specific allele names like "HLA-A03":
    insuffient_mask = (
        (~iedb_df["Allele Name"].str.upper().str.startswith("H2-")) &
        (~iedb_df["Allele Name"].str.upper().str.startswith("H-2-")) &
        (~iedb_df["Allele Name"].str.upper().str.startswith("MAMU")) &
        (iedb_df["Allele Name"].str.findall("[0-9]").str.len() < 4)
    )
    print("Dropping %d records with insufficiently-specific allele names:" %
        insuffient_mask.sum())
    print(iedb_df.loc[insuffient_mask]["Allele Name"].value_counts())
    iedb_df = iedb_df.loc[~insuffient_mask]

    iedb_df["allele"] = iedb_df["Allele Name"].map(normalize_allele_name)
    print("Dropping un-parseable alleles: %s" % ", ".join(
        iedb_df.loc[iedb_df.allele == "UNKNOWN"]["Allele Name"].unique()))
    iedb_df = iedb_df.loc[iedb_df.allele != "UNKNOWN"]

    print("IEDB measurements per allele:\n%s" % iedb_df.allele.value_counts())

    quantitative = iedb_df.loc[iedb_df["Units"] == "nM"].copy()
    quantitative["measurement_kind"] = "affinity"
    quantitative["measurement_type"] = "quantitative"
    quantitative["measurement_inequality"] = quantitative[
        "Measurement Inequality"
    ].fillna("=").map(lambda s: {">=": ">", "<=": "<"}.get(s, s))
    print("Quantitative measurements: %d" % len(quantitative))

    qualitative = iedb_df.loc[iedb_df["Units"].isnull()].copy()
    qualitative["measurement_type"] = "qualitative"
    qualitative["measurement_kind"] = qualitative[
        "Method/Technique"
    ].str.contains("mass spec").map({
        True: "mass_spec",
        False: "affinity",
    })
    print("Qualitative measurements: %d" % len(qualitative))

    qualitative["Quantitative measurement"] = (
        qualitative["Qualitative Measure"].map(QUALITATIVE_TO_AFFINITY))
    qualitative["measurement_inequality"] = (
        qualitative["Qualitative Measure"].map(QUALITATIVE_TO_INEQUALITY))

    print("Qualitative measurements (possibly after dropping MS): %d" % (
        len(qualitative)))

    iedb_df = pandas.concat(
        (
            ([quantitative]) +
            ([qualitative] if include_qualitative else [])),
        ignore_index=True)

    print("IEDB measurements per allele:\n%s" % iedb_df.allele.value_counts())

    print("Subselecting to valid peptides. Starting with: %d" % len(iedb_df))
    iedb_df["Description"] = iedb_df.Description.str.strip()
    iedb_df = iedb_df.loc[
        iedb_df.Description.str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")
    ]
    print("Now: %d" % len(iedb_df))

    print("Annotating last author and category")
    iedb_df["last_author"] = iedb_df.Authors.map(
        lambda x: (
            x.split(";")[-1]
            .split(",")[-1]
            .split(" ")[-1]
            .strip()
            .replace("*", ""))).values
    iedb_df["category"] = (
        iedb_df["last_author"] + " - " + iedb_df["Method/Technique"]).values

    train_data = pandas.DataFrame()
    train_data["peptide"] = iedb_df.Description.values
    train_data["measurement_value"] = iedb_df[
        "Quantitative measurement"
    ].values
    train_data["measurement_source"] = iedb_df.category.values
    train_data["measurement_inequality"] = iedb_df.measurement_inequality.values

    train_data["allele"] = iedb_df["allele"].values
    train_data["original_allele"] = iedb_df["Allele Name"].values
    train_data["measurement_type"] = iedb_df["measurement_type"].values
    train_data["measurement_kind"] = iedb_df["measurement_kind"].values
    train_data = train_data.drop_duplicates().reset_index(drop=True)

    return train_data


def load_data_additional_ms(filename):
    df = pandas.read_csv(filename)
    print("Loaded additional MS", filename, df.shape)
    print(df)
    print("Entries:", len(df))

    print("Subselecting to monoallelic")
    df = df.loc[
        df.format == "MONOALLELIC"
    ].copy()
    print("Now", len(df))

    df["allele"] = df["hla"].map(normalize_allele_name)
    assert not (df.allele == "UNKNOWN").any()
    df["measurement_value"] = QUALITATIVE_TO_AFFINITY["Positive"]
    df["measurement_inequality"] = "<"
    df["measurement_type"] = "qualitative"
    df["measurement_kind"] = "mass_spec"
    df["measurement_source"] = "MS:pmid:" + df["original_pmid"].map(str)
    df["original_allele"] = ""
    return df


def run():
    args = parser.parse_args(sys.argv[1:])

    dfs = []
    for filename in args.data_iedb:
        df = load_data_iedb(filename)
        dfs.append(df)
    for filename in args.data_kim2014:
        df = load_data_kim2014(filename)
        df["allele_peptide"] = df.allele + "_" + df.peptide

        # Give precedence to IEDB data.
        if dfs:
            iedb_df = dfs[0]
            iedb_df["allele_peptide"] = iedb_df.allele + "_" + iedb_df.peptide
            print("Dropping kim2014 data present in IEDB.")
            df = df.loc[
                ~df.allele_peptide.isin(iedb_df.allele_peptide)
            ]
            print("Kim2014 data now: %s" % str(df.shape))
        dfs.append(df)
    for filename in args.data_systemhc_atlas:
        df = load_data_systemhc_atlas(filename)
        dfs.append(df)

    for filename in args.data_additional_ms:
        df = load_data_additional_ms(filename)
        dfs.append(df)

    df = pandas.concat(dfs, ignore_index=True)
    print("Combined df: %s" % (str(df.shape)))

    print("Removing combined duplicates")
    df = df.drop_duplicates(
        ["allele", "peptide", "measurement_value", "measurement_kind"])
    print("New combined df: %s" % (str(df.shape)))

    df = df[[
        "allele",
        "peptide",
        "measurement_value",
        "measurement_inequality",
        "measurement_type",
        "measurement_kind",
        "measurement_source",
        "original_allele",
    ]].sort_values(["allele", "peptide"]).dropna()

    print("Final combined df: %s" % (str(df.shape)))

    print("Measurement sources:")
    print(df.measurement_source.value_counts())

    print("Measurement kind:")
    print(df.measurement_kind.value_counts())

    print("Measurement source / kind:")
    print(
        df.groupby(
            ["measurement_source", "measurement_kind"]
        ).peptide.count().sort_values())

    def write(write_df, filename):
        filename = os.path.abspath(filename)
        write_df.to_csv(filename, index=False)
        print("Wrote [%d lines]: %s" % (len(write_df), filename))

    write(df, args.out_csv)
    if args.out_affinity_csv:
        write(
            df.loc[df.measurement_kind == "affinity"],
            args.out_affinity_csv)
    if args.out_mass_spec_csv:
        write(
            df.loc[df.measurement_kind == "mass_spec"],
            args.out_mass_spec_csv)


if __name__ == '__main__':
    run()

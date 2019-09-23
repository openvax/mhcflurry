"""
Filter and combine various peptide/MHC datasets to derive a composite training set,
optionally including eluted peptides identified by mass-spec.
"""
import sys
import argparse
import os
import collections

import pandas

import mhcnames


def normalize_allele_name(s):
    try:
        return mhcnames.normalize_allele_name(s)
    except Exception:
        return "UNKNOWN"


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--item",
    nargs="+",
    action="append",
    metavar="PMID FILE, ... FILE",
    default=[],
    help="Item to curate: PMID and list of files")
parser.add_argument(
    "--out",
    metavar="OUT.csv",
    help="Out file path")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Leave user in pdb if PMID is unsupported")

HANDLERS = {}


def load(filenames, **kwargs):
    result = {}
    for filename in filenames:
        if filename.endswith(".csv"):
            result[filename] = pandas.read_csv(filename, **kwargs)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            result[filename] = pandas.read_excel(filename, **kwargs)
        else:
            result[filename] = filename

    return result


def debug(*filenames):
    loaded = load(filenames)
    import ipdb
    ipdb.set_trace()


def handle_pmid_27600516(filename):
    df = pandas.read_csv(filename)

    sample_to_peptides = {}
    current_sample = None
    for peptide in df.peptide:
        if peptide.startswith("#"):
            current_sample = peptide[1:]
            sample_to_peptides[current_sample] = []
        else:
            assert current_sample is not None
            sample_to_peptides[current_sample].append(peptide.strip().upper())

    rows = []
    for (sample, peptides) in sample_to_peptides.items():
        for peptide in sorted(set(peptides)):
            rows.append([sample, peptide])

    result = pandas.DataFrame(rows, columns=["sample_id", "peptide"])
    result["sample_type"] = "melanoma_cell_line"
    return result


def handle_pmid_23481700(filename):
    df = pandas.read_excel(filename)
    peptides = df.iloc[10:,0].values
    assert peptides[0] == "TPSLVKSTSQL"
    assert peptides[-1] == "LPHSVNSKL"

    result = pandas.DataFrame({
        "peptide": peptides,
    })
    result["sample_id"] = "23481700"
    result["sample_type"] = "B-LCL"
    return result


def handle_pmid_24616531(filename):
    df = pandas.read_excel(filename, sheetname="EThcD")
    peptides = df.Sequence.values
    assert peptides[0] == "APFLRIAF"
    assert peptides[-1] == "WRQAGLSYIRYSQI"

    result = pandas.DataFrame({
        "peptide": peptides,
    })
    result["sample_id"] = "24616531"
    result["sample_type"] = "B-lymphoblastoid"
    result["cell_line"] = "GR"
    result["pulldown_antibody"] = "W6/32"

    # Note: this publication lists hla as "HLA-A*01,-03, B*07,-27, and -C*02,-07"
    # we are guessing the exact 4 digit alleles based on this.
    result["hla"] = "HLA-A*01:01 HLA-A*03:01 HLA-B*07:02 HLA-B*27:05 HLA-C*02:02 HLA-C*07:01"
    return result


def handle_pmid_25576301(filename):
    df = pandas.read_excel(filename, sheetname="Peptides")
    peptides = df.Sequence.values   
    assert peptides[0] == "AAAAAAAQSVY"
    assert peptides[-1] == "YYYNGKAVY"

    column_to_sample = {}
    for s in [c for c in df if c.startswith("Intensity ")]:
        assert s[-2] == "-"
        column_to_sample[s] = s.replace("Intensity ", "")[:-2].strip()

    intensity_columns = list(column_to_sample)

    rows = []
    for _, row in df.iterrows():
        x1 = row[intensity_columns]
        x2 = x1[x1 > 0].index.map(column_to_sample).value_counts()
        x3 = x2[x2 >= 2]  # require at least two replicates for each peptide
        for sample in x3.index:
            rows.append((row.Sequence, sample))

    result = pandas.DataFrame(rows, columns=["peptide", "sample_id"])
    result["cell_line"] = ""
    result["pulldown_antibody"] = "W6/32"

    allele_map = {
        'Fib': "HLA-A*03:01	HLA-A*23:01	HLA-B*08:01	HLA-B*15:18	HLA-C*07:02	HLA-C*07:04",
        'HCC1937': "HLA-A*23:01 HLA-A*24:02 HLA-B*07:02 HLA-B*40:01 HLA-C*03:04 HLA-C*07:02",
        'SupB15WT': None,  # four digit alleles unknown, will drop sample
        'SupB15RT': None,
        'HCT116': "HLA-A*01:01 HLA-A*02:01 HLA-B*45:01 HLA-B*18:01 HLA-C*05:01 HLA-C*07:01",

        # Homozygous at HLA-A:
        'HCC1143': "HLA-A*31:01 HLA-A*31:01 HLA-B*35:08 HLA-B*37:01 HLA-C*04:01 HLA-C*06:02",

        # Homozygous everywhere:
        'JY': "HLA-A*02:01 HLA-A*02:01 HLA-B*07:02 HLA-B*07:02 HLA-C*07:02 HLA-C*07:02",
    }

    sample_type = {
        'Fib': "fibroblast",
        'HCC1937': "basal like breast cancer",
        'SupB15WT': None,
        'SupB15RT': None,
        'HCT116': "colon carcinoma",
        'HCC1143': "basal like breast cancer",
        'JY': "B-cell",
    }
    result["hla"] = result.sample_id.map(allele_map)
    print("Entries before dropping samples with unknown alleles", len(result))
    result = result.loc[~result.hla.isnull()]
    print("Entries after dropping samples with unknown alleles", len(result))
    result["sample_type"] = result.sample_id.map(sample_type)
    print(result.head(3))
    return result



# Hack to add all functions with names like handle_pmid_XXXX to HANDLERS dict.
for (key, value) in list(locals().items()):
    if key.startswith("handle_pmid_"):
        HANDLERS[key.replace("handle_pmid_", "")] = value


def run():
    args = parser.parse_args(sys.argv[1:])

    dfs = []
    for item_tpl in args.item:
        (pmid, filenames) = (item_tpl[0], item_tpl[1:])
        print("Processing item", pmid, *[os.path.abspath(f) for f in filenames])

        df = None
        if pmid in HANDLERS:
            df = HANDLERS[pmid](*filenames)
        elif args.debug:
            debug(*filenames)
        else:
            raise NotImplementedError(args.pmid)

        if df is not None:
            df["pmid"] = pmid
            print("*** PMID %s: %d peptides ***" % (pmid, len(df)))
            print("Counts by sample id:")
            print(df.groupby("sample_id").peptide.nunique())
            print("")
            print("Counts by sample type:")
            print(df.groupby("sample_type").peptide.nunique())
            print("****************************")

            dfs.append(df)


    df = pandas.concat(dfs, ignore_index=True)
    df.to_csv(args.out, index=False)
    print("Wrote: %s" % args.out)

if __name__ == '__main__':
    run()

"""
Filter and combine various peptide/MHC datasets to derive a composite training set,
optionally including eluted peptides identified by mass-spec.
"""
import sys
import argparse
import os
import json
import collections
from six.moves import StringIO

import pandas

import mhcnames


def normalize_allele_name(s):
    try:
        return mhcnames.normalize_allele_name(s)
    except Exception:
        return "UNKNOWN"


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--ms-item",
    nargs="+",
    action="append",
    metavar="PMID FILE, ... FILE",
    default=[],
    help="Mass spec item to curate: PMID and list of files")
parser.add_argument(
    "--expression-item",
    nargs="+",
    action="append",
    metavar="LABEL FILE, ... FILE",
    default=[],
    help="Expression data to curate: dataset label and list of files")
parser.add_argument(
    "--ms-out",
    metavar="OUT.csv",
    help="Out file path (MS data)")
parser.add_argument(
    "--expression-out",
    metavar="OUT.csv",
    help="Out file path (RNA-seq expression)")
parser.add_argument(
    "--expression-metadata-out",
    metavar="OUT.csv",
    help="Out file path for expression metadata, i.e. which samples used")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Leave user in pdb if PMID is unsupported")

PMID_HANDLERS = {}
EXPRESSION_HANDLERS = {}

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
    """Gloger, ..., Neri Cancer Immunol Immunother 2016 [PMID 27600516]"""
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

    result_df = pandas.DataFrame(rows, columns=["sample_id", "peptide"])
    result_df["sample_type"] = "melanoma_cell_line"
    result_df["cell_line"] = result_df.sample_id
    result_df["mhc_class"] = "I"
    result_df["pulldown_antibody"] = "W6/32"
    result_df["format"] = "multiallelic"
    result_df["hla"] = result_df.sample_id.map({
        "FM-82": "HLA-A*02:01 HLA-A*01:01 HLA-B*08:01 HLA-B*15:01 HLA-C*03:04 HLA-C*07:01",
        "FM-93/2": "HLA-A*02:01 HLA-A*26:01 HLA-B*40:01 HLA-B*44:02 HLA-C*03:04 HLA-C*05:01",
        "Mel-624": "HLA-A*02:01 HLA-A*03:01 HLA-B*07:02 HLA-B*14:01 HLA-C*07:02 HLA-C*08:02",
        "MeWo": "HLA-A*02:01 HLA-A*26:01 HLA-B*14:02 HLA-B*38:01 HLA-C*08:02 HLA-C*12:03",
        "SK-Mel-5": "HLA-A*02:01 HLA-A*11:01 HLA-B*40:01 HLA-C*03:03",
    })
    return result_df


def handle_pmid_23481700(filename):
    """Hassan, ..., van Veelen Mol Cell Proteomics 2015 [PMID 23481700]"""
    df = pandas.read_excel(filename, skiprows=10)
    assert df["Peptide sequence"].iloc[0] == "TPSLVKSTSQL"
    assert df["Peptide sequence"].iloc[-1] == "LPHSVNSKL"

    hla = {
        "JY": "HLA-A*02:01 HLA-B*07:02 HLA-C*07:02",
        "HHC": "HLA-A*02:01 HLA-B*07:02 HLA-B*44:02 HLA-C*05:01 HLA-C*07:02",
    }

    results = []
    for sample_id in ["JY", "HHC"]:
        hits_df = df.loc[
            df["Int %s" % sample_id].map(
                lambda x: {"n.q.": 0, "n.q": 0}.get(x, x)).astype(float) > 0
        ]
        result_df = pandas.DataFrame({
            "peptide": hits_df["Peptide sequence"].dropna().values,
        })
        result_df["sample_id"] = sample_id
        result_df["cell_line"] = "B-LCL-" + sample_id
        result_df["hla"] = hla[sample_id]
        result_df["sample_type"] = "B-LCL"
        result_df["mhc_class"] = "I"
        result_df["format"] = "multiallelic"
        result_df["pulldown_antibody"] = "W6/32"
        results.append(result_df)

    result_df = pandas.concat(results, ignore_index=True)

    # Rename samples to avoid a collision with the JY sample in PMID 25576301.
    result_df.sample_id = result_df.sample_id.map({
        "JY": "JY.2015",
        "HHC": "HHC.2015",
    })
    return result_df


def handle_pmid_24616531(filename):
    """Mommen, ..., Heck PNAS 2014 [PMID 24616531]"""
    df = pandas.read_excel(filename, sheet_name="EThcD")
    peptides = df.Sequence.values
    assert peptides[0] == "APFLRIAF"
    assert peptides[-1] == "WRQAGLSYIRYSQI"

    result_df = pandas.DataFrame({
        "peptide": peptides,
    })
    result_df["sample_id"] = "24616531"
    result_df["sample_type"] = "B-LCL"
    result_df["cell_line"] = "GR"
    result_df["pulldown_antibody"] = "W6/32"

    # Note: this publication lists hla as "HLA-A*01,-03, B*07,-27, and -C*02,-07"
    # we are guessing the exact 4 digit alleles based on this.
    result_df["hla"] = "HLA-A*01:01 HLA-A*03:01 HLA-B*07:02 HLA-B*27:05 HLA-C*02:02 HLA-C*07:01"
    result_df["mhc_class"] = "I"
    result_df["format"] = "multiallelic"
    return result_df


def handle_pmid_25576301(filename):
    """Bassani-Sternberg, ..., Mann Mol Cell Proteomics 2015 [PMID 25576301]"""
    df = pandas.read_excel(filename, sheet_name="Peptides")
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

    result_df = pandas.DataFrame(rows, columns=["peptide", "sample_id"])
    result_df["pulldown_antibody"] = "W6/32"
    result_df["mhc_class"] = "I"
    result_df["format"] = "multiallelic"

    allele_map = {
        'Fib': "HLA-A*03:01 HLA-A*23:01 HLA-B*08:01 HLA-B*15:18 HLA-C*07:02 HLA-C*07:04",
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
    cell_line = {
        'Fib': None,
        'HCC1937': "HCC1937",
        'SupB15WT': None,
        'SupB15RT': None,
        'HCT116': "HCT116",
        'HCC1143': "HCC1143",
        'JY': "JY",
    }
    result_df["hla"] = result_df.sample_id.map(allele_map)
    print("Entries before dropping samples with unknown alleles", len(result_df))
    result_df = result_df.loc[~result_df.hla.isnull()]
    print("Entries after dropping samples with unknown alleles", len(result_df))
    result_df["sample_type"] = result_df.sample_id.map(sample_type)
    result_df["cell_line"] = result_df.sample_id.map(cell_line)
    print(result_df.head(3))
    return result_df


def handle_pmid_26992070(*filenames):
    """Ritz, ..., Fugmann Proteomics 2016 [PMID 26992070]"""
    # Although this publication seems to suggest that HEK293 are C*07:02
    # (figure 3B), in a subsequent publication [PMID 28834231] this group
    # gives the HEK293 HLA type as HLA‐A*03:01, HLA‐B*07:02, and HLA‐C*07:01.
    # We are therefore using the HLA‐C*07:01 (i.e. the latter) typing results
    # here.
    allele_text = """
        Cell line	HLA-A 1	HLA-A 2	HLA-B 1	HLA-B 2	HLA-C 1	HLA-C 2
        HEK293	03:01	03:01	07:02	07:02	07:01	07:01
        HL-60	01:01	01:01	57:01	57:01	06:02	06:02
        RPMI8226	30:01	68:02	15:03	15:10	02:10	03:04
        MAVER-1	24:02	26:01	38:01	44:02	05:01	12:03
        THP-1	02:01	24:02	15:11	35:01	03:03	03:03
    """
    allele_info = pandas.read_csv(
        StringIO(allele_text), sep="\t", index_col=0)
    allele_info.index = allele_info.index.str.strip()
    for gene in ["A", "B", "C"]:
        for num in ["1", "2"]:
            allele_info[
                "HLA-%s %s" % (gene, num)
            ] = "HLA-" + gene + "*" + allele_info["HLA-%s %s" % (gene, num)]
    cell_line_to_allele = allele_info.apply(" ".join, axis=1)

    sheets = {}
    for f in filenames:
        if f.endswith(".xlsx"):
            d = pandas.read_excel(f, sheet_name=None, skiprows=1)
            sheets.update(d)

    dfs = []
    for cell_line in cell_line_to_allele.index:
        # Using data from DeepQuanTR, which appears to be a consensus between
        # two other methods used.
        sheet = sheets[cell_line + "_DeepQuanTR"]
        replicated = sheet.loc[
            sheet[[c for c in sheet if "Sample" in c]].fillna(0).sum(1) > 1
        ]
        df = pandas.DataFrame({
            'peptide': replicated.Sequence.values
        })
        df["sample_id"] = cell_line
        df["hla"] = cell_line_to_allele.get(cell_line)
        dfs.append(df)

    result_df = pandas.concat(dfs, ignore_index=True)
    result_df["pulldown_antibody"] = "W6/32"
    result_df["cell_line"] = result_df["sample_id"]
    result_df["sample_type"] = result_df.sample_id.map({
        "HEK293": "hek",
        "HL-60": "neutrophil",
        "RPMI8226": "b-cell",
        "MAVER-1": "b-LCL",
        "THP-1": "monocyte",
    })
    result_df["mhc_class"] = "I"
    result_df["format"] = "multiallelic"
    return result_df


def handle_pmid_27412690(filename):
    """Shraibman, ..., Admon Mol Cell Proteomics 2016 [PMID 27412690]"""
    hla_types = {
        "U-87": "HLA-A*02:01 HLA-B*44:02 HLA-C*05:01",
        "T98G": "HLA-A*02:01 HLA-B*39:06 HLA-C*07:02",
        "LNT-229": "HLA-A*03:01 HLA-B*35:01 HLA-C*04:01",
    }
    sample_id_to_cell_line = {
        "U-87": "U-87",
        "T98G": "T98G",
        "LNT-229": "LNT-229",
        "U-87+DAC": "U-87",
        "T98G+DAC": "T98G",
        "LNT-229+DAC": "LNT-229",
    }

    df = pandas.read_excel(filename)
    assert df.Sequence.iloc[0] == "AAAAAAGSGTPR"

    intensity_col_to_sample_id = {}
    for col in df:
        if col.startswith("Intensity "):
            sample_id = col.split()[1]
            assert sample_id in sample_id_to_cell_line, (col, sample_id)
            intensity_col_to_sample_id[col] = sample_id

    dfs = []
    for (sample_id, cell_line) in sample_id_to_cell_line.items():
        intensity_cols = [
            c for (c, v) in intensity_col_to_sample_id.items()
            if v == sample_id
        ]
        hits_df = df.loc[
            (df[intensity_cols] > 0).sum(1) > 1
        ]
        result_df = pandas.DataFrame({
            "peptide": hits_df.Sequence.values,
        })
        result_df["sample_id"] = sample_id
        result_df["cell_line"] = cell_line
        result_df["hla"] = hla_types[cell_line]

        dfs.append(result_df)

    result_df = pandas.concat(dfs, ignore_index=True)
    result_df["sample_type"] = "glioblastoma"
    result_df["pulldown_antibody"] = "W6/32"
    result_df["mhc_class"] = "I"
    result_df["format"] = "multiallelic"
    return result_df


def handle_pmid_28832583(*filenames):
    """Bassani-Sternberg, ..., Gfeller PLOS Comp. Bio. 2017 [PMID 28832583]"""
    # This work also reanalyzes data from
    # Pearson, ..., Perreault J Clin Invest 2016 [PMID 27841757]

    (filename_dataset1, filename_dataset2) = sorted(filenames)

    dataset1 = pandas.read_csv(filename_dataset1, sep="\t")
    dataset2 = pandas.read_csv(filename_dataset2, sep="\t")
    df = pandas.concat([dataset1, dataset2], ignore_index=True, sort=False)

    info_text = """
    cell_line	origin	original_pmid	allele1	allele2	allele3	allele4	allele5	allele6
    CD165	B-cell	28832583	HLA-A*02:05	HLA-A*24:02	HLA-B*15:01	HLA-B*50:01	HLA-C*03:03	HLA-C*06:02
    CM467	B-cell	28832583	HLA-A*01:01	HLA-A*24:02	HLA-B*13:02	HLA-B*39:06	HLA-C*06:02	HLA-C*12:03
    GD149	B-cell	28832583	HLA-A*01:01	HLA-A*24:02	HLA-B*38:01	HLA-B*44:03	HLA-C*06:02	HLA-C*12:03
    MD155	B-cell	28832583	HLA-A*02:01	HLA-A*24:02	HLA-B*15:01	HLA-B*18:01	HLA-C*03:03	HLA-C*07:01
    PD42	B-cell	28832583	HLA-A*02:06	HLA-A*24:02	HLA-B*07:02	HLA-B*55:01	HLA-C*01:02	HLA-C*07:02
    RA957	B-cell	28832583	HLA-A*02:20	HLA-A*68:01	HLA-B*35:03	HLA-B*39:01	HLA-C*04:01	HLA-C*07:02
    TIL1	TIL	28832583	HLA-A*02:01	HLA-A*02:01	HLA-B*18:01	HLA-B*38:01	HLA-C*05:01	
    TIL3	TIL	28832583	HLA-A*01:01	HLA-A*23:01	HLA-B*07:02	HLA-B*15:01	HLA-C*12:03	HLA-C*14:02
    Apher1	Leukapheresis	28832583	HLA-A*03:01	HLA-A*29:02	HLA-B*44:02	HLA-B*44:03	HLA-C*12:03	HLA-C*16:01
    Apher6	Leukapheresis	28832583	HLA-A*02:01	HLA-A*03:01	HLA-B*07:02		HLA-C*07:02	
    pat_AC2	B-LCL	27841757	HLA-A*03:01	HLA-A*32:01	HLA-B*27:05	HLA-B*45:01		
    pat_C	B-LCL	27841757	HLA-A*02:01	HLA-A*03:01	HLA-B*07:02		HLA-C*07:02	
    pat_CELG	B-LCL	27841757	HLA-A*02:01	HLA-A*24:02	HLA-B*15:01	HLA-B*73:01	HLA-C*03:03	HLA-C*15:05
    pat_CP2	B-LCL	27841757	HLA-A*11:01		HLA-B*14:02	HLA-B*44:02		
    pat_FL	B-LCL	27841757	HLA-A*03:01	HLA-A*11:01	HLA-B*44:03	HLA-B*50:01		
    pat_J	B-LCL	27841757	HLA-A*02:01	HLA-A*03:01	HLA-B*07:02		HLA-C*07:02	
    pat_JPB3	B-LCL	27841757	HLA-A*02:01	HLA-A*11:01	HLA-B*27:05	HLA-B*56:01		
    pat_JT2	B-LCL	27841757	HLA-A*11:01		HLA-B*18:03	HLA-B*35:01		
    pat_M	B-LCL	27841757	HLA-A*03:01	HLA-A*29:02	HLA-B*08:01	HLA-B*44:03	HLA-C*07:01	HLA-C*16:01
    pat_MA	B-LCL	27841757	HLA-A*02:01	HLA-A*29:02	HLA-B*44:03	HLA-B*57:01	HLA-C*07:01	HLA-C*16:01
    pat_ML	B-LCL	27841757	HLA-A*02:01	HLA-A*11:01	HLA-B*40:01	HLA-B*44:03		
    pat_NS2	B-LCL	27841757	HLA-A*02:01		HLA-B*13:02	HLA-B*41:01		
    pat_NT	B-LCL	27841757	HLA-A*01:01	HLA-A*32:01	HLA-B*08:01			
    pat_PF1	B-LCL	27841757	HLA-A*01:01	HLA-A*02:01	HLA-B*07:02	HLA-B*44:03	HLA-C*07:02	HLA-C*16:01
    pat_R	B-LCL	27841757	HLA-A*03:01	HLA-A*29:02	HLA-B*08:01	HLA-B*44:03	HLA-C*07:01	HLA-C*16:01
    pat_RT	B-LCL	27841757	HLA-A*01:01	HLA-A*02:01	HLA-B*18:01	HLA-B*39:24	HLA-C*05:01	HLA-C*07:01
    pat_SR	B-LCL	27841757	HLA-A*02:01	HLA-A*23:01	HLA-B*18:01	HLA-B*44:03		
    pat_ST	B-LCL	27841757	HLA-A*03:01	HLA-A*24:02	HLA-B*07:02	HLA-B*27:05
    """
    info_df = pandas.read_csv(StringIO(info_text), sep="\t", index_col=0)
    info_df.index = info_df.index.str.strip()

    info_df["hla"] = info_df[
        [c for c in info_df if c.startswith("allele")]
    ].fillna("").apply(" ".join, axis=1)

    results = []
    for col in df.columns:
        if col.startswith("Intensity "):
            sample_id = col.replace("Intensity ", "")
            assert sample_id in info_df.index, sample_id
            peptides = df.loc[df[col].fillna(0) > 0].Sequence.unique()
            result_df = pandas.DataFrame({"peptide": peptides})
            result_df["sample_id"] = sample_id
            result_df["hla"] = info_df.loc[sample_id].hla
            result_df["sample_type"] = info_df.loc[sample_id].origin
            result_df["original_pmid"] = str(
                info_df.loc[sample_id].original_pmid)
            results.append(result_df)

    result_df = pandas.concat(results, ignore_index=True)
    samples = result_df.sample_id.unique()
    for sample_id in info_df.index:
        assert sample_id in samples, (sample_id, samples)

    result_df["mhc_class"] = "I"
    result_df["format"] = "multiallelic"
    result_df["cell_line"] = ""
    result_df["pulldown_antibody"] = "W6/32"
    return result_df


PMID_31495665_SAMPLE_TYPES = {
        "HLA-DR_Lung": "lung",
        "HLA-DR_PBMC_HDSC": "pbmc",
        "HLA-DR_PBMC_RG1095": "pbmc",
        "HLA-DR_PBMC_RG1104": "pbmc",
        "HLA-DR_PBMC_RG1248": "pbmc",
        "HLA-DR_Spleen": "spleen",
        "MAPTAC_A*02:01": "mix:a375,expi293,hek293,hela",
        "MAPTAC_A*11:01": "mix:expi293,hela",
        "MAPTAC_A*32:01": "mix:a375,expi293,hela",
        "MAPTAC_B*07:02": "mix:a375,expi293,hela",
        "MAPTAC_B*45:01": "expi293",
        "MAPTAC_B*52:01": "mix:a375,expi293",
        "MAPTAC_C*03:03": "expi293",
        "MAPTAC_C*06:02": "mix:a375,expi293",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm+": "expi293",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm-": "expi293",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm+": "expi293",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm-": "expi293",
        "MAPTAC_DRB1*01:01": "mix:a375,b721,expi293,kg1,k562",
        "MAPTAC_DRB1*03:01": "expi293",
        "MAPTAC_DRB1*04:01": "expi293",
        "MAPTAC_DRB1*07:01": "mix:expi293,hek293",
        "MAPTAC_DRB1*11:01": "mix:expi293,k562,kg1",
        "MAPTAC_DRB1*12:01_dm+": "expi293",
        "MAPTAC_DRB1*12:01_dm-": "expi293",
        "MAPTAC_DRB1*15:01": "expi293",
        "MAPTAC_DRB3*01:01_dm+": "expi293",
        "MAPTAC_DRB3*01:01_dm-": "expi293",
}
CELL_LINE_MIXTURES = sorted(
    set(
        x for x in PMID_31495665_SAMPLE_TYPES.values()
        if x.startswith("mix:")))


def handle_pmid_31495665(filename):
    """Abelin, ..., Rooney Immunity 2019 [PMID 31495665]"""
    hla_type = {
        "HLA-DR_A375": None,
        "HLA-DR_Lung": "DRB1*01:01 DRB1*03:01 DRB3*01:01",
        "HLA-DR_PBMC_HDSC": "DRB1*03:01 DRB1*11:01 DRB3*01:01 DRB3*02:02",
        "HLA-DR_PBMC_RG1095": "HLA-DRA1*01:01-DRB1*03:01 HLA-DRA1*01:01-DRB1*11:01 HLA-DRA1*01:01-DRB3*01:01 HLA-DRA1*01:01-DRB3*02:02",
        "HLA-DR_PBMC_RG1104": "DRB1*01:01 DRB1*11:01 DRB3*02:02",
        "HLA-DR_PBMC_RG1248": "DRB1*03:01 DRB1*03:01 DRB3*01:01 DRB3*01:01",
        "HLA-DR_SILAC_Donor1_10minLysate": None,
        "HLA-DR_SILAC_Donor1_5hrLysate": None,
        "HLA-DR_SILAC_Donor1_DConly": None,
        "HLA-DR_SILAC_Donor1_UVovernight": None,
        "HLA-DR_SILAC_Donor2_DC_UV_16hr": None,
        "HLA-DR_SILAC_Donor2_DC_UV_24hr": None,
        "HLA-DR_Spleen": "DRB1*04:01 DRB4*01:03 DRB1*15:03 DRB5*01:01",
        "MAPTAC_A*02:01": "HLA-A*02:01",
        "MAPTAC_A*11:01": "HLA-A*11:01",
        "MAPTAC_A*32:01": "HLA-A*32:01",
        "MAPTAC_B*07:02": "HLA-B*07:02",
        "MAPTAC_B*45:01": "HLA-B*45:01",
        "MAPTAC_B*52:01": "HLA-B*52:01",
        "MAPTAC_C*03:03": "HLA-C*03:03",
        "MAPTAC_C*06:02": "HLA-C*06:02",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm+": "HLA-DPB1*06:01-DPA1*01:03",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm-": "HLA-DPB1*06:01-DPA1*01:03",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm+": "HLA-DQB1*06:04-DQA1*01:02",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm-": "HLA-DQB1*06:04-DQA1*01:02",
        "MAPTAC_DRB1*01:01": "HLA-DRA1*01:01-DRB1*01:01",
        "MAPTAC_DRB1*03:01": "HLA-DRA1*01:01-DRB1*03:01",
        "MAPTAC_DRB1*04:01": "HLA-DRA1*01:01-DRB1*04:01",
        "MAPTAC_DRB1*07:01": "HLA-DRA1*01:01-DRB1*07:01",
        "MAPTAC_DRB1*11:01": "HLA-DRA1*01:01-DRB1*11:01",
        "MAPTAC_DRB1*12:01_dm+": "HLA-DRA1*01:01-DRB1*12:01",
        "MAPTAC_DRB1*12:01_dm-": "HLA-DRA1*01:01-DRB1*12:01",
        "MAPTAC_DRB1*15:01": "HLA-DRA1*01:01-DRB1*15:01",
        "MAPTAC_DRB3*01:01_dm+": "HLA-DRA1*01:01-DRB3*01:01",
        "MAPTAC_DRB3*01:01_dm-": "HLA-DRA1*01:01-DRB3*01:01",
    }
    pulldown_antibody = {
        "HLA-DR_Lung": "L243 (HLA-DR)",
        "HLA-DR_PBMC_HDSC": "tal1b5 (HLA-DR)",
        "HLA-DR_PBMC_RG1095": "tal1b5 (HLA-DR)",
        "HLA-DR_PBMC_RG1104": "tal1b5 (HLA-DR)",
        "HLA-DR_PBMC_RG1248": "tal1b5 (HLA-DR)",
        "HLA-DR_Spleen": "L243 (HLA-DR)",
        "MAPTAC_A*02:01": "MAPTAC",
        "MAPTAC_A*11:01": "MAPTAC",
        "MAPTAC_A*32:01": "MAPTAC",
        "MAPTAC_B*07:02": "MAPTAC",
        "MAPTAC_B*45:01": "MAPTAC",
        "MAPTAC_B*52:01": "MAPTAC",
        "MAPTAC_C*03:03": "MAPTAC",
        "MAPTAC_C*06:02": "MAPTAC",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm+": "MAPTAC",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm-": "MAPTAC",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm+": "MAPTAC",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm-": "MAPTAC",
        "MAPTAC_DRB1*01:01": "MAPTAC",
        "MAPTAC_DRB1*03:01": "MAPTAC",
        "MAPTAC_DRB1*04:01": "MAPTAC",
        "MAPTAC_DRB1*07:01": "MAPTAC",
        "MAPTAC_DRB1*11:01": "MAPTAC",
        "MAPTAC_DRB1*12:01_dm+": "MAPTAC",
        "MAPTAC_DRB1*12:01_dm-": "MAPTAC",
        "MAPTAC_DRB1*15:01": "MAPTAC",
        "MAPTAC_DRB3*01:01_dm+": "MAPTAC",
        "MAPTAC_DRB3*01:01_dm-": "MAPTAC",
    }
    format = {
        "HLA-DR_Lung": "DR-specific",
        "HLA-DR_PBMC_HDSC": "DR-specific",
        "HLA-DR_PBMC_RG1095": "DR-specific",
        "HLA-DR_PBMC_RG1104": "DR-specific",
        "HLA-DR_PBMC_RG1248": "DR-specific",
        "HLA-DR_Spleen": "DR-specific",
        "MAPTAC_A*02:01": "monoallelic",
        "MAPTAC_A*11:01": "monoallelic",
        "MAPTAC_A*32:01": "monoallelic",
        "MAPTAC_B*07:02": "monoallelic",
        "MAPTAC_B*45:01": "monoallelic",
        "MAPTAC_B*52:01": "monoallelic",
        "MAPTAC_C*03:03": "monoallelic",
        "MAPTAC_C*06:02": "monoallelic",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm+": "monoallelic",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm-": "monoallelic",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm+": "monoallelic",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm-": "monoallelic",
        "MAPTAC_DRB1*01:01": "monoallelic",
        "MAPTAC_DRB1*03:01": "monoallelic",
        "MAPTAC_DRB1*04:01": "monoallelic",
        "MAPTAC_DRB1*07:01": "monoallelic",
        "MAPTAC_DRB1*11:01": "monoallelic",
        "MAPTAC_DRB1*12:01_dm+": "monoallelic",
        "MAPTAC_DRB1*12:01_dm-": "monoallelic",
        "MAPTAC_DRB1*15:01": "monoallelic",
        "MAPTAC_DRB3*01:01_dm+": "monoallelic",
        "MAPTAC_DRB3*01:01_dm-": "monoallelic",
    }
    mhc_class = {
        "HLA-DR_Lung": "II",
        "HLA-DR_PBMC_HDSC": "II",
        "HLA-DR_PBMC_RG1095": "II",
        "HLA-DR_PBMC_RG1104": "II",
        "HLA-DR_PBMC_RG1248": "II",
        "HLA-DR_Spleen": "II",
        "MAPTAC_A*02:01": "I",
        "MAPTAC_A*11:01": "I",
        "MAPTAC_A*32:01": "I",
        "MAPTAC_B*07:02": "I",
        "MAPTAC_B*45:01": "I",
        "MAPTAC_B*52:01": "I",
        "MAPTAC_C*03:03": "I",
        "MAPTAC_C*06:02": "I",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm+": "II",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm-": "II",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm+": "II",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm-": "II",
        "MAPTAC_DRB1*01:01": "II",
        "MAPTAC_DRB1*03:01": "II",
        "MAPTAC_DRB1*04:01": "II",
        "MAPTAC_DRB1*07:01": "II",
        "MAPTAC_DRB1*11:01": "II",
        "MAPTAC_DRB1*12:01_dm+": "II",
        "MAPTAC_DRB1*12:01_dm-": "II",
        "MAPTAC_DRB1*15:01": "II",
        "MAPTAC_DRB3*01:01_dm+": "II",
        "MAPTAC_DRB3*01:01_dm-": "II",
    }
    cell_line = {
        "HLA-DR_Lung": "",
        "HLA-DR_PBMC_HDSC": "",
        "HLA-DR_PBMC_RG1095": "",
        "HLA-DR_PBMC_RG1104": "",
        "HLA-DR_PBMC_RG1248": "",
        "HLA-DR_Spleen": "",
        "MAPTAC_A*02:01": "",
        "MAPTAC_A*11:01": "",
        "MAPTAC_A*32:01": "",
        "MAPTAC_B*07:02": "",
        "MAPTAC_B*45:01": "expi293",
        "MAPTAC_B*52:01": "",
        "MAPTAC_C*03:03": "expi293",
        "MAPTAC_C*06:02": "",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm+": "expi293",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm-": "expi293",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm+": "expi293",  # don't actually see this in DataS1A!
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm-": "expi293",
        "MAPTAC_DRB1*01:01": "",
        "MAPTAC_DRB1*03:01": "expi293",
        "MAPTAC_DRB1*04:01": "expi293",
        "MAPTAC_DRB1*07:01": "",
        "MAPTAC_DRB1*11:01": "",
        "MAPTAC_DRB1*12:01_dm+": "expi293",
        "MAPTAC_DRB1*12:01_dm-": "expi293",
        "MAPTAC_DRB1*15:01": "expi293",
        "MAPTAC_DRB3*01:01_dm+": "expi293",
        "MAPTAC_DRB3*01:01_dm-": "expi293",
    }


    df = pandas.read_excel(filename, sheet_name="DataS1B")
    results = []
    for sample_id in df.columns:
        if hla_type[sample_id] is None:
            print("Intentionally skipping", sample_id)
            continue

        result_df = pandas.DataFrame({
            "peptide": df[sample_id].dropna().values,
        })
        result_df["sample_id"] = sample_id
        result_df["hla"] = hla_type[sample_id]
        result_df["pulldown_antibody"] = pulldown_antibody[sample_id]
        result_df["format"] = format[sample_id]
        result_df["mhc_class"] = mhc_class[sample_id]
        result_df["sample_type"] = PMID_31495665_SAMPLE_TYPES[sample_id]
        result_df["cell_line"] = cell_line[sample_id]
        results.append(result_df)
    result_df = pandas.concat(results, ignore_index=True)
    return result_df


def handle_pmid_27869121(filename):
    """Bassani-Sternberg, ..., Krackhardt Nature Comm. 2016 [PMID 27869121]"""
    # Although this dataset has class II data also, we are only extracting
    # class I for now.
    df = pandas.read_excel(filename, skiprows=1)

    # Taking these from:
    # Supplementary Table 2: Information of patients selected for neoepitope
    # identification
    # For the Mel5 sample, only two-digit alleles are shown (A*01, A*25,
    # B*08, B*18) so we are skipping that sample for now.
    hla_df = pandas.DataFrame([
        ("Mel-8", "HLA-A*01:01 HLA-A*03:01 HLA-B*07:02 HLA-B*08:01 HLA-C*07:01 HLA-C*07:02"),
        ("Mel-12", "HLA-A*01:01 HLA-B*08:01 HLA-C*07:01"),
        ("Mel-15", "HLA-A*03:01 HLA-A*68:01 HLA-B*27:05 HLA-B*35:03 HLA-C*02:02 HLA-C*04:01"),
        ("Mel-16", "HLA-A*01:01 HLA-A*24:02 HLA-B*07:02 HLA-B*08:01 HLA-C*07:01 HLA-C*07:02"),
    ], columns=["sample_id", "hla"]).set_index("sample_id")

    # We assert below that none of the class I hit peptides were found in any
    # of the class II pull downs.
    class_ii_cols = [
        c for c in df.columns if c.endswith("HLA-II (arbitrary units)")
    ]
    class_ii_hits = set(df.loc[
        (df[class_ii_cols].fillna(0.0).sum(1) > 0)
    ].Sequence.unique())

    results = []
    for (sample_id, hla) in hla_df.hla.items():
        intensity_col = "Intensity %s_HLA-I (arbitrary units)" % sample_id
        sub_df = df.loc[
            (df[intensity_col].fillna(0.0) > 0)
        ]
        filtered_sub_df = sub_df.loc[
            (~sub_df.Sequence.isin(class_ii_hits))
        ]
        peptides = filtered_sub_df.Sequence.unique()
        assert not any(p in class_ii_hits for p in peptides)

        result_df = pandas.DataFrame({
            "peptide": peptides,
        })
        result_df["sample_id"] = sample_id
        result_df["hla"] = hla_df.loc[sample_id, "hla"]
        result_df["pulldown_antibody"] = "W6/32"
        result_df["format"] = "multiallelic"
        result_df["mhc_class"] = "I"
        result_df["sample_type"] = "melanoma_met"
        result_df["cell_line"] = None
        results.append(result_df)

    result_df = pandas.concat(results, ignore_index=True)
    return result_df


def handle_pmid_31154438(*filenames):
    """Shraibman, ..., Admon Mol Cell Proteomics 2019 [PMID 31154438]"""
    # Note: this publication also includes analyses of the secreted HLA
    # peptidedome (sHLA) but we are using only the data from membrane-bound
    # HLA.
    (xls, txt) = sorted(filenames, key=lambda s: not s.endswith(".xlsx"))

    info = pandas.read_excel(xls, skiprows=1)
    df = pandas.read_csv(txt, sep="\t", skiprows=1)

    hla_df = info.loc[
        ~info["mHLA tissue sample"].isnull()
    ].set_index("mHLA tissue sample")[["HLA typing"]]

    def fix_hla(string):
        result = []
        alleles = string.split(";")
        for a in alleles:
            a = a.strip()
            if "/" in a:
                (a1, a2) = a.split("/")
                a2 = a1[:2] + a2
                lst = [a1, a2]
            else:
                lst = [a]
            for a in lst:
                normalized = normalize_allele_name(a)
                # Ignore class II
                if normalized[4] in ("A", "B", "C"):
                    result.append(normalized)
        return " ".join(result)

    hla_df["hla"] = hla_df["HLA typing"].map(fix_hla)

    results = []
    for (sample_id, hla) in hla_df.hla.items():
        intensity_col = "Intensity %s" % sample_id
        sub_df = df.loc[
            (df[intensity_col].fillna(0.0) > 0)
        ]
        peptides = sub_df.Sequence.unique()

        result_df = pandas.DataFrame({
            "peptide": peptides,
        })
        result_df["sample_id"] = sample_id
        result_df["hla"] = hla_df.loc[sample_id, "hla"]
        result_df["pulldown_antibody"] = "W6/32"
        result_df["format"] = "multiallelic"
        result_df["mhc_class"] = "I"
        result_df["sample_type"] = "glioblastoma_tissue"
        result_df["cell_line"] = None
        results.append(result_df)

    result_df = pandas.concat(results, ignore_index=True)
    return result_df


def handle_pmid_31844290(*filenames):
    """Sarkizova, ..., Keskin Nature Biotechnology 2019 [PMID 31844290]"""
    (mono_filename, multi_filename) = sorted(filenames)

    # Monoallelic
    mono = pandas.read_excel(mono_filename, sheet_name=None)
    dfs = []
    for (key, value) in mono.items():
        if key == 'Sheet1':
            continue
        allele = normalize_allele_name("HLA-" + key)
        assert allele != "UNKNOWN"
        df = pandas.DataFrame({"peptide": value.sequence.values})
        df["sample_id"] = "keskin_%s" % key
        df["hla"] = allele
        df["pulldown_antibody"] = "W6/32"
        df["format"] = "monoallelic"
        df["mhc_class"] = "I"
        df["sample_type"] = "B-CELL"
        df["cell_line"] = "b721"
        dfs.append(df)

    # Multiallelic
    multi = pandas.read_excel(multi_filename, sheet_name=None)
    metadata = multi['Tissue Sample Characteristics']
    allele_table = metadata.drop_duplicates(
        "Clinical ID").set_index("Clinical ID").loc[
        :, [c for c in metadata if c.startswith("HLA-")]
    ]
    allele_table = allele_table.loc[~allele_table.index.isnull()]
    allele_table = allele_table.loc[allele_table["HLA-A"] != 'n.d.']
    allele_table = allele_table.applymap(
        lambda s: s[1:] if s.startswith("-") else s)
    allele_table = allele_table.applymap(
        lambda s: "B5101" if s == "B51" else s)
    allele_table = allele_table.applymap(normalize_allele_name)

    sample_info = metadata.drop_duplicates(
        "Clinical ID").set_index("Clinical ID")[['Cancer type', 'IP Ab']]
    sample_info = sample_info.loc[~sample_info.index.isnull()].fillna(
        method='ffill')
    sample_info = sample_info.loc[sample_info.index.isin(allele_table.index)]
    sample_info = sample_info.loc[allele_table.index]
    sample_info["hla"] = [
        " ".join(row).replace("HLA-A*31:0102", "HLA-A*31:01")  # fix a typo
        for _, row in allele_table.iterrows()
    ]
    sample_info["sample_type"] = sample_info['Cancer type'].map({
        'CLL': "B-CELL",
        'GBM': "GLIOBLASTOMA_TISSUE",
        'Melanoma': "MELANOMA",
        "Ovarian": "OVARY",
        'ccRCC': "KIDNEY",
    })
    assert not sample_info["sample_type"].isnull().any()
    assert not "UNKNOWN" in sample_info["hla"].any()

    for (key, value) in multi.items():
        if key == 'Tissue Sample Characteristics':
            continue
        for (directory, sub_df) in value.groupby("directory"):
            if 'Pat7' in directory or 'Pat9' in directory:
                print("Skipping due to no HLA typing", directory)
                continue
            try:
                (sample_id,) = sample_info.loc[
                    sample_info.index.map(
                        lambda idx: (
                            idx in directory or
                            idx.replace("-", "_").replace("MEL_", "") in directory or
                            idx.replace(" ", "_") in directory
                        ))
                ].index
            except ValueError as e:
                print(directory, e)
                import ipdb ; ipdb.set_trace()
            info = sample_info.loc[sample_id]
            df = pandas.DataFrame({"peptide": sub_df.sequence.values})
            df["sample_id"] = "keskin_%s" % sample_id.replace(" ", "_")
            df["hla"] = info['hla']
            df["pulldown_antibody"] = info['IP Ab']
            df["format"] = "multiallelic"
            df["mhc_class"] = "I"
            df["sample_type"] = info['sample_type']
            df["cell_line"] = None
            dfs.append(df)

    result_df = pandas.concat(dfs, ignore_index=True)
    result_df["peptide"] = result_df.peptide.str.upper()
    return result_df


EXPRESSION_GROUPS_ROWS = []


def make_expression_groups(dataset_identifier, df, groups):
    result_df = pandas.DataFrame(index=df.index)
    for (label, columns) in groups.items():
        for col in columns:
            if col not in df.columns:
                raise ValueError(
                    "Missing: %s. Available: %s" % (col, df.columns.tolist()))
        result_df[label] = df[columns].mean(1)
        EXPRESSION_GROUPS_ROWS.append((dataset_identifier, label, columns))
    return result_df


def handle_expression_GSE113126(*filenames):
    """
    Barry, ..., Krummel Nature Medicine 2018 [PMID 29942093]

    This is the melanoma met RNA-seq dataset.

    """

    df = pandas.read_csv(filenames[0], sep="\t", index_col=0)
    df = df[[]]  # no columns

    for filename in filenames:
        df[os.path.basename(filename)] = pandas.read_csv(
            filename, sep="\t", index_col=0)["TPM"]

    assert len(df.columns) == len(filenames)

    groups = {
        "sample_type:MELANOMA_MET": df.columns.tolist(),
    }
    return [make_expression_groups("GSE113126", df, groups)]


def handle_expression_expression_atlas_22460905(filename):
    df = pandas.read_csv(filename, sep="\t", skiprows=4, index_col=0)
    del df["Gene Name"]
    df.columns = df.columns.str.lower()
    df = df.fillna(0.0)

    def matches(*strings):
        return [c for c in df.columns if all(s in c for s in strings)]

    groups = {
        "sample_type:B-LCL": (
            matches("b-cell", "lymphoblast") + matches("b acute lymphoblastic")),
        "sample_type:B-CELL": matches("b-cell"),
        "sample_type:B721-LIKE": matches("b-cell"),
        "sample_type:MELANOMA_CELL_LINE": matches("melanoma"),
        "sample_type:MELANOMA": matches("melanoma"),
        "sample_type:A375-LIKE": matches("melanoma"),
        "sample_type:KG1-LIKE": matches("myeloid leukemia"),

        # Using a fibrosarcoma cell line for our fibroblast sample.
        "sample_type:FIBROBLAST": ['fibrosarcoma, ht-1080'],

        # For GBM tissue we are just using a mixture of cell lines.
        "sample_type:GLIOBLASTOMA_TISSUE": matches("glioblastoma"),

        "cell_line:THP-1": ["childhood acute monocytic leukemia, thp-1"],
        "cell_line:HL-60": ["adult acute myeloid leukemia, hl-60"],
        "cell_line:U-87": ['glioblastoma, u-87 mg'],
        "cell_line:LNT-229": ['glioblastoma, ln-229'],
        "cell_line:T98G": ['glioblastoma, t98g'],
        "cell_line:SK-MEL-5": ['cutaneous melanoma, sk-mel-5'],
        'cell_line:MEWO': ['melanoma, mewo'],
        "cell_line:HCC1937": ['breast ductal adenocarcinoma, hcc1937'],
        "cell_line:HCT116": ['colon carcinoma, hct 116'],
        "cell_line:HCC1143": ['breast ductal adenocarcinoma, hcc1143'],
    }
    return [make_expression_groups("expression_atlas_22460905", df, groups)]


def handle_expression_human_protein_atlas(*filenames):
    (cell_line_filename,) = [f for f in filenames if "celline" in f]
    (blood_filename,) = [f for f in filenames if "blood" in f]
    (gtex_filename,) = [f for f in filenames if "gtex" in f]

    cell_line_df = pandas.read_csv(cell_line_filename, sep="\t")
    blood_df = pandas.read_csv(blood_filename, sep="\t", index_col=0)
    gtex_df = pandas.read_csv(gtex_filename, sep="\t")

    cell_line_df = cell_line_df.pivot(
        index="Gene", columns="Cell line", values="TPM")

    gtex_df = gtex_df.pivot(
        index="Gene", columns="Tissue", values="TPM")

    return [
        make_expression_groups(
            "human_protein_atlas:%s" % os.path.basename(blood_filename),
            blood_df,
            groups={
                "sample_type:PBMC": [
                    c for c in blood_df.columns if "total PBMC" in c
                ],

                # for samples labeled leukapheresis we also use PBMC
                "sample_type:LEUKAPHERESIS": [
                    c for c in blood_df.columns if "total PBMC" in c
                ],

                # for samples labeled TIL we are also using PBMC
                "sample_type:TIL": [
                    c for c in blood_df.columns if "total PBMC" in c
                ],
            }),
        make_expression_groups(
            "human_protein_atlas:%s" % os.path.basename(cell_line_filename),
            cell_line_df,
            groups={
                "cell_line:HELA": ['HeLa'],
                "cell_line:K562": ["K-562"],
                "cell_line:HEK293": ['HEK 293'],
                "cell_line:RPMI8226": ['RPMI-8226'],
                "cell_line:EXPI293": ['HEK 293'],  # EXPI293 derived from HEK293
            }),
        make_expression_groups(
            "human_protein_atlas:%s" % os.path.basename(gtex_filename),
            gtex_df,
            groups={
                "sample_type:LUNG": ["lung"],
                "sample_type:SPLEEN": ["spleen"],
                "sample_type:OVARY": ["ovary"],
                "sample_type:KIDNEY": ["kidney"],
            }),
    ]


def make_expression_mixtures(expression_df):
    global CELL_LINE_MIXTURES
    groups = {}
    for mix in CELL_LINE_MIXTURES:
        components = []
        for item in mix.replace("mix:", "").upper().split(","):
            if "cell_line:%s" % item in expression_df.columns:
                components.append("cell_line:%s" % item)
            else:
                print("No cell line, falling back on similar: ", item)
                components.append("sample_type:%s-LIKE" % item)
        groups["sample_type:" + mix.upper()] = components
    missing = set()
    for some in groups.values():
        for item in some:
            if item not in expression_df.columns:
                missing.add(item)
    if missing:
        raise ValueError(
            "Missing [%d]: %s. Available: %s" % (
                len(missing), missing, expression_df.columns.tolist()))
    return make_expression_groups("mixtures", expression_df, groups)


# Add all functions with names like handle_pmid_XXXX to PMID_HANDLERS dict.
for (key, value) in list(locals().items()):
    if key.startswith("handle_pmid_"):
        PMID_HANDLERS[key.replace("handle_pmid_", "")] = value
    elif key.startswith("handle_expression_"):
        EXPRESSION_HANDLERS[key.replace("handle_expression_", "")] = value


def run():
    args = parser.parse_args(sys.argv[1:])

    expression_dfs = []
    for (i, item_tpl) in enumerate(args.expression_item):
        (label, filenames) = (item_tpl[0], item_tpl[1:])
        label = label.replace("-", "_")
        print(
            "Processing expression item %d of %d" % (i + 1, len(args.expression_item)),
            label,
            *[os.path.abspath(f) for f in filenames])

        expression_dfs_for_item = []
        handler = None
        if label in EXPRESSION_HANDLERS:
            handler = EXPRESSION_HANDLERS[label]
            expression_dfs_for_item = handler(*filenames)
        elif args.debug:
            debug(*filenames)
        else:
            raise NotImplementedError(label)

        if expression_dfs_for_item:
            print(
                "Processed expression data",
                label,
                "result dataframes",
                len(expression_dfs_for_item))
            print(*[e.columns for e in expression_dfs_for_item])
            expression_dfs.extend(expression_dfs_for_item)

    expression_df = expression_dfs[0]
    for other in expression_dfs[1:]:
        expression_df = pandas.merge(
            expression_df, other, how='outer', left_index=True, right_index=True)

    print("Genes in each expression dataframe: ",
        *[len(e) for e in expression_dfs])
    print("Genes in merged expression dataframe", len(expression_df))

    if CELL_LINE_MIXTURES:
        print("Generating cell line mixtures.")
        expression_mixture_df = make_expression_mixtures(expression_df)
        expression_df = pandas.merge(
            expression_df,
            expression_mixture_df,
            how='outer',
            left_index=True,
            right_index=True)

    ms_dfs = []
    for (i, item_tpl) in enumerate(args.ms_item):
        (pmid, filenames) = (item_tpl[0], item_tpl[1:])
        print(
            "Processing MS item %d of %d" % (i + 1, len(args.ms_item)),
            pmid,
            *[os.path.abspath(f) for f in filenames])

        ms_df = None
        handler = None
        if pmid in PMID_HANDLERS:
            handler = PMID_HANDLERS[pmid]
            ms_df = handler(*filenames)
        elif args.debug:
            debug(*filenames)
        else:
            raise NotImplementedError(pmid)

        if ms_df is not None:
            ms_df["pmid"] = pmid
            if "original_pmid" not in ms_df.columns:
                ms_df["original_pmid"] = pmid
            if "expression_dataset" not in ms_df.columns:
                ms_df["expression_dataset"] = ""
            ms_df = ms_df.applymap(str).applymap(str.upper)
            ms_df["sample_id"] = ms_df.sample_id.str.replace(" ", "")
            print("*** PMID %s: %d peptides ***" % (pmid, len(ms_df)))
            if handler is not None:
                print(handler.__doc__)
            print("Counts by sample id:")
            print(ms_df.groupby("sample_id").peptide.nunique())
            print("")
            print("Counts by sample type:")
            print(ms_df.groupby("sample_type").peptide.nunique())
            print("****************************")

            for value in ms_df.expression_dataset.unique():
                if value and value not in expression_df.columns:
                    raise ValueError("No such expression dataset", value)

            ms_dfs.append(ms_df)

    ms_df = pandas.concat(ms_dfs, ignore_index=True, sort=False)
    ms_df["cell_line"] = ms_df["cell_line"].fillna("")
    ms_df["hla"] = ms_df["hla"].str.strip().str.replace(r'\s+', ' ')

    sample_table = ms_df[
        ["sample_id", "pmid", "expression_dataset", "cell_line", "sample_type"]
    ].drop_duplicates().set_index("sample_id")

    sample_id_to_expression_dataset = sample_table.expression_dataset.to_dict()
    for (sample_id, value) in sorted(sample_id_to_expression_dataset.items()):
        if value:
            print("Expression dataset for sample", sample_id, "already assigned")
            continue
        cell_line_col = "cell_line:" + sample_table.loc[sample_id, "cell_line"]
        sample_type_col = "sample_type:" + (
            sample_table.loc[sample_id, "sample_type"])

        expression_dataset = None
        for col in [cell_line_col, sample_type_col]:
            if col in expression_df.columns:
                expression_dataset = col
                break

        if not expression_dataset:
            print("*" * 20)
            print("No expression dataset for sample ", sample_id)
            print("Sample info:")
            print(sample_table.loc[sample_id])
            print("*" * 20)

        sample_id_to_expression_dataset[sample_id] = expression_dataset
        print(
            "Sample", sample_id, "assigned exp. dataset", expression_dataset)

    print("Expression dataset usage:")
    print(pandas.Series(sample_id_to_expression_dataset).value_counts())

    missing = [
        key for (key, value) in
        sample_id_to_expression_dataset.items()
        if value is None
    ]
    if missing:
        print("Missing expression data for samples", *missing)
        print(
            "Missing cell lines: ",
            *sample_table.loc[missing, "cell_line"].dropna().drop_duplicates().tolist())
        print("Missing sample types: ", *sample_table.loc[
            missing, "sample_type"].dropna().drop_duplicates().tolist())
        if args.debug:
            import ipdb; ipdb.set_trace()
        else:
            raise ValueError("Missing expression data for samples: ", missing)

    ms_df["expression_dataset"] = ms_df.sample_id.map(
        sample_id_to_expression_dataset)

    cols = [
        "pmid",
        "sample_id",
        "peptide",
        "format",
        "mhc_class",
        "hla",
        "expression_dataset",
    ]
    cols += [c for c in sorted(ms_df.columns) if c not in cols]
    ms_df = ms_df[cols]

    null_df = ms_df.loc[ms_df.isnull().any(1)]
    if len(null_df) > 0:
        print("Nulls:")
        print(null_df)
    else:
        print("No nulls.")

    # Each sample should be coming from only one experiment.
    assert ms_df.groupby("sample_id").pmid.nunique().max() == 1, (
        ms_df.groupby("sample_id").pmid.nunique().sort_values())

    expression_df.to_csv(args.expression_out, index=True)
    print("Wrote: %s" % os.path.abspath(args.expression_out))

    ms_df.to_csv(args.ms_out, index=False)
    print("Wrote: %s" % os.path.abspath(args.ms_out))

    if args.expression_metadata_out is not None:
        expression_metadata_df = pandas.DataFrame(
            EXPRESSION_GROUPS_ROWS,
            columns=["expression_dataset", "label", "samples"])
        expression_metadata_df["samples"] = expression_metadata_df[
            "samples"
        ].map(json.dumps)
        expression_metadata_df.to_csv(args.expression_metadata_out, index=False)
        print("Wrote: %s" % os.path.abspath(args.expression_metadata_out))

if __name__ == '__main__':
    run()

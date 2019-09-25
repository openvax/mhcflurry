"""
Filter and combine various peptide/MHC datasets to derive a composite training set,
optionally including eluted peptides identified by mass-spec.
"""
import sys
import argparse
import os
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
    result_df["sample_type"] = "B-lymphoblastoid"
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
    result_df["cell_line"] = ""
    result_df["pulldown_antibody"] = "W6/32"
    result_df["mhc_class"] = "I"
    result_df["format"] = "multiallelic"

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
    result_df["hla"] = result_df.sample_id.map(allele_map)
    print("Entries before dropping samples with unknown alleles", len(result_df))
    result_df = result_df.loc[~result_df.hla.isnull()]
    print("Entries after dropping samples with unknown alleles", len(result_df))
    result_df["sample_type"] = result_df.sample_id.map(sample_type)
    print(result_df.head(3))
    return result_df


def handle_pmid_26992070(*filenames):
    """Ritz, ..., Fugmann Proteomics 2016 [PMID 26992070]"""
    allele_text = """
        Cell line	HLA-A 1	HLA-A 2	HLA-B 1	HLA-B 2	HLA-C 1	HLA-C 2
        HEK293	03:01	03:01	07:02	07:02	07:02	07:02
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
            ] = "HLA-" + gene + allele_info["HLA-%s %s" % (gene, num)]
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
        "MAVER-1": "b-lymphoblast",
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
    PD42	B cell	28832583	HLA-A*02:06	HLA-A*24:02	HLA-B*07:02	HLA-B*55:01	HLA-C*01:02	HLA-C*07:02
    RA957	B cell	28832583	HLA-A*02:20	HLA-A*68:01	HLA-B*35:03	HLA-B*39:01	HLA-C*04:01	HLA-C*07:02
    TIL1	TIL	28832583	HLA-A*02:01	HLA-A*02:01	HLA-B*18:01	HLA-B*38:01	HLA-C*05:01	
    TIL3	TIL	28832583	HLA-A*01:01	HLA-A*23:01	HLA-B*07:02	HLA-B*15:01	HLA-C*12:03	HLA-C*14:02
    Apher1	Leukapheresis	28832583	HLA-A*03:01	HLA-A*29:02	HLA-B*44:02	HLA-B*44:03	HLA-C*12:03	HLA-C*16:01
    Apher6	Leukapheresis	28832583	HLA-A*02:01	HLA-A*03:01	HLA-B*07:02		HLA-C*07:02	
    pat_AC2	B lymphoblast	27841757	HLA-A*03:01	HLA-A*32:01	HLA-B*27:05	HLA-B*45:01		
    pat_C	B lymphoblast	27841757	HLA-A*02:01	HLA-A*03:01	HLA-B*07:02		HLA-C*07:02	
    pat_CELG	B lymphoblast	27841757	HLA-A*02:01	HLA-A*24:02	HLA-B*15:01	HLA-B*73:01	HLA-C*03:03	HLA-C*15:05
    pat_CP2	B lymphoblast	27841757	HLA-A*11:01		HLA-B*14:02	HLA-B*44:02		
    pat_FL	B lymphoblast	27841757	HLA-A*03:01	HLA-A*11:01	HLA-B*44:03	HLA-B*50:01		
    pat_J	B lymphoblast	27841757	HLA-A*02:01	HLA-A*03:01	HLA-B*07:02		HLA-C*07:02	
    pat_JPB3	B lymphoblast	27841757	HLA-A*02:01	HLA-A*11:01	HLA-B*27:05	HLA-B*56:01		
    pat_JT2	B lymphoblast	27841757	HLA-A*11:01		HLA-B*18:03	HLA-B*35:01		
    pat_M	B lymphoblast	27841757	HLA-A*03:01	HLA-A*29:02	HLA-B*08:01	HLA-B*44:03	HLA-C*07:01	HLA-C*16:01
    pat_MA	B lymphoblast	27841757	HLA-A*02:01	HLA-A*29:02	HLA-B*44:03	HLA-B*57:01	HLA-C*07:01	HLA-C*16:01
    pat_ML	B lymphoblast	27841757	HLA-A*02:01	HLA-A*11:01	HLA-B*40:01	HLA-B*44:03		
    pat_NS2	B lymphoblast	27841757	HLA-A*02:01		HLA-B*13:02	HLA-B*41:01		
    pat_NT	B lymphoblast	27841757	HLA-A*01:01	HLA-A*32:01	HLA-B*08:01			
    pat_PF1	B lymphoblast	27841757	HLA-A*01:01	HLA-A*02:01	HLA-B*07:02	HLA-B*44:03	HLA-C*07:02	HLA-C*16:01
    pat_R	B lymphoblast	27841757	HLA-A*03:01	HLA-A*29:02	HLA-B*08:01	HLA-B*44:03	HLA-C*07:01	HLA-C*16:01
    pat_RT	B lymphoblast	27841757	HLA-A*01:01	HLA-A*02:01	HLA-B*18:01	HLA-B*39:24	HLA-C*05:01	HLA-C*07:01
    pat_SR	B lymphoblast	27841757	HLA-A*02:01	HLA-A*23:01	HLA-B*18:01	HLA-B*44:03		
    pat_ST	B lymphoblast	27841757	HLA-A*03:01	HLA-A*24:02	HLA-B*07:02	HLA-B*27:05
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
        "MAPTAC_B*45:01": "",
        "MAPTAC_B*52:01": "",
        "MAPTAC_C*03:03": "",
        "MAPTAC_C*06:02": "",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm+": "expi293",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm-": "expi293",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm+": "expi293",  # don't actually see this in DataS1A!
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm-": "expi293",
        "MAPTAC_DRB1*01:01": "",
        "MAPTAC_DRB1*03:01": "",
        "MAPTAC_DRB1*04:01": "",
        "MAPTAC_DRB1*07:01": "",
        "MAPTAC_DRB1*11:01": "",
        "MAPTAC_DRB1*12:01_dm+": "",
        "MAPTAC_DRB1*12:01_dm-": "",
        "MAPTAC_DRB1*15:01": "",
        "MAPTAC_DRB3*01:01_dm+": "",
        "MAPTAC_DRB3*01:01_dm-": "",
    }
    sample_type = {
        "HLA-DR_Lung": "lung",
        "HLA-DR_PBMC_HDSC": "lung",
        "HLA-DR_PBMC_RG1095": "lung",
        "HLA-DR_PBMC_RG1104": "lung",
        "HLA-DR_PBMC_RG1248": "lung",
        "HLA-DR_Spleen": "spleen",
        "MAPTAC_A*02:01": "mixed",
        "MAPTAC_A*11:01": "mixed",
        "MAPTAC_A*32:01": "mixed",
        "MAPTAC_B*07:02": "mixed",
        "MAPTAC_B*45:01": "mixed",
        "MAPTAC_B*52:01": "mixed",
        "MAPTAC_C*03:03": "mixed",
        "MAPTAC_C*06:02": "mixed",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm+": "mixed",
        "MAPTAC_DPB1*06:01/DPA1*01:03_dm-": "mixed",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm+": "mixed",
        "MAPTAC_DQB1*06:04/DQA1*01:02_dm-": "mixed",
        "MAPTAC_DRB1*01:01": "mixed",
        "MAPTAC_DRB1*03:01": "mixed",
        "MAPTAC_DRB1*04:01": "mixed",
        "MAPTAC_DRB1*07:01": "mixed",
        "MAPTAC_DRB1*11:01": "mixed",
        "MAPTAC_DRB1*12:01_dm+": "mixed",
        "MAPTAC_DRB1*12:01_dm-": "mixed",
        "MAPTAC_DRB1*15:01": "mixed",
        "MAPTAC_DRB3*01:01_dm+": "mixed",
        "MAPTAC_DRB3*01:01_dm-": "mixed",
    }

    df = pandas.read_excel(filename, sheetname="DataS1B")
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
        result_df["sample_type"] = sample_type[sample_id]
        result_df["cell_line"] = cell_line[sample_id]
        results.append(result_df)
    result_df = pandas.concat(results, ignore_index=True)
    return result_df


# Add all functions with names like handle_pmid_XXXX to HANDLERS dict.
for (key, value) in list(locals().items()):
    if key.startswith("handle_pmid_"):
        HANDLERS[key.replace("handle_pmid_", "")] = value


def run():
    args = parser.parse_args(sys.argv[1:])

    dfs = []
    for (i, item_tpl) in enumerate(args.item):
        (pmid, filenames) = (item_tpl[0], item_tpl[1:])
        print(
            "Processing item %d / %d" % (i + 1, len(args.item)),
            pmid,
            *[os.path.abspath(f) for f in filenames])

        df = None
        handler = None
        if pmid in HANDLERS:
            handler = HANDLERS[pmid]
            df = handler(*filenames)
        elif args.debug:
            debug(*filenames)
        else:
            raise NotImplementedError(args.pmid)

        if df is not None:
            df["pmid"] = pmid
            if "original_pmid" not in df.columns:
                df["original_pmid"] = pmid
            df = df.applymap(str).applymap(str.upper)
            print("*** PMID %s: %d peptides ***" % (pmid, len(df)))
            if handler is not None:
                print(handler.__doc__)
            print("Counts by sample id:")
            print(df.groupby("sample_id").peptide.nunique())
            print("")
            print("Counts by sample type:")
            print(df.groupby("sample_type").peptide.nunique())
            print("****************************")

            dfs.append(df)

    df = pandas.concat(dfs, ignore_index=True, sort=False)

    df["cell_line"] = df["cell_line"].fillna("")

    cols = ["pmid", "sample_id", "peptide", "format", "mhc_class", "hla", ]
    cols += [c for c in sorted(df.columns) if c not in cols]
    df = df[cols]

    null_df = df.loc[df.isnull().any(1)]
    if len(null_df) > 0:
        print("Nulls:")
        print(null_df)
    else:
        print("No nulls.")

    df.to_csv(args.out, index=False)
    print("Wrote: %s" % os.path.abspath(args.out))

if __name__ == '__main__':
    run()

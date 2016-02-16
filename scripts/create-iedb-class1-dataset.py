#!/usr/bin/env python

"""
Turn a raw CSV snapshot of the IEDB contents into a usable
class I binding prediction dataset by grouping all unique pMHCs
"""
from collections import defaultdict
from os.path import join
import pickle

import numpy as np
import pandas as pd

from mhcflurry.paths import CLASS1_DATA_DIRECTORY

IEDB_SOURCE_FILENAME = "mhc_ligand_full.csv"
IEDB_SOURCE_PATH = join(CLASS1_DATA_DIRECTORY, IEDB_SOURCE_FILENAME)

OUTPUT_FILENAME = "iedb_human_class1_assay_datasets.pickle"
OUTPUT_PATH = join(CLASS1_DATA_DIRECTORY, OUTPUT_FILENAME)

if __name__ == "__main__":
    df = pd.read_csv(
        IEDB_SOURCE_PATH,
        error_bad_lines=False,
        encoding="latin-1",
        header=[0, 1])
    alleles = df["MHC"]["Allele Name"]
    n = len(alleles)
    print("# total: %d" % n)

    mask = np.zeros(n, dtype=bool)
    patterns = [
        "HLA-A",
        "HLA-B",
        "HLA-C",
        # "H-2-D",
        # "H-2-K",
        # "H-2-L",
    ]
    for pattern in patterns:
        pattern_mask = alleles.str.startswith(pattern)
        print("# %s: %d" % (pattern, pattern_mask.sum()))
        mask |= pattern_mask
    df = df[mask]
    print("# entries matching allele masks: %d" % (len(df)))
    assay_group = df["Assay"]["Assay Group"]
    assay_method = df["Assay"]["Method/Technique"]
    groups = df.groupby([assay_group, assay_method])
    print("---")
    print("Assays")
    assay_dataframes = {}
    # create a dataframe for every distinct kind of assay which is used
    # by IEDB submitters to measure peptide-MHC affinity or stability
    for (assay_group, assay_method), group_data in sorted(
            groups, key=lambda x: len(x[1]), reverse=True):
        print("%s (%s): %d" % (assay_group, assay_method, len(group_data)))
        group_alleles = group_data["MHC"]["Allele Name"]
        group_peptides = group_data["Epitope"]["Description"]
        distinct_pmhc = group_data.groupby([group_alleles, group_peptides])
        columns = defaultdict(list)
        for (allele, peptide), pmhc_group in distinct_pmhc:
            columns["mhc"].append(allele)
            columns["peptide"].append(peptide)
            # performing median in log space since in two datapoint case
            # we don't want to take e.g. (10 + 1000) / 2.0 = 505
            # but would prefer something like 10 ** ( (1 + 3) / 2.0) = 100
            columns["value"].append(
                np.exp(
                    np.median(
                        np.log(
                            pmhc_group["Assay"]["Quantitative measurement"]))))
            qualitative = pmhc_group["Assay"]["Qualitative Measure"]
            columns["percent_positive"].append(
                qualitative.str.startswith("Positive").mean())
            columns["count"].append(
                pmhc_group["Assay"]["Quantitative measurement"].count())
        assay_dataframes[(assay_group, assay_method)] = pd.DataFrame(
            columns,
            columns=[
                "mhc",
                "peptide",
                "value",
                "percent_positive",
                "count"])
        print("# distinct pMHC entries: %d" % len(columns["mhc"]))
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(assay_dataframes, f, pickle.HIGHEST_PROTOCOL)

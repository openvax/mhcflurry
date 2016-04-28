#!/usr/bin/env python

# Copyright (c) 2016. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Turn a raw CSV snapshot of the IEDB contents into a usable
class I binding prediction dataset by grouping all unique pMHCs
"""
from collections import defaultdict
from os import makedirs
from os.path import join, exists
import pickle
import argparse

import numpy as np
import pandas as pd

from mhcflurry.paths import CLASS1_DATA_DIRECTORY


IEDB_SOURCE_FILENAME = "mhc_ligand_full.csv"
PICKLE_OUTPUT_FILENAME = "iedb_human_class1_assay_datasets.pickle"

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--input-csv",
    default=IEDB_SOURCE_FILENAME,
    help="CSV file with IEDB's MHC binding data. Default: %(default)s")

parser.add_argument(
    "--output-dir",
    default=CLASS1_DATA_DIRECTORY,
    help="Directory to write output pickle file. Default: %(default)s")

parser.add_argument(
    "--output-pickle-filename",
    default=PICKLE_OUTPUT_FILENAME,
    help="Path to .pickle file containing dictionary of IEDB assay datasets. "
    "Default: %(default)s")

parser.add_argument(
    "--alleles",
    metavar="ALLELE",
    nargs="+",
    default=[],
    help="Restrict dataset to specified alleles")

def filter_class1_alleles(df):
    mhc_class = df["MHC"]["MHC allele class"]
    print("MHC class counts: \n%s" % (mhc_class.value_counts(),))
    class1_mask = mhc_class == "I"
    return df[class1_mask]

def filter_allele_names(df):
    alleles = df["MHC"]["Allele Name"]
    invalid_allele_mask = alleles.str.contains(" ") | alleles.str.contains("/")
    invalid_alleles = alleles[invalid_allele_mask]
    print("-- Invalid allele names: %s" % (list(sorted(set(invalid_alleles)))))
    print("Dropping %d with complex alleles (e.g. descriptions of mutations)" % (
        len(invalid_alleles),))
    return df[~invalid_allele_mask]

def filter_affinity_values(df):
    affinities = df["Assay"]["Quantitative measurement"]
    finite_affinity_mask = ~affinities.isnull() & np.isfinite(affinities)
    invalid_affinity_mask = ~finite_affinity_mask

    print("Dropping %d rows without finite affinity measurements" % (
        invalid_affinity_mask.sum(),))
    return df[finite_affinity_mask]

def filter_mhc_dataframe(df):
    filter_functions = [
        filter_class1_alleles,
        filter_allele_names,
        filter_affinity_values,
    ]

    for fn in filter_functions:
        df = fn(df)

    return df


def groupby_assay(df):
    assay_group = df["Assay"]["Assay Group"]
    assay_method = df["Assay"]["Method/Technique"]
    groups = df.groupby([assay_group, assay_method])

    # speed up repeated calls to np.log by caching log affinities as a column
    # in the dataframe
    df["_log_affinity"] = np.log(df["Assay"]["Quantitative measurement"])

    # speed up computing percent positive with the helper column
    qualitative = df["Assay"]["Qualitative Measure"]
    df["_qualitative_positive"] = qualitative.str.startswith("Positive")
    print("---")
    print("Assays")
    assay_dataframes = {}
    # create a dataframe for every distinct kind of assay which is used
    # by IEDB submitters to measure peptide-MHC affinity or stability
    for (assay_group, assay_method), group_data in sorted(
            groups,
            key=lambda x: len(x[1]),
            reverse=True):
        print("- %s (%s): %d" % (assay_group, assay_method, len(group_data)))
        group_alleles = group_data["MHC"]["Allele Name"]
        group_peptides = group_data["Epitope"]["Description"]
        distinct_pmhc = group_data.groupby([group_alleles, group_peptides])
        columns = defaultdict(list)
        for (allele, peptide), pmhc_group in distinct_pmhc:
            columns["mhc"].append(allele)
            columns["peptide"].append(peptide)
            positive = pmhc_group["_qualitative_positive"]
            count = len(pmhc_group)
            if count == 1:
                ic50 = pmhc_group["Assay"]["Quantitative measurement"].mean()
            else:
                ic50 = np.exp(np.mean(pmhc_group["_log_affinity"]))
            # averaging the log affinities preserves orders of magnitude better
            columns["value"].append(ic50)
            columns["percent_positive"].append(positive.mean())
            columns["count"].append(count)
        assay_dataframes[(assay_group, assay_method)] = pd.DataFrame(
            columns,
            columns=[
                "mhc",
                "peptide",
                "value",
                "percent_positive",
                "count"])
        print("# distinct pMHC entries: %d" % len(columns["mhc"]))
    return assay_dataframes

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(
        args.input_csv,
        error_bad_lines=False,
        encoding="latin-1",
        header=[0, 1])

    df = filter_mhc_dataframe(df)

    alleles = df["MHC"]["Allele Name"]

    n = len(alleles)

    print("# Class I rows: %d" % n)
    print("# Class I alleles: %d" % len(set(alleles)))
    print("Unique alleles: %s" % list(sorted(set(alleles))))

    if args.alleles:
        print("User-supplied allele whitelist: %s" % (args.alleles,))
        mask = np.zeros(n, dtype=bool)
        for pattern in args.alleles:
            pattern_mask = alleles.str.startswith(pattern)
            print("# %s: %d" % (pattern, pattern_mask.sum()))
            mask |= pattern_mask
        df = df[mask]
        print("# entries matching alleles %s: %d" % (
            args.alleles,
            len(df)))

    assay_dataframes = groupby_assay(df)

    if not exists(args.output_dir):
        makedirs(args.output_dir)

    output_path = join(args.output_dir, args.output_pickle_filename)

    with open(output_path, "wb") as f:
        pickle.dump(assay_dataframes, f, pickle.HIGHEST_PROTOCOL)

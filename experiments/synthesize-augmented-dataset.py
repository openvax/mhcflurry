#!/usr/bin/env python
#
# Copyright (c) 2015. Mount Sinai School of Medicine
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

from collections import OrderedDict
import argparse

import mhcflurry
import pandas as pd

from dataset_paths import PETERS2009_CSV_PATH
from synthetic_data import synthesize_affinities_for_all_alleles, load_sims_dict

parser = argparse.ArgumentParser()

parser.add_argument(
    "--binding-data-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--output-csv",
    help="Path for CSV containing synthesized dataset of affinities",
    required=True)

parser.add_argument(
    "--max-ic50",
    default=50000.0,
    type=float)

parser.add_argument(
    "--only-human",
    default=False,
    action="store_true")

parser.add_argument(
    "--allele-similarity-csv",
    required=True)

parser.add_argument(
    "--smoothing-coef",
    default=0.01,
    type=float,
    help="Smoothing value used for peptides with low weight across alleles")

parser.add_argument(
    "--similarity-exponent",
    default=2.0,
    type=float,
    help="Affinities are synthesized by adding up y_ip * sim(i,j) ** exponent")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    sims_dict = load_sims_dict(args.allele_similarity_csv)
    print("Loaded similarities between %d allele pairs (%d unique)" % (
        len(sims_dict), len(set(a for (a, _) in sims_dict.keys()))))

    allele_datasets = mhcflurry.data.load_allele_datasets(
        args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human)
    print("Loaded binding data for %d alleles" % len(allele_datasets))

    synthetic_data = synthesize_affinities_for_all_alleles(
        allele_datasets=allele_datasets,
        pairwise_allele_similarities=sims_dict,
        smoothing=args.smoothing_coef,
        exponent=args.similarity_exponent)

    synthetic_data_dict = OrderedDict([
        ("mhc", []),
        ("sequence", []),
        ("ic50", []),
    ])

    for allele, allele_entries in synthetic_data.items():
        print(allele, len(allele_entries))
        for (peptide, regression_value) in allele_entries.items():
            synthetic_data_dict["mhc"].append(allele)
            synthetic_data_dict["sequence"].append(peptide)
            ic50 = args.max_ic50 ** (1.0 - regression_value)
            synthetic_data_dict["ic50"].append(ic50)

    synthetic_data_df = pd.DataFrame(synthetic_data_dict)

    print("Created dataset with %d synthesized pMHC affinities" % (
        len(synthetic_data_df),))

    synthetic_data_df.to_csv(args.output_csv, index=False)

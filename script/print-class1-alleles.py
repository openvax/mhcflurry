#!/usr/bin/env python

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

"""
Print list of supported class I alleles for which
trained models are available
"""

import argparse
import os

import pandas as pd

from mhcflurry.paths import CLASS1_MODEL_DIRECTORY, CLASS1_DATA_CSV_PATH

parser = argparse.ArgumentParser()
parser.add_argument(
    "--with-peptide-lengths",
    default=False,
    action="store_true")

parser.add_argument(
    "--with-dataset-size",
    default=False,
    action="store_true")

parser.add_argument(
    "--all",
    default=False,
    action="store_true",
    help="Include serotypes (like 'A2') which include multiple 4-digit types")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.with_dataset_size:
        df = pd.read_csv(CLASS1_DATA_CSV_PATH)
        allele_sizes = {
            allele: len(group) for (allele, group) in df.groupby("mhc")
        }
    else:
        allele_sizes = None

    for filename in os.listdir(CLASS1_MODEL_DIRECTORY):
        allele = filename.replace(".hdf", "")
        if len(allele) >= 5:
            allele = "HLA-%s*%s:%s" % (allele[0], allele[1:3], allele[3:])
        elif args.all:
            allele = "HLA-%s" % allele
        else:
            # skipping serotype names like A2 or B7
            continue

        line = allele

        if args.with_peptide_lengths:
            line += "\t8,9,10,11,12"
        if args.with_dataset_size:
            line += "\t%d" % allele_sizes[allele]
        print(line)

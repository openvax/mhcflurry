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

from mhcflurry.paths import CLASS1_MODEL_DIRECTORY

parser = argparse.ArgumentParser()
parser.add_argument(
    "--with-peptide-lengths",
    default=False,
    action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    for filename in os.listdir(CLASS1_MODEL_DIRECTORY):
        allele = filename.replace(".hdf", "")
        if len(allele) < 5:
            # skipping serotype names like A2 or B7
            continue
        allele = "HLA-%s*%s:%s" % (allele[0], allele[1:3], allele[3:])
        if args.with_peptide_lengths:
            print("%s\t8,9,10,11,12" % allele)
        else:
            print(allele)
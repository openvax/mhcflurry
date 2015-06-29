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


import argparse

from mhcflurry.common import parse_int_list, split_peptide_sequences

parser = argparse.ArgumentParser()

parser.add_argument("--mhc",
    default="HLA-A*02:01",
    help="Comma separated list of MHC alleles")

parser.add_argument("--sequence",
    required=True,
    type=split_peptide_sequences,
    help="Comma separated list of protein sequences")

parser.add_argument("--fasta-file",
    help="FASTA file of protein sequences to chop up into peptides")

parser.add_argument("--peptide-lengths",
    default=[9],
    type=parse_int_list,
    help="Comma separated list of peptide length, e.g. 8,9,10,11")

if __name__ == "__main__":
    args = parser.parse_args()

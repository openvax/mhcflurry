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

from __future__ import (
    print_function,
    division,
    absolute_import,
)


def parse_int_list(s):
    return [int(part.strip() for part in s.split(","))]


def split_uppercase_sequences(s):
    return [part.strip().upper() for part in s.split(",")]


def normalize_allele_name(allele_name):
    """
    Only works for mouse, human, and rhesus monkey alleles.

    TODO: use the same logic as mhctools for MHC name parsing.
    Possibly even worth its own small repo called something like "mhcnames"
    """
    allele_name = allele_name.upper()
    if allele_name.startswith("MAMU"):
        prefix = "Mamu-"
    elif allele_name.startswith("H-2") or allele_name.startswith("H2"):
        prefix = "H-2-"
    else:
        prefix = ""
    # old school HLA-C serotypes look like "Cw"
    allele_name = allele_name.replace("CW", "C")
    patterns = [
        "HLA-",
        "H-2",
        "H2",
        "MAMU",
        "-",
        "*",
        ":"
    ]
    for pattern in patterns:
        allele_name = allele_name.replace(pattern, "")
    return "%s%s" % (prefix, allele_name)


def split_allele_names(s):
    return [
        normalize_allele_name(part.strip())
        for part
        in s.split(",")
    ]

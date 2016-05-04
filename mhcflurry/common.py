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

from __future__ import (
    print_function,
    division,
    absolute_import,
)

def parse_int_list(s):
    return [int(part.strip() for part in s.split(","))]


def split_uppercase_sequences(s):
    return [part.strip().upper() for part in s.split(",")]

MHC_PREFIXES = [
    "HLA",
    "H-2",
    "Mamu",
    "Patr",
    "Gogo",
    "ELA",
]


def normalize_allele_name(allele_name, default_prefix="HLA"):
    """
    Only works for a small number of species.

    TODO: use the same logic as mhctools for MHC name parsing.
    Possibly even worth its own small repo called something like "mhcnames"
    """
    allele_name = allele_name.upper()
    # old school HLA-C serotypes look like "Cw"
    allele_name = allele_name.replace("CW", "C")

    prefix = default_prefix
    for candidate in MHC_PREFIXES:
        if (allele_name.startswith(candidate.upper()) or
                allele_name.startswith(candidate.replace("-", "").upper())):
            prefix = candidate
            allele_name = allele_name[len(prefix):]
            break
    for pattern in MHC_PREFIXES + ["-", "*", ":"]:
        allele_name = allele_name.replace(pattern, "")
    return "%s%s" % (prefix + "-" if prefix else "", allele_name)


def split_allele_names(s):
    return [
        normalize_allele_name(part.strip())
        for part
        in s.split(",")
    ]

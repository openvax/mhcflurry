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

from __future__ import print_function, division, absolute_import
from math import exp, log
import itertools
from collections import defaultdict

import numpy as np

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


def geometric_mean(xs, weights=None):
    """
    Geometric mean of a collection of affinity measurements, with optional
    sample weights.
    """
    if len(xs) == 1:
        return xs[0]
    elif weights is None:
        return exp(sum(log(xi) for xi in xs) / len(xs))
    sum_weighted_log = sum(log(xi) * wi for (xi, wi) in zip(xs, weights))
    denom = sum(weights)
    return exp(sum_weighted_log / denom)


def all_combinations(**dict_of_lists):
    """
    Iterator that generates all combinations of parameters given in the
    kwargs dictionary which is expected to map argument names to lists
    of possible values.
    """
    arg_names = dict_of_lists.keys()
    value_lists = dict_of_lists.values()
    for combination_of_values in itertools.product(*value_lists):
        yield dict(zip(arg_names, combination_of_values))


def groupby_indices(iterable, key_fn=lambda x: x):
    """
    Returns diçtionary mapping unique values to list of indices that had
    those values.
    """
    index_groups = defaultdict(list)
    for i, x in enumerate(key_fn(x) for x in iterable):
        index_groups[x].append(i)
    return index_groups

def shuffle_split_list(indices, fraction=0.5):
    """
    Split a list of indices into two sub-lists, with an optional parameter
    controlling what fraction of the indices go to the left list.
    """
    indices = np.asarray(indices)
    np.random.shuffle(indices)
    n = len(indices)

    left_count = int(np.ceil(fraction * n))

    if n > 1 and left_count == 0:
        left_count = 1
    elif n > 1 and left_count == n:
        left_count = n - 1

    return indices[:left_count], indices[left_count:]

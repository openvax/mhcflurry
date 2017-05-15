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
import logging
import hashlib
import time
import sys
from os import environ

import numpy as np
import pandas


class UnsupportedAllele(Exception):
    pass


def parse_int_list(s):
    return [int(part.strip()) for part in s.split(",")]


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
    Returns dictionary mapping unique values to list of indices that
    had those values.
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


def dataframe_cryptographic_hash(df):
    """
    Return a cryptographic (i.e. collisions extremely unlikely) hash
    of a dataframe. Suitible for using as a cache key.

    Parameters
    -----------
    df : pandas.DataFrame or pandas.Series

    Returns
    -----------
    string
    """
    start = time.time()
    result = hashlib.sha1(df.to_msgpack()).hexdigest()
    logging.info(
        "Generated dataframe hash in %0.2f sec" % (time.time() - start))
    return result


def freeze_object(o):
    """
    Recursively convert nested dicts and lists into frozensets and tuples.
    """
    if isinstance(o, dict):
        return frozenset({k: freeze_object(v) for k, v in o.items()}.items())
    if isinstance(o, list):
        return tuple(freeze_object(v) for v in o)
    return o


def configure_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s:"
        " %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        level=level)


def describe_nulls(df, related_df_with_same_index_to_describe=None):
    """
    Return a string describing the positions of any nan or inf values
    in a dataframe.

    If related_df_with_same_index_to_describe is specified, it should be
    a dataframe with the same index as df. Positions corresponding to
    where df has null values will also be printed from this dataframe.
    """
    if isinstance(df, pandas.Series):
        df = df.to_frame()
    with pandas.option_context('mode.use_inf_as_null', True):
        null_counts_by_col = pandas.isnull(df).sum(axis=0)
        null_rows = pandas.isnull(df).sum(axis=1) > 0
        return (
            "Columns with nulls:\n%s, related rows with nulls:\n%s, "
            "full df:\n%s" % (
                null_counts_by_col.index[null_counts_by_col > 0],
                related_df_with_same_index_to_describe.ix[null_rows]
                if related_df_with_same_index_to_describe is not None
                else "(n/a)",
                str(df.ix[null_rows])))


def raise_or_debug(exception):
    """
    Raise the exception unless the MHCFLURRY_DEBUG environment variable is set,
    in which case drop into ipython debugger (ipdb).
    """
    if environ.get("MHCFLURRY_DEBUG"):
        import ipdb
        ipdb.set_trace()
    raise exception


def assert_no_null(df, message=''):
    """
    Raise an assertion error if the given DataFrame has any nan or inf values.
    """
    if hasattr(df, 'count'):
        with pandas.option_context('mode.use_inf_as_null', True):
            failed = df.count().sum() != df.size
    else:
        failed = np.isnan(df).sum() > 0
    if failed:
        raise_or_debug(
            AssertionError(
                "%s %s" % (message, describe_nulls(df))))


def drop_nulls_and_warn(df, related_df_with_same_index_to_describe=None):
    """
    Return a new DataFrame that is a copy of the given DataFrame where any
    rows with nulls have been removed, and a warning about them logged.
    """
    with pandas.option_context('mode.use_inf_as_null', True):
        new_df = df.dropna()
    if df.shape != new_df.shape:
        logging.warn(
            "Dropped rows with null or inf: %s -> %s:\n%s" % (
                df.shape,
                new_df.shape,
                describe_nulls(df, related_df_with_same_index_to_describe)))
    return new_df

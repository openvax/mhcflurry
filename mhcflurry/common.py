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
import itertools
import collections
import logging
import hashlib
import time
import sys
from os import environ

import numpy
import pandas

from . import amino_acid


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
        failed = numpy.isnan(df).sum() > 0
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


def amino_acid_distribution(peptides, smoothing=0.0):
    """
    Compute the fraction of each amino acid across a collection of peptides.
    
    Parameters
    ----------
    peptides : list of string
    smoothing : float, optional
        Small number (e.g. 0.01) to add to all amino acid fractions. The higher
        the number the more uniform the distribution.

    Returns
    -------
    pandas.Series indexed by amino acids
    """
    peptides = pandas.Series(peptides)
    aa_counts = pandas.Series(peptides.map(collections.Counter).sum())
    normalized = aa_counts / aa_counts.sum()
    if smoothing:
        normalized += smoothing
        normalized /= normalized.sum()
    return normalized


def random_peptides(num, length=9, distribution=None):
    """
    Generate random peptides (kmers).

    Parameters
    ----------
    num : int
        Number of peptides to return

    length : int
        Length of each peptide

    distribution : pandas.Series
        Maps 1-letter amino acid abbreviations to
        probabilities. If not specified a uniform
        distribution is used.

    Returns
    ----------
    list of string

    """
    if num == 0:
        return []
    if distribution is None:
        distribution = pandas.Series(
            1, index=sorted(amino_acid.COMMON_AMINO_ACIDS))
        distribution /= distribution.sum()

    return [
        ''.join(peptide_sequence)
        for peptide_sequence in
        numpy.random.choice(
            distribution.index,
            p=distribution.values,
            size=(int(num), int(length)))
    ]

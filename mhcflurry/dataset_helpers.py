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
from math import log, exp

import numpy as np

from .common import normalize_allele_name

def infer_csv_separator(filename):
    """
    Determine if file is separated by comma, tab, or whitespace.
    Default to whitespace if the others are not detected.

    Returns (sep, delim_whitespace)
    """
    for candidate in [",", "\t"]:
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                if candidate in line:
                    return candidate, False
    return None, True


def load_dataframe(
        filename,
        sep=None,
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name=None,
        peptide_length_column_name="peptide_length",
        ic50_column_name="meas"):
    """
    Load a dataframe of peptide-MHC affinity measurements

    filename : str
        CSV/TSV filename with columns:
            - 'species'
            - 'mhc'
            - 'sequence'
            - 'meas'

    sep : str, optional
        Separator in CSV file, default is to let Pandas infer

    peptide_column_name : str, optional
        Default behavior is to try  {"sequence", "peptide", "peptide_sequence"}

    Returns pair of DataFrame and name of column containing peptide sequences.
    """
    if sep is None:
        sep, delim_whitespace = infer_csv_separator(filename)
    else:
        delim_whitespace = False

    df = pd.read_csv(
        filename,
        sep=sep,
        delim_whitespace=delim_whitespace,
        engine="c")

    # hack: get around variability of column naming by checking if
    # the peptide_column_name is actually present and if not try "peptide"
    if peptide_column_name is None:
        columns = set(df.keys())
        for candidate in ["sequence", "peptide", "peptide_sequence"]:
            if candidate in columns:
                peptide_column_name = candidate
                break
        if peptide_column_name is None:
            raise ValueError(
                "Couldn't find peptide column name, candidates: %s" % (
                    columns))
    return df, peptide_column_name

def prepare_peptides_array(peptides):
    """
    Parameters
    ----------
    peptides : sequence of str

    Returns an array of strings
    """
    require_iterable_of(peptides, string_types)
    peptides = np.asarray(peptides)
    n_peptides = len(peptides)
    if len(peptides) == 0:
        raise ValueError("No peptides given")
    return np.asarray(peptides)

def prepare_ic50_values_array(ic50_values,  required_count=None):
    """
    Parameters
    ----------
    ic50_values : List, tuple or array float

    required_count : int, optional

    Checks that all IC50 values are valid and returns an array.
    """
    ic50_values = np.asarray(ic50_values)
    if len(ic50_values) == 0:
        raise ValueError("No IC50 values given")
    elif required_count is not None and len(ic50_values) != required_count:
        raise ValueError(
            "Number of IC50 values (%d) must match number of peptides (%d)" % (
                len(ic50_values),
                required_count))
    n_negative_values = (ic50_values < 0).sum()
    if n_negative_values > 0:
        raise ValueError(
            "Found %d invalid negative IC50 values" % n_negative_values)
    return ic50_values

def prepare_alleles_array(alleles, required_count=None):
    """
    Parameters
    ----------
    alleles : list of str

    required_count : int, optional

    Returns array of normalized MHC allele names.
    """
    require_iterable_of(alleles, string_types)
    if len(alleles) == 0:
        raise ValueError("No alleles given")
    elif required_count is not None and len(alleles) != required_count:
        raise ValueError(
            "Number of alleles (%d) must match number of peptides (%d)" % (
                len(alleles),
                required_count))
    return np.asarray(
        [normalize_allele_name(allele) for allele in alleles])

def transform_ic50_values_into_regression_targets(ic50_values, max_ic50):
    """
    Rescale IC50 values, assumed to be between (0, infinity)
    to the range [0,1] where 0 indicates non-binding and 1 indicates
    strong binding.

    Parameters
    ----------
    ic50_values : numpy.ndarray of float

    max_ic50 : float

    Returns numpy.ndarray of float
    """
    log_ic50 = np.log(ic50) / np.log(max_ic50)
    result = 1.0 - log_ic50
    # clamp to values between 0, 1
    return  np.minimum(1.0, np.maximum(0.0, result))

def transform_regression_targets_into_ic50_values(y, max_ic50):
    """
    Given a value between [0,1] transform it into a nM IC50 measurment in the
    range [0, max_ic50]
    """
    return max_ic50 ** (1 - y)

def combine_multiple_peptide_ic50_measurements(
        peptides,
        ic50_values,
        weights=None):
    """
    If a peptide's IC50 to a particular allele is measured multiple times
    then we want to reduce it to a single estimate by taking the
    geometric mean of the affinities.

    Parameters
    ----------
    peptides : list or array of string

    ic50_values : list or array of float

    weights : list or array of float, optional
        Sample weights associated with each measurment. If not given, then
        each sample gets a weight of 1.0
    """
    if weights is None:
        weights = np.ones(len(peptides), dtype=float)

    if len(peptides) != len(ic50_values):
        raise ValueError("Expected same number of peptides (%d) as IC50 values (%d)" % (
            len(peptides),
            len(ic50_values)))

    if len(peptides) != len(weights):
        raise ValueError("Expected same number of peptides (%d) as weights (%d)" % (
            len(peptides),
            len(weights)))

    # each peptide mapping to a list of log(IC50) paired with a weight
    measurements_with_weights = defaultdict(list)
    for (i, peptide) in enumerate(peptides):
        ic50 = ic50_values[i]
        weights = weights[i]
        measurements_with_weights.append((ic50, weight))

    result_peptides = []
    result_ic50_values = []
    for peptide, affinity_tuples in measurements_with_weights.items():
        if len(affinity_tuples) == 1:
            # special case when there's only one entry
            combined_ic50 = affinity_tuples[0][0]
        else:
            denom = 0.0
            total = 0.0
            for ic50, weight in affinity_tuples:
                total += log(log_ic50) * weight
                denom += weight

            if denom > 0:
                combined_log_ic50 = total / denom
                combined_ic50 = exp(log_ic50)
            else:
                # if none of the samples had non-zero weight then don't use them
                continue
        result_peptides.append(peptide)
        result_ic50_values.append(combined_ic50)
    n_combined = len(result_peptides)
    assert len(result_ic50_values) == n_combined
    peptides_array = prepare_peptides_array(result_peptides)
    ic50_values_array = prepare_ic50_values_array(
        ic50_values=result_ic50_values, required_count=n_combined)
    return peptides_array, ic50_values_array

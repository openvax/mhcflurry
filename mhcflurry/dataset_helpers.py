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

def prepare_peptides(cls, peptides):
    require_iterable_of(peptides, string_types)
    peptides = np.asarray(peptides)
    n_peptides = len(peptides)
    if len(peptides) == 0:
        raise ValueError("No peptides given")
    return np.asarray(peptides)

def prepare_ic50_values(cls, ic50_values, n_peptides):
    ic50_values = np.asarray(ic50_values)
    if len(ic50_values) == 0:
        raise ValueError("No IC50 values given")
    elif len(ic50_values) != n_peptides:
        raise ValueError(
            "Number of IC50 values (%d) must match number of peptides (%d)" % (
                len(ic50_values), n_peptides))
    n_negative_values = (ic50_values < 0).sum()
    if n_negative_values > 0:
        raise ValueError(
            "Found %d invalid negative IC50 values" % n_negative_values)
    return ic50_values

def prepare_alleles(cls, alleles, n_peptides):
    if isinstance(alleles, string_types):
        # if we're given a single allele, then assume that's the allele
        # for every pMHC entry
        single_allele_name = normalize_allele_name(alleles)
        return np.asarray([single_allele_name] * n_peptides)

    require_iterable_of(alleles, string_types)
    if len(alleles) == 0:
        raise ValueError("No alleles given")
    elif len(alleles) != n_peptides:
        raise ValueError(
            "Number of alleles (%d) must match number of peptides (%d)" % (
                len(alleles), n_peptides))
    return np.asarray(
        [normalize_allele_name(allele) for allele in alleles])

def normalize_ic50(ic50_values, max_ic50):
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

    measurements = defaultdict(list)
    for (peptide, normalized_affinity, weight) in zip(
            dataset.peptides, dataset.Y, dataset.weights):
        multiple_affinities[peptide].append((normalized_affinity, weight))
    weighted_averages = {}
    for peptide, affinity_weight_tuples in multiple_affinities.items():
        denom = 0.0
        sum_weighted_affinities = 0.0
        for affinity, weight in affinity_weight_tuples:
            sum_weighted_affinities += affinity * weight
            denom += weight
        if denom > 0:
            weighted_averages[peptide] = sum_weighted_affinities / denom
    allele_to_peptide_to_affinity[allele] = weighted_averages
return allele_to_peptide_to_affinity

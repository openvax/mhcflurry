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
from collections import namedtuple, defaultdict

import pandas as pd
import numpy as np

from .common import normalize_allele_name, ic50_to_regression_target
from .amino_acid import common_amino_acids
from .peptide_encoding import (
    indices_to_hotshot_encoding,
    fixed_length_index_encoding,
    check_valid_index_encoding_array,
)
from .class1_allele_specific_hyperparameters import MAX_IC50

index_encoding = common_amino_acids.index_encoding

AlleleData = namedtuple(
    "AlleleData",
    [
        "X_index",    # index-based featue encoding of fixed length peptides
        "X_binary",  # binary encoding of fixed length peptides
        "Y",     # regression encoding of IC50 (log scaled between 0..1)
        "peptides",  # list of fixed length peptide string
        "ic50",      # IC50 value associated with each entry
        "original_peptides",  # original peptides may be of different lengths
        "original_lengths",   # len(original_peptide)
        "substring_counts",   # how many substrings were extracted from
                              # each original peptide string
        "weights",    # 1.0 / count
        "max_ic50",   # maximum IC50 value used for encoding Y
    ])


def _infer_csv_separator(filename):
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
        max_ic50=MAX_IC50,
        sep=None,
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name=None,
        filter_peptide_length=None,
        ic50_column_name="meas",
        only_human=False):
    """
    Load a dataframe of peptide-MHC affinity measurements

    filename : str
        TSV filename with columns:
            - 'species'
            - 'mhc'
            - 'peptide_length'
            - 'sequence'
            - 'meas'

    max_ic50 : float
        Treat IC50 scores above this value as all equally bad
        (transform them to 0.0 in the regression output)

    sep : str, optional
        Separator in CSV file, default is to let Pandas infer

    peptide_column_name : str, optional
        Default behavior is to try  {"sequence", "peptide", "peptide_sequence"}

    filter_peptide_length : int, optional
        Which length peptides to use (default=load all lengths)

    only_human : bool
        Only load entries from human MHC alleles

    Returns DataFrame augmented with extra column:
        - "regression_output" : 1.0 - log(ic50)/log(max_ic50), limited to [0,1]
    """
    if sep is None:
        sep, delim_whitespace = _infer_csv_separator(filename)
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
    if only_human:
        human_mask = df[species_column_name] == "human"
        df = df[human_mask]

    if filter_peptide_length:
        length_mask = df[peptide_column_name].str.len() == filter_peptide_length
        df = df[length_mask]

    df[allele_column_name] = df[allele_column_name].map(normalize_allele_name)
    ic50 = np.array(df[ic50_column_name])
    df["regression_output"] = ic50_to_regression_target(ic50, max_ic50=max_ic50)
    return df, peptide_column_name


def load_allele_dicts(
        filename,
        max_ic50=MAX_IC50,
        regression_output=False,
        sep=None,
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name=None,
        ic50_column_name="meas",
        only_human=False,
        min_allele_size=1):
    """
    Parsing CSV of binding data into dictionary of dictionaries.
    The outer key is an allele name, the inner key is a peptide sequence,
    and the inner value is an IC50 or log-transformed value between [0,1]
    """
    binding_df, peptide_column_name = load_dataframe(
        filename=filename,
        max_ic50=max_ic50,
        sep=sep,
        species_column_name=species_column_name,
        allele_column_name=allele_column_name,
        peptide_column_name=peptide_column_name,
        ic50_column_name=ic50_column_name,
        only_human=only_human)
    # map peptides to either the raw IC50 or rescaled log IC50 depending
    # on the `regression_output` parameter
    output_column_name = (
        "regression_output"
        if regression_output
        else ic50_column_name
    )
    return {
        allele_name: {
            peptide: output
            for (peptide, output)
            in zip(group[peptide_column_name], group[output_column_name])
        }
        for (allele_name, group)
        in binding_df.groupby(allele_column_name)
        if len(group) >= min_allele_size
    }


def encode_peptide_to_affinity_dict(
        peptide_to_affinity_dict,
        peptide_length=9,
        flatten_binary_encoding=True,
        allow_unknown_amino_acids=True):
    """
    Given a dictionary mapping from peptide sequences to affinity values, return
    both index and binary encodings of fixed length peptides, and
    a vector of their affinities.

    Parameters
    ----------
    peptide_to_affinity_dict : dict
        Keys are peptide strings (of multiple lengths), each mapping to a
        continuous affinity value.

    peptide_length : int
        Length of vector encoding

    flatten_binary_encoding : bool
        Should the binary encoding of a peptide be two-dimensional (9x20)
        or a flattened 1d vector

    allow_unknown_amino_acids : bool
        When extending a short vector to the desired peptide length, should
        we insert every possible amino acid or a designated character "X"
        indicating an unknown amino acid.

    Returns tuple with the following fields:
        - kmer_peptides: fixed length peptide strings
        - original_peptides: variable length peptide strings
        - counts: how many fixed length peptides were made from this original
        - X_index: index encoding of fixed length peptides
        - X_binary: binary encoding of fixed length peptides
        - Y: affinity values associated with original peptides
    """
    raw_peptides = list(sorted(peptide_to_affinity_dict.keys()))
    X_index, kmer_peptides, original_peptides, counts = \
        fixed_length_index_encoding(
            peptides=raw_peptides,
            desired_length=peptide_length,
            start_offset_shorten=0,
            end_offset_shorten=0,
            start_offset_extend=0,
            end_offset_extend=0,
            allow_unknown_amino_acids=allow_unknown_amino_acids)

    n_samples = len(kmer_peptides)

    assert n_samples == len(original_peptides), \
        "Mismatch between # of samples (%d) and # of peptides (%d)" % (
            n_samples, len(original_peptides))
    assert n_samples == len(counts), \
        "Mismatch between # of samples (%d) and # of counts (%d)" % (
            n_samples, len(counts))
    assert n_samples == len(X_index), \
        "Mismatch between # of sample (%d) and index feature vectors (%d)" % (
            n_samples, len(X_index))
    X_index = check_valid_index_encoding_array(X_index, allow_unknown_amino_acids)
    n_indices = 20 + allow_unknown_amino_acids
    X_binary = indices_to_hotshot_encoding(
        X_index,
        n_indices=n_indices)

    assert X_binary.shape[0] == X_index.shape[0], \
        ("Mismatch between number of samples for index encoding (%d)"
         " vs. binary encoding (%d)") % (
            X_binary.shape[0],
            X_index.shape[0])

    if flatten_binary_encoding:
        # collapse 3D input into 2D matrix
        n_binary_features = peptide_length * n_indices
        X_binary = X_binary.reshape((n_samples, n_binary_features))

    # easier to work with counts when they're an array instead of list
    counts = np.array(counts)

    Y = np.array([peptide_to_affinity_dict[p] for p in original_peptides])
    assert n_samples == len(Y), \
        "Mismatch between # peptides %d and # regression outputs %d" % (
            n_samples, len(Y))
    return (kmer_peptides, original_peptides, counts, X_index, X_binary, Y)

def create_allele_data_from_peptide_to_ic50_dict(
        peptide_to_ic50_dict,
        max_ic50=MAX_IC50,
        kmer_length=9,
        flatten_binary_encoding=True):
    """
    Parameters
    ----------
    peptide_to_ic50_dict : dict
        Dictionary mapping peptides of different lengths to IC50 binding
        affinity values.

    max_ic50 : float
        Maximum IC50 value used as the cutoff for affinity of 0.0 when
        transforming from IC50 to regression targets.

    kmer_length : int
        What length substrings will be fed to a fixed-length predictor?

    flatten_binary_encoding : bool
        Should hotshot encodings of amino acid inputs be flattened into a 1D
        vector or have two dimensions (where the first represents position)?

    Return an AlleleData object.
    """
    Y_dict = {
        peptide: ic50_to_regression_target(ic50, max_ic50)
        for (peptide, ic50)
        in peptide_to_ic50_dict.items()
    }
    (kmer_peptides, original_peptides, counts, X_index, X_binary, Y_kmer) = \
        encode_peptide_to_affinity_dict(
            Y_dict,
            peptide_length=kmer_length,
            flatten_binary_encoding=flatten_binary_encoding)

    ic50_array = np.array([peptide_to_ic50_dict[p] for p in original_peptides])
    assert len(kmer_peptides) == len(ic50_array), \
        "Mismatch between # of peptides %d and # IC50 outputs %d" % (
            len(kmer_peptides), len(ic50_array))

    return AlleleData(
        X_index=X_index,
        X_binary=X_binary,
        Y=Y_kmer,
        ic50=ic50_array,
        peptides=np.array(kmer_peptides),
        original_peptides=np.array(original_peptides),
        original_lengths=np.array(
            [len(peptide) for peptide in original_peptides]),
        substring_counts=counts,
        weights=1.0 / counts,
        max_ic50=max_ic50)


def load_allele_datasets(
        filename,
        peptide_length=9,
        use_multiple_peptide_lengths=True,
        max_ic50=MAX_IC50,
        flatten_binary_encoding=True,
        sep=None,
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name=None,
        peptide_length_column_name="peptide_length",
        ic50_column_name="meas",
        only_human=False,
        shuffle=True):
    """
    Loads an IEDB dataset, extracts "hot-shot" encoding of fixed length peptides
    and log-transforms the IC50 measurement. Returns dictionary mapping allele
    names to AlleleData objects (containing fields X, Y, ic50)

    Parameters
    ----------
    filename : str
        TSV filename with columns:
            - 'species'
            - 'mhc'
            - 'peptide_length'
            - 'sequence'
            - 'meas'

    peptide_length : int
        Which length peptides to use (default=9)

    use_multiple_peptide_lengths : bool
        If a peptide is shorter than `peptide_length`, expand it into many
        peptides of the appropriate length by inserting all combinations of
        amino acids. Similarly, if a peptide is longer than `peptide_length`,
        shorten it by deleting stretches of contiguous amino acids at all
        peptide positions.

    max_ic50 : float
        Treat IC50 scores above this value as all equally bad
        (transform them to 0.0 in the rescaled output)

    flatten_binary_encoding : bool
        If False, returns a (n_samples, peptide_length, 20) matrix, otherwise
        returns the 2D flattened version of the same data.

    sep : str, optional
        Separator in CSV file, default is to let Pandas infer

    peptide_column_name : str, optional
        Default behavior is to try {"sequence", "peptide", "peptide_sequence"}

    only_human : bool
        Only load entries from human MHC alleles
    """
    df, peptide_column_name = load_dataframe(
        filename=filename,
        max_ic50=max_ic50,
        sep=sep,
        species_column_name=species_column_name,
        allele_column_name=allele_column_name,
        peptide_column_name=peptide_column_name,
        filter_peptide_length=None if use_multiple_peptide_lengths else peptide_length,
        ic50_column_name=ic50_column_name,
        only_human=only_human)

    allele_groups = {}
    for allele, group in df.groupby(allele_column_name):
        assert allele not in allele_groups, \
            "Duplicate datasets for %s" % allele

        raw_peptides = group[peptide_column_name]

        # filter lengths in case user wants to drop peptides that are longer
        # or shorter than the desired fixed length
        if not use_multiple_peptide_lengths:
            drop_mask = raw_peptides.str.len() != peptide_length
            group = group[~drop_mask]
            raw_peptides = raw_peptides[~drop_mask]

        # convert from a Pandas column to a list, since that's easier to
        # interact with later
        raw_peptides = list(raw_peptides)

        # create dictionaries of outputs from which we can look up values
        # after peptides have been expanded
        ic50_dict = {
            peptide: ic50
            for (peptide, ic50)
            in zip(raw_peptides, group[ic50_column_name])
        }
        allele_groups[allele] = create_allele_data_from_peptide_to_ic50_dict(
            ic50_dict,
            max_ic50=max_ic50)
    return allele_groups


def collapse_multiple_peptide_entries(allele_datasets):
    """
    If an (allele, peptide) pair occurs multiple times then reduce it
    to a single entry by taking the weighted average of affinity values.

    Returns a dictionary of alleles, each of which maps to a dictionary of
    peptides that map to affinity values.
    """
    allele_to_peptide_to_affinity = {}
    for (allele, dataset) in allele_datasets.items():
        multiple_affinities = defaultdict(list)
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

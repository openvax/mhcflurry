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

from .amino_acid import common_amino_acids
from .peptide_encoding import (
    indices_to_hotshot_encoding,
    fixed_length_index_encoding,
    check_valid_index_encoding_array,
)


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
        only_human=False):
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

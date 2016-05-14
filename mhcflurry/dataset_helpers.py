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

from typechecks import require_instance
import numpy as np
import pandas as pd

from .common import normalize_allele_name
from .peptide_encoding import (
    indices_to_hotshot_encoding,
    fixed_length_index_encoding,
    check_valid_index_encoding_array,
)

def check_pMHC_affinity_arrays(alleles, peptides, affinities, sample_weights):
    """
    Make sure that we have the same number of peptides, affinity values,
    and weights.
    """
    require_instance(alleles, np.ndarray)
    require_instance(peptides, np.ndarray)
    require_instance(affinities, np.ndarray)
    require_instance(sample_weights, np.ndarray)

    if len(alleles.shape) != 1:
        raise ValueError("Expected 1d array of alleles but got shape %s" % (
            alleles.shape,))
    if len(peptides.shape) != 1:
        raise ValueError("Expected 1d array of peptides but got shape %s" % (
            peptides.shape,))
    if len(affinities.shape) != 1:
        raise ValueError("Expected 1d array of affinity values but got shape %s" % (alleles.shape,))
    if len(sample_weights.shape) != 1:
        raise ValueError("Expected 1d array of sample weights but got shape %s" % (
            sample_weights.shape,))

    n = len(alleles)
    if len(peptides) != n:
        raise ValueError("Expected %d peptides but got %d" % (n, len(peptides)))
    if len(affinities) != n:
        raise ValueError("Expected %d affinity values but got %d" % (n, len(affinities)))
    if len(sample_weights) != n:
        raise ValueError("Expected %d sample weights but got %d" % (n, len(sample_weights)))


def prepare_pMHC_affinity_arrays(alleles, peptides, affinities, sample_weights=None):
    """
    Converts every sequence to an array and if sample_weights is missing then
    create an array of ones.
    """
    alleles = np.asarray(alleles)
    peptides = np.asarray(peptides)
    affinities = np.asarray(affinities)
    if sample_weights is None:
        sample_weights = np.ones(len(alleles), dtype=float)
    check_pMHC_affinity_arrays(
        alleles=alleles,
        peptides=peptides,
        affinities=affinities,
        sample_weights=sample_weights)
    return alleles, peptides, affinities, sample_weights


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
        allele_column_name=None,
        peptide_column_name=None,
        affinity_column_name=None,
        filter_peptide_length=None,
        normalize_allele_names=True):
    """
    Load a dataframe of peptide-MHC affinity measurements

    filename : str
        TSV filename with columns:
            - 'species'
            - 'mhc'
            - 'peptide_length'
            - 'sequence'
            - 'meas'

    sep : str, optional
        Separator in CSV file, default is to let Pandas infer

    allele_column_name : str, optional
        Default behavior is to try {"mhc", "allele", "hla"}

    peptide_column_name : str, optional
        Default behavior is to try  {"sequence", "peptide", "peptide_sequence"}

    affinity_column_name : str, optional
        Default behavior is to try {"meas", "ic50", "affinity", "aff"}

    filter_peptide_length : int, optional
        Which length peptides to use (default=load all lengths)

    normalize_allele_names : bool
        Normalize MHC names or leave them alone

    Returns:
        - DataFrame
        - peptide column name
        - allele column name
        - affinity column name
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

    columns = set(df.keys())

    if allele_column_name is None:
        for candidate in ["mhc", "allele", "hla"]:
            if candidate in columns:
                allele_column_name = candidate
                break
        if allele_column_name is None:
            raise ValueError(
                "Couldn't find alleles, available columns: %s" % (
                    columns,))

    if peptide_column_name is None:
        for candidate in ["sequence", "peptide", "peptide_sequence"]:
            if candidate in columns:
                peptide_column_name = candidate
                break
        if peptide_column_name is None:
            raise ValueError(
                "Couldn't find peptides, available columns: %s" % (
                    columns,))

    if affinity_column_name is None:
        for candidate in ["meas", "ic50", "affinity"]:
            if candidate in columns:
                affinity_column_name = candidate
                break
        if affinity_column_name is None:
            raise ValueError(
                "Couldn't find affinity values, available columns: %s" % (
                    columns,))
    if filter_peptide_length:
        length_mask = df[peptide_column_name].str.len() == filter_peptide_length
        df = df[length_mask]
    df[allele_column_name] = df[allele_column_name].map(normalize_allele_name)
    return df, allele_column_name, peptide_column_name, affinity_column_name


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

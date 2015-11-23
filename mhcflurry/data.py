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
from collections import namedtuple

import pandas as pd
import numpy as np

from .common import normalize_allele_name
from .peptide_encoding import (
    fixed_length_index_encoding,
    indices_to_hotshot_encoding,
)

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
        peptide_length=None,
        max_ic50=50000.0,
        sep=None,
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name=None,
        peptide_length_column_name="peptide_length",
        ic50_column_name="meas",
        only_human=True):
    """
    Load a dataframe of peptide-MHC affinity measurements

    filename : str
        TSV filename with columns:
            - 'species'
            - 'mhc'
            - 'peptide_length'
            - 'sequence'
            - 'meas'

     peptide_length : int, optional
        Which length peptides to use (default=load all lengths)

    max_ic50 : float
        Treat IC50 scores above this value as all equally bad
        (transform them to 0.0 in the regression output)

    sep : str, optional
        Separator in CSV file, default is to let Pandas infer

    peptide_column_name : str, optional
        Default behavior is to try  {"sequence", "peptide", "peptide_sequence"}

    only_human : bool
        Only load entries from human MHC alleles

    Returns DataFrame augmented with extra columns:
        - "log_ic50" : log(ic50) / log(max_ic50)
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
    if peptide_length is not None:
        length_mask = df[peptide_length_column_name] == peptide_length
        df = df[length_mask]

    df[allele_column_name] = df[allele_column_name].map(normalize_allele_name)
    ic50 = np.array(df[ic50_column_name])
    log_ic50 = np.log(ic50) / np.log(max_ic50)
    df["log_ic50"] = log_ic50
    regression_output = 1.0 - log_ic50
    # clamp to values between 0, 1
    regression_output = np.maximum(regression_output, 0.0)
    regression_output = np.minimum(regression_output, 1.0)
    df["regression_output"] = regression_output
    return df, peptide_column_name


def load_allele_dicts(
        filename,
        max_ic50=50000.0,
        regression_output=False,
        sep=None,
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name=None,
        peptide_length_column_name="peptide_length",
        ic50_column_name="meas",
        only_human=True):
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
        peptide_length_column_name=peptide_length_column_name,
        ic50_column_name=ic50_column_name,
        only_human=only_human)
    return {
        allele_name: {
            row[peptide_column_name]: (
                row["regression_output"]
                if regression_output
                else row[ic50_column_name]
            )
            for (_, row) in group.iterrows()
        }
        for (allele_name, group) in binding_df.groupby(allele_column_name)
    }


def load_allele_datasets(
        filename,
        peptide_length=9,
        use_multiple_peptide_lengths=True,
        max_ic50=50000.0,
        flatten_binary_encoding=True,
        sep=None,
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name=None,
        peptide_length_column_name="peptide_length",
        ic50_column_name="meas",
        only_human=True):
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
        peptide_length=peptide_length,
        species_column_name=species_column_name,
        allele_column_name=allele_column_name,
        peptide_column_name=peptide_column_name,
        peptide_length_column_name=peptide_length_column_name,
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
        # convert numberical values from a Pandas column to arrays
        ic50 = np.array(group[ic50_column_name])
        Y = np.array(group["regression_output"])

        X_index, original_peptides, counts = fixed_length_index_encoding(
            peptides=raw_peptides,
            desired_length=peptide_length)

        X_binary = indices_to_hotshot_encoding(X_index, n_indices=20)
        assert X_binary.shape[0] == X_index.shape[0], \
            ("Mismatch between number of samples for index encoding (%d)"
             " vs. binary encoding (%d)") % (
                X_binary.shape[0],
                X_index.shape[0])
        n_samples = X_binary.shape[0]

        if flatten_binary_encoding:
            # collapse 3D input into 2D matrix
            n_binary_features = peptide_length * 20
            X_binary = X_binary.reshape((n_samples, n_binary_features))

        # easier to work with counts when they're an array instead of list
        counts = np.array(counts)

        allele_groups[allele] = AlleleData(
            X_index=X_index,
            X_binary=X_binary,
            Y=Y,
            ic50=ic50,
            peptides=raw_peptides,
            original_peptides=original_peptides,
            original_lengths=[len(peptide) for peptide in original_peptides],
            substring_counts=counts,
            weights=1.0 / counts)
    return allele_groups

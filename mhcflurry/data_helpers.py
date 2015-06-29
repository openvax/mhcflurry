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

from collections import namedtuple
import pandas as pd
import numpy as np

from pepdata.amino_acid import amino_acid_letter_indices

AlleleData = namedtuple("AlleleData", "X Y peptides ic50")

def hotshot_encoding(peptides, peptide_length):
    """
    Encode a set of equal length peptides as a binary matrix,
    where each letter is transformed into a length 20 vector with a single
    element that is 1 (and the others are 0).
    """
    shape = (len(peptides), peptide_length, 20)
    X = np.zeros(shape, dtype=bool)
    for i, peptide in enumerate(peptides):
        for j, amino_acid in enumerate(peptide):
            k = amino_acid_letter_indices[amino_acid]
            X[i, j, k] = 1
    return X

def index_encoding(peptides, peptide_length):
    """
    Encode a set of equal length peptides as a vector of their
    amino acid indices.
    """
    X = np.zeros((len(peptides), peptide_length), dtype=int)
    for i, peptide in enumerate(peptides):
        for j, amino_acid in enumerate(peptide):
            X[i, j] = amino_acid_letter_indices[amino_acid]
    return X

def normalize_allele_name(allele_name):
    allele_name = allele_name.upper()
    # old school HLA-C serotypes look like "Cw"
    allele_name = allele_name.replace("CW", "C")
    patterns = [
        "HLA-",
        "-",
        "*",
        ":"
    ]
    for pattern in patterns:
        allele_name = allele_name.replace(pattern, "")
    return allele_name

def load_data(
        filename,
        peptide_length=9,
        max_ic50=5000.0,
        binary_encoding=True,
        flatten_binary_encoding=True,
        sep="\t",
        species_column_name="species",
        allele_column_name="mhc",
        peptide_column_name="sequence",
        peptide_length_column_name="peptide_length",
        ic50_column_name="meas"):
    """
    Loads an IEDB dataset, extracts "hot-shot" encoding of fixed length peptides
    and log-transforms the IC50 measurement. Returns (groups, df),
    where groups is a dictionary mapping allele names to (X, Y, original_ic50)
    and df is the full dataframe.

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

    max_ic50 : float
        Treat IC50 scores above this value as all equally bad
        (transform them to 0.0 in the rescaled output)

    binary_encoding : bool
        Encode amino acids of each peptide as indices or binary vectors

    flatten_features : bool
        If False, returns a (n_samples, peptide_length, 20) matrix, otherwise
        returns the 2D flattened version of the same data.
    """
    df = pd.read_csv(filename, sep=sep)
    human_mask = df[species_column_name] == "human"
    length_mask = df[peptide_length_column_name] == peptide_length
    df = df[human_mask & length_mask]
    allele_groups = {}
    for allele, group in df.groupby(allele_column_name):
        allele = normalize_allele_name(allele)
        ic50 = np.array(group[ic50_column_name])
        Y = np.maximum(0.0, 1.0 - np.log(ic50) / np.log(max_ic50))
        peptides = list(group[peptide_column_name])
        if binary_encoding:
            X = hotshot_encoding(peptides, peptide_length=peptide_length)
            if flatten_binary_encoding:
                # collapse 3D input into 2D matrix
                X = X.reshape((X.shape[0], peptide_length * 20))
        else:
            X = index_encoding(peptides, peptide_length=peptide_length)
        allele_groups[allele] = AlleleData(
            X=X,
            Y=Y,
            ic50=ic50,
            peptides=peptides)
    return allele_groups, df


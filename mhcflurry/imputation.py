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
from collections import defaultdict
import numpy as np

def prune_dense_matrix_and_labels(
        X,
        peptide_list,
        allele_list,
        min_observations_per_peptide=1,
        min_observations_per_allele=1):
    """
    Filter the dense matrix of pMHC binding affinities according to
    the given minimum number of row/column observations.

    Parameters
    ----------
    X : numpy.ndarray
        Incomplete dense matrix of pMHC affinity with n_peptides rows and
        n_alleles columns.

    peptide_list : list of str
        Expected to have n_peptides entries

    allele_list : list of str
        Expected to have n_alleles entries

    min_observations_per_peptide : int
        Drop peptide rows with fewer than this number of observed values.

    min_observations_per_allele : int
        Drop allele columns with fewer than this number of observed values.
    """
    observed_mask = np.isfinite(X)
    n_observed_per_peptide = observed_mask.sum(axis=1)
    too_few_peptide_observations = (
        n_observed_per_peptide < min_observations_per_peptide)
    if too_few_peptide_observations.any():
        drop_peptide_indices = np.where(too_few_peptide_observations)[0]
        keep_peptide_indices = np.where(~too_few_peptide_observations)[0]
        print("Dropping %d peptides with <%d observations" % (
            len(drop_peptide_indices),
            min_observations_per_peptide))
        X = X[keep_peptide_indices]
        observed_mask = observed_mask[keep_peptide_indices]
        peptide_list = [peptide_list[i] for i in keep_peptide_indices]

    n_observed_per_allele = observed_mask.sum(axis=0)
    too_few_allele_observations = (
        n_observed_per_allele < min_observations_per_peptide)
    if too_few_peptide_observations.any():
        drop_allele_indices = np.where(too_few_allele_observations)[0]
        keep_allele_indices = np.where(~too_few_allele_observations)[0]
        print("Dropping %d alleles with <%d observations: %s" % (
            len(drop_allele_indices),
            min_observations_per_allele,
            [allele_list[i] for i in drop_allele_indices]))
        X = X[:, keep_allele_indices]
        observed_mask = observed_mask[:, keep_allele_indices]
        allele_list = [allele_list[i] for i in keep_allele_indices]
    return X, peptide_list, allele_list

def create_incompelte_dense_pMHC_matrix(
        allele_data_dict,
        min_observations_per_peptide=1,
        min_observations_per_allele=1):
    """
    Given a dictionary mapping each allele name onto an AlleleData object,
    returns a tuple with a dense matrix of affinities, a list of peptide labels
    for each row and a list of allele labels for each column.

    Parameters
    ----------
    allele_data_dict : dict
        Dictionary mapping allele names to AlleleData objects

    imputer : object
        Expected to have a method imputer.complete(X) which takes an array
        with missing entries marked by NaN and returns a completed array.

    min_observations_per_peptide : int
        Drop peptide rows with fewer than this number of observed values.

    min_observations_per_allele : int
        Drop allele columns with fewer than this number of observed values.
    """
    peptide_to_allele_to_affinity_dict = defaultdict(dict)
    for (allele_name, allele_data) in allele_data_dict.items():
        for peptide, affinity in zip(
                allele_data.original_peptides,
                allele_data.Y):
            if allele_name not in peptide_to_allele_to_affinity_dict[peptide]:
                peptide_to_allele_to_affinity_dict[peptide][allele_name] = affinity

    n_binding_values = sum(
        len(allele_dict)
        for allele_dict in
        allele_to_peptide_to_affinity.values()
    )
    print("Collected %d binding values for %d alleles" % (
        n_binding_values,
        len(peptide_to_allele_to_affinity_dict)))
    X, peptide_list, allele_list = \
        dense_matrix_from_nested_dictionary(peptide_to_allele_to_affinity)
    return prune_data(
        X,
        peptide_list,
        allele_list,
        min_observations_per_peptide=min_observations_per_peptide,
        min_observations_per_allele=min_observations_per_allele)


def create_imputed_dataset(
        allele_data_dict,
        imputer,
        min_observations_per_peptide=1,
        min_observations_per_allele=1):
    """
    Parameters
    ----------
    allele_data_dict : dict
        Dictionary mapping allele names to AlleleData objects

    imputer : object
        Expected to have a method imputer.complete(X) which takes an array
        with missing entries marked by NaN and returns a completed array.

    min_observations_per_peptide : int
        Drop peptide rows with fewer than this number of observed values.

    min_observations_per_allele : int
        Drop allele columns with fewer than this number of observed values.
    """
    X_incomplete, peptide_list, allele_list = create_incompelte_dense_pMHC_matrix(
        allele_data_dict=allele_data_dict,
        min_observations_per_peptide=min_observations_per_peptide,
        min_observations_per_allele=min_observations_per_allele)
    X_complete = impute.complete(X_incomplete)

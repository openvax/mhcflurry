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
from collections import defaultdict
import logging

import numpy as np
from fancyimpute.knn import KNN
from fancyimpute.iterative_svd import IterativeSVD
from fancyimpute.simple_fill import SimpleFill
from fancyimpute.soft_impute import SoftImpute
from fancyimpute.mice import MICE

from .dataset import Dataset

def _check_dense_pMHC_array(X, peptide_list, allele_list):
    if len(peptide_list) != len(set(peptide_list)):
        raise ValueError("Duplicate peptides detected in peptide list")
    if len(allele_list) != len(set(allele_list)):
        raise ValueError("Duplicate alleles detected in allele list")
    n_rows, n_cols = X.shape
    if n_rows != len(peptide_list):
        raise ValueError(
            "Expected dense array with shape %s to have %d rows" % (
                X.shape, len(peptide_list)))
    if n_cols != len(allele_list):
        raise ValueError(
            "Expected dense array with shape %s to have %d columns" % (
                X.shape, len(allele_list)))

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
    _check_dense_pMHC_array(X, peptide_list, allele_list)
    return X, peptide_list, allele_list

def dense_pMHC_matrix_to_nested_dict(X, peptide_list, allele_list):
    """
    Converts a dense matrix of (n_peptides, n_alleles) floats to a nested
    dictionary from allele -> peptide -> affinity.
    """
    allele_to_peptide_to_ic50_dict = defaultdict(dict)
    for row_index, peptide in enumerate(peptide_list):
        for column_index, allele_name in enumerate(allele_list):
            affinity = X[row_index, column_index]
            if np.isfinite(affinity):
                allele_to_peptide_to_ic50_dict[allele_name][peptide] = affinity
    return allele_to_peptide_to_ic50_dict

def create_imputed_datasets(
        dataset,
        imputer,
        min_observations_per_peptide=2,
        min_observations_per_allele=2):
    """
    Parameters
    ----------
    allele_data_dict : Dataset

    imputer : object
        Expected to have a method imputer.complete(X) which takes an array
        with missing entries marked by NaN and returns a completed array.

    min_observations_per_peptide : int
        Drop peptide rows with fewer than this number of observed values.

    min_observations_per_allele : int
        Drop allele columns with fewer than this number of observed values.

    Returns dictionary mapping allele names to AlleleData objects containing
    imputed affinity values.
    """
    assert isinstance(dataset, Dataset), "Expected Dataset, got %s" % (type(dataset),)
    X_incomplete, peptide_list, allele_list = dataset.to_dense_pMHC_affinity_matrix()

    _check_dense_pMHC_array(X_incomplete, peptide_list, allele_list)

    # drop alleles and peptides with small amounts of data
    X_incomplete, peptide_list, allele_list = prune_dense_matrix_and_labels(
        X_incomplete, peptide_list, allele_list,
        min_observations_per_peptide=min_observations_per_peptide,
        min_observations_per_allele=min_observations_per_allele)
    X_incomplete_log = np.log(X_incomplete)

    if np.isnan(X_incomplete).sum() == 0:
        # if all entries in the matrix are already filled in then don't
        # try using an imputation algorithm since it might raise an
        # exception.
        logging.warn("No missing values, using original data instead of imputation")
        X_complete_log = X_incomplete_log
    else:
        X_complete_log = imputer.complete(X_incomplete_log)
    X_complete = np.exp(X_complete_log)
    allele_to_peptide_to_affinity_dict = dense_pMHC_matrix_to_nested_dict(
        X=X_complete,
        peptide_list=peptide_list,
        allele_list=allele_list)
    return Dataset.from_nested_dictionary(allele_to_peptide_to_affinity_dict)

def imputer_from_name(imputation_method_name, **kwargs):
    """
    Helper function for constructing an imputation object from a name given
    typically from a commandline argument.
    """
    imputation_method_name = imputation_method_name.strip().lower()
    if imputation_method_name == "mice":
        kwargs["n_burn_in"] = kwargs.get("n_burn_in", 5)
        kwargs["n_imputations"] = kwargs.get("n_imputations", 25)
        kwargs["min_value"] = kwargs.get("min_value", 0)
        kwargs["max_value"] = kwargs.get("max_value", 1)
        return MICE(**kwargs)
    elif imputation_method_name == "knn":
        kwargs["k"] = kwargs.get("k", 3)
        kwargs["orientation"] = kwargs.get("orientation", "columns")
        kwargs["print_interval"] = kwargs.get("print_interval", 10)
        return KNN(**kwargs)
    elif imputation_method_name == "svd":
        kwargs["rank"] = kwargs.get("rank", 10)
        return IterativeSVD(**kwargs)
    elif imputation_method_name == "svt" or imputation_method_name == "softimpute":
        return SoftImpute(**kwargs)
    elif imputation_method_name == "mean":
        return SimpleFill("mean", **kwargs)
    elif imputation_method_name == "none":
        return None
    else:
        raise ValueError("Invalid imputation method: %s" % imputation_method_name)

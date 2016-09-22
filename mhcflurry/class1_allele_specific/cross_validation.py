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
import collections
import logging


import pepdata

from .train import impute_and_select_allele, AlleleSpecificTrainTestFold
from ..parallelism import get_default_executor

gbmr4_transformer = pepdata.reduced_alphabet.make_alphabet_transformer("gbmr4")


def default_projector(peptide):
    """
    Given a peptide, return a list of projections for it. The projections are:
        - the gbmr4 reduced representation
        - for all positions in the peptide, the peptide with a "." replacing
          the residue at that position

    Peptides with overlapping projections are considered similar when doing
    cross validation.

    Parameters
    ----------
    peptide : string

    Returns
    ----------
    string list
    """
    def projections(peptide, edit_distance=1):
        if edit_distance == 0:
            return set([peptide])
        return set.union(*[
            projections(p, edit_distance - 1)
            for p in (
                peptide[0:i] + "." + peptide[(i + 1):]
                for i in range(len(peptide)))
        ])
    return sorted(projections(peptide)) + [gbmr4_transformer(peptide)]


def similar_peptides(set1, set2, projector=default_projector):
    """
    Given two sets of peptides, return a list of the peptides whose reduced
    representations are found in both sets.

    Parameters
    ----------
    projector : (string -> string) or (string -> string list)
        Function giving projection(s) of a peptide

    Returns
    ----------
    string list
    """
    result = collections.defaultdict(lambda: ([], []))
    for (index, peptides) in enumerate([set1, set2]):
        for peptide in peptides:
            projections = projector(peptide)
            if not isinstance(projections, list):
                projections = [projections]
            for projection in projections:
                result[projection][index].append(peptide)

    common = set()
    for (peptides1, peptides2) in result.values():
        if peptides1 and peptides2:
            common.update(peptides1 + peptides2)

    return sorted(common)


def cross_validation_folds(
        train_data,
        alleles=None,
        n_folds=3,
        drop_similar_peptides=False,
        imputer=None,
        impute_kwargs={
            'min_observations_per_peptide': 2,
            'min_observations_per_allele': 2,
        },
        executor=None):
    '''
    Split a Dataset into n_folds cross validation folds for each allele,
    optionally performing imputation.

    Parameters
    -----------
    train_data : mhcflurry.Dataset

    alleles : string list, optional
        Alleles to run cross validation on. Default: all alleles in
        train_data.

    n_folds : int, optional
        Number of cross validation folds for each allele.

    drop_similar_peptides : boolean, optional
        For each fold, remove peptides from the test data that are similar
        to peptides in the train data. Similarity is defined as in the
        similar_peptides function.

    imputer : fancyimpute.Solver, optional
        Imputer to use. If not specified, no imputation is done.

    impute_kwargs : dict, optional
        Additional kwargs to pass to mhcflurry.Dataset.impute_missing_values.

    n_jobs : integer, optional
        The number of jobs to run in parallel. If -1, then the number of jobs
        is set to the number of cores.

    Returns
    -----------
    list of AlleleSpecificTrainTestFold of length num alleles * n_folds

    '''
    if executor is None:
        executor = get_default_executor()

    if alleles is None:
        alleles = train_data.unique_alleles()

    result_folds = []
    for allele in alleles:
        logging.info("Allele: %s" % allele)
        cv_iter = train_data.cross_validation_iterator(
            allele, n_folds=n_folds, shuffle=True)
        for (all_allele_train_split, full_test_split) in cv_iter:
            peptides_to_remove = []
            if drop_similar_peptides:
                peptides_to_remove = similar_peptides(
                    all_allele_train_split.get_allele(allele).peptides,
                    full_test_split.get_allele(allele).peptides
                )

            if peptides_to_remove:
                test_split = full_test_split.drop_allele_peptide_lists(
                    [allele] * len(peptides_to_remove),
                    peptides_to_remove)
                logging.info(
                    "After dropping similar peptides, test size %d->%d" % (
                        len(full_test_split), len(test_split)))
            else:
                test_split = full_test_split

            if imputer is not None:
                imputation_future = executor.submit(
                    impute_and_select_allele,
                    all_allele_train_split,
                    imputer=imputer,
                    allele=allele,
                    **impute_kwargs)
            else:
                imputation_future = None

            train_split = all_allele_train_split.get_allele(allele)
            fold = AlleleSpecificTrainTestFold(
                allele=allele,
                train=train_split,
                imputed_train=imputation_future,
                test=test_split)
            result_folds.append(fold)

    return [
        result_fold._replace(imputed_train=(
            result_fold.imputed_train.result()
            if result_fold.imputed_train is not None
            else None))
        for result_fold in result_folds
    ]

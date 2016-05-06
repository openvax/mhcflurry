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

from .regression_target import regression_target_to_ic50, MAX_IC50
from .dataset import Dataset


class IC50PredictorBase(object):
    """
    Base class for all mhcflurry predictors which predict IC50 values
    (using any representation of peptides)
    """
    def __init__(
            self,
            name,
            verbose,
            max_ic50=MAX_IC50):
        self.name = name
        self.max_ic50 = max_ic50
        self.verbose = verbose

    def __repr__(self):
        return "%s(name=%s, max_ic50=%f)" % (
            self.__class__.__name__,
            self.name,
            self.model,
            self.max_ic50)

    def __str__(self):
        return repr(self)

    def predict_scores(self, peptides, combine_fn=np.mean):
        raise NotImplementedError(
            "predict_scores expected to be implemented in sub-class")

    def predict(self, peptides):
        """
        Predict IC50 affinities for peptides of any length
        """
        scores = self.predict_scores(peptides)
        return regression_target_to_ic50(scores, max_ic50=self.max_ic50)

    def fit_dictionary(self, peptide_to_ic50_dict, **kwargs):
        """
        Fit the model parameters using the given peptide->IC50 dictionary,
        all samples are given the same weight.

        Parameters
        ----------
        peptide_to_ic50_dict : dict
            Dictionary that maps peptides to IC50 values.
        """
        dataset = Dataset.from_peptide_to_affinity_dictionary(
            allele_name=self.name,
            peptide_to_affinity_dict=peptide_to_ic50_dict)
        return self.fit_dataset(dataset, **kwargs)

    def fit_sequences(
            self,
            peptides,
            affinities,
            sample_weights=None,
            alleles=None, **kwargs):
        if alleles is None:
            alleles = [self.name] * len(peptides)
        dataset = Dataset.from_sequences(
            alleles=alleles,
            peptides=peptides,
            affinities=affinities,
            sample_weights=sample_weights)
        return self.fit_dataset(dataset, **kwargs)

    def fit_dataset(self, dataset, pretraining_dataset=None, *args, **kwargs):
        raise NotImplementedError(
            "fit_dataset expected to be implemented in sub-class")

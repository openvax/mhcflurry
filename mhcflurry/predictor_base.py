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

import numpy as np

from itertools import groupby

from .peptide_encoding import fixed_length_from_many_peptides
from .amino_acid import (
    common_amino_acid_letters,
    amino_acids_with_unknown,
    common_amino_acids
)


class PredictorBase(object):
    """
    Base class for all mhcflurry predictors (including the Ensemble class)
    """

    def __init__(
            self,
            name,
            max_ic50,
            allow_unknown_amino_acids,
            verbose):
        self.name = name
        self.max_ic50 = max_ic50
        self.allow_unknown_amino_acids = allow_unknown_amino_acids
        self.verbose = verbose

    def log_to_ic50(self, log_value):
        """
        Convert neural network output to IC50 values between 0.0 and
        self.max_ic50 (typically 5000, 20000 or 50000)
        """
        return self.max_ic50 ** (1.0 - log_value)

    def encode_9mer_peptides(self, peptides):
        if self.allow_unknown_amino_acids:
            return amino_acids_with_unknown.index_encoding(peptides, 9)
        else:
            return common_amino_acids.index_encoding(peptides, 9)

    def predict_9mer_peptides_ic50(self, peptides):
        return self.log_to_ic50(self.predict_9mer_peptides(peptides))

    def predict_peptides_ic50(self, peptides):
        """
        Predict IC50 affinities for peptides of any length
        """
        return self.log_to_ic50(
            self.predict_peptides(peptides))

    def predict_peptides(
            self,
            peptides,
            combine_fn=np.mean):
        """
        Given a list of peptides of any length, returns an array of predicted
        normalized affinity values. Unlike IC50, a higher value here
        means a stronger affinity. Peptides of lengths other than 9 are
        transformed into a set of 9mers either by deleting or inserting
        amino acid characters. The prediction for a single peptide will be
        the average of expanded 9mers.
        """
        results_dict = {}
        for length, group_peptides in groupby(peptides, lambda x: len(x)):
            group_peptides = list(group_peptides)
            expanded_peptides, _, _ = fixed_length_from_many_peptides(
                peptides=group_peptides,
                desired_length=9,
                insert_amino_acid_letters=(
                    ["X"] if self.allow_unknown_amino_acids
                    else common_amino_acid_letters))

            n_group = len(group_peptides)
            n_expanded = len(expanded_peptides)
            expansion_factor = int(n_expanded / n_group)
            raw_y = self.predict_9mer_peptides(expanded_peptides)
            if expansion_factor == 1:
                log_ic50s = raw_y
            else:
                # if peptides were a different length than the predictor's
                # expected input length, then let's take the median prediction
                # of each expanded peptide set
                log_ic50s = np.zeros(n_group)
                # take the median of each group of log(IC50) values
                for i in range(n_group):
                    start = i * expansion_factor
                    end = (i + 1) * expansion_factor
                    log_ic50s[i] = combine_fn(raw_y[start:end])
            assert len(group_peptides) == len(log_ic50s)
            for peptide, log_ic50 in zip(group_peptides, log_ic50s):
                results_dict[peptide] = log_ic50
        return np.array([results_dict[p] for p in peptides])

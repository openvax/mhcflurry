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

from collections import defaultdict

import numpy as np
from six import string_types

from .peptide_encoding import fixed_length_index_encoding
from .amino_acid import (
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

    def encode_peptides(self, peptides):
        """
        Parameters
        ----------
        peptides : str list
            Peptide strings of any length

        Encode peptides of any length into fixed length vectors.
        Returns 2d array of encoded peptides and 1d array indicating the
        original peptide index for each row.
        """
        indices = []
        encoded_matrices = []
        for i, peptide in enumerate(peptides):
            matrix, _, _, _ = fixed_length_index_encoding(
                peptides=[peptide],
                desired_length=9,
                allow_unknown_amino_acids=self.allow_unknown_amino_acids)
            encoded_matrices.append(matrix)
            indices.extend([i] * len(matrix))
        combined_matrix = np.concatenate(encoded_matrices)
        index_array = np.array(indices)
        expected_shape = (len(index_array), 9)
        assert combined_matrix.shape == expected_shape, \
            "Expected shape %s but got %s" % (expected_shape, combined_matrix.shape)
        return combined_matrix, index_array

    def predict_9mer_peptides(self, peptides):
        """
        Predict binding affinity for 9mer peptides
        """
        if any(len(peptide) != 9 for peptide in peptides):
            raise ValueError("Can only predict 9mer peptides")
        X, _ = self.encode_peptides(peptides)
        return self.predict(X)

    def predict_9mer_peptides_ic50(self, peptides):
        return self.log_to_ic50(self.predict_9mer_peptides(peptides))

    def predict_peptides_ic50(self, peptides):
        """
        Predict IC50 affinities for peptides of any length
        """
        return self.log_to_ic50(
            self.predict_peptides(peptides))

    def predict(self, X):
        raise ValueError("Method 'predict' not yet implemented for %s!" % (
            self.__class__.__name__,))

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
        if isinstance(peptides, string_types):
            raise TypeError("Input must be a list of peptides, not %s : %s" % (
                peptides, type(peptides)))

        input_matrix, original_peptide_indices = self.encode_peptides(peptides)
        # non-9mer peptides get multiple predictions, which are then combined
        # with the combine_fn argument
        multiple_predictions_dict = defaultdict(list)
        fixed_length_predictions = self.predict(input_matrix)
        for i, yi in enumerate(fixed_length_predictions):
            original_peptide_index = original_peptide_indices[i]
            original_peptide = peptides[original_peptide_index]
            multiple_predictions_dict[original_peptide].append(yi)

        combined_predictions_dict = {
            p: combine_fn(ys)
            for (p, ys) in multiple_predictions_dict.items()
        }
        return np.array([combined_predictions_dict[p] for p in peptides])

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

import numpy as np
from six import string_types

from .peptide_encoding import fixed_length_index_encoding
from .amino_acid import (
    amino_acids_with_unknown,
    common_amino_acids
)
from .regression_target import regression_target_to_ic50, MAX_IC50
from .ic50_predictor_base import IC50PredictorBase

class Class1AlleleSpecificKmerIC50PredictorBase(IC50PredictorBase):
    """
    Base class for all mhcflurry predictors which used fixed-length
    k-mer representation of peptides and don't require scanning over
    a longer sequence to find a binding core (like you might for Class II).
    """
    def __init__(
            self,
            name,
            allow_unknown_amino_acids,
            verbose,
            max_ic50=MAX_IC50,
            kmer_size=9):
        self.name = name
        self.max_ic50 = max_ic50
        self.allow_unknown_amino_acids = allow_unknown_amino_acids
        self.verbose = verbose
        self.kmer_size = kmer_size

    def __repr__(self):
        return "%s(name=%s, max_ic50=%f, allow_unknown_amino_acids=%s, kmer_size=%d)" % (
            self.__class__.__name__,
            self.name,
            self.max_ic50,
            self.allow_unknown_amino_acids,
            self.kmer_size)

    def __str__(self):
        return repr(self)

    @property
    def amino_acids(self):
        """
        Amino acid alphabet used for encoding peptides, may include
        "X" if allow_unknown_amino_acids is True.
        """
        if self.allow_unknown_amino_acids:
            return amino_acids_with_unknown
        else:
            return common_amino_acids

    @property
    def max_amino_acid_encoding_value(self):
        return len(self.amino_acids)

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
                desired_length=self.kmer_size,
                allow_unknown_amino_acids=self.allow_unknown_amino_acids)
            encoded_matrices.append(matrix)
            indices.extend([i] * len(matrix))
        combined_matrix = np.concatenate(encoded_matrices)
        index_array = np.array(indices)
        expected_shape = (len(index_array), self.kmer_size)
        assert combined_matrix.shape == expected_shape, \
            "Expected shape %s but got %s" % (expected_shape, combined_matrix.shape)
        return combined_matrix, index_array

    def predict_scores_for_kmer_peptides(self, peptides):
        """
        Predict binding affinity for 9mer peptides
        """
        if any(len(peptide) != self.kmer_size for peptide in peptides):
            raise ValueError("Can only predict 9mer peptides")
        X, _ = self.encode_peptides(peptides)
        return self.predict_scores_for_kmer_encoded_array(X)

    def predict_ic50_for_kmer_peptides(self, peptides):
        scores = self.predict_scores_for_kmer_peptides(peptides)
        return regression_target_to_ic50(scores, max_ic50=self.max_ic50)

    def predict_scores(self, peptides, combine_fn=np.mean):
        """
        Given a list of peptides of any length, returns an array of predicted
        normalized affinity values. Unlike IC50, a higher value here
        means a stronger affinity. Peptides of lengths other than 9 are
        transformed into a set of k-mers either by deleting or inserting
        amino acid characters. The prediction for a single peptide will be
        the average of expanded k-mers.
        """
        if isinstance(peptides, string_types):
            raise TypeError("Input must be a list of peptides, not %s : %s" % (
                peptides, type(peptides)))

        input_matrix, original_peptide_indices = self.encode_peptides(peptides)
        # peptides of lengths other than self.kmer_size get multiple predictions,
        # which are then combined with the combine_fn argument
        multiple_predictions_dict = defaultdict(list)
        fixed_length_predictions = \
            self.predict_scores_for_kmer_encoded_array(input_matrix)
        for i, yi in enumerate(fixed_length_predictions):
            original_peptide_index = original_peptide_indices[i]
            original_peptide = peptides[original_peptide_index]
            multiple_predictions_dict[original_peptide].append(yi)
        combined_predictions_dict = {
            p: combine_fn(ys) if len(ys) > 1 else ys[0]
            for (p, ys) in multiple_predictions_dict.items()
        }
        return np.array([combined_predictions_dict[p] for p in peptides])

    def fit_dataset(self, dataset, pretraining_dataset=None, *args, **kwargs):
        """
        Fit the model parameters on the given training data.

        Parameters
        ----------
        dataset : Dataset

        pretraining_dataset : Dataset

        **kwargs : dict
            Extra arguments are passed on to the fit_encoded_kmer_arrays()
            method.
        """
        X, Y, sample_weights, _ = dataset.kmer_index_encoding(
            kmer_size=self.kmer_size,
            allow_unknown_amino_acids=self.allow_unknown_amino_acids)

        if pretraining_dataset is None:
            X_pretrain = Y_pretrain = sample_weights_pretrain = None
        else:
            X_pretrain, Y_pretrain, sample_weights_pretrain = \
                pretraining_dataset.kmer_index_encoding(
                    kmer_size=self.kmer_size,
                    allow_unknown_amino_acids=self.allow_unknown_amino_acids)
        return self.fit_arrays(
            X=X,
            Y=Y,
            sample_weights=sample_weights,
            X_pretrain=X_pretrain,
            Y_pretrain=Y_pretrain,
            sample_weights_pretrain=sample_weights,
            **kwargs)
